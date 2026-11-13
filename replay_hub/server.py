
import json
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2

# ============================================================
# PATHS AND CONFIG
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.resolve()

CATALOG_FILE = HERE / "rallies.json"
INDEX_FILE = HERE / "index.html"

ASSET_DIR = HERE / "assets"

ASSET_TYPES = {
    "replay.css": "text/css; charset=utf-8",
    "replay.js": "text/javascript; charset=utf-8",
    "library.css": "text/css; charset=utf-8",
    "library.js": "text/javascript; charset=utf-8",
}

CACHE_ROOT = ROOT / "outputs" / "replay_hub_cache"

HOST = "127.0.0.1"
PORT = 8765

JPEG_QUALITY = 82

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# In-memory cache of validated datasets.
RALLIES = {}
DEFAULT_RALLY = None

# Prevent concurrent requests from rebuilding the same cache.
FRAME_LOCKS = {}


# ============================================================
# FILE HELPERS
# ============================================================

def resolve_project_file(relative_path):
    """
    All catalog paths must stay inside our project.
    Never accept arbitrary file paths from HTTP requests.
    """
    path = (ROOT / relative_path).resolve()

    if not path.is_relative_to(ROOT):
        raise ValueError(
            f"Catalog path escapes project: {relative_path}"
        )

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# VALIDATE INDIVIDUAL RALLY
# ============================================================

def validate_rally(config):
    rally_id = config["id"]

    if not ID_PATTERN.fullmatch(rally_id):
        raise ValueError(
            f"Invalid rally ID: {rally_id}"
        )

    timeline_file = resolve_project_file(
        config["timeline"]
    )

    metadata_file = resolve_project_file(
        config["metadata"]
    )

    video_file = resolve_project_file(
        config["video"]
    )

    timeline = load_json(timeline_file)
    metadata = load_json(metadata_file)

    if not isinstance(timeline, list) or not timeline:
        raise ValueError(
            f"{rally_id}: timeline must be a nonempty list"
        )

    count = int(metadata["total_frames"])
    fps = float(metadata["fps"])
    first_frame = int(metadata["start_frame"])

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"{rally_id}: invalid FPS"
        )

    if count != len(timeline):
        raise ValueError(
            f"{rally_id}: timeline length mismatch"
        )

    if "end_frame_exclusive" in metadata:
        if int(metadata["end_frame_exclusive"]) != first_frame + count:
            raise ValueError(
                f"{rally_id}: invalid end frame"
            )

    for index, row in enumerate(timeline):
        expected_frame = first_frame + index

        if int(row["frame"]) != expected_frame:
            raise ValueError(
                f"{rally_id}: frame mismatch at index {index}"
            )

        if int(row["local_frame"]) != index:
            raise ValueError(
                f"{rally_id}: local frame mismatch at {index}"
            )

        expected_time = expected_frame / fps

        if abs(float(row["time"]) - expected_time) > 0.002:
            raise ValueError(
                f"{rally_id}: timestamp mismatch at {index}"
            )

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        raise RuntimeError(
            f"{rally_id}: cannot open video"
        )

    try:
        video_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        video_fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    finally:
        cap.release()

    if video_count != count:
        raise ValueError(
            f"{rally_id}: video has {video_count} frames, "
            f"timeline has {count}"
        )

    if abs(video_fps - fps) > 0.001:
        raise ValueError(
            f"{rally_id}: video FPS mismatch"
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            f"{rally_id}: invalid video resolution"
        )

    info = {
        "id": rally_id,
        "title": config["title"],
        "description": config.get("description", ""),
        "frame_count": count,
        "fps": fps,
        "width": width,
        "height": height,
        "start_frame": first_frame,
        "end_frame": first_frame + count - 1,
        "start_seconds": first_frame / fps,
        "end_seconds": (first_frame + count) / fps,
        "duration_seconds": count / fps,
        "has_players": any(
            bool(row.get("players_sampled"))
            for row in timeline
        ),
        "shuttle_visible_frames": sum(
            bool(row.get("shuttle", {}).get("visible"))
            for row in timeline
        ),
    }

    return {
        "info": info,
        "timeline": timeline,
        "metadata": metadata,
        "video_file": video_file,
        "cache_dir": CACHE_ROOT / rally_id,
        "lock": threading.Lock(),
    }


# ============================================================
# LOAD CATALOG
# ============================================================

def load_catalog():
    global RALLIES, DEFAULT_RALLY

    catalog = load_json(CATALOG_FILE)
    configs = catalog.get("rallies", [])

    if not configs:
        raise ValueError("Rally catalog is empty")

    registered = {}

    for config in configs:
        rally_id = config["id"]

        if rally_id in registered:
            raise ValueError(
                f"Duplicate rally ID: {rally_id}"
            )

        rally = validate_rally(config)
        registered[rally_id] = rally

        info = rally["info"]

        print(
            f"  {rally_id}: "
            f"{info['frame_count']} frames, "
            f"{info['duration_seconds']:.2f}s, "
            f"{info['shuttle_visible_frames']} shuttle predictions"
        )

    default_id = catalog.get(
        "default_rally",
        next(iter(registered))
    )

    if default_id not in registered:
        raise ValueError(
            f"Default rally not registered: {default_id}"
        )

    RALLIES = registered
    DEFAULT_RALLY = default_id


# ============================================================
# FRAME CACHING
# ============================================================

def ensure_frames(rally):
    """
    Extract frames once per rally.

    Cache is rebuilt if the source video's timestamp,
    size, or expected frame count changes.
    """

    video_file = rally["video_file"]
    cache_dir = rally["cache_dir"]
    count = rally["info"]["frame_count"]

    manifest_file = cache_dir / "cache_manifest.json"

    with rally["lock"]:
        source_stat = video_file.stat()

        expected_manifest = {
            "source_path": str(video_file),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "frame_count": count,
            "jpeg_quality": JPEG_QUALITY,
        }

        current_manifest = None

        if manifest_file.is_file():
            try:
                current_manifest = load_json(manifest_file)
            except (OSError, ValueError):
                pass

        if current_manifest == expected_manifest:
            # Check every filename, not just the first/last.
            if all(
                (cache_dir / f"{i:04d}.jpg").is_file()
                for i in range(count)
            ):
                return

        print(
            f"Preparing frame cache for {rally['info']['id']}..."
        )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Invalidate old cache manifest before writing.
        manifest_file.unlink(missing_ok=True)

        cap = cv2.VideoCapture(str(video_file))

        if not cap.isOpened():
            raise RuntimeError(
                "Cannot open video for frame extraction"
            )

        written = 0

        try:
            for index in range(count):
                ok, frame = cap.read()

                if not ok:
                    raise RuntimeError(
                        f"Cannot read source frame {index}"
                    )

                destination = (
                    cache_dir / f"{index:04d}.jpg"
                )

                success = cv2.imwrite(
                    str(destination),
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        JPEG_QUALITY,
                    ]
                )

                if not success:
                    raise RuntimeError(
                        f"Cannot write {destination}"
                    )

                written += 1

        finally:
            cap.release()

        if written != count:
            raise RuntimeError(
                "Incomplete frame cache"
            )

        temp_manifest = (
            cache_dir / "cache_manifest.tmp"
        )

        temp_manifest.write_text(
            json.dumps(expected_manifest, indent=2),
            encoding="utf-8"
        )

        temp_manifest.replace(manifest_file)

        print(
            f"Cached {written} frames for "
            f"{rally['info']['id']}"
        )


# ============================================================
# HTTP HANDLER
# ============================================================

class ReplayHandler(BaseHTTPRequestHandler):
    def send_bytes(self, data, content_type, cache=False):
        self.send_response(200)

        self.send_header(
            "Content-Type",
            content_type
        )

        self.send_header(
            "Content-Length",
            str(len(data))
        )

        self.send_header(
            "Cache-Control",
            "public, max-age=3600"
            if cache
            else "no-store"
        )

        self.send_header(
            "X-Content-Type-Options",
            "nosniff"
        )

        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value):
        self.send_bytes(
            json.dumps(value).encode("utf-8"),
            "application/json; charset=utf-8"
        )

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path in ("/", "/index.html"):
                return self.send_bytes(
                    INDEX_FILE.read_bytes(),
                    "text/html; charset=utf-8"
                )

            if path in ("/library", "/library.html"):
                return self.send_bytes(
                    (HERE / "library.html").read_bytes(),
                    "text/html; charset=utf-8"
                )

            if path.startswith("/assets/"):
                filename = path.removeprefix("/assets/")

                content_type = ASSET_TYPES.get(filename)

                if content_type is None:
                    self.send_error(404)
                    return

                asset_path = ASSET_DIR / filename

                if not asset_path.is_file():
                    self.send_error(404)
                    return

                return self.send_bytes(
                    asset_path.read_bytes(),
                    content_type
                )

            if path == "/api/rallies":
                return self.send_json({
                    "default_rally": DEFAULT_RALLY,
                    "rallies": [
                        rally["info"]
                        for rally in RALLIES.values()
                    ],
                })

            if path == "/api/timeline":
                rally_id = query.get(
                    "rally", [DEFAULT_RALLY]
                )[0]

                rally = RALLIES.get(rally_id)

                if rally is None:
                    self.send_error(
                        404, "Unknown rally"
                    )
                    return

                return self.send_json({
                    "rally": rally["info"],
                    "timeline": rally["timeline"],
                    "metadata": rally["metadata"],
                    "video": {
                        "width": rally["info"]["width"],
                        "height": rally["info"]["height"],
                        "fps": rally["info"]["fps"],
                        "frames": rally["info"]["frame_count"],
                    },
                })

            match = re.fullmatch(
                r"/frames/([a-z0-9_-]+)/(\d{4,6})\.jpg",
                path
            )

            if match:
                rally_id = match.group(1)
                index = int(match.group(2))

                rally = RALLIES.get(rally_id)

                if rally is None:
                    self.send_error(404)
                    return

                if not 0 <= index < rally["info"]["frame_count"]:
                    self.send_error(404)
                    return

                ensure_frames(rally)

                frame_file = (
                    rally["cache_dir"] / f"{index:04d}.jpg"
                )

                return self.send_bytes(
                    frame_file.read_bytes(),
                    "image/jpeg",
                    cache=True
                )

            self.send_error(404)

        except (BrokenPipeError, ConnectionResetError):
            pass

        except Exception as error:
            print(
                f"HTTP error for {path}: {error}"
            )

            try:
                self.send_error(
                    500, "Internal server error"
                )
            except OSError:
                pass


# ============================================================
# MAIN
# ============================================================

def main():
    print("========== REPLAY HUB V0.3 ==========")
    print("Loading and validating rally catalog...")

    load_catalog()

    print()
    print("Registered rallies:", len(RALLIES))
    print("Default:", DEFAULT_RALLY)

    server = ThreadingHTTPServer(
        (HOST, PORT),
        ReplayHandler
    )

    print()
    print(f"Open: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nStopping Replay Hub...")

    finally:
        server.server_close()


if __name__ == "__main__":
    main()
