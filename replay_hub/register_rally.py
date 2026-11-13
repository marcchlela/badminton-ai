
import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.resolve()
CATALOG = HERE / "rallies.json"

ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def project_file(value):
    path = (ROOT / value).resolve()

    if not path.is_relative_to(ROOT):
        raise ValueError("File must be inside badminton-ai")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def validate(timeline_path, metadata_path, video_path):
    timeline = read_json(project_file(timeline_path))
    metadata = read_json(project_file(metadata_path))
    video = project_file(video_path)

    if not isinstance(timeline, list) or not timeline:
        raise ValueError("Timeline must be a nonempty list")

    count = int(metadata["total_frames"])
    fps = float(metadata["fps"])
    first = int(metadata["start_frame"])

    if count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frame count or FPS")

    if len(timeline) != count:
        raise ValueError("Timeline frame count mismatch")

    for i, row in enumerate(timeline):
        if int(row["frame"]) != first + i:
            raise ValueError(f"Frame mismatch at index {i}")

        if int(row["local_frame"]) != i:
            raise ValueError(f"Local frame mismatch at index {i}")

        if abs(float(row["time"]) - (first + i) / fps) > 0.002:
            raise ValueError(f"Timestamp mismatch at index {i}")

        if not isinstance(row.get("shuttle"), dict):
            raise ValueError(f"Missing shuttle data at index {i}")

        if not isinstance(row.get("players"), dict):
            raise ValueError(f"Missing player data at index {i}")

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise ValueError("Cannot open rally video")

    try:
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if video_frames != count:
        raise ValueError(
            f"Video frames: {video_frames}; expected: {count}"
        )

    if abs(video_fps - fps) > 0.001:
        raise ValueError("Video FPS does not match timeline")

    if width <= 0 or height <= 0:
        raise ValueError("Invalid video dimensions")

    return {
        "frames": count,
        "fps": fps,
        "duration": count / fps,
        "resolution": f"{width}x{height}",
        "first_frame": first,
        "last_frame": first + count - 1,
        "shuttle_predictions": sum(
            bool(row["shuttle"].get("visible"))
            for row in timeline
        )
    }


def print_rallies(catalog):
    print("\n========== REGISTERED RALLIES ==========")

    for entry in catalog["rallies"]:
        marker = (
            " [default]"
            if entry["id"] == catalog["default_rally"]
            else ""
        )
        print(f"{entry['id']}{marker}")
        print(f"  {entry['title']}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and register Badminton AI rallies."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--add", action="store_true")

    parser.add_argument("--id")
    parser.add_argument("--title")
    parser.add_argument("--description", default="")
    parser.add_argument("--timeline")
    parser.add_argument("--metadata")
    parser.add_argument("--video")
    parser.add_argument("--set-default", action="store_true")

    args = parser.parse_args()
    catalog = read_json(CATALOG)

    if args.list:
        print_rallies(catalog)
        return

    if args.check:
        print("========== VALIDATING CATALOG ==========")

        for entry in catalog["rallies"]:
            info = validate(
                entry["timeline"],
                entry["metadata"],
                entry["video"]
            )

            print(
                f"{entry['id']}: PASS | "
                f"{info['frames']} frames | "
                f"{info['duration']:.2f}s | "
                f"{info['shuttle_predictions']} shuttle predictions"
            )

        print("\nAll registered rallies validated.")
        return

    required = {
        "--id": args.id,
        "--title": args.title,
        "--timeline": args.timeline,
        "--metadata": args.metadata,
        "--video": args.video,
    }

    missing = [
        key for key, value in required.items()
        if not value
    ]

    if missing:
        parser.error("Missing: " + ", ".join(missing))

    if not ID_PATTERN.fullmatch(args.id):
        raise ValueError(
            "ID must contain lowercase letters, digits, "
            "hyphens or underscores"
        )

    if any(r["id"] == args.id for r in catalog["rallies"]):
        raise ValueError(
            f"Rally already registered: {args.id}"
        )

    if len(args.title) > 150:
        raise ValueError("Title is too long")

    info = validate(
        args.timeline,
        args.metadata,
        args.video
    )

    entry = {
        "id": args.id,
        "title": args.title,
        "description": args.description,
        "timeline": args.timeline,
        "metadata": args.metadata,
        "video": args.video,
    }

    catalog["rallies"].append(entry)

    if args.set_default:
        catalog["default_rally"] = args.id

    # Validate the updated catalog before saving.
    ids = [r["id"] for r in catalog["rallies"]]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate rally IDs")

    if catalog["default_rally"] not in ids:
        raise ValueError("Invalid default rally")

    # Atomic write: keep original catalog intact on failure.
    temp_name = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=HERE,
            prefix=".rallies_",
            suffix=".tmp",
            delete=False
        ) as f:
            temp_name = f.name
            json.dump(catalog, f, indent=2, ensure_ascii=False)
            f.write("\n")

        os.replace(temp_name, CATALOG)

    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)

    print("\n========== RALLY REGISTERED ==========")
    print("ID:", args.id)
    print("Title:", args.title)

    for key, value in info.items():
        print(f"{key}: {value}")

    print("\nUpdated:", CATALOG)
    print("Restart the Replay Hub server to load the new rally.")


if __name__ == "__main__":
    main()
