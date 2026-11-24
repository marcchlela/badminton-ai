
"""
Badminton AI — Phase 6.4.4
Manual contact annotation dataset exporter.

Exports:
  manifest.json
  samples.json
  samples.csv
  frame_features.csv
  optional source JPEG frames

No model training is performed by this script.
Original annotations, videos and tracking files are read-only.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "replay_hub" / "rallies.json"
ANNOTATIONS = ROOT / "outputs" / "annotations"
OUTPUTS = ROOT / "outputs" / "ml_datasets"

FEATURE_COLUMNS = [
    "sample_id",
    "rally_id",
    "event_id",
    "relative_frame",
    "local_frame",
    "source_frame",
    "time_seconds",
    "is_contact_frame",
    "shuttle_visible",
    "shuttle_x_px",
    "shuttle_y_px",
    "players_sampled",
    "near_x_m",
    "near_y_m",
    "far_x_m",
    "far_y_m",
    "near_bbox_x",
    "near_bbox_y",
    "near_bbox_w",
    "near_bbox_h",
    "far_bbox_x",
    "far_bbox_y",
    "far_bbox_w",
    "far_bbox_h",
]

SAMPLE_COLUMNS = [
    "sample_id",
    "rally_id",
    "event_id",
    "local_frame",
    "source_frame",
    "time_seconds",
    "player",
    "shot",
    "confidence",
    "contact_label",
    "shot_label_available",
    "window_start",
    "window_end",
    "window_frames",
    "full_window",
    "notes",
]


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_file(value):
    path = (ROOT / value).resolve()

    if not path.is_relative_to(ROOT):
        raise ValueError("Catalog path escapes project")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def number(value):
    if value is None or isinstance(value, bool):
        return None

    try:
        result = float(value)
    except (ValueError, TypeError):
        return None

    return result if math.isfinite(result) else None


def vector(value, count):
    if not isinstance(value, (list, tuple)):
        return [None] * count

    if len(value) != count:
        return [None] * count

    return [number(item) for item in value]


def csv_value(value):
    return "" if value is None else value


def write_csv(path, fieldnames, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: csv_value(row.get(key))
                for key in fieldnames
            })


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def load_rally(rally_id):
    catalog = read_json(CATALOG)

    entry = next(
        (
            item for item in catalog["rallies"]
            if item["id"] == rally_id
        ),
        None,
    )

    if entry is None:
        raise ValueError(
            f"Rally not registered: {rally_id}"
        )

    timeline_file = resolve_file(entry["timeline"])
    metadata_file = resolve_file(entry["metadata"])
    video_file = resolve_file(entry["video"])

    timeline = read_json(timeline_file)
    metadata = read_json(metadata_file)

    if not isinstance(timeline, list) or not timeline:
        raise ValueError("Invalid or empty timeline")

    fps = number(metadata.get("fps"))
    start = metadata.get("start_frame")
    count = metadata.get("total_frames")

    if fps is None or fps <= 0:
        raise ValueError("Invalid metadata FPS")

    if type(start) is not int or type(count) is not int:
        raise ValueError("Invalid metadata frame fields")

    if count != len(timeline):
        raise ValueError("Timeline length mismatch")

    for i, row in enumerate(timeline):
        if row.get("local_frame") != i:
            raise ValueError(
                f"Local frame mismatch: {i}"
            )

        if row.get("frame") != start + i:
            raise ValueError(
                f"Source frame mismatch: {i}"
            )

        t = number(row.get("time"))

        if t is None or abs(t - (start + i) / fps) > 0.002:
            raise ValueError(
                f"Timestamp mismatch: {i}"
            )

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        raise ValueError("Cannot open rally video")

    try:
        video_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        video_fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    if video_frames != count:
        raise ValueError(
            "Rally video and timeline length mismatch"
        )

    if abs(video_fps - fps) > 0.001:
        raise ValueError(
            "Rally video and timeline FPS mismatch"
        )

    return {
        "entry": entry,
        "timeline": timeline,
        "metadata": metadata,
        "timeline_file": timeline_file,
        "video_file": video_file,
        "fps": fps,
        "start_frame": start,
        "frame_count": count,
    }


def load_annotations(rally_id):
    path = ANNOTATIONS / f"{rally_id}.json"

    if not path.exists():
        return [], None

    data = read_json(path)

    if data.get("rally_id") != rally_id:
        raise ValueError(
            "Annotation rally ID mismatch"
        )

    events = data.get("events")

    if not isinstance(events, list):
        raise ValueError(
            "Invalid annotation event list"
        )

    return events, path


def features_for_frame(sample_id, event_id, rally_id,
                       row, center):
    local_frame = row["local_frame"]

    shuttle = row.get("shuttle") or {}
    players = row.get("players") or {}

    near = players.get("Near") or {}
    far = players.get("Far") or {}

    # Raw, sparse values only.
    # We do not use browser-side interpolation,
    # smoothing, or schematic shuttle coordinates.
    shuttle_xy = (
        vector(shuttle.get("pixel"), 2)
        if shuttle.get("visible") is True
        else [None, None]
    )

    sampled = row.get("players_sampled") is True

    near_xy = (
        vector(near.get("court_m"), 2)
        if sampled else [None, None]
    )

    far_xy = (
        vector(far.get("court_m"), 2)
        if sampled else [None, None]
    )

    near_box = (
        vector(near.get("bbox"), 4)
        if sampled else [None] * 4
    )

    far_box = (
        vector(far.get("bbox"), 4)
        if sampled else [None] * 4
    )

    return {
        "sample_id": sample_id,
        "rally_id": rally_id,
        "event_id": event_id,
        "relative_frame": local_frame - center,
        "local_frame": local_frame,
        "source_frame": row["frame"],
        "time_seconds": row["time"],
        "is_contact_frame": int(
            local_frame == center
        ),
        "shuttle_visible": int(
            shuttle.get("visible") is True
        ),
        "shuttle_x_px": shuttle_xy[0],
        "shuttle_y_px": shuttle_xy[1],
        "players_sampled": int(sampled),
        "near_x_m": near_xy[0],
        "near_y_m": near_xy[1],
        "far_x_m": far_xy[0],
        "far_y_m": far_xy[1],
        "near_bbox_x": near_box[0],
        "near_bbox_y": near_box[1],
        "near_bbox_w": near_box[2],
        "near_bbox_h": near_box[3],
        "far_bbox_x": far_box[0],
        "far_bbox_y": far_box[1],
        "far_bbox_w": far_box[2],
        "far_bbox_h": far_box[3],
    }


def select_contacts(events, timeline, fps,
                    start_frame, allowed):
    selected = []
    skipped = {
        "non_contact": 0,
        "confidence_excluded": 0,
    }

    seen_ids = set()

    for event in events:
        event_id = event.get("id")

        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Annotation missing event ID")

        if event_id in seen_ids:
            raise ValueError("Duplicate annotation ID")

        seen_ids.add(event_id)

        if event.get("kind") != "contact":
            skipped["non_contact"] += 1
            continue

        confidence = event.get(
            "confidence", "uncertain"
        )

        if confidence not in {
            "confirmed", "approximate", "uncertain"
        }:
            confidence = "uncertain"

        if confidence not in allowed:
            skipped["confidence_excluded"] += 1
            continue

        index = event.get("local_frame")

        if type(index) is not int:
            raise ValueError(
                f"Invalid frame for event {event_id}"
            )

        if not 0 <= index < len(timeline):
            raise ValueError(
                f"Event outside rally: {event_id}"
            )

        original = start_frame + index

        if event.get("source_frame") != original:
            raise ValueError(
                f"Source frame mismatch in {event_id}"
            )

        expected_time = original / fps
        stored_time = number(
            event.get("time_seconds")
        )

        if (
            stored_time is None
            or abs(stored_time - expected_time) > 0.002
        ):
            raise ValueError(
                f"Event timestamp mismatch: {event_id}"
            )

        player = event.get("player", "Unknown")
        shot = event.get("shot", "unknown")

        if player not in {"Near", "Far", "Unknown"}:
            raise ValueError(
                f"Invalid player: {event_id}"
            )

        if shot not in {
            "unknown", "serve", "clear", "drop",
            "smash", "lift", "drive", "net",
            "push", "other",
        }:
            raise ValueError(
                f"Invalid shot type: {event_id}"
            )

        selected.append({
            **event,
            "confidence": confidence,
        })

    selected.sort(
        key=lambda event: (
            event["local_frame"],
            event["id"],
        )
    )

    return selected, skipped


def extract_source_frames(video_file, directory,
                          local_indices):
    directory.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        raise RuntimeError("Cannot open source video")

    try:
        for index in sorted(set(local_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Could not decode frame {index}"
                )

            destination = (
                directory / f"{index:06d}.jpg"
            )

            if not cv2.imwrite(
                str(destination),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            ):
                raise RuntimeError(
                    f"Could not write {destination}"
                )
    finally:
        cap.release()


def build_dataset(args):
    rally = load_rally(args.rally)
    timeline = rally["timeline"]

    events, annotation_path = load_annotations(
        args.rally
    )

    allowed = {"confirmed"}

    if args.include_approximate:
        allowed.add("approximate")

    contacts, skipped = select_contacts(
        events,
        timeline,
        rally["fps"],
        rally["start_frame"],
        allowed,
    )

    print(f"Rally: {args.rally}")
    print(f"Total annotations: {len(events)}")
    print(f"Eligible contacts: {len(contacts)}")
    print(
        "Excluded by confidence:",
        skipped["confidence_excluded"]
    )

    if not contacts:
        print(
            "No eligible contacts. Nothing exported.\n"
            "Review and confirm contact annotations "
            "in the Replay Hub first."
        )
        return

    OUTPUTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        OUTPUTS / args.name
    )

    if destination.exists():
        raise FileExistsError(
            f"Dataset already exists: {destination}\n"
            "Choose a new --name to preserve older exports."
        )

    temp = Path(
        tempfile.mkdtemp(
            prefix=".export_",
            dir=OUTPUTS,
        )
    )

    try:
        sample_rows = []
        feature_rows = []
        sample_json = []
        frame_indices = set()

        for event in contacts:
            center = event["local_frame"]

            first = max(
                0,
                center - args.radius,
            )

            last = min(
                len(timeline) - 1,
                center + args.radius,
            )

            complete = (
                first == center - args.radius
                and last == center + args.radius
            )

            sample_id = (
                args.rally + "_" +
                event["id"]
            )

            item = {
                "sample_id": sample_id,
                "rally_id": args.rally,
                "event_id": event["id"],
                "local_frame": center,
                "source_frame": event["source_frame"],
                "time_seconds": event["time_seconds"],
                "player": event["player"],
                "shot": event["shot"],
                "confidence": event["confidence"],
                "contact_label": 1,
                "shot_label_available": int(
                    event["shot"] != "unknown"
                ),
                "window_start": first,
                "window_end": last,
                "window_frames": last - first + 1,
                "full_window": int(complete),
                "notes": event.get("notes", ""),
            }

            sample_rows.append(item)

            frame_details = []

            for index in range(first, last + 1):
                row = timeline[index]

                feature = features_for_frame(
                    sample_id,
                    event["id"],
                    args.rally,
                    row,
                    center,
                )

                feature_rows.append(feature)
                frame_indices.add(index)

                frame_details.append({
                    "local_frame": index,
                    "source_frame": row["frame"],
                    "relative_frame": index - center,
                    "image_path": (
                        f"frames/{index:06d}.jpg"
                        if args.frames
                        else None
                    ),
                })

            sample_json.append({
                **item,
                "frames": frame_details,
            })

        write_csv(
            temp / "samples.csv",
            SAMPLE_COLUMNS,
            sample_rows,
        )

        write_csv(
            temp / "frame_features.csv",
            FEATURE_COLUMNS,
            feature_rows,
        )

        write_json(
            temp / "samples.json",
            sample_json,
        )

        if args.frames:
            print(
                f"Extracting {len(frame_indices)} "
                "unique original frames..."
            )

            extract_source_frames(
                rally["video_file"],
                temp / "frames",
                frame_indices,
            )

        video_stat = rally["video_file"].stat()

        manifest = {
            "schema_version": 1,
            "dataset_name": args.name,
            "task": "manual_racket_contact_windows",
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "rally_id": args.rally,
            "rally_title": rally["entry"]["title"],
            "fps": rally["fps"],
            "total_rally_frames": rally["frame_count"],
            "source_frame_start": rally["start_frame"],
            "window_radius_frames": args.radius,
            "allowed_confidence": sorted(allowed),
            "contact_samples": len(sample_rows),
            "feature_rows": len(feature_rows),
            "unique_exported_frames": len(frame_indices),
            "frames_included": args.frames,
            "excluded_annotations": skipped,
            "source": {
                "timeline_path": rally["entry"]["timeline"],
                "timeline_sha256": sha256(
                    rally["timeline_file"]
                ),
                "annotation_path": (
                    str(annotation_path.relative_to(ROOT))
                    if annotation_path else None
                ),
                "annotation_sha256": (
                    sha256(annotation_path)
                    if annotation_path else None
                ),
                "video_path": rally["entry"]["video"],
                "video_size_bytes": video_stat.st_size,
                "video_mtime_ns": video_stat.st_mtime_ns,
            },
            "limitations": [
                "Only human-annotated contacts are positive labels.",
                "Unannotated frames are not negative examples.",
                "Unknown shot types are excluded from shot classification labels.",
                "Court positions are sparse measured/recorded values; missing values stay missing.",
                "Shuttle pixels are image-space coordinates, not 3D positions.",
                "Overlapping rallies can duplicate source footage.",
                "No train/validation/test split has been assigned.",
            ],
        }

        write_json(
            temp / "manifest.json",
            manifest,
        )

        # Publish only a fully completed export.
        temp.rename(destination)

    except Exception:
        shutil.rmtree(
            temp,
            ignore_errors=True,
        )
        raise

    print("\nEXPORT COMPLETE")
    print("Location:", destination)
    print("Contact samples:", len(sample_rows))
    print("Tracking rows:", len(feature_rows))
    print("Unique frame indexes:", len(frame_indices))
    print("Extracted JPEGs:", args.frames)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export manually labeled badminton "
            "contact windows for future ML experiments."
        )
    )

    parser.add_argument(
        "--rally",
        required=True,
        help="Registered rally ID",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Unique output dataset folder name",
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=10,
        help="Frames before and after contact (default 10)",
    )

    parser.add_argument(
        "--frames",
        action="store_true",
        help="Include original source-frame JPEGs",
    )

    parser.add_argument(
        "--include-approximate",
        action="store_true",
        help="Also include approximate contact annotations",
    )

    args = parser.parse_args()

    if args.radius < 0 or args.radius > 100:
        parser.error("--radius must be between 0 and 100")

    if (
        not args.name
        or len(args.name) > 80
        or not all(
            c.isascii() and (
                c.isalnum() or c in "-_"
            )
            for c in args.name
        )
    ):
        parser.error(
            "--name must contain only ASCII letters, "
            "digits, hyphens or underscores"
        )

    build_dataset(args)


if __name__ == "__main__":
    main()
