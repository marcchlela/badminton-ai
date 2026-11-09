
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path("data/raw/high_quality_match.mp4")
LABELS = Path("outputs/shuttle/hd_visibility_audit.json")

V1_PATH = Path(
    "outputs/shuttle/trajectory_v1/tracking_positions.csv"
)
V4_PATH = Path(
    "outputs/shuttle/trajectory_v4/optimized_positions.csv"
)

OUTPUT = Path(
    "outputs/shuttle/trajectory_v4/candidate_inspection.jpg"
)

# Known failures, plus some successful comparison frames.
TIMES = [
    200.0, 200.8, 201.0,
    202.0, 202.6,
    205.6, 206.2, 206.4,
    206.6, 206.8, 207.0, 209.8
]

CROP_SIZE = 160
DISPLAY_SIZE = 180


def load_positions(path):
    with path.open(newline="", encoding="utf-8") as f:
        return {
            int(row["frame"]): row
            for row in csv.DictReader(f)
        }


def get_point(row, x_key="x", y_key="y"):
    if not row:
        return None

    x = row.get(x_key)
    y = row.get(y_key)

    if x in (None, "") or y in (None, ""):
        return None

    return float(x), float(y)


def crop_around(frame, point):
    if point is None:
        tile = np.full(
            (DISPLAY_SIZE, DISPLAY_SIZE, 3),
            45,
            dtype=np.uint8
        )
        cv2.putText(
            tile, "MISSING",
            (35, DISPLAY_SIZE // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )
        return tile

    height, width = frame.shape[:2]
    half = CROP_SIZE // 2

    x, y = int(round(point[0])), int(round(point[1]))

    x1 = x - half
    y1 = y - half
    x2 = x1 + CROP_SIZE
    y2 = y1 + CROP_SIZE

    # Fixed-size crop with padding near image boundaries.
    crop = np.full(
        (CROP_SIZE, CROP_SIZE, 3),
        30,
        dtype=np.uint8
    )

    sx1 = max(x1, 0)
    sy1 = max(y1, 0)
    sx2 = min(x2, width)
    sy2 = min(y2, height)

    if sx2 > sx1 and sy2 > sy1:
        dx1 = sx1 - x1
        dy1 = sy1 - y1

        crop[
            dy1:dy1 + sy2 - sy1,
            dx1:dx1 + sx2 - sx1
        ] = frame[sy1:sy2, sx1:sx2]

    center = CROP_SIZE // 2

    cv2.drawMarker(
        crop, (center, center),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2
    )

    return cv2.resize(
        crop,
        (DISPLAY_SIZE, DISPLAY_SIZE),
        interpolation=cv2.INTER_LINEAR
    )


def error(point, truth):
    if point is None or truth is None:
        return None
    return math.dist(point, truth)


def fmt_error(value):
    return "missing" if value is None else f"{value:.1f}px"


def add_header(tile, title, subtitle):
    header = np.full(
        (56, DISPLAY_SIZE, 3),
        35,
        dtype=np.uint8
    )

    cv2.putText(
        header, title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 1
    )

    cv2.putText(
        header, subtitle,
        (8, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (170, 220, 255), 1
    )

    return np.vstack([header, tile])


with LABELS.open(encoding="utf-8") as f:
    annotations = json.load(f)["annotations"]

manual = {
    int(record["frame"]): tuple(record["pixel"])
    for record in annotations
    if record["status"] == "visible"
    and record["pixel"] is not None
}

v1 = load_positions(V1_PATH)
v4 = load_positions(V4_PATH)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
rows = []

try:
    for time in TIMES:
        frame_id = int(round(time * fps))

        truth = manual.get(frame_id)
        row1 = v1.get(frame_id)
        row4 = v4.get(frame_id)

        if truth is None:
            print(f"{time:.1f}s: no visible reference; skipping")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()

        if not ok:
            print(f"{time:.1f}s: failed to decode")
            continue

        top1 = get_point(row1, "top1_x", "top1_y")
        tracked_v1 = get_point(row1)
        tracked_v4 = get_point(row4)

        locations = [
            ("MANUAL", truth),
            ("TOP-1", top1),
            ("V1 TRACKER", tracked_v1),
            ("V4 TRACKER", tracked_v4),
        ]

        tiles = []

        for name, point in locations:
            tile = crop_around(frame, point)

            subtitle = (
                "reference label"
                if name == "MANUAL"
                else fmt_error(error(point, truth))
            )

            tiles.append(add_header(tile, name, subtitle))

        combined = np.hstack(tiles)

        time_strip = np.full(
            (32, combined.shape[1], 3),
            18,
            dtype=np.uint8
        )

        cv2.putText(
            time_strip,
            f"Video time: {time:.2f}s",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 1
        )

        rows.append(np.vstack([time_strip, combined]))

        print(
            f"{time:.1f}s | "
            f"Top1={fmt_error(error(top1, truth))} | "
            f"V1={fmt_error(error(tracked_v1, truth))} | "
            f"V4={fmt_error(error(tracked_v4, truth))}"
        )

finally:
    cap.release()

if not rows:
    raise SystemExit("No comparison frames produced")

sheet = np.vstack(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(OUTPUT), sheet):
    raise RuntimeError("Failed to save contact sheet")

print("\nSaved:", OUTPUT)
print("Done!")
