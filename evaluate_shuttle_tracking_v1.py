
import csv
import json
import math
from pathlib import Path

POSITIONS = Path(
    "outputs/shuttle/trajectory_v1/tracking_positions.csv"
)
LABELS = Path(
    "outputs/shuttle/hd_visibility_audit.json"
)

RADIUS = 18.0

with POSITIONS.open(newline="", encoding="utf-8") as f:
    positions = {
        int(row["frame"]): row
        for row in csv.DictReader(f)
    }

with LABELS.open(encoding="utf-8") as f:
    annotations = json.load(f)["annotations"]


def read_point(row, x_key, y_key):
    if not row or not row[x_key] or not row[y_key]:
        return None

    return float(row[x_key]), float(row[y_key])


def error(point, truth):
    if point is None:
        return None

    return math.dist(point, truth)


groups = {
    "both_correct": [],
    "tracker_only": [],
    "top1_only": [],
    "both_wrong": [],
}

all_errors = []

print("\n======= TRACKER VS TOP-1 =======")

for annotation in annotations:
    if annotation["status"] != "visible":
        continue

    if annotation["pixel"] is None:
        continue

    frame = int(annotation["frame"])
    row = positions.get(frame)

    if row is None:
        continue

    truth = tuple(annotation["pixel"])

    top1 = read_point(row, "top1_x", "top1_y")
    tracked = read_point(row, "x", "y")

    top1_error = error(top1, truth)
    track_error = error(tracked, truth)

    top1_hit = (
        top1_error is not None
        and top1_error <= RADIUS
    )

    track_hit = (
        track_error is not None
        and track_error <= RADIUS
    )

    if top1_hit and track_hit:
        category = "both_correct"
    elif track_hit:
        category = "tracker_only"
    elif top1_hit:
        category = "top1_only"
    else:
        category = "both_wrong"

    groups[category].append({
        "time": float(annotation["time"]),
        "top1_error": top1_error,
        "tracker_error": track_error,
    })

    if track_error is not None:
        all_errors.append(track_error)


for category, entries in groups.items():
    print(f"\n{category}: {len(entries)}")

    if category != "both_correct":
        for entry in entries:
            top1_text = (
                f"{entry['top1_error']:.1f}px"
                if entry["top1_error"] is not None
                else "missing"
            )

            tracker_text = (
                f"{entry['tracker_error']:.1f}px"
                if entry["tracker_error"] is not None
                else "missing"
            )

            print(
                f"  {entry['time']:.1f}s | "
                f"Top-1: {top1_text} | "
                f"Tracker: {tracker_text}"
            )


print("\n======= ERROR DISTRIBUTION =======")

if all_errors:
    sorted_errors = sorted(all_errors)

    median = sorted_errors[len(sorted_errors) // 2]

    print(f"Mean: {sum(all_errors) / len(all_errors):.1f}px")
    print(f"Median: {median:.1f}px")
    print(f"Worst: {max(all_errors):.1f}px")
    print(
        "Large errors (>50px):",
        sum(e > 50 for e in all_errors)
    )

print("\nDone.")
