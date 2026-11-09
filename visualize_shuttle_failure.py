
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("outputs/shuttle")

CACHE = ROOT / "trajectory_v2/motion_candidates.json"
TRACK = ROOT / "trajectory_v1/tracking_positions.csv"
LABELS = ROOT / "hd_visibility_audit.json"

OUT = ROOT / "trajectory_v3/failure_analysis.png"

START = 205.8
END = 207.1


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def point(row, x_key="x", y_key="y"):
    if row is None:
        return None

    x = row.get(x_key, "")
    y = row.get(y_key, "")

    if x == "" or y == "":
        return None

    return float(x), float(y)


with CACHE.open(encoding="utf-8") as f:
    cache = json.load(f)

with LABELS.open(encoding="utf-8") as f:
    labels = json.load(f)["annotations"]

tracking_rows = load_csv(TRACK)

fps = float(cache["fps"])
first_frame = int(cache["first_frame"])

tracking = {
    int(row["frame"]): row
    for row in tracking_rows
}

manual = {
    int(a["frame"]): a["pixel"]
    for a in labels
    if a["status"] == "visible"
    and a["pixel"] is not None
}

top1_points = []
tracker_points = []
reference_points = []

for index, candidates in enumerate(cache["candidates"]):
    frame = first_frame + index
    time = frame / fps

    if not START <= time <= END:
        continue

    if candidates:
        top1_points.append({
            "time": time,
            "x": float(candidates[0][0]),
            "y": float(candidates[0][1]),
        })

    tracked = point(tracking.get(frame))

    if tracked is not None:
        tracker_points.append({
            "time": time,
            "x": tracked[0],
            "y": tracked[1],
        })

    if frame in manual:
        x, y = manual[frame]

        reference_points.append({
            "time": time,
            "x": x,
            "y": y,
        })


# ============================================================
# PLOT POSITION VS TIME
# ============================================================

fig, axes = plt.subplots(
    2, 1,
    figsize=(13, 9),
    sharex=True
)

series = [
    (top1_points, "Top-1 candidate", "#3498db"),
    (tracker_points, "V1 tracker", "#e67e22"),
]

for axis, coordinate in zip(axes, ["x", "y"]):

    for points, name, color in series:
        axis.plot(
            [p["time"] for p in points],
            [p[coordinate] for p in points],
            marker=".",
            markersize=4,
            linewidth=1.4,
            color=color,
            label=name
        )

    axis.scatter(
        [p["time"] for p in reference_points],
        [p[coordinate] for p in reference_points],
        color="#2ecc71",
        marker="X",
        s=95,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
        label="Manual shuttle label"
    )

    axis.set_ylabel(
        f"Image {coordinate.upper()} (pixels)"
    )

    axis.grid(alpha=0.25)
    axis.legend(loc="best")

axes[0].set_title(
    "Shuttle Tracking Failure: 205.8–207.1 Seconds"
)

axes[1].set_xlabel("Video time (seconds)")

# In image coordinates, Y increases downward.
axes[1].invert_yaxis()

plt.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=160)
plt.close()

print("\n========== FAILURE VISUALIZATION ==========")
print("Top-1 observations:", len(top1_points))
print("Tracker observations:", len(tracker_points))
print("Manual references:", len(reference_points))
print("Saved:", OUT)
print("Done!")
