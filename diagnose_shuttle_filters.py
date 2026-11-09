
import cv2
import csv
import json
import math
import numpy as np
from collections import Counter
from pathlib import Path

# ============================================================
# CONFIGURATION — SAME AS OUR HD BASELINE
# ============================================================

VIDEO = Path("data/raw/high_quality_match.mp4")
LABELS = Path("outputs/shuttle/hd_visibility_audit.json")
OUTPUT_DIR = Path("outputs/shuttle/filter_diagnostics")

FRAME_GAP = 2
MOTION_THRESHOLD = 22
MIN_BRIGHTNESS = 115

MIN_AREA = 1
MAX_AREA = 90
MAX_SIDE = 20

MATCH_RADIUS = 12
LOCAL_RADIUS = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

with open(LABELS, "r", encoding="utf-8") as f:
    dataset = json.load(f)

annotations = [
    a for a in dataset["annotations"]
    if a["status"] == "visible"
    and a["pixel"] is not None
]

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Visible references: {len(annotations)}")
print(f"Video frames: {total_frames}")

# ============================================================
# HELPERS
# ============================================================

def read_frame(index):
    if index < 0 or index >= total_frames:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()

    return frame if ok else None


def get_motion_data(frame_id):
    indices = [
        frame_id - FRAME_GAP,
        frame_id - 1,
        frame_id,
        frame_id + 1,
        frame_id + FRAME_GAP
    ]

    frames = [read_frame(i) for i in indices]

    if any(f is None for f in frames):
        return None

    grays = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        for f in frames
    ]

    background = np.median(
        np.stack([
            grays[0],
            grays[1],
            grays[3],
            grays[4]
        ]),
        axis=0
    ).astype(np.uint8)

    gray = grays[2]
    difference = cv2.absdiff(gray, background)

    motion_pass = difference >= MOTION_THRESHOLD
    brightness_pass = gray >= MIN_BRIGHTNESS

    mask = (
        motion_pass & brightness_pass
    ).astype(np.uint8)

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    candidates = []
    component_details = {}

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        area_ok = MIN_AREA <= area <= MAX_AREA
        side_ok = w <= MAX_SIDE and h <= MAX_SIDE

        accepted = bool(area_ok and side_ok)

        component_details[i] = {
            "area": int(area),
            "width": int(w),
            "height": int(h),
            "area_ok": bool(area_ok),
            "side_ok": bool(side_ok),
            "accepted": accepted
        }

        if not accepted:
            continue

        region = labels[y:y+h, x:x+w] == i

        motion_values = difference[y:y+h, x:x+w][region]
        brightness_values = gray[y:y+h, x:x+w][region]

        if len(motion_values) == 0:
            continue

        score = (
            float(np.mean(motion_values))
            * float(np.mean(brightness_values)) / 255.0
            * math.sqrt(float(area))
        )

        cx, cy = centers[i]

        candidates.append({
            "component_id": i,
            "x": float(cx),
            "y": float(cy),
            "score": float(score)
        })

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    return {
        "frame": frames[2],
        "gray": gray,
        "difference": difference,
        "motion_pass": motion_pass,
        "brightness_pass": brightness_pass,
        "mask": mask,
        "labels": labels,
        "components": component_details,
        "candidates": candidates
    }


def local_window(array, x, y, radius):
    height, width = array.shape[:2]

    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)

    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)

    return array[y1:y2, x1:x2]


def nearby_component_ids(labels, x, y, radius):
    """Components with foreground pixels near the manual point."""

    height, width = labels.shape

    x1 = max(0, x - radius)
    x2 = min(width, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(height, y + radius + 1)

    region = labels[y1:y2, x1:x2]

    yy, xx = np.ogrid[y1:y2, x1:x2]

    circle = (xx - x)**2 + (yy - y)**2 <= radius**2

    ids = np.unique(region[circle])

    return [int(i) for i in ids if i != 0]


def find_rank(candidates, x, y):
    """Rank of first candidate within the matching radius."""

    for rank, candidate in enumerate(candidates, start=1):
        distance = math.hypot(
            candidate["x"] - x,
            candidate["y"] - y
        )

        if distance <= MATCH_RADIUS:
            return rank

    return None


# ============================================================
# DIAGNOSE EACH ANNOTATED FRAME
# ============================================================

rows = []
failure_counts = Counter()

try:
    for annotation in annotations:
        t = float(annotation["time"])
        frame_id = int(annotation["frame"])

        x, y = map(
            lambda v: int(round(v)),
            annotation["pixel"]
        )

        result = get_motion_data(frame_id)

        if result is None:
            print(f"{t:.1f}s: could not read neighboring frames")
            continue

        gray = result["gray"]
        diff = result["difference"]
        labels = result["labels"]

        height, width = gray.shape

        if not (0 <= x < width and 0 <= y < height):
            print(f"{t:.1f}s: invalid manual point")
            continue

        brightness = int(gray[y, x])
        motion = int(diff[y, x])

        # Center-pixel diagnostic
        center_bright = brightness >= MIN_BRIGHTNESS
        center_motion = motion >= MOTION_THRESHOLD

        # Local diagnostic: allows for small annotation offsets.
        local_brightness = local_window(
            gray, x, y, LOCAL_RADIUS
        )

        local_motion = local_window(
            diff, x, y, LOCAL_RADIUS
        )

        local_mask = local_window(
            result["mask"], x, y, LOCAL_RADIUS
        )

        local_bright_max = int(np.max(local_brightness))
        local_motion_max = int(np.max(local_motion))

        local_motion_bright_pixels = int(
            np.count_nonzero(local_mask)
        )

        # Check actual components near the label.
        component_ids = nearby_component_ids(
            labels, x, y, MATCH_RADIUS
        )

        nearby_components = [
            result["components"][i]
            for i in component_ids
        ]

        accepted_components = [
            c for c in nearby_components
            if c["accepted"]
        ]

        rejected_size = [
            c for c in nearby_components
            if not c["area_ok"] or not c["side_ok"]
        ]

        rank = find_rank(
            result["candidates"], x, y
        )

        if rank is not None and rank <= 5:
            reason = "top5_hit"

        elif rank is not None:
            reason = "ranking_failure"

        elif not nearby_components:
            reason = "no_motion_brightness_component"

        elif rejected_size:
            reason = "component_size_or_shape"

        elif accepted_components:
            reason = "component_centroid_too_far"

        else:
            reason = "other"

        failure_counts[reason] += 1

        row = {
            "time": t,
            "frame": frame_id,
            "manual_x": x,
            "manual_y": y,

            "center_brightness": brightness,
            "center_motion": motion,

            "center_brightness_pass": center_bright,
            "center_motion_pass": center_motion,

            "local_max_brightness": local_bright_max,
            "local_max_motion": local_motion_max,
            "local_foreground_pixels": local_motion_bright_pixels,

            "nearby_component_count": len(nearby_components),
            "nearby_accepted_count": len(accepted_components),
            "nearby_rejected_size_count": len(rejected_size),

            "candidate_count": len(result["candidates"]),
            "matching_candidate_rank": rank,
            "top5_hit": rank is not None and rank <= 5,

            "diagnosis": reason
        }

        rows.append(row)

        rank_text = (
            str(rank) if rank is not None else "NONE"
        )

        print(
            f"{t:6.1f}s | "
            f"brightness={brightness:3d} | "
            f"motion={motion:3d} | "
            f"components={len(nearby_components):2d} | "
            f"rank={rank_text:>4s} | "
            f"{reason}"
        )

finally:
    cap.release()


# ============================================================
# EXPORT CSV AND SUMMARY
# ============================================================

csv_path = OUTPUT_DIR / "filter_diagnostics.csv"
json_path = OUTPUT_DIR / "summary.json"

if rows:
    with open(
        csv_path, "w",
        newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

summary = {
    "visible_frames_analyzed": len(rows),
    "thresholds": {
        "motion": MOTION_THRESHOLD,
        "brightness": MIN_BRIGHTNESS,
        "max_area": MAX_AREA,
        "max_side": MAX_SIDE,
        "match_radius": MATCH_RADIUS
    },
    "diagnoses": dict(failure_counts),
    "center_brightness_failures": sum(
        not row["center_brightness_pass"]
        for row in rows
    ),
    "center_motion_failures": sum(
        not row["center_motion_pass"]
        for row in rows
    ),
    "no_local_foreground": sum(
        row["local_foreground_pixels"] == 0
        for row in rows
    ),
    "top5_hits": sum(
        row["top5_hit"] for row in rows
    ),
    "top100_hits": sum(
        row["matching_candidate_rank"] is not None
        and row["matching_candidate_rank"] <= 100
        for row in rows
    )
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n========== FILTER DIAGNOSTICS ==========")
print("Frames analyzed:", len(rows))

for reason, count in sorted(failure_counts.items()):
    print(f"  {reason}: {count}")

print(
    "Center brightness failures:",
    summary["center_brightness_failures"]
)
print(
    "Center motion failures:",
    summary["center_motion_failures"]
)
print(
    "No local foreground:",
    summary["no_local_foreground"]
)

print(
    f"Top-5: {summary['top5_hits']}/{len(rows)}"
)
print(
    f"Top-100: {summary['top100_hits']}/{len(rows)}"
)

print("\nSaved:")
print(csv_path)
print(json_path)
print("Done.")
