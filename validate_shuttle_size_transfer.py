
import cv2
import json
import math
import numpy as np
from pathlib import Path

# ======================================================
# CONFIGURATION
# ======================================================

VIDEO = Path("data/raw/clip.mp4")
LABELS = Path("outputs/shuttle/visibility_audit.json")

OUTPUT = Path(
    "outputs/shuttle/size_transfer_720p.json"
)

FRAME_GAP = 2
MOTION_THRESHOLD = 22
MIN_BRIGHTNESS = 115
MATCH_RADIUS = 12

CONFIGS = {
    "baseline": (90, 20),
    "relaxed_1": (180, 30),
    "relaxed_2": (360, 45),
}

# ======================================================
# LOAD ANNOTATIONS
# ======================================================

with open(LABELS, encoding="utf-8") as f:
    annotations = [
        record
        for record in json.load(f)["annotations"]
        if record["status"] == "visible"
        and record["pixel"] is not None
    ]

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit(f"Cannot open: {VIDEO}")

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


def read_frame(index):
    if not 0 <= index < total_frames:
        return None

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(index)
    )

    ok, frame = cap.read()
    return frame if ok else None


# ======================================================
# DETECT MOTION COMPONENTS
# ======================================================

def get_components(frame_id):
    offsets = [-2, -1, 0, 1, 2]

    frames = [
        read_frame(frame_id + offset)
        for offset in offsets
    ]

    if any(frame is None for frame in frames):
        return None

    grays = [
        cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY
        )
        for frame in frames
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

    difference = cv2.absdiff(
        grays[2], background
    )

    mask = (
        (difference >= MOTION_THRESHOLD)
        & (grays[2] >= MIN_BRIGHTNESS)
    ).astype(np.uint8)

    count, label_map, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    flat_labels = label_map.ravel()

    motion_sums = np.bincount(
        flat_labels,
        weights=difference.ravel(),
        minlength=count
    )

    brightness_sums = np.bincount(
        flat_labels,
        weights=grays[2].ravel(),
        minlength=count
    )

    components = []

    for i in range(1, count):
        x, y, w, h, area = map(
            int, stats[i]
        )

        if area < 1:
            continue

        mean_motion = motion_sums[i] / area
        mean_brightness = brightness_sums[i] / area

        score = (
            mean_motion
            * mean_brightness / 255.0
            * math.sqrt(area)
        )

        cx, cy = centers[i]

        components.append({
            "x": float(cx),
            "y": float(cy),
            "area": area,
            "width": w,
            "height": h,
            "score": float(score)
        })

    return components


# ======================================================
# EVALUATION
# ======================================================

def evaluate(components, truth, limits):
    max_area, max_side = limits

    candidates = [
        c for c in components
        if c["area"] <= max_area
        and c["width"] <= max_side
        and c["height"] <= max_side
    ]

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    tx, ty = truth

    rank = None

    for i, c in enumerate(candidates, start=1):
        distance = math.hypot(
            c["x"] - tx,
            c["y"] - ty
        )

        if distance <= MATCH_RADIUS:
            rank = i
            break

    return {
        "rank": rank,
        "top1": rank is not None and rank <= 1,
        "top5": rank is not None and rank <= 5,
        "top20": rank is not None and rank <= 20,
        "any": rank is not None
    }


results = {
    name: [] for name in CONFIGS
}

try:
    for record in annotations:
        frame_id = int(record["frame"])
        timestamp = float(record["time"])

        components = get_components(frame_id)

        if components is None:
            print(f"Skipping {timestamp:.1f}s")
            continue

        print(f"Processing {timestamp:.1f}s")

        for name, limits in CONFIGS.items():
            measurement = evaluate(
                components,
                record["pixel"],
                limits
            )

            results[name].append(measurement)

finally:
    cap.release()


# ======================================================
# SUMMARY
# ======================================================

summary = {}

print("\n====== 720P TRANSFER VALIDATION ======")

for name, measurements in results.items():
    n = len(measurements)

    def hits(key):
        return sum(
            int(row[key])
            for row in measurements
        )

    summary[name] = {
        "frames": n,
        "top1": hits("top1"),
        "top5": hits("top5"),
        "top20": hits("top20"),
        "any_rank": hits("any")
    }

    print(f"\n{name}")
    print(f"  Top-1: {hits('top1')}/{n}")
    print(f"  Top-5: {hits('top5')}/{n}")
    print(f"  Top-20: {hits('top20')}/{n}")
    print(f"  Any rank: {hits('any')}/{n}")


OUTPUT.parent.mkdir(
    parents=True, exist_ok=True
)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved:", OUTPUT)
print("Done!")
