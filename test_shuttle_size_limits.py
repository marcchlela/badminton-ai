
import cv2
import csv
import json
import math
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

VIDEO = Path("data/raw/high_quality_match.mp4")
LABELS = Path("outputs/shuttle/hd_visibility_audit.json")
OUT = Path("outputs/shuttle/size_experiment")
OUT.mkdir(parents=True, exist_ok=True)

FRAME_GAP = 2
MOTION_THRESHOLD = 22
MIN_BRIGHTNESS = 115
MATCH_RADIUS = 12

# Only these two limits change between experiments.
CONFIGS = {
    "baseline":   (90, 20),
    "relaxed_1":  (180, 30),
    "relaxed_2":  (360, 45),
    "relaxed_3":  (720, 65),
}

with open(LABELS, encoding="utf-8") as f:
    annotations = [
        a for a in json.load(f)["annotations"]
        if a["status"] == "visible"
        and a["pixel"] is not None
    ]

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# ============================================================
# FRAME PROCESSING
# ============================================================

def read_frame(frame_id):
    if not 0 <= frame_id < total_frames:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    return frame if ok else None


def get_components(frame_id):
    indices = [
        frame_id - 2,
        frame_id - 1,
        frame_id,
        frame_id + 1,
        frame_id + 2,
    ]

    frames = [read_frame(i) for i in indices]

    if any(frame is None for frame in frames):
        return None

    grays = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for frame in frames
    ]

    background = np.median(
        np.stack([
            grays[0],
            grays[1],
            grays[3],
            grays[4],
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

    count, component_map, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    # Calculate component averages efficiently.
    flat_labels = component_map.ravel()

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

    max_allowed_area = max(
        config[0] for config in CONFIGS.values()
    )

    max_allowed_side = max(
        config[1] for config in CONFIGS.values()
    )

    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])

        if area < 1 or area > max_allowed_area:
            continue

        if w > max_allowed_side or h > max_allowed_side:
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
            "score": float(score),
        })

    return components


# ============================================================
# EVALUATION
# ============================================================

def evaluate(components, truth, max_area, max_side):
    candidates = [
        component for component in components
        if component["area"] <= max_area
        and component["width"] <= max_side
        and component["height"] <= max_side
    ]

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    tx, ty = truth

    rank = None

    for i, candidate in enumerate(candidates, start=1):
        distance = math.hypot(
            candidate["x"] - tx,
            candidate["y"] - ty
        )

        if distance <= MATCH_RADIUS:
            rank = i
            break

    return {
        "candidate_count": len(candidates),
        "rank": rank,
        "top1": rank is not None and rank <= 1,
        "top5": rank is not None and rank <= 5,
        "top20": rank is not None and rank <= 20,
        "top100": rank is not None and rank <= 100,
        "any_rank": rank is not None,
    }


results = {
    name: [] for name in CONFIGS
}

csv_rows = []

try:
    for annotation in annotations:
        time = float(annotation["time"])
        frame_id = int(annotation["frame"])
        truth = annotation["pixel"]

        components = get_components(frame_id)

        if components is None:
            print(f"Skipping {time:.1f}s")
            continue

        print(f"\n{time:.1f}s")

        for name, (max_area, max_side) in CONFIGS.items():
            metrics = evaluate(
                components,
                truth,
                max_area,
                max_side
            )

            results[name].append(metrics)

            csv_rows.append({
                "time": time,
                "configuration": name,
                "max_area": max_area,
                "max_side": max_side,
                "candidate_count": metrics["candidate_count"],
                "matching_rank": metrics["rank"],
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "top20": metrics["top20"],
                "top100": metrics["top100"],
                "any_rank": metrics["any_rank"],
            })

            print(
                f"  {name:10s} | "
                f"rank={str(metrics['rank']):>5s} | "
                f"top5={metrics['top5']}"
            )

finally:
    cap.release()


# ============================================================
# SUMMARY
# ============================================================

print("\n========== SIZE EXPERIMENT ==========")

summary = {}

for name, measurements in results.items():
    n = len(measurements)

    if n == 0:
        continue

    def hits(key):
        return sum(int(m[key]) for m in measurements)

    avg_candidates = sum(
        m["candidate_count"]
        for m in measurements
    ) / n

    summary[name] = {
        "frames": n,
        "top1": hits("top1"),
        "top5": hits("top5"),
        "top20": hits("top20"),
        "top100": hits("top100"),
        "any_rank": hits("any_rank"),
        "avg_candidate_count": round(avg_candidates, 1)
    }

    print(f"\n{name} (n={n})")
    print(f"  Top-1:   {hits('top1')}/{n}")
    print(f"  Top-5:   {hits('top5')}/{n}")
    print(f"  Top-20:  {hits('top20')}/{n}")
    print(f"  Top-100: {hits('top100')}/{n}")
    print(f"  Any rank: {hits('any_rank')}/{n}")
    print(f"  Avg candidates: {avg_candidates:.1f}")


csv_path = OUT / "results.csv"
json_path = OUT / "summary.json"

if csv_rows:
    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(csv_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(csv_rows)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nSaved:")
print(csv_path)
print(json_path)
print("Experiment complete.")
