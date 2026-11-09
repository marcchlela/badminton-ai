
import csv
import json
import math
from pathlib import Path

ROOT = Path("outputs/shuttle")

CACHE = ROOT / "trajectory_v2/motion_candidates.json"
V1_CSV = ROOT / "trajectory_v1/tracking_positions.csv"
V2_CSV = ROOT / "trajectory_v2/tracking_positions.csv"
LABELS = ROOT / "hd_visibility_audit.json"

MATCH_RADIUS = 18.0

# Problem areas we want to inspect.
WINDOWS = [
    (201.80, 202.20),
    (204.80, 205.20),
    (205.90, 207.00),
    (209.55, 209.95),
]


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return {
            int(row["frame"]): row
            for row in csv.DictReader(f)
        }


def get_point(row):
    if row is None:
        return None

    if not row.get("x") or not row.get("y"):
        return None

    return float(row["x"]), float(row["y"])


def distance(a, b):
    if a is None or b is None:
        return None

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def fmt(value):
    if value is None:
        return "   ---"

    return f"{value:6.1f}"


def get_top1(candidates):
    if not candidates:
        return None

    return candidates[0][:2]


def get_best_match(candidates, truth):
    if not candidates:
        return None, None

    best_rank = None
    best_distance = float("inf")

    for rank, candidate in enumerate(candidates, start=1):
        d = distance(candidate[:2], truth)

        if d < best_distance:
            best_distance = d
            best_rank = rank

    return best_rank, best_distance


with CACHE.open(encoding="utf-8") as f:
    cache = json.load(f)

v1 = load_csv(V1_CSV)
v2 = load_csv(V2_CSV)

with LABELS.open(encoding="utf-8") as f:
    annotations = json.load(f)["annotations"]

candidates_by_frame = {
    int(cache["first_frame"]) + i: candidates
    for i, candidates in enumerate(cache["candidates"])
}

fps = float(cache["fps"])

visible_labels = {
    int(a["frame"]): tuple(a["pixel"])
    for a in annotations
    if a["status"] == "visible"
    and a["pixel"] is not None
}


print("\n========== SHUTTLE FAILURE INSPECTION ==========")
print("Cached frames:", len(candidates_by_frame))
print("FPS:", fps)
print("Visible labels:", len(visible_labels))


# ============================================================
# PART A: ALL LABELED FAILURE FRAMES
# ============================================================

print("\n========== LABELED V1 FAILURES ==========")

failure_frames = []

for frame, truth in sorted(visible_labels.items()):
    candidates = candidates_by_frame.get(frame, [])

    old_point = get_point(v1.get(frame))
    new_point = get_point(v2.get(frame))

    v1_error = distance(old_point, truth)
    v2_error = distance(new_point, truth)

    if v1_error is not None and v1_error <= MATCH_RADIUS:
        continue

    failure_frames.append(frame)

    rank, nearest_error = get_best_match(
        candidates, truth
    )

    top1 = get_top1(candidates)

    print(
        f"{frame / fps:6.2f}s | "
        f"V1 error={fmt(v1_error)} | "
        f"V2 error={fmt(v2_error)} | "
        f"Top1 error={fmt(distance(top1, truth))} | "
        f"Nearest candidate rank={rank} | "
        f"Nearest error={fmt(nearest_error)}"
    )


# ============================================================
# PART B: NEIGHBORING FRAME MOVEMENT
# ============================================================

print("\n========== TRAJECTORY WINDOWS ==========")

for start, end in WINDOWS:
    print(f"\n--- {start:.2f}s to {end:.2f}s ---")

    previous_v1 = None
    previous_v2 = None

    start_frame = math.ceil(start * fps - 1e-6)
    end_frame = math.floor(end * fps + 1e-6)

    for frame in range(start_frame, end_frame + 1):
        if frame not in candidates_by_frame:
            continue

        candidates = candidates_by_frame[frame]

        v1_point = get_point(v1.get(frame))
        v2_point = get_point(v2.get(frame))
        top1_point = get_top1(candidates)

        v1_jump = distance(v1_point, previous_v1)
        v2_jump = distance(v2_point, previous_v2)

        truth = visible_labels.get(frame)

        v1_error = distance(v1_point, truth)
        v2_error = distance(v2_point, truth)

        changed = (
            v1_point is not None
            and v2_point is not None
            and distance(v1_point, v2_point) > 1
        )

        label_text = ""
        if truth is not None:
            label_text = (
                f" | LABEL V1={fmt(v1_error)}"
                f" V2={fmt(v2_error)}"
            )

        print(
            f"{frame / fps:6.2f}s | "
            f"candidates={len(candidates):2d} | "
            f"V1 jump={fmt(v1_jump)} | "
            f"V2 jump={fmt(v2_jump)} | "
            f"V1-to-Top1={fmt(distance(v1_point, top1_point))}"
            f"{' | CHANGED' if changed else ''}"
            f"{label_text}"
        )

        previous_v1 = v1_point
        previous_v2 = v2_point


# ============================================================
# PART C: SUMMARY
# ============================================================

print("\n========== SUMMARY ==========")
print("Labeled V1 failure frames:", len(failure_frames))

all_frames = sorted(candidates_by_frame)

large_jumps_v1 = []
large_jumps_v2 = []

for current_frame, next_frame in zip(
    all_frames, all_frames[1:]
):
    if next_frame != current_frame + 1:
        continue

    for records, destination in [
        (v1, large_jumps_v1),
        (v2, large_jumps_v2)
    ]:
        p1 = get_point(records.get(current_frame))
        p2 = get_point(records.get(next_frame))

        jump = distance(p1, p2)

        if jump is not None and jump > 150:
            destination.append(
                (next_frame / fps, round(jump, 1))
            )

print("V1 jumps exceeding 150px:", len(large_jumps_v1))
print("V2 jumps exceeding 150px:", len(large_jumps_v2))

print("\nV1 large jumps:")
for time, jump in large_jumps_v1[:20]:
    print(f"  {time:.2f}s: {jump}px")

print("\nV2 large jumps:")
for time, jump in large_jumps_v2[:20]:
    print(f"  {time:.2f}s: {jump}px")

print("\nDone. No video decoding required.")
