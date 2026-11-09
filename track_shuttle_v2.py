
import csv
import json
import math
from pathlib import Path

import cv2

import track_shuttle_v1 as v1

# ============================================================
# CONFIGURATION
# ============================================================

OUT = Path("outputs/shuttle/trajectory_v2")
CACHE = OUT / "motion_candidates.json"

OUT.mkdir(parents=True, exist_ok=True)

# Set False to skip creating another replay video.
RENDER_VIDEO = True

# Conservative temporal correction settings.
DISAGREEMENT_THRESHOLD = 80
TOP1_NEIGHBOR_MAX_MOVE = 110
TOP1_INTERPOLATION_TOLERANCE = 45

# ============================================================
# STEP 1 — LOAD OR CREATE CANDIDATE CACHE
# ============================================================

def load_candidates():
    if CACHE.exists():
        print("Loading saved motion candidates...")

        with CACHE.open(encoding="utf-8") as f:
            data = json.load(f)

        if (
            data["video"] != str(v1.VIDEO)
            or data["start_sec"] != v1.START_SEC
            or data["end_sec"] != v1.END_SEC
            or data["top_candidates"] != v1.TOP_CANDIDATES
            or data["motion_threshold"] != v1.MOTION_THRESHOLD
            or data["min_brightness"] != v1.MIN_BRIGHTNESS
        ):
            raise RuntimeError(
                "Candidate cache settings differ. "
                "Delete the cache to regenerate it."
            )

        candidates = [
            [tuple(item) for item in frame]
            for frame in data["candidates"]
        ]

        return data, candidates

    print("Creating candidate cache (first run only)...")

    cap = cv2.VideoCapture(str(v1.VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {v1.VIDEO}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        first = round(v1.START_SEC * fps)
        stop = round(v1.END_SEC * fps)

        if fps <= 0 or first < 2 or stop + 2 > total:
            raise RuntimeError("Invalid video range")

        scale = height / 720

        max_area = round(90 * scale**2)
        max_side = round(20 * scale)

        candidates = v1.extract_candidates(
            cap,
            first,
            stop,
            total,
            max_area,
            max_side
        )

        data = {
            "video": str(v1.VIDEO),
            "start_sec": v1.START_SEC,
            "end_sec": v1.END_SEC,
            "top_candidates": v1.TOP_CANDIDATES,
            "motion_threshold": v1.MOTION_THRESHOLD,
            "min_brightness": v1.MIN_BRIGHTNESS,
            "fps": fps,
            "width": width,
            "height": height,
            "first_frame": first,
            "max_area": max_area,
            "max_side": max_side,
            "candidates": candidates
        }

        with CACHE.open("w", encoding="utf-8") as f:
            json.dump(data, f)

        print("Cache saved:", CACHE)

        return data, candidates

    finally:
        cap.release()


# ============================================================
# STEP 2 — LOCAL TOP-1 CONSISTENCY
# ============================================================

def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def top1_at(all_candidates, index):
    if not 0 <= index < len(all_candidates):
        return None

    candidates = all_candidates[index]

    return candidates[0] if candidates else None


def top1_is_consistent(all_candidates, index):
    """
    Check whether Top-1 at this frame agrees with its
    immediate temporal neighbors.

    This checks local smoothness, not true shuttle identity.
    """

    previous = top1_at(all_candidates, index - 1)
    current = top1_at(all_candidates, index)
    following = top1_at(all_candidates, index + 1)

    if any(p is None for p in (previous, current, following)):
        return False

    if distance(previous, current) > TOP1_NEIGHBOR_MAX_MOVE:
        return False

    if distance(current, following) > TOP1_NEIGHBOR_MAX_MOVE:
        return False

    predicted = (
        (previous[0] + following[0]) / 2,
        (previous[1] + following[1]) / 2
    )

    return (
        distance(current, predicted)
        <= TOP1_INTERPOLATION_TOLERANCE
    )


# ============================================================
# STEP 3 — EXPERIMENTAL V2 CORRECTION
# ============================================================

def correct_trajectory(all_candidates, v1_path):
    corrected = list(v1_path)
    corrections = []

    for index, old_choice in enumerate(v1_path):
        if old_choice is None:
            continue

        candidates = all_candidates[index]

        if not candidates:
            continue

        if old_choice == 0:
            continue

        original = candidates[old_choice]
        top1 = candidates[0]

        disagreement = distance(original, top1)

        if disagreement < DISAGREEMENT_THRESHOLD:
            continue

        if not top1_is_consistent(all_candidates, index):
            continue

        corrected[index] = 0

        corrections.append({
            "index": index,
            "old_rank": old_choice + 1,
            "new_rank": 1,
            "disagreement_px": round(disagreement, 1)
        })

    return corrected, corrections


# ============================================================
# STEP 4 — COMPARE TRACKERS AGAINST MANUAL LABELS
# ============================================================

def compare_trackers(all_candidates, old_path, new_path, first, height):
    with v1.LABELS.open(encoding="utf-8") as f:
        labels = json.load(f)["annotations"]

    radius = 12 * height / 720

    counts = {
        "both_correct": 0,
        "v2_only": 0,
        "v1_only": 0,
        "both_wrong": 0
    }

    for annotation in labels:
        if (
            annotation["status"] != "visible"
            or annotation["pixel"] is None
        ):
            continue

        index = int(annotation["frame"]) - first

        if not 0 <= index < len(all_candidates):
            continue

        candidates = all_candidates[index]
        truth = annotation["pixel"]

        def is_hit(choice):
            if choice is None:
                return False

            if not 0 <= choice < len(candidates):
                return False

            return distance(candidates[choice], truth) <= radius

        old_hit = is_hit(old_path[index])
        new_hit = is_hit(new_path[index])

        if old_hit and new_hit:
            counts["both_correct"] += 1
        elif new_hit:
            counts["v2_only"] += 1
        elif old_hit:
            counts["v1_only"] += 1
        else:
            counts["both_wrong"] += 1

    return counts


# ============================================================
# MAIN
# ============================================================

def main():
    print("========== SHUTTLE TRACKER V2 ==========")

    data, all_candidates = load_candidates()

    print("\nReconstructing v1 trajectory...")
    old_path = v1.link_candidates(all_candidates)

    print("\nApplying experimental v2 corrections...")
    new_path, corrections = correct_trajectory(
        all_candidates,
        old_path
    )

    print("Corrections applied:", len(corrections))

    first = data["first_frame"]
    fps = data["fps"]
    height = data["height"]
    width = data["width"]

    rows = []

    for i, (candidates, choice) in enumerate(
        zip(all_candidates, new_path)
    ):
        chosen = (
            candidates[choice]
            if choice is not None
            else None
        )

        rows.append({
            "frame": first + i,
            "time": (first + i) / fps,
            "height": height,
            "candidates": candidates,
            "choice": choice,
            "chosen": chosen
        })

    metrics = v1.evaluate(rows, fps, first)

    comparison = compare_trackers(
        all_candidates,
        old_path,
        new_path,
        first,
        height
    )

    print("\n========== V1 VS V2 ==========")

    for key, value in comparison.items():
        print(f"{key}: {value}")

    print("\n========== V2 EVALUATION ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n========== CORRECTIONS ==========")

    for correction in corrections:
        frame_index = correction["index"]
        timestamp = (first + frame_index) / fps

        print(
            f"{timestamp:.2f}s | "
            f"rank {correction['old_rank']} -> 1 | "
            f"disagreement {correction['disagreement_px']}px"
        )

    csv_path = OUT / "tracking_positions.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame", "time", "x", "y",
                "candidate_rank", "top1_x", "top1_y"
            ]
        )

        writer.writeheader()

        for row in rows:
            chosen = row["chosen"]
            top1 = (
                row["candidates"][0]
                if row["candidates"] else None
            )

            writer.writerow({
                "frame": row["frame"],
                "time": round(row["time"], 4),
                "x": round(chosen[0], 2) if chosen else "",
                "y": round(chosen[1], 2) if chosen else "",
                "candidate_rank": (
                    row["choice"] + 1
                    if chosen is not None else ""
                ),
                "top1_x": round(top1[0], 2) if top1 else "",
                "top1_y": round(top1[1], 2) if top1 else ""
            })

    report = {
        "comparison": comparison,
        "evaluation": metrics,
        "corrections": corrections,
        "note": (
            "Experimental development-set comparison. "
            "Correction rule does not use manual labels."
        )
    }

    with (OUT / "evaluation.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    if RENDER_VIDEO:
        print("\nRendering v2 replay...")
        v1.OUT = OUT

        v1.render_video(
            rows, fps, first, width, height
        )

    print("\nSaved to:", OUT)
    print("Done!")


if __name__ == "__main__":
    main()
