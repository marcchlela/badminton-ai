
import cv2
import json
import math
import numpy as np
from pathlib import Path

DATASETS = [
    (
        "720p",
        Path("data/raw/clip.mp4"),
        Path("outputs/shuttle/visibility_audit.json")
    ),
    (
        "1080p",
        Path("data/raw/high_quality_match.mp4"),
        Path("outputs/shuttle/hd_visibility_audit.json")
    )
]

OUTPUT = Path("outputs/shuttle/adaptive_size_test.json")

BASE_HEIGHT = 720
BASE_AREA = 90
BASE_SIDE = 20
BASE_RADIUS = 12

MOTION_THRESHOLD = 22
MIN_BRIGHTNESS = 115


def read_frame(cap, index, total):
    if not 0 <= index < total:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return frame if ok else None


def find_candidates(cap, frame_id, total, max_area, max_side):
    indices = [
        frame_id - 2,
        frame_id - 1,
        frame_id,
        frame_id + 1,
        frame_id + 2
    ]

    frames = [read_frame(cap, i, total) for i in indices]

    if any(frame is None for frame in frames):
        return None

    grays = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for frame in frames
    ]

    background = np.median(
        np.stack([grays[0], grays[1], grays[3], grays[4]]),
        axis=0
    ).astype(np.uint8)

    diff = cv2.absdiff(grays[2], background)

    mask = (
        (diff >= MOTION_THRESHOLD)
        & (grays[2] >= MIN_BRIGHTNESS)
    ).astype(np.uint8)

    count, component_map, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    flat = component_map.ravel()

    motion_sums = np.bincount(
        flat, weights=diff.ravel(), minlength=count
    )

    bright_sums = np.bincount(
        flat, weights=grays[2].ravel(), minlength=count
    )

    candidates = []

    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])

        if not (
            1 <= area <= max_area
            and w <= max_side
            and h <= max_side
        ):
            continue

        score = (
            motion_sums[i] / area
            * bright_sums[i] / area
            / 255.0
            * math.sqrt(area)
        )

        cx, cy = centers[i]

        candidates.append((float(cx), float(cy), float(score)))

    candidates.sort(key=lambda item: item[2], reverse=True)

    return candidates


def get_rank(candidates, truth, radius):
    tx, ty = truth

    for rank, (x, y, score) in enumerate(candidates, start=1):
        if math.hypot(x - tx, y - ty) <= radius:
            return rank

    return None


results = {}

for name, video_path, label_path in DATASETS:
    with open(label_path, encoding="utf-8") as f:
        annotations = [
            a for a in json.load(f)["annotations"]
            if a["status"] == "visible"
            and a["pixel"] is not None
        ]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")

    try:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scale = height / BASE_HEIGHT

        # Scale widths linearly, areas quadratically.
        adaptive_area = round(BASE_AREA * scale**2)
        adaptive_side = round(BASE_SIDE * scale)
        adaptive_radius = BASE_RADIUS * scale

        configurations = {
            "fixed_baseline": (
                BASE_AREA, BASE_SIDE, BASE_RADIUS
            ),
            "adaptive": (
                adaptive_area, adaptive_side, adaptive_radius
            )
        }

        results[name] = {}

        print(f"\n========== {name} ==========")

        for config_name, settings in configurations.items():
            max_area, max_side, radius = settings

            ranks = []

            for annotation in annotations:
                candidates = find_candidates(
                    cap,
                    int(annotation["frame"]),
                    total,
                    max_area,
                    max_side
                )

                if candidates is None:
                    continue

                rank = get_rank(
                    candidates,
                    annotation["pixel"],
                    radius
                )

                ranks.append(rank)

            n = len(ranks)

            def hits(limit):
                return sum(
                    rank is not None and rank <= limit
                    for rank in ranks
                )

            result = {
                "frames": n,
                "max_area": max_area,
                "max_side": max_side,
                "match_radius": radius,
                "top1": hits(1),
                "top5": hits(5),
                "top20": hits(20),
                "any_rank": sum(r is not None for r in ranks)
            }

            results[name][config_name] = result

            print(f"\n{config_name}")
            print(f"  Max area: {max_area}")
            print(f"  Max side: {max_side}")
            print(f"  Match radius: {radius:.1f}px")
            print(f"  Top-1: {result['top1']}/{n}")
            print(f"  Top-5: {result['top5']}/{n}")
            print(f"  Top-20: {result['top20']}/{n}")
            print(f"  Any rank: {result['any_rank']}/{n}")

    finally:
        cap.release()


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved:", OUTPUT)
print("Experiment complete.")
