
import cv2
import csv
import json
import numpy as np
from pathlib import Path

VIDEO = Path("data/raw/high_quality_match.mp4")
LABELS = Path("outputs/shuttle/hd_visibility_audit.json")
OUT = Path("outputs/shuttle/hd_motion_top100")
OUT.mkdir(parents=True, exist_ok=True)

# Exploratory settings, not optimized against labels.
FRAME_GAP = 2
MOTION_THRESHOLD = 22
MIN_BRIGHTNESS = 115
MIN_AREA = 1
MAX_AREA = 90
MAX_SIDE = 20
MAX_CANDIDATES = 100
MATCH_RADIUS = 12  # pixels, for evaluation only

with open(LABELS, encoding="utf-8") as f:
    annotations = json.load(f)["annotations"]

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def read_frame(index):
    if not 0 <= index < total_frames:
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    return frame if ok else None


def detect(frame_id):
    indices = [
        frame_id - FRAME_GAP,
        frame_id - 1,
        frame_id,
        frame_id + 1,
        frame_id + FRAME_GAP
    ]

    frames = [read_frame(i) for i in indices]

    if any(f is None for f in frames):
        return None, []

    current = frames[2]

    grays = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        for f in frames
    ]

    # Median of neighboring frames approximates the
    # local background around the current time.
    background = np.median(
        np.stack([
            grays[0], grays[1],
            grays[3], grays[4]
        ]),
        axis=0
    ).astype(np.uint8)

    difference = cv2.absdiff(grays[2], background)

    # Require both apparent movement and brightness.
    mask = (
        (difference >= MOTION_THRESHOLD)
        & (grays[2] >= MIN_BRIGHTNESS)
    ).astype(np.uint8)

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
    )

    candidates = []

    for i in range(1, count):
        x, y, w, h, area = stats[i]

        if not MIN_AREA <= area <= MAX_AREA:
            continue

        if w > MAX_SIDE or h > MAX_SIDE:
            continue

        region = labels[y:y+h, x:x+w] == i

        motion_values = difference[y:y+h, x:x+w][region]
        brightness_values = grays[2][y:y+h, x:x+w][region]

        if len(motion_values) == 0:
            continue

        # Heuristic ranking, independent of the labels.
        score = (
            float(np.mean(motion_values))
            * float(np.mean(brightness_values)) / 255.0
            * np.sqrt(float(area))
        )

        cx, cy = centers[i]

        candidates.append({
            "x": round(float(cx), 2),
            "y": round(float(cy), 2),
            "score": round(score, 3),
            "area": int(area)
        })

    candidates.sort(
        key=lambda c: c["score"],
        reverse=True
    )

    return current, candidates[:MAX_CANDIDATES]


def evaluate(candidates, truth):
    if truth is None:
        return None, None, None, None, None

    distances = [
        float(np.hypot(c["x"] - truth[0],
                       c["y"] - truth[1]))
        for c in candidates
    ]

    if not distances:
        return False, False, False, False, None

    top1_hit = distances[0] <= MATCH_RADIUS
    top5_hit = min(distances[:5]) <= MATCH_RADIUS

    top20_hit = min(distances[:20]) <= MATCH_RADIUS
    top100_hit = min(distances) <= MATCH_RADIUS

    return top1_hit, top5_hit, top20_hit, top100_hit, min(distances)


rows = []
tiles = []
hits_top1 = 0
hits_top5 = 0
hits_top20 = 0
hits_top100 = 0
visible_count = 0

for annotation in annotations:
    frame_id = annotation["frame"]
    timestamp = annotation["time"]
    status = annotation["status"]
    truth = annotation.get("pixel")

    frame, candidates = detect(frame_id)

    if frame is None:
        print(f"Skipping frame {frame_id}")
        continue

    top1, top5, top20, top100, nearest = evaluate(candidates, truth)

    if status == "visible" and truth is not None:
        visible_count += 1
        hits_top1 += int(bool(top1))
        hits_top5 += int(bool(top5))
        hits_top20 += int(bool(top20))
        hits_top100 += int(bool(top100))

    rows.append({
        "time": timestamp,
        "frame": frame_id,
        "label_status": status,
        "candidate_count": len(candidates),
        "top1_hit": top1,
        "top5_hit": top5,
        "nearest_distance_px": nearest
    })

    preview = frame.copy()

    # Red = automatic candidates
    for rank, c in enumerate(candidates[:5], start=1):
        point = (
            int(round(c["x"])),
            int(round(c["y"]))
        )

        cv2.circle(
            preview, point, 9,
            (0, 0, 255), 2
        )

        cv2.putText(
            preview, str(rank),
            (point[0] + 10, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )

    # Cyan = manual reference (not used for detection)
    if truth is not None:
        p = tuple(map(int, truth))

        cv2.drawMarker(
            preview, p,
            (255, 255, 0),
            cv2.MARKER_CROSS,
            25, 2
        )

    tile = cv2.resize(
        preview, (480, 270),
        interpolation=cv2.INTER_AREA
    )

    cv2.rectangle(
        tile, (0, 0), (480, 42),
        (15, 15, 15), -1
    )

    result = (
        "HIT" if top5 else "MISS"
        if status == "visible" else "UNLABELED"
    )

    cv2.putText(
        tile,
        f"{timestamp:.1f}s | {status} | {result}",
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2
    )

    tiles.append(tile)

    print(
        f"{timestamp:5.1f}s | "
        f"{status:11s} | "
        f"candidates={len(candidates):2d} | "
        f"top5={top5}"
    )

cap.release()

# Save CSV
csv_path = OUT / "results.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

# Save comparison contact sheet
columns = 3
tile_w, tile_h = 480, 270
num_rows = int(np.ceil(len(tiles) / columns))

canvas = np.full(
    (num_rows * tile_h, columns * tile_w, 3),
    (15, 15, 15), dtype=np.uint8
)

for i, tile in enumerate(tiles):
    y = (i // columns) * tile_h
    x = (i % columns) * tile_w

    canvas[
        y:y+tile_h,
        x:x+tile_w
    ] = tile

image_path = OUT / "candidate_review.jpg"
cv2.imwrite(str(image_path), canvas)

print("\n========== MOTION BASELINE ==========")
print("Visible reference frames:", visible_count)
print(f"Top-1 hits: {hits_top1}/{visible_count}")
print(f"Top-5 hits: {hits_top5}/{visible_count}")
print(f"Top-20 hits: {hits_top20}/{visible_count}")
print(f"Top-100 hits: {hits_top100}/{visible_count}")

print("\nSaved:")
print(csv_path)
print(image_path)
print("\nNote: candidate ranking is heuristic.")
print("This is not yet a trained shuttle detector.")
