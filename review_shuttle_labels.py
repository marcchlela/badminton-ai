
import cv2
import json
import math
import numpy as np
from pathlib import Path

VIDEO = Path("data/raw/clip.mp4")
LABELS = Path("outputs/shuttle/visibility_audit.json")
OUTPUT = Path("outputs/shuttle/label_review.jpg")

# Original frame + magnified shuttle region
FULL_W, FULL_H = 320, 180
ZOOM_SIZE = 160
ZOOM_RADIUS = 12

TILE_W = 520
TILE_H = 245
COLUMNS = 4

with open(LABELS, "r", encoding="utf-8") as f:
    dataset = json.load(f)

annotations = [
    a for a in dataset["annotations"]
    if a["status"] == "visible"
    and a["pixel"] is not None
]

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO}")

tiles = []

for annotation in annotations:
    frame_id = int(annotation["frame"])
    timestamp = float(annotation["time"])
    x, y = map(int, annotation["pixel"])

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()

    if not ok:
        print(f"Could not read frame {frame_id}")
        continue

    h, w = frame.shape[:2]

    if not (0 <= x < w and 0 <= y < h):
        print(f"Invalid point at {timestamp:.2f}s")
        continue

    tile = np.full(
        (TILE_H, TILE_W, 3),
        (20, 20, 20),
        dtype=np.uint8
    )

    # Full frame preview
    preview = cv2.resize(frame, (FULL_W, FULL_H))
    sx = FULL_W / w
    sy = FULL_H / h

    px = round(x * sx)
    py = round(y * sy)

    cv2.circle(
        preview, (px, py),
        9, (0, 255, 255), 2
    )

    tile[42:42 + FULL_H, 8:8 + FULL_W] = preview

    # Magnify a 24x24 original-pixel region.
    # Padding keeps the crop centered near image edges.
    r = ZOOM_RADIUS

    padded = cv2.copyMakeBorder(
        frame, r, r, r, r,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    # Pixel (x,y) is at (x+r,y+r) in padded image.
    crop = padded[
        y:y + 2*r,
        x:x + 2*r
    ]

    zoom = cv2.resize(
        crop,
        (ZOOM_SIZE, ZOOM_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # Corner markers instead of obscuring the shuttle.
    center = ZOOM_SIZE // 2
    cv2.circle(
        zoom, (center, center),
        15, (0, 255, 255), 2
    )

    tile[
        42:42 + ZOOM_SIZE,
        345:345 + ZOOM_SIZE
    ] = zoom

    cv2.putText(
        tile,
        f"{timestamp:.2f}s | ({x}, {y})",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255),
        2, cv2.LINE_AA
    )

    cv2.putText(
        tile, "FULL FRAME",
        (10, TILE_H - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (180, 180, 180), 1
    )

    cv2.putText(
        tile, "12px ZOOM",
        (345, TILE_H - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (180, 180, 180), 1
    )

    tiles.append(tile)

cap.release()

if not tiles:
    raise SystemExit("No visible shuttle labels found.")

rows = math.ceil(len(tiles) / COLUMNS)

canvas = np.full(
    (rows * TILE_H, COLUMNS * TILE_W, 3),
    (12, 12, 12),
    dtype=np.uint8
)

for i, tile in enumerate(tiles):
    row = i // COLUMNS
    col = i % COLUMNS

    y0 = row * TILE_H
    x0 = col * TILE_W

    canvas[
        y0:y0 + TILE_H,
        x0:x0 + TILE_W
    ] = tile

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(OUTPUT), canvas):
    raise SystemExit("Failed to save image.")

print(f"Reviewed {len(tiles)} visible annotations")
print(f"Saved: {OUTPUT}")
print("No existing annotations were modified.")
