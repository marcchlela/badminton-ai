
import cv2
import math
import numpy as np
from pathlib import Path

VIDEO = Path("data/raw/high_quality_match.mp4")
OUTPUT = Path("outputs/hd_preview.jpg")

INTERVAL = 10
TILE_W = 320
TILE_H = 180
HEADER = 30
COLUMNS = 5

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frames / fps

tiles = []

for second in range(0, math.ceil(duration), INTERVAL):
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        round(second * fps)
    )

    ok, frame = cap.read()

    if not ok:
        continue

    preview = cv2.resize(
        frame, (TILE_W, TILE_H),
        interpolation=cv2.INTER_AREA
    )

    tile = np.full(
        (TILE_H + HEADER, TILE_W, 3),
        (18, 18, 18),
        dtype=np.uint8
    )

    tile[HEADER:] = preview

    minutes, seconds = divmod(second, 60)

    cv2.putText(
        tile,
        f"{minutes:02d}:{seconds:02d}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    tiles.append(tile)

cap.release()

rows = math.ceil(len(tiles) / COLUMNS)

sheet = np.full(
    (rows * (TILE_H + HEADER),
     COLUMNS * TILE_W, 3),
    (12, 12, 12),
    dtype=np.uint8
)

for i, tile in enumerate(tiles):
    row = i // COLUMNS
    col = i % COLUMNS

    y = row * (TILE_H + HEADER)
    x = col * TILE_W

    sheet[
        y:y + TILE_H + HEADER,
        x:x + TILE_W
    ] = tile

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(OUTPUT), sheet)

print("Video duration:", round(duration, 2), "seconds")
print("Frames previewed:", len(tiles))
print("Saved:", OUTPUT)
