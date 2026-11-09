
import cv2
import numpy as np
from pathlib import Path

VIDEO = Path("data/raw/high_quality_match.mp4")
OUTPUT = Path("outputs/hd_rally_preview.jpg")

START = 200
END = 210
STEP = 1

TILE_W = 480
TILE_H = 270
HEADER = 32
COLS = 3

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
tiles = []

for second in range(START, END + 1, STEP):
    frame_id = round(second * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()

    if not ok:
        continue

    preview = cv2.resize(
        frame, (TILE_W, TILE_H),
        interpolation=cv2.INTER_AREA
    )

    tile = np.zeros(
        (TILE_H + HEADER, TILE_W, 3),
        dtype=np.uint8
    )

    tile[HEADER:] = preview

    minutes, seconds = divmod(second, 60)

    cv2.putText(
        tile,
        f"{minutes:02d}:{seconds:02d}",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2
    )

    tiles.append(tile)

cap.release()

rows = int(np.ceil(len(tiles) / COLS))

sheet = np.zeros(
    (rows * (TILE_H + HEADER),
     COLS * TILE_W, 3),
    dtype=np.uint8
)

for i, tile in enumerate(tiles):
    y = (i // COLS) * (TILE_H + HEADER)
    x = (i % COLS) * TILE_W

    sheet[
        y:y + TILE_H + HEADER,
        x:x + TILE_W
    ] = tile

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(OUTPUT), sheet)

print("Frames previewed:", len(tiles))
print("Saved:", OUTPUT)
