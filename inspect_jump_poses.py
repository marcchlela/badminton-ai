
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

VIDEO = "data/raw/clip.mp4"
OUT = Path("outputs/pose_motion/jump_inspection.jpg")

CASES = [
    ("20_23", 22.3, "Far"),
    ("20_23", 22.4, "Far"),
    ("20_23", 22.5, "Far"),
    ("20_23", 22.6, "Far"),
    ("20_23", 22.7, "Far"),
    ("20_23", 22.8, "Far"),
    ("36_40", 36.9, "Near"),
    ("36_40", 37.2, "Near"),
    ("36_40", 37.5, "Near"),
    ("36_40", 39.6, "Near"),
    ("36_40", 39.9, "Near"),
]

data = {
    name: pd.read_csv(
        f"outputs/pose_motion/rally_{name}_features.csv"
    )
    for name in ("20_23", "36_40")
}

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("Cannot open video")

tiles = []

for rally, time, side in CASES:
    df = data[rally]

    row = df[
        (abs(df["time"] - time) < 0.001)
        & (df["side"] == side)
    ]

    if row.empty:
        continue

    r = row.iloc[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame"]))
    ok, frame = cap.read()
    if not ok:
        continue

    height, width = frame.shape[:2]

    x = float(r["hip_x_px"]) if pd.notna(r["hip_x_px"]) else np.nan
    bx_top = int(r["bbox_top_px"])
    bx_bottom = int(r["bbox_bottom_px"])

    # Center crop around the player using available pose
    # coordinates when present. Otherwise use video center.
    if pd.notna(x):
        cx = int(x)
    elif side == "Far":
        cx = int(width * 0.5)
    else:
        cx = int(width * 0.5)

    cy = (bx_top + bx_bottom) // 2
    half_w = 125
    half_h = 135

    x1 = max(0, cx - half_w)
    x2 = min(width, cx + half_w)
    y1 = max(0, cy - half_h)
    y2 = min(height, cy + half_h)

    crop = frame[y1:y2, x1:x2].copy()

    if pd.notna(r["ankle_x_px"]) and pd.notna(r["ankle_y_px"]):
        px = int(r["ankle_x_px"]) - x1
        py = int(r["ankle_y_px"]) - y1
        cv2.circle(crop, (px, py), 6, (0, 255, 255), -1)

    tile = cv2.resize(crop, (320, 320))

    cv2.rectangle(tile, (0, 0), (320, 58), (0, 0, 0), -1)

    label = str(r["manual_label"])
    cv2.putText(
        tile, f"{time:.1f}s {side}",
        (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2
    )
    cv2.putText(
        tile, label, (8, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (0, 255, 255), 1
    )

    tiles.append(tile)

cap.release()

cols = 3
rows = int(np.ceil(len(tiles) / cols))

canvas = np.zeros(
    (rows * 320, cols * 320, 3), dtype=np.uint8
)

for i, tile in enumerate(tiles):
    y = (i // cols) * 320
    x = (i % cols) * 320
    canvas[y:y+320, x:x+320] = tile

OUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUT), canvas)

print("Saved:", OUT)
