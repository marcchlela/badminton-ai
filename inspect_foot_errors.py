
import cv2
import json
import math
import numpy as np
from pathlib import Path

VIDEO = Path("data/raw/clip.mp4")
TRACKING = Path("outputs/tracking_v3.json")
MANUAL = Path("outputs/manual_evaluation.json")
OUTPUT = Path("outputs/evaluation/foot_error_collage.jpg")

TIMES = [20.0, 20.6, 21.4, 21.7,
         22.2, 22.4, 22.6, 22.9]

THUMB_W = 640
THUMB_H = 360
COLS = 4

GREEN = (80, 220, 100)
PINK = (185, 90, 240)
CYAN = (255, 210, 80)
WHITE = (255, 255, 255)
BLACK = (15, 15, 15)


def load(path):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


tracking = {
    round(float(r["time"]), 2): r
    for r in load(TRACKING)
}

manual = {
    round(float(r["time"]), 2): r
    for r in load(MANUAL)["annotations"]
}

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit("Could not open source video.")

fps = cap.get(cv2.CAP_PROP_FPS)
tiles = []


def dot(frame, point, color, radius=7):
    x, y = map(lambda n: int(round(n)), point)

    cv2.circle(frame, (x, y), radius + 2,
               BLACK, -1, cv2.LINE_AA)

    cv2.circle(frame, (x, y), radius,
               color, -1, cv2.LINE_AA)


def text(frame, message, x, y, color=WHITE):
    cv2.putText(
        frame, message, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.63, color, 2, cv2.LINE_AA
    )


try:
    for timestamp in TIMES:
        record = tracking.get(timestamp)
        reference = manual.get(timestamp)

        if record is None or reference is None:
            print(f"Missing data at {timestamp:.2f}s")
            continue

        frame_id = round(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ok, frame = cap.read()

        if not ok:
            print(f"Could not read {timestamp:.2f}s")
            continue

        h, w = frame.shape[:2]
        messages = []

        for side in ("Near", "Far"):
            detected = record["players"].get(side)
            annotated = reference["players"].get(side)

            if detected is None or annotated is None:
                continue

            bbox = detected.get("bbox")
            foot = detected.get("pixel")
            truth = annotated.get("pixel")

            if bbox is not None:
                x, y, bw, bh = map(int, bbox)

                cv2.rectangle(
                    frame, (x, y),
                    (x + bw, y + bh),
                    GREEN, 2
                )

                text(frame, side, x, max(y - 8, 50), GREEN)

            if foot is not None:
                dot(frame, foot, PINK)

            if truth is not None:
                dot(frame, truth, CYAN)

                if foot is not None:
                    a = tuple(map(int, foot))
                    b = tuple(map(int, truth))

                    cv2.line(
                        frame, a, b,
                        WHITE, 2, cv2.LINE_AA
                    )

                    error_px = math.dist(foot, truth)

                    messages.append(
                        f"{side}: {error_px:.1f}px"
                    )

            else:
                messages.append(
                    f"{side}: ground position estimated"
                )

        # Resize only AFTER drawing in original coordinates.
        tile = cv2.resize(
            frame, (THUMB_W, THUMB_H),
            interpolation=cv2.INTER_AREA
        )

        # Timestamp header
        cv2.rectangle(
            tile, (0, 0),
            (THUMB_W, 37),
            BLACK, -1
        )

        text(tile, f"TIME {timestamp:.2f}s", 12, 26)

        # Error labels
        cv2.rectangle(
            tile, (0, THUMB_H - 50),
            (THUMB_W, THUMB_H),
            BLACK, -1
        )

        for i, message in enumerate(messages[:2]):
            text(
                tile, message,
                12 + i * 310,
                THUMB_H - 17
            )

        tiles.append(tile)

finally:
    cap.release()


if not tiles:
    raise SystemExit("No valid frames found.")

rows = math.ceil(len(tiles) / COLS)

sheet = np.full(
    (rows * THUMB_H + 65,
     COLS * THUMB_W, 3),
    BLACK, dtype=np.uint8
)

for i, tile in enumerate(tiles):
    row = i // COLS
    col = i % COLS

    y = row * THUMB_H
    x = col * THUMB_W

    sheet[
        y:y + THUMB_H,
        x:x + THUMB_W
    ] = tile


legend_y = rows * THUMB_H + 40

text(sheet, "GREEN: YOLO BOX", 20, legend_y, GREEN)
text(sheet, "PINK: YOLO FOOT", 440, legend_y, PINK)
text(sheet, "CYAN: MANUAL FOOT", 900, legend_y, CYAN)

OUTPUT.parent.mkdir(
    parents=True, exist_ok=True
)

if not cv2.imwrite(str(OUTPUT), sheet):
    raise SystemExit("Could not save collage.")

print(f"Saved: {OUTPUT}")
print(f"Frames compared: {len(tiles)}")
