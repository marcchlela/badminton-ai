
import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

VIDEO_PATH = Path("data/raw/clip.mp4")
MODEL_PATH = Path("models/yolov8n.onnx")
OUTPUT_PATH = Path("outputs/tracking_sample.json")

START_SECONDS = 20.0
DURATION_SECONDS = 3.0
SAMPLE_FPS = 10

MODEL_W = 640
MODEL_H = 480

CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45

COURT_W = 6.10
COURT_L = 13.40


# ============================================================
# 2. LOAD CALIBRATION
# ============================================================

with open("outputs/court_points.json") as f:
    image_points = np.float32(json.load(f))

court_points = np.float32([
    [0, 0],
    [COURT_W, 0],
    [COURT_W, 4.72],
    [0, 4.72]
])

H = cv2.getPerspectiveTransform(image_points, court_points)


def to_court(x, y):
    point = np.float32([[[x, y]]])
    result = cv2.perspectiveTransform(point, H)
    return result[0, 0]


# ============================================================
# 3. LOAD MODEL AND VIDEO
# ============================================================

if not MODEL_PATH.exists():
    raise SystemExit("Model not found")

net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise SystemExit("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise SystemExit("Invalid video FPS")

print(f"Video FPS: {fps}")
print(f"Sampling {SAMPLE_FPS} frames per second")


# ============================================================
# 4. DETECT PLAYERS IN ONE FRAME
# ============================================================

def detect(frame):
    height, width = frame.shape[:2]

    # Letterbox preprocessing
    scale = min(MODEL_W / width, MODEL_H / height)

    new_w = round(width * scale)
    new_h = round(height * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    pad_x = (MODEL_W - new_w) // 2
    pad_y = (MODEL_H - new_h) // 2

    padded = np.full(
        (MODEL_H, MODEL_W, 3),
        114,
        dtype=np.uint8
    )

    padded[
        pad_y:pad_y + new_h,
        pad_x:pad_x + new_w
    ] = resized

    blob = cv2.dnn.blobFromImage(
        padded,
        scalefactor=1 / 255.0,
        size=(MODEL_W, MODEL_H),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    output = net.forward()

    if output.shape != (1, 84, 6300):
        raise ValueError(
            f"Unexpected model output: {output.shape}"
        )

    predictions = output[0].T

    boxes = []
    scores = []

    for prediction in predictions:
        class_scores = prediction[4:]

        # COCO class 0 = person
        if int(np.argmax(class_scores)) != 0:
            continue

        confidence = float(class_scores[0])

        if confidence < CONF_THRESHOLD:
            continue

        cx, cy, w, h = prediction[:4]

        x1 = int(np.clip(
            (cx - w / 2 - pad_x) / scale,
            0, width - 1
        ))

        y1 = int(np.clip(
            (cy - h / 2 - pad_y) / scale,
            0, height - 1
        ))

        x2 = int(np.clip(
            (cx + w / 2 - pad_x) / scale,
            0, width - 1
        ))

        y2 = int(np.clip(
            (cy + h / 2 - pad_y) / scale,
            0, height - 1
        ))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    selected = {}

    for index in np.asarray(indices, dtype=int).reshape(-1):
        x, y, w, h = boxes[index]

        foot_x = x + w / 2
        foot_y = y + h

        court_x, court_y = to_court(foot_x, foot_y)

        # Only keep on-court detections
        if not (
            0 <= court_x <= COURT_W
            and 0 <= court_y <= COURT_L
        ):
            continue

        side = "Near" if court_y < COURT_L / 2 else "Far"

        player = {
            "side": side,
            "confidence": float(scores[index]),
            "pixel": [float(foot_x), float(foot_y)],
            "court_m": [float(court_x), float(court_y)]
        }

        # One player per half for this singles prototype
        previous = selected.get(side)

        if previous is None or (
            player["confidence"] > previous["confidence"]
        ):
            selected[side] = player

    return selected


# ============================================================
# 5. PROCESS SAMPLED VIDEO FRAMES
# ============================================================

results = []

sample_count = round(
    DURATION_SECONDS * SAMPLE_FPS
)

try:
    for i in range(sample_count):
        timestamp = START_SECONDS + i / SAMPLE_FPS
        frame_number = round(timestamp * fps)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = cap.read()

        if not success:
            print(f"Could not read frame {frame_number}")
            continue

        players = detect(frame)

        record = {
            "frame": frame_number,
            "time": timestamp,
            "players": players
        }

        results.append(record)

        print(f"\nTime {timestamp:.2f}s")

        for side in ("Near", "Far"):
            if side in players:
                p = players[side]
                x, y = p["court_m"]
                print(
                    f"  {side}: ({x:.2f}, {y:.2f}) m "
                    f"| confidence {p['confidence']:.2f}"
                )
            else:
                print(f"  {side}: NOT DETECTED")

finally:
    cap.release()


# ============================================================
# 6. SAVE TRACKING DATA
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {OUTPUT_PATH}")
print(f"Processed {len(results)} sampled frames")
