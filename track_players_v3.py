
import argparse
import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# 1. CONFIG
# ============================================================

parser = argparse.ArgumentParser(description="Track players in a fixed-camera rally.")
parser.add_argument("--video", type=Path, default=Path("data/raw/clip.mp4"))
parser.add_argument("--model", type=Path, default=Path("models/yolov8n.onnx"))
parser.add_argument("--calibration", type=Path, default=Path("outputs/court_points.json"))
parser.add_argument("--output", type=Path, default=Path("outputs/tracking_v3.json"))
parser.add_argument("--start", type=float, default=20.0)
parser.add_argument("--duration", type=float, default=3.0)
args = parser.parse_args()

VIDEO = args.video
MODEL = args.model
CALIBRATION = args.calibration
OUTPUT = args.output

START = args.start
DURATION = args.duration
SAMPLE_FPS = 10

MODEL_W, MODEL_H = 640, 480
MIN_CONF = 0.15
UPDATE_CONF = 0.28
NORMAL_CONF = 0.35

COURT_W, COURT_L = 6.10, 13.40
NET_Y = 6.70

MAX_PIXEL_MATCH = {
    "Near": 145,
    "Far": 105
}

MAX_COURT_UPDATE = 1.80
MAX_PREDICTION_AGE = 0.80

# ============================================================
# 2. COURT CALIBRATION
# ============================================================

with open(CALIBRATION, "r", encoding="utf-8") as f:
    image_points = np.float32(json.load(f))

court_points = np.float32([
    [0, 0],
    [6.10, 0],
    [6.10, 4.72],
    [0, 4.72]
])

H = cv2.getPerspectiveTransform(
    image_points, court_points
)


def to_court(x, y):
    p = np.float32([[[x, y]]])
    result = cv2.perspectiveTransform(p, H)[0, 0]
    return result.astype(float).tolist()


def valid_court(point, side):
    x, y = point

    if not (0 <= x <= COURT_W):
        return False

    if side == "Near":
        return 0 <= y <= NET_Y

    return NET_Y <= y <= COURT_L


# ============================================================
# 3. KALMAN FILTER
# ============================================================

def make_filter(dt, position_noise, velocity_noise,
                measurement_noise):

    kf = cv2.KalmanFilter(4, 2)

    # [x, y, vx, vy]
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kf.processNoiseCov = np.diag([
        position_noise,
        position_noise,
        velocity_noise,
        velocity_noise
    ]).astype(np.float32)

    kf.measurementNoiseCov = (
        np.eye(2, dtype=np.float32) * measurement_noise
    )

    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.zeros((4, 1), dtype=np.float32)

    return kf


def initialize(kf, position):
    x, y = position

    kf.statePost = np.array([
        [x], [y], [0], [0]
    ], dtype=np.float32)

    kf.errorCovPost = np.eye(4, dtype=np.float32)


def predict(kf):
    state = kf.predict()
    return state[:2, 0].astype(float)


def correct(kf, position):
    measurement = np.array(
        position, dtype=np.float32
    ).reshape(2, 1)

    state = kf.correct(measurement)
    return state[:2, 0].astype(float)


class PlayerTrack:

    def __init__(self, side):
        self.side = side

        # Tracks visible body movement in image pixels
        self.image_kf = make_filter(
            1 / SAMPLE_FPS,
            position_noise=4.0,
            velocity_noise=12.0,
            measurement_noise=16.0
        )

        # Tracks estimated floor position in metres
        self.court_kf = make_filter(
            1 / SAMPLE_FPS,
            position_noise=0.01,
            velocity_noise=0.03,
            measurement_noise=0.04
        )

        self.initialized = False
        self.last_ground_time = None

    def start(self, detection, time):
        initialize(
            self.image_kf,
            detection["center"]
        )

        initialize(
            self.court_kf,
            detection["court_m"]
        )

        self.initialized = True
        self.last_ground_time = time


# ============================================================
# 4. YOLO DETECTOR
# ============================================================

net = cv2.dnn.readNetFromONNX(str(MODEL))

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit("Could not open source video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise SystemExit("Invalid FPS.")


def detect_people(frame):
    height, width = frame.shape[:2]

    scale = min(
        MODEL_W / width,
        MODEL_H / height
    )

    new_w = round(width * scale)
    new_h = round(height * scale)

    resized = cv2.resize(
        frame, (new_w, new_h)
    )

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
        1 / 255.0,
        (MODEL_W, MODEL_H),
        swapRB=True
    )

    net.setInput(blob)
    output = net.forward()

    if output.shape != (1, 84, 6300):
        raise ValueError(
            f"Unexpected YOLO output {output.shape}"
        )

    boxes = []
    scores = []

    for pred in output[0].T:
        class_scores = pred[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if class_id != 0 or confidence < MIN_CONF:
            continue

        cx, cy, w, h = pred[:4]

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

        boxes.append([
            x1, y1, x2 - x1, y2 - y1
        ])
        scores.append(confidence)

    indices = cv2.dnn.NMSBoxes(
        boxes, scores, MIN_CONF, 0.45
    )

    detections = []

    for index in np.asarray(
        indices, dtype=int
    ).reshape(-1):

        x, y, w, h = boxes[index]

        center = [
            x + w / 2,
            y + h / 2
        ]

        foot = [
            x + w / 2,
            y + h
        ]

        detections.append({
            "bbox": [x, y, w, h],
            "center": center,
            "pixel": foot,
            "court_m": to_court(*foot),
            "confidence": scores[index]
        })

    return detections


# ============================================================
# 5. TRACKING
# ============================================================

tracks = {
    "Near": PlayerTrack("Near"),
    "Far": PlayerTrack("Far")
}

results = []

try:
    for i in range(round(DURATION * SAMPLE_FPS)):
        time = START + i / SAMPLE_FPS
        frame_id = round(time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()

        if not ok:
            print(f"Could not read frame {frame_id}")
            continue

        detections = detect_people(frame)
        used = set()

        predicted_image = {}
        predicted_court = {}

        for side, track in tracks.items():
            if track.initialized:
                predicted_image[side] = predict(
                    track.image_kf
                )
                predicted_court[side] = predict(
                    track.court_kf
                )

        record = {
            "time": round(time, 3),
            "frame": frame_id,
            "players": {}
        }

        for side, track in tracks.items():
            chosen = None
            chosen_index = None

            if not track.initialized:
                # Bootstrap using plausible ground positions.
                candidates = [
                    (idx, d)
                    for idx, d in enumerate(detections)
                    if idx not in used
                    and d["confidence"] >= UPDATE_CONF
                    and valid_court(d["court_m"], side)
                ]

                if candidates:
                    chosen_index, chosen = max(
                        candidates,
                        key=lambda item: item[1]["confidence"]
                    )

            else:
                # Match the player's BODY in image space.
                # Ground projection is NOT used for association.
                best_distance = float("inf")

                for idx, d in enumerate(detections):
                    if idx in used:
                        continue

                    center = np.array(d["center"])

                    distance = float(np.linalg.norm(
                        center - predicted_image[side]
                    ))

                    # Keep associations reasonably local.
                    if (
                        distance < MAX_PIXEL_MATCH[side]
                        and distance < best_distance
                    ):
                        best_distance = distance
                        chosen_index = idx
                        chosen = d

            if chosen_index is not None:
                used.add(chosen_index)

            if chosen is not None and not track.initialized:
                track.start(chosen, time)

                image_xy = np.array(chosen["center"])
                ground_xy = np.array(chosen["court_m"])

                source = (
                    "measured"
                    if chosen["confidence"] >= NORMAL_CONF
                    else "low_confidence"
                )
                ground_updated = True

            elif track.initialized:
                image_xy = predicted_image[side]
                ground_xy = predicted_court[side]

                source = "predicted"
                ground_updated = False

                if chosen is not None:
                    # Always use a matched detection to update
                    # IMAGE tracking, including during jumps.
                    image_xy = correct(
                        track.image_kf,
                        chosen["center"]
                    )

                    candidate_ground = np.array(
                        chosen["court_m"]
                    )

                    ground_distance = float(np.linalg.norm(
                        candidate_ground - ground_xy
                    ))

                    # Update court position ONLY when the
                    # foot projection looks trustworthy.
                    if (
                        chosen["confidence"] >= UPDATE_CONF
                        and valid_court(candidate_ground, side)
                        and ground_distance <= MAX_COURT_UPDATE
                    ):
                        ground_xy = correct(
                            track.court_kf,
                            candidate_ground
                        )

                        track.last_ground_time = time
                        ground_updated = True

                        source = (
                            "measured"
                            if chosen["confidence"] >= NORMAL_CONF
                            else "low_confidence"
                        )
                    else:
                        source = "predicted_from_track"

            else:
                print(f"{time:.2f}s {side}: not initialized")
                continue

            # Don't keep an unsupported prediction forever.
            age = time - track.last_ground_time

            if age > MAX_PREDICTION_AGE and not ground_updated:
                source = "unavailable"
                court_output = None
            else:
                court_output = ground_xy.tolist()

            record["players"][side] = {
                "side": side,
                "confidence": (
                    chosen["confidence"]
                    if chosen is not None else None
                ),
                "pixel": (
                    chosen["pixel"]
                    if chosen is not None else None
                ),
                "bbox": (
                    chosen["bbox"]
                    if chosen is not None else None
                ),
                "image_center": image_xy.tolist(),
                "court_m": court_output,
                "raw_court_m": (
                    chosen["court_m"]
                    if chosen is not None else None
                ),
                "visual_detection": chosen is not None,
                "ground_updated": ground_updated,
                "ground_age_seconds": round(age, 3),
                "position_source": source
            }

            print(
                f"{time:.2f}s {side}: {source}"
                f" | visual={chosen is not None}"
                f" | ground_updated={ground_updated}"
            )

        results.append(record)

finally:
    cap.release()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {OUTPUT}")
print(f"Processed: {len(results)} frames")
