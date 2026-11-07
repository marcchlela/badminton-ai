import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# 1. CONFIG
# ============================================================

VIDEO_PATH = Path("data/raw/clip.mp4")
MODEL_PATH = Path("models/yolov8n.onnx")
COURT_POINTS_PATH = Path("outputs/court_points.json")
OUTPUT_PATH = Path("outputs/tracking_kalman.json")

START_SECONDS = 20.0
DURATION_SECONDS = 3.0
SAMPLE_FPS = 10

MODEL_W = 640
MODEL_H = 480

DETECT_CONF = 0.15          # collect more candidates
NORMAL_CONF = 0.35          # normal good detection
LOW_CONF_UPDATE = 0.28      # allow lower-confidence updates if plausible
NMS_THRESHOLD = 0.45

COURT_W = 6.10
COURT_L = 13.40
NET_Y = COURT_L / 2.0
NEAR_SHORT_SERVICE_Y = NET_Y - 1.98

IMAGE_HALF_Y = 360          # heuristic split in 720p broadcast frame
MAX_PREDICT_GAP = 0.70      # seconds
MAX_MATCH_DIST = 1.80       # metres in court space

# ============================================================
# 2. LOAD COURT CALIBRATION
# ============================================================

with open(COURT_POINTS_PATH, "r", encoding="utf-8") as f:
    image_points = np.float32(json.load(f))

court_points = np.float32([
    [0.0, 0.0],
    [COURT_W, 0.0],
    [COURT_W, NEAR_SHORT_SERVICE_Y],
    [0.0, NEAR_SHORT_SERVICE_Y],
])

H = cv2.getPerspectiveTransform(image_points, court_points)

def to_court(x, y):
    pt = np.float32([[[x, y]]])
    projected = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(projected[0]), float(projected[1])

# ============================================================
# 3. LOAD VIDEO + MODEL
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise SystemExit("Could not open source video.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise SystemExit("Invalid video FPS.")

net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print(f"Video FPS: {fps}")
print(f"Sampling {SAMPLE_FPS} FPS")
print("Kalman tracker enabled")

# ============================================================
# 4. KALMAN FILTER
# ============================================================

def create_kalman(dt):
    """
    State = [x, y, vx, vy]
    Measurement = [x, y]
    """
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    kf.processNoiseCov = np.array([
        [1e-2, 0,    0,    0],
        [0,    1e-2, 0,    0],
        [0,    0,    2e-2, 0],
        [0,    0,    0,    2e-2],
    ], dtype=np.float32)

    kf.measurementNoiseCov = np.array([
        [5e-2, 0],
        [0, 5e-2],
    ], dtype=np.float32)

    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.zeros((4, 1), dtype=np.float32)

    return kf

def init_track(side, dt):
    return {
        "side": side,
        "kf": create_kalman(dt),
        "initialized": False,
        "last_measured_time": None,
        "last_output_time": None,
    }

dt = 1.0 / SAMPLE_FPS

tracks = {
    "Near": init_track("Near", dt),
    "Far": init_track("Far", dt),
}

def get_predicted_position(track):
    predicted = track["kf"].predict()
    x = float(predicted[0, 0])
    y = float(predicted[1, 0])
    return np.array([x, y], dtype=float)

def correct_track(track, x, y):
    measurement = np.array([[x], [y]], dtype=np.float32)
    corrected = track["kf"].correct(measurement)
    return np.array(
        [float(corrected[0, 0]), float(corrected[1, 0])],
        dtype=float
    )

def initialize_track(track, x, y, current_time):
    track["kf"].statePost = np.array([
        [x],
        [y],
        [0.0],
        [0.0],
    ], dtype=np.float32)
    track["initialized"] = True
    track["last_measured_time"] = current_time
    track["last_output_time"] = current_time

# ============================================================
# 5. DETECTION
# ============================================================

def detect_people(frame):
    height, width = frame.shape[:2]

    scale = min(MODEL_W / width, MODEL_H / height)
    new_w = round(width * scale)
    new_h = round(height * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    pad_x = (MODEL_W - new_w) // 2
    pad_y = (MODEL_H - new_h) // 2

    padded = np.full((MODEL_H, MODEL_W, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    blob = cv2.dnn.blobFromImage(
        padded,
        scalefactor=1.0 / 255.0,
        size=(MODEL_W, MODEL_H),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    output = net.forward()

    if output.shape != (1, 84, 6300):
        raise ValueError(f"Unexpected model output: {output.shape}")

    boxes = []
    scores = []

    for pred in output[0].T:
        class_scores = pred[4:]
        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])

        if class_id != 0:
            continue

        if conf < DETECT_CONF:
            continue

        cx, cy, w, h = pred[:4]

        x1 = int(np.clip((cx - w / 2 - pad_x) / scale, 0, width - 1))
        y1 = int(np.clip((cy - h / 2 - pad_y) / scale, 0, height - 1))
        x2 = int(np.clip((cx + w / 2 - pad_x) / scale, 0, width - 1))
        y2 = int(np.clip((cy + h / 2 - pad_y) / scale, 0, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(conf)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        DETECT_CONF,
        NMS_THRESHOLD
    )

    detections = []

    for idx in np.asarray(indices, dtype=int).reshape(-1):
        x, y, w, h = boxes[idx]
        conf = scores[idx]

        foot_x = x + w / 2
        foot_y = y + h

        court_x, court_y = to_court(foot_x, foot_y)

        on_court = (
            0.0 <= court_x <= COURT_W and
            0.0 <= court_y <= COURT_L
        )

        image_side = "Near" if foot_y >= IMAGE_HALF_Y else "Far"

        detections.append({
            "confidence": float(conf),
            "bbox": [int(x), int(y), int(w), int(h)],
            "pixel": [float(foot_x), float(foot_y)],
            "court_m": [float(court_x), float(court_y)],
            "on_court": bool(on_court),
            "image_side": image_side,
        })

    return detections

# ============================================================
# 6. MATCH DETECTIONS TO TRACK
# ============================================================

def choose_candidate(side, detections, predicted_position):
    """
    Prefer same-side detections.
    Use prediction to choose the most plausible one.
    """
    candidates = [d for d in detections if d["image_side"] == side]

    if not candidates:
        return None

    if predicted_position is None:
        # Prefer on-court candidates, then higher confidence
        candidates.sort(
            key=lambda d: (d["on_court"], d["confidence"]),
            reverse=True
        )
        return candidates[0]

    best = None
    best_score = None

    for d in candidates:
        cx, cy = d["court_m"]

        # clamp for rough distance comparison
        if side == "Near":
            cy_cmp = np.clip(cy, 0.0, NET_Y)
        else:
            cy_cmp = np.clip(cy, NET_Y, COURT_L + 1.5)

        cx_cmp = np.clip(cx, -0.5, COURT_W + 0.5)

        dist = np.linalg.norm(
            np.array([cx_cmp, cy_cmp]) - predicted_position
        )

        penalty = 0.0 if d["on_court"] else 0.7
        score = dist + penalty - 0.15 * d["confidence"]

        if best_score is None or score < best_score:
            best_score = score
            best = d

    if best is None:
        return None

    if predicted_position is not None:
        cx, cy = best["court_m"]
        cy_cmp = np.clip(
            cy,
            0.0 if side == "Near" else NET_Y,
            NET_Y if side == "Near" else COURT_L + 1.5
        )
        cx_cmp = np.clip(cx, -0.5, COURT_W + 0.5)

        dist = np.linalg.norm(
            np.array([cx_cmp, cy_cmp]) - predicted_position
        )

        if dist > MAX_MATCH_DIST:
            return None

    return best

# ============================================================
# 7. MAIN LOOP
# ============================================================

results = []

sample_count = round(DURATION_SECONDS * SAMPLE_FPS)

try:
    for i in range(sample_count):
        timestamp = START_SECONDS + i / SAMPLE_FPS
        frame_number = round(timestamp * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()

        if not ok:
            print(f"Could not read frame {frame_number}")
            continue

        detections = detect_people(frame)

        record = {
            "frame": frame_number,
            "time": round(timestamp, 3),
            "players": {}
        }

        print(f"\nTime {timestamp:.2f}s")

        for side in ("Near", "Far"):
            track = tracks[side]

            predicted = None
            if track["initialized"]:
                predicted = get_predicted_position(track)

            chosen = choose_candidate(side, detections, predicted)

            output_player = None

            if chosen is not None:
                cx, cy = chosen["court_m"]
                conf = chosen["confidence"]

                # Case 1: track not initialized yet
                if not track["initialized"]:
                    if chosen["on_court"]:
                        initialize_track(track, cx, cy, timestamp)
                        output_player = {
                            "side": side,
                            "confidence": conf,
                            "pixel": chosen["pixel"],
                            "court_m": [cx, cy],
                            "bbox": chosen["bbox"],
                            "position_source": "measured"
                            if conf >= NORMAL_CONF else "low_confidence",
                        }

                else:
                    # Case 2: usable on-court measurement
                    if chosen["on_court"] and conf >= LOW_CONF_UPDATE:
                        filtered = correct_track(track, cx, cy)
                        track["last_measured_time"] = timestamp
                        track["last_output_time"] = timestamp

                        output_player = {
                            "side": side,
                            "confidence": conf,
                            "pixel": chosen["pixel"],
                            "court_m": filtered.tolist(),
                            "bbox": chosen["bbox"],
                            "position_source": "measured"
                            if conf >= NORMAL_CONF else "low_confidence",
                        }

                    # Case 3: seen visually, but ground point unreliable
                    elif predicted is not None:
                        output_player = {
                            "side": side,
                            "confidence": conf,
                            "pixel": chosen["pixel"],
                            "court_m": predicted.tolist(),
                            "bbox": chosen["bbox"],
                            "raw_court_m": chosen["court_m"],
                            "position_source": "predicted_from_track",
                        }
                        track["last_output_time"] = timestamp

            # Case 4: no usable detection -> pure prediction
            if output_player is None and track["initialized"]:
                last_seen = track["last_measured_time"]

                if last_seen is not None and (timestamp - last_seen) <= MAX_PREDICT_GAP:
                    predicted_now = predicted if predicted is not None else get_predicted_position(track)

                    output_player = {
                        "side": side,
                        "confidence": None,
                        "pixel": None,
                        "court_m": predicted_now.tolist(),
                        "bbox": None,
                        "position_source": "predicted",
                    }
                    track["last_output_time"] = timestamp

            if output_player is not None:
                record["players"][side] = output_player

                x_m, y_m = output_player["court_m"]
                src = output_player["position_source"]

                print(
                    f"  {side}: ({x_m:.2f}, {y_m:.2f}) m "
                    f"| {src}"
                )
            else:
                print(f"  {side}: NOT AVAILABLE")

        results.append(record)

finally:
    cap.release()

# ============================================================
# 8. SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {OUTPUT_PATH}")
print(f"Processed {len(results)} samples")