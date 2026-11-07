
import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/yolov8n.onnx")
IMAGE_PATH = Path("outputs/calibration_frame.jpg")

DETECTION_IMAGE_PATH = Path("outputs/detected_people.jpg")
EVALUATION_IMAGE_PATH = Path("outputs/detection_evaluation.jpg")
AUTO_PLAYERS_PATH = Path("outputs/auto_players.json")

MODEL_WIDTH = 640
MODEL_HEIGHT = 480

CONFIDENCE_THRESHOLD = 0.35
NMS_THRESHOLD = 0.45

# COCO class ID for a person
PERSON_CLASS = 0

COURT_WIDTH = 6.10
COURT_LENGTH = 13.40
NET_Y = COURT_LENGTH / 2.0
NEAR_SHORT_SERVICE_Y = NET_Y - 1.98


# ============================================================
# 2. LOAD IMAGE AND YOLO MODEL
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise SystemExit("Could not load calibration frame.")

if not MODEL_PATH.exists():
    raise SystemExit("YOLO model not found.")

height, width = image.shape[:2]

net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

print(f"Original image: {width} x {height}")
print("Model loaded successfully")


# ============================================================
# 3. PREPROCESS IMAGE
# ============================================================

# Preserve aspect ratio using letterboxing
scale = min(
    MODEL_WIDTH / width,
    MODEL_HEIGHT / height
)

new_width = round(width * scale)
new_height = round(height * scale)

resized = cv2.resize(
    image,
    (new_width, new_height)
)

pad_x = (MODEL_WIDTH - new_width) // 2
pad_y = (MODEL_HEIGHT - new_height) // 2

letterbox = np.full(
    (MODEL_HEIGHT, MODEL_WIDTH, 3),
    114,
    dtype=np.uint8
)

letterbox[
    pad_y:pad_y + new_height,
    pad_x:pad_x + new_width
] = resized

# BGR -> RGB, normalize to 0-1,
# and convert into NCHW tensor format
blob = cv2.dnn.blobFromImage(
    letterbox,
    scalefactor=1.0 / 255.0,
    size=(MODEL_WIDTH, MODEL_HEIGHT),
    swapRB=True,
    crop=False
)

print(f"Neural network input: {blob.shape}")


# ============================================================
# 4. RUN MODEL INFERENCE
# ============================================================

net.setInput(blob)
output = net.forward()

print(f"Model output shape: {output.shape}")

# This specific YOLOv8 ONNX model:
# (1, 84, 6300)
if (
    output.ndim != 3
    or output.shape[0] != 1
    or output.shape[1] != 84
):
    raise SystemExit(
        "Unexpected YOLO model output format."
    )

predictions = output[0].T


# ============================================================
# 5. EXTRACT PERSON DETECTIONS
# ============================================================

boxes = []
confidences = []

for prediction in predictions:

    class_scores = prediction[4:]

    class_id = int(np.argmax(class_scores))
    confidence = float(class_scores[class_id])

    if class_id != PERSON_CLASS:
        continue

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    cx, cy, w, h = prediction[:4]

    # Undo preprocessing to recover original pixels
    x1 = (cx - w / 2 - pad_x) / scale
    y1 = (cy - h / 2 - pad_y) / scale
    x2 = (cx + w / 2 - pad_x) / scale
    y2 = (cy + h / 2 - pad_y) / scale

    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y2 = int(np.clip(y2, 0, height - 1))

    if x2 <= x1 or y2 <= y1:
        continue

    boxes.append([
        x1,
        y1,
        x2 - x1,
        y2 - y1
    ])

    confidences.append(confidence)


# ============================================================
# 6. NON-MAXIMUM SUPPRESSION
# ============================================================

indices = cv2.dnn.NMSBoxes(
    boxes,
    confidences,
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD
)

indices = np.asarray(indices, dtype=int).reshape(-1)

print(f"Candidate boxes: {len(boxes)}")
print(f"Detections after NMS: {len(indices)}")


# ============================================================
# 7. DRAW PERSON BOUNDING BOXES
# ============================================================

preview = image.copy()

for index in indices:

    x, y, w, h = boxes[int(index)]
    confidence = confidences[int(index)]

    cv2.rectangle(
        preview,
        (x, y),
        (x + w, y + h),
        (0, 255, 255),
        2
    )

    cv2.putText(
        preview,
        f"Person {confidence:.2f}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2
    )

    print(
        f"Person: confidence={confidence:.2f}, "
        f"box=({x}, {y}, {w}, {h})"
    )

cv2.imwrite(str(DETECTION_IMAGE_PATH), preview)

print(f"\nSaved: {DETECTION_IMAGE_PATH}")


# ============================================================
# 8. LOAD COURT CALIBRATION
# ============================================================

with open("outputs/court_points.json", "r") as f:
    calibration_pixels = np.float32(json.load(f))

court_reference = np.float32([
    [0.0, 0.0],
    [COURT_WIDTH, 0.0],
    [COURT_WIDTH, NEAR_SHORT_SERVICE_Y],
    [0.0, NEAR_SHORT_SERVICE_Y]
])

H = cv2.getPerspectiveTransform(
    calibration_pixels,
    court_reference
)

with open("outputs/player_positions.json", "r") as f:
    manual_players = json.load(f)


def pixel_to_court(x, y):
    """Convert image pixels to badminton court metres."""

    point = np.float32([[[x, y]]])

    transformed = cv2.perspectiveTransform(
        point, H
    )

    return transformed[0, 0]


# ============================================================
# 9. CONVERT AI DETECTIONS TO COURT POSITIONS
# ============================================================

selected_players = {}

for index in indices:

    x, y, w, h = boxes[int(index)]
    confidence = confidences[int(index)]

    # Approximate ground-contact point
    foot_x = x + w / 2
    foot_y = y + h

    court_x, court_y = pixel_to_court(
        foot_x,
        foot_y
    )

    # Discard detections outside the court
    if not (
        0 <= court_x <= COURT_WIDTH
        and 0 <= court_y <= COURT_LENGTH
    ):
        continue

    # Singles: select one detection per court half
    side = "Near" if court_y < NET_Y else "Far"

    candidate = {
        "name": side,
        "confidence": float(confidence),
        "pixel": [
            float(foot_x),
            float(foot_y)
        ],
        "court_m": [
            float(court_x),
            float(court_y)
        ],
        "bbox": [
            int(x), int(y),
            int(w), int(h)
        ]
    }

    previous = selected_players.get(side)

    if (
        previous is None
        or confidence > previous["confidence"]
    ):
        selected_players[side] = candidate


# ============================================================
# 10. EVALUATE AGAINST MANUAL ANNOTATIONS
# ============================================================

print("\n========== PLAYER POSITION EVALUATION ==========")

evaluation_image = preview.copy()

for i, side in enumerate(["Near", "Far"]):

    if side not in selected_players:
        print(f"{side}: No player detected")
        continue

    detected = selected_players[side]
    manual = manual_players[i]

    ai_pixel = np.array(
        detected["pixel"],
        dtype=float
    )

    manual_pixel = np.array(
        manual["pixel"],
        dtype=float
    )

    ai_court = np.array(
        detected["court_m"],
        dtype=float
    )

    manual_court = pixel_to_court(
        *manual_pixel
    )

    # Euclidean distance in image coordinates
    pixel_error = np.linalg.norm(
        ai_pixel - manual_pixel
    )

    # Euclidean distance in court coordinates
    court_error = np.linalg.norm(
        ai_court - manual_court
    )

    print(f"\n{side} player")
    print(f"  Confidence: {detected['confidence']:.2f}")
    print(f"  AI pixel position: {ai_pixel}")
    print(f"  Manual pixel position: {manual_pixel}")
    print(f"  Pixel error: {pixel_error:.2f} px")
    print(f"  Court error: {court_error:.3f} m")

    # Green: AI-estimated foot position
    cv2.circle(
        evaluation_image,
        tuple(np.round(ai_pixel).astype(int)),
        6,
        (0, 255, 0),
        -1
    )

    # Red: manually annotated position
    cv2.circle(
        evaluation_image,
        tuple(manual_pixel.astype(int)),
        5,
        (0, 0, 255),
        -1
    )


# ============================================================
# 11. SAVE AUTOMATIC PLAYER POSITIONS
# ============================================================

# The calibration image is video frame 500 at 25 FPS.
results = {
    "frame": 500,
    "time_seconds": 20.0,
    "players": [
        selected_players[side]
        for side in ["Near", "Far"]
        if side in selected_players
    ]
}

AUTO_PLAYERS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    AUTO_PLAYERS_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

if not cv2.imwrite(
    str(EVALUATION_IMAGE_PATH),
    evaluation_image
):
    raise SystemExit("Failed to save evaluation image.")

print(f"\nSaved {AUTO_PLAYERS_PATH}")
print(f"Saved {EVALUATION_IMAGE_PATH}")
