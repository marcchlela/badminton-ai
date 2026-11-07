
import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

VIDEO = Path("data/raw/clip.mp4")
MODEL = Path("models/yolov8n.onnx")
OUTPUT = Path("outputs/debug_detections")

TIMES = [
    21.7,
    22.3,
    22.4,
    22.5,
    22.6,
    22.7,
    22.8,
    22.9
]

MODEL_W = 640
MODEL_H = 480

DEBUG_CONF = 0.15
NORMAL_CONF = 0.35
NMS_THRESHOLD = 0.45

COURT_W = 6.10
COURT_L = 13.40

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD COURT CALIBRATION
# ============================================================

with open("outputs/court_points.json", "r") as f:
    image_points = np.float32(json.load(f))

court_points = np.float32([
    [0, 0],
    [COURT_W, 0],
    [COURT_W, 4.72],
    [0, 4.72]
])

H = cv2.getPerspectiveTransform(
    image_points, court_points
)


def to_court(x, y):
    p = np.float32([[[x, y]]])
    return cv2.perspectiveTransform(p, H)[0, 0]


# ============================================================
# 3. LOAD MODEL AND VIDEO
# ============================================================

net = cv2.dnn.readNetFromONNX(str(MODEL))
cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit("Could not open source video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise SystemExit("Invalid video FPS.")

print(f"Video FPS: {fps}")
print(f"Diagnostic threshold: {DEBUG_CONF}")
print(f"Normal threshold: {NORMAL_CONF}")


# ============================================================
# 4. DETECT ALL POSSIBLE PERSONS
# ============================================================

def detect_people(frame):
    height, width = frame.shape[:2]

    scale = min(
        MODEL_W / width,
        MODEL_H / height
    )

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
        swapRB=True
    )

    net.setInput(blob)
    output = net.forward()

    if output.shape != (1, 84, 6300):
        raise ValueError(
            f"Unexpected output shape: {output.shape}"
        )

    boxes = []
    scores = []

    for pred in output[0].T:
        class_scores = pred[4:]

        if int(np.argmax(class_scores)) != 0:
            continue

        conf = float(class_scores[0])

        if conf < DEBUG_CONF:
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

        scores.append(conf)

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        DEBUG_CONF,
        NMS_THRESHOLD
    )

    detections = []

    for index in np.asarray(
        indices, dtype=int
    ).reshape(-1):

        x, y, w, h = boxes[index]
        conf = scores[index]

        foot_x = x + w / 2
        foot_y = y + h

        court_x, court_y = to_court(
            foot_x, foot_y
        )

        on_court = (
            0 <= court_x <= COURT_W
            and 0 <= court_y <= COURT_L
        )

        passes_conf = conf >= NORMAL_CONF

        accepted = on_court and passes_conf

        detections.append({
            "confidence": conf,
            "bbox": [x, y, w, h],
            "foot": [foot_x, foot_y],
            "court": [float(court_x), float(court_y)],
            "on_court": bool(on_court),
            "passes_conf": bool(passes_conf),
            "accepted": bool(accepted)
        })

    return detections


# ============================================================
# 5. VISUALIZE DIAGNOSTIC DETECTIONS
# ============================================================

def annotate(frame, detections, timestamp):
    result = frame.copy()

    for d in detections:
        x, y, w, h = d["bbox"]
        fx, fy = d["foot"]

        if d["accepted"]:
            color = (0, 255, 0)       # Green
            status = "ACCEPT"
        elif not d["on_court"]:
            color = (0, 0, 255)       # Red
            status = "OUTSIDE"
        else:
            color = (0, 255, 255)     # Yellow
            status = "LOW CONF"

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.circle(
            result,
            (round(fx), round(fy)),
            5,
            color,
            -1
        )

        label = (
            f"{d['confidence']:.2f} {status}"
        )

        cv2.putText(
            result,
            label,
            (x, max(y - 6, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        result,
        f"TIME: {timestamp:.2f}s",
        (440, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return result


# ============================================================
# 6. PROCESS PROBLEMATIC FRAMES
# ============================================================

contact_frames = []
all_results = []

try:
    for timestamp in TIMES:
        frame_id = round(timestamp * fps)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_id
        )

        ok, frame = cap.read()

        if not ok:
            print(f"Failed at {timestamp:.2f}s")
            continue

        detections = detect_people(frame)

        print(f"\n===== {timestamp:.2f}s =====")

        for d in detections:
            cx, cy = d["court"]

            print(
                f"Conf={d['confidence']:.2f} "
                f"Court=({cx:.2f}, {cy:.2f}) "
                f"OnCourt={d['on_court']} "
                f"Accepted={d['accepted']}"
            )

        annotated = annotate(
            frame, detections, timestamp
        )

        filename = OUTPUT / (
            f"frame_{timestamp:.1f}s.jpg"
        )

        cv2.imwrite(str(filename), annotated)

        # Smaller preview for contact sheet
        preview = cv2.resize(
            annotated,
            (640, 360)
        )

        contact_frames.append(preview)

        all_results.append({
            "time": timestamp,
            "frame": frame_id,
            "detections": detections
        })

finally:
    cap.release()


# ============================================================
# 7. SAVE RESULTS
# ============================================================

with open(
    OUTPUT / "diagnostics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(all_results, f, indent=2)

if contact_frames:
    cols = 2
    rows = (
        len(contact_frames) + cols - 1
    ) // cols

    sheet = np.full(
        (rows * 360, cols * 640, 3),
        (20, 20, 20),
        dtype=np.uint8
    )

    for i, frame in enumerate(contact_frames):
        row = i // cols
        col = i % cols

        sheet[
            row * 360:(row + 1) * 360,
            col * 640:(col + 1) * 640
        ] = frame

    cv2.imwrite(
        str(OUTPUT / "contact_sheet.jpg"),
        sheet
    )

print("\nDiagnostic analysis complete.")
print(f"Saved results in: {OUTPUT}")
