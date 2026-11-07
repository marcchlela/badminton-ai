
import cv2
import csv
import json
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = Path("data/raw/clip.mp4")
TRACKING = Path("outputs/tracking_v3.json")
MANUAL = Path("outputs/manual_evaluation.json")
CALIBRATION = Path("outputs/court_points.json")

OUT = Path("outputs/pose")
OUT.mkdir(parents=True, exist_ok=True)

TIMES = [
    20.0, 20.6, 21.4, 21.7,
    22.2, 22.4, 22.6, 22.9
]

MODEL_NAME = "yolo11n-pose.pt"
IMAGE_SIZE = 960
MIN_ANKLE_CONF = 0.30
MIN_BOX_IOU = 0.10

TILE_W, TILE_H = 640, 360
COLS = 2

# OpenCV uses BGR
GREEN = (90, 225, 110)
CYAN = (255, 220, 80)
PINK = (200, 80, 245)
YELLOW = (60, 220, 255)
WHITE = (245, 245, 245)
DARK = (18, 18, 18)

# COCO pose keypoints
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Skeleton connections
SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


# ============================================================
# LOAD REFERENCES
# ============================================================

def load_json(path):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


tracking = {
    round(float(r["time"]), 2): r
    for r in load_json(TRACKING)
}

annotations = {
    round(float(r["time"]), 2): r
    for r in load_json(MANUAL)["annotations"]
}

image_points = np.float32(load_json(CALIBRATION))
court_points = np.float32([
    [0, 0],
    [6.10, 0],
    [6.10, 4.72],
    [0, 4.72]
])

H = cv2.getPerspectiveTransform(
    image_points, court_points
)


def to_court(pixel):
    point = np.float32([[pixel]])
    result = cv2.perspectiveTransform(point, H)[0, 0]
    return result.astype(float).tolist()


def distance(a, b):
    if a is None or b is None:
        return None

    return float(math.dist(a, b))


def box_iou(a, b):
    """Compare two boxes: a=xyxy, b=xywh."""

    ax1, ay1, ax2, ay2 = a

    bx1, by1, bw, bh = b
    bx2 = bx1 + bw
    by2 = by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bw) * max(0, bh)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


# ============================================================
# DRAWING
# ============================================================

def draw_text(image, text, x, y,
              color=WHITE, scale=0.55):

    cv2.putText(
        image, str(text), (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, 2, cv2.LINE_AA
    )


def draw_dot(image, point, color, radius=6):
    x, y = map(lambda v: int(round(v)), point)

    cv2.circle(
        image, (x, y), radius + 2,
        DARK, -1, cv2.LINE_AA
    )

    cv2.circle(
        image, (x, y), radius,
        color, -1, cv2.LINE_AA
    )


def draw_skeleton(image, keypoints, confidence):
    for a, b in SKELETON:
        if (
            confidence[a] < MIN_ANKLE_CONF
            or confidence[b] < MIN_ANKLE_CONF
        ):
            continue

        pa = tuple(np.round(keypoints[a]).astype(int))
        pb = tuple(np.round(keypoints[b]).astype(int))

        cv2.line(
            image, pa, pb,
            GREEN, 2, cv2.LINE_AA
        )

    for i, point in enumerate(keypoints):
        if confidence[i] >= MIN_ANKLE_CONF:
            draw_dot(image, point, GREEN, 3)


# ============================================================
# LOAD POSE MODEL
# ============================================================

print("Loading YOLO11n-Pose...")
print("The model downloads automatically on first use.")

model = YOLO(MODEL_NAME)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit("Cannot open video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise SystemExit("Invalid video FPS.")


# ============================================================
# PROCESS TEST FRAMES
# ============================================================

tiles = []
records = []

try:
    for timestamp in TIMES:

        print(f"\nProcessing {timestamp:.2f}s")

        tracked = tracking.get(timestamp)
        manual = annotations.get(timestamp)

        if tracked is None or manual is None:
            print("  Missing tracking or annotation data")
            continue

        frame_id = round(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ok, frame = cap.read()

        if not ok:
            print("  Failed to read frame")
            continue

        result = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=0.20,
            device="cpu",
            verbose=False
        )[0]

        if (
            result.boxes is None
            or result.keypoints is None
            or len(result.boxes) == 0
        ):
            print("  No poses detected")
            continue

        pose_boxes = (
            result.boxes.xyxy.cpu().numpy()
        )

        pose_points = (
            result.keypoints.xy.cpu().numpy()
        )

        if result.keypoints.conf is not None:
            pose_conf = (
                result.keypoints.conf.cpu().numpy()
            )
        else:
            pose_conf = np.ones(
                pose_points.shape[:2],
                dtype=float
            )

        # Match pose detections to our existing player boxes.
        # Greedy one-to-one matching prevents the same pose
        # from being assigned to both players.
        candidates = []

        for side in ("Near", "Far"):
            person = tracked["players"].get(side)

            if person is None or person.get("bbox") is None:
                continue

            for pose_idx, pose_box in enumerate(pose_boxes):

                overlap = box_iou(
                    pose_box,
                    person["bbox"]
                )

                if overlap >= MIN_BOX_IOU:
                    candidates.append(
                        (overlap, side, pose_idx)
                    )

        candidates.sort(reverse=True)

        assignments = {}
        used_pose = set()

        for overlap, side, pose_idx in candidates:

            if side in assignments or pose_idx in used_pose:
                continue

            assignments[side] = (pose_idx, overlap)
            used_pose.add(pose_idx)

        messages = []

        for side in ("Near", "Far"):

            manual_player = manual["players"].get(side)
            tracked_player = tracked["players"].get(side)

            if manual_player is None or tracked_player is None:
                continue

            if side not in assignments:
                messages.append(f"{side}: no matched pose")
                print(f"  {side}: no matched pose")
                continue

            pose_idx, overlap = assignments[side]

            points = pose_points[pose_idx]
            confidence = pose_conf[pose_idx]

            draw_skeleton(frame, points, confidence)

            # Compare ankle candidates with our manual point.
            # The lower ankle means lower ON THE IMAGE,
            # not necessarily a foot touching the ground.
            ankles = []

            for index, label in [
                (LEFT_ANKLE, "left"),
                (RIGHT_ANKLE, "right")
            ]:
                if confidence[index] >= MIN_ANKLE_CONF:
                    ankles.append({
                        "label": label,
                        "xy": points[index].astype(float).tolist(),
                        "confidence": float(confidence[index])
                    })

            ankle_midpoint = None
            lower_ankle = None

            if len(ankles) == 2:
                ankle_midpoint = np.mean(
                    [a["xy"] for a in ankles],
                    axis=0
                ).tolist()

            if ankles:
                lower_ankle = max(
                    ankles,
                    key=lambda a: a["xy"][1]
                )["xy"]

            foot = tracked_player.get("pixel")
            reference_pixel = manual_player.get("pixel")
            reference_court = manual_player.get("court_m")

            if foot is not None:
                draw_dot(frame, foot, PINK)

            if reference_pixel is not None:
                draw_dot(frame, reference_pixel, CYAN)

            if ankle_midpoint is not None:
                draw_dot(frame, ankle_midpoint, YELLOW, 8)

            if len(ankles) == 1:
                draw_dot(frame, ankles[0]["xy"], YELLOW, 8)

            # Pose-to-reference errors in metres.
            # A valid manual reference must be visible-ground.
            is_grounded_reference = (
                manual_player.get("method")
                == "visible_foot_contact"
                and reference_court is not None
            )

            def metric(point):
                if point is None or not is_grounded_reference:
                    return None

                return distance(
                    to_court(point),
                    reference_court
                )

            bbox_error = metric(foot)
            midpoint_error = metric(ankle_midpoint)
            lower_error = metric(lower_ankle)

            # Optimistic diagnostic only:
            # choose the closer ankle using the reference.
            # This cannot be used by a real tracker.
            best_ankle_error = None

            if is_grounded_reference and ankles:
                best_ankle_error = min(
                    metric(a["xy"]) for a in ankles
                )

            records.append({
                "time": timestamp,
                "side": side,
                "reference_method": manual_player.get("method"),
                "pose_box_iou": round(float(overlap), 4),
                "ankles_detected": len(ankles),
                "left_ankle_conf": float(
                    confidence[LEFT_ANKLE]
                ),
                "right_ankle_conf": float(
                    confidence[RIGHT_ANKLE]
                ),
                "bbox_error_m": bbox_error,
                "ankle_midpoint_error_m": midpoint_error,
                "lower_ankle_error_m": lower_error,
                "best_ankle_oracle_error_m": best_ankle_error
            })

            info = (
                f"{side}: {len(ankles)} ankle(s)"
                f" | bbox={bbox_error:.2f}m"
                if bbox_error is not None
                else f"{side}: {len(ankles)} ankle(s)"
            )

            messages.append(info)
            print(" ", info)

        # Use original pixels for geometry, then resize.
        tile = cv2.resize(
            frame,
            (TILE_W, TILE_H),
            interpolation=cv2.INTER_AREA
        )

        cv2.rectangle(
            tile, (0, 0),
            (TILE_W, 35), DARK, -1
        )

        draw_text(
            tile,
            f"POSE ANALYSIS - {timestamp:.2f}s",
            10, 25
        )

        cv2.rectangle(
            tile,
            (0, TILE_H - 50),
            (TILE_W, TILE_H),
            DARK, -1
        )

        for i, message in enumerate(messages[:2]):
            draw_text(
                tile, message,
                8, TILE_H - 30 + i * 21,
                scale=0.45
            )

        tiles.append(tile)

finally:
    cap.release()


# ============================================================
# SAVE COLLAGE
# ============================================================

if not tiles:
    raise SystemExit("No output frames generated.")

rows = math.ceil(len(tiles) / COLS)

collage = np.full(
    (rows * TILE_H + 65, COLS * TILE_W, 3),
    DARK, dtype=np.uint8
)

for i, tile in enumerate(tiles):
    row = i // COLS
    col = i % COLS

    x = col * TILE_W
    y = row * TILE_H

    collage[
        y:y + TILE_H,
        x:x + TILE_W
    ] = tile

legend_y = rows * TILE_H + 40

draw_text(collage, "GREEN: SKELETON", 15, legend_y, GREEN)
draw_text(collage, "PINK: YOLO FOOT", 340, legend_y, PINK)
draw_text(collage, "CYAN: MANUAL", 650, legend_y, CYAN)
draw_text(collage, "YELLOW: ANKLE MIDPOINT", 925, legend_y, YELLOW)

image_path = OUT / "pose_collage.jpg"

if not cv2.imwrite(str(image_path), collage):
    raise SystemExit("Failed to save collage.")


# ============================================================
# SAVE METRICS
# ============================================================

csv_path = OUT / "pose_comparison.csv"

with open(
    csv_path, "w", newline="", encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "time", "side", "reference_method",
            "pose_box_iou", "ankles_detected",
            "left_ankle_conf", "right_ankle_conf",
            "bbox_error_m",
            "ankle_midpoint_error_m",
            "lower_ankle_error_m",
            "best_ankle_oracle_error_m"
        ]
    )

    writer.writeheader()
    writer.writerows(records)


print("\n========== POSE EXPERIMENT ==========")

for side in ("Near", "Far"):
    subset = [
        r for r in records
        if r["side"] == side
        and r["reference_method"] == "visible_foot_contact"
    ]

    print(f"\n{side} player")

    for key in (
        "bbox_error_m",
        "ankle_midpoint_error_m",
        "lower_ankle_error_m",
        "best_ankle_oracle_error_m"
    ):
        values = [
            r[key] for r in subset
            if r[key] is not None
        ]

        avg = (
            f"{sum(values) / len(values):.3f}m"
            if values else "N/A"
        )

        print(f"  {key}: {avg} (n={len(values)})")

print("\nSaved:")
print(image_path)
print(csv_path)
print("Done!")
