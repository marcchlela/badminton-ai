
import cv2
import csv
import json
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO

VIDEO = Path("data/raw/clip.mp4")
TRACKING = Path("outputs/tracking_v3.json")
MANUAL = Path("outputs/manual_evaluation.json")
CALIBRATION = Path("outputs/court_points.json")

OUT = Path("outputs/pose")
OUT.mkdir(parents=True, exist_ok=True)

TIMES = [20.0, 20.6, 21.4, 21.7,
         22.2, 22.4, 22.6, 22.9]

MIN_CONF = 0.15
ANKLE_CONF = 0.30

# Larger crop means more context, but less detail.
CROP_SCALE_X = 1.8
CROP_SCALE_Y = 1.5

SKELETON = [
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

GREEN = (80, 220, 100)
PINK = (185, 90, 240)
CYAN = (255, 210, 80)
YELLOW = (60, 220, 255)
DARK = (15, 15, 15)
WHITE = (255, 255, 255)


def load(path):
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

src_points = np.float32(load(CALIBRATION))
dst_points = np.float32([
    [0, 0], [6.10, 0],
    [6.10, 4.72], [0, 4.72]
])

H = cv2.getPerspectiveTransform(
    src_points, dst_points
)


def to_court(point):
    p = np.float32([[point]])
    return cv2.perspectiveTransform(p, H)[0, 0].tolist()


def error_metres(pixel, reference):
    if pixel is None or reference is None:
        return None
    return math.dist(to_court(pixel), reference)


def make_crop(frame, bbox):
    height, width = frame.shape[:2]
    x, y, w, h = bbox

    cx = x + w / 2
    cy = y + h / 2

    crop_w = max(w * CROP_SCALE_X, 80)
    crop_h = max(h * CROP_SCALE_Y, 100)

    x1 = max(0, int(cx - crop_w / 2))
    y1 = max(0, int(cy - crop_h / 2))
    x2 = min(width, int(cx + crop_w / 2))
    y2 = min(height, int(cy + crop_h / 2))

    return frame[y1:y2, x1:x2], x1, y1


def iou_xyxy_xywh(box, target):
    ax1, ay1, ax2, ay2 = map(float, box)
    x, y, w, h = map(float, target)
    bx1, by1, bx2, by2 = x, y, x + w, y + h

    intersection = (
        max(0, min(ax2, bx2) - max(ax1, bx1))
        * max(0, min(ay2, by2) - max(ay1, by1))
    )

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, w) * max(0, h)
    union = area_a + area_b - intersection

    return intersection / union if union else 0


def dot(frame, point, color, radius=5):
    p = tuple(np.round(point).astype(int))
    cv2.circle(frame, p, radius + 2, DARK, -1)
    cv2.circle(frame, p, radius, color, -1)


def label(frame, message, x, y, scale=0.45):
    cv2.putText(
        frame, message, (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, WHITE, 1, cv2.LINE_AA
    )


print("Loading pose model...")
model = YOLO("yolo11s-pose.pt")

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise SystemExit("Invalid FPS")

records = []
tiles = []

try:
    for timestamp in TIMES:
        frame_id = round(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()

        if not ok:
            print(f"Failed to read {timestamp}")
            continue

        frame_record = tracking[timestamp]["players"]
        refs = manual[timestamp]["players"]

        print(f"\n{timestamp:.2f}s")

        for side in ("Near", "Far"):
            tracked = frame_record.get(side)
            reference = refs.get(side)

            if tracked is None or reference is None:
                continue

            bbox = tracked.get("bbox")
            if bbox is None:
                continue

            crop, ox, oy = make_crop(frame, bbox)

            if crop.size == 0:
                continue

            # Pose operates on the cropped player region.
            result = model.predict(
                crop,
                imgsz=960,
                conf=MIN_CONF,
                device="cpu",
                verbose=False
            )[0]

            best_index = None
            best_overlap = 0.0

            if (
                result.boxes is not None
                and result.keypoints is not None
            ):
                boxes = result.boxes.xyxy.cpu().numpy()

                # Find the pose matching the ORIGINAL
                # YOLO player box, not simply the first pose.
                for idx, crop_box in enumerate(boxes):
                    global_box = crop_box + np.array(
                        [ox, oy, ox, oy]
                    )

                    overlap = iou_xyxy_xywh(
                        global_box, bbox
                    )

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_index = idx

            ankle_points = []
            midpoint = None
            lower_ankle = None

            if (
                best_index is not None
                and best_overlap >= 0.10
            ):
                points = (
                    result.keypoints.xy.cpu().numpy()
                    [best_index].copy()
                )

                if result.keypoints.conf is not None:
                    conf = (
                        result.keypoints.conf.cpu().numpy()
                        [best_index]
                    )
                else:
                    conf = np.ones(17)

                # Convert local crop positions back to
                # coordinates of the original 1280x720 video.
                points[:, 0] += ox
                points[:, 1] += oy

                for a, b in SKELETON:
                    if (
                        conf[a] >= ANKLE_CONF
                        and conf[b] >= ANKLE_CONF
                    ):
                        cv2.line(
                            frame,
                            tuple(np.round(points[a]).astype(int)),
                            tuple(np.round(points[b]).astype(int)),
                            GREEN, 2, cv2.LINE_AA
                        )

                for j in range(17):
                    if conf[j] >= ANKLE_CONF:
                        dot(frame, points[j], GREEN, 3)

                for j in (15, 16):
                    if conf[j] >= ANKLE_CONF:
                        ankle_points.append(
                            points[j].tolist()
                        )

                if len(ankle_points) == 2:
                    midpoint = np.mean(
                        ankle_points, axis=0
                    ).tolist()

                if ankle_points:
                    lower_ankle = max(
                        ankle_points,
                        key=lambda p: p[1]
                    )

            bbox_foot = tracked.get("pixel")
            ref_pixel = reference.get("pixel")
            ref_court = reference.get("court_m")

            grounded = (
                reference.get("method")
                == "visible_foot_contact"
            )

            if bbox_foot is not None:
                dot(frame, bbox_foot, PINK, 7)

            if ref_pixel is not None:
                dot(frame, ref_pixel, CYAN, 7)

            if midpoint is not None:
                dot(frame, midpoint, YELLOW, 7)

            def metric(point):
                if not grounded:
                    return None
                return error_metres(point, ref_court)

            row = {
                "time": timestamp,
                "side": side,
                "reference_type": reference.get("method"),
                "pose_matched": (
                    best_index is not None
                    and best_overlap >= 0.10
                ),
                "pose_iou": best_overlap,
                "ankles_found": len(ankle_points),
                "bbox_error_m": metric(bbox_foot),
                "midpoint_error_m": metric(midpoint),
                "lower_ankle_error_m": metric(lower_ankle)
            }

            records.append(row)

            print(
                f"  {side}: "
                f"pose={row['pose_matched']} "
                f"ankles={len(ankle_points)} "
                f"bbox={row['bbox_error_m']} "
                f"midpoint={row['midpoint_error_m']}"
            )

        tile = cv2.resize(frame, (640, 360))
        cv2.rectangle(tile, (0, 0), (640, 34), DARK, -1)
        label(
            tile,
            f"CROPPED POSE - {timestamp:.2f}s",
            12, 23, 0.65
        )

        tiles.append(tile)

finally:
    cap.release()


# ============================================================
# OUTPUT
# ============================================================

columns = 2
tile_w, tile_h = 640, 360
rows = math.ceil(len(tiles) / columns)

collage = np.full(
    (rows * tile_h, columns * tile_w, 3),
    DARK, np.uint8
)

for i, tile in enumerate(tiles):
    x = (i % columns) * tile_w
    y = (i // columns) * tile_h
    collage[y:y + tile_h, x:x + tile_w] = tile

image_path = OUT / "cropped_pose_collage.jpg"
csv_path = OUT / "cropped_pose_comparison.csv"

cv2.imwrite(str(image_path), collage)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=[
            "time", "side", "reference_type",
            "pose_matched", "pose_iou",
            "ankles_found", "bbox_error_m",
            "midpoint_error_m", "lower_ankle_error_m"
        ]
    )
    writer.writeheader()
    writer.writerows(records)


print("\n========== RESULTS ==========")

for side in ("Near", "Far"):
    subset = [r for r in records if r["side"] == side]

    matched = sum(
        r["pose_matched"] for r in subset
    )

    print(f"\n{side}: matched {matched}/{len(subset)}")

    for key in ("bbox_error_m", "midpoint_error_m"):
        values = [
            r[key] for r in subset
            if r[key] is not None
        ]

        if values:
            print(
                f"  {key}: "
                f"{np.mean(values):.3f}m "
                f"(n={len(values)})"
            )
        else:
            print(f"  {key}: N/A")

print("\nSaved:")
print(image_path)
print(csv_path)
