
import argparse
import cv2
import json
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

parser = argparse.ArgumentParser(description="Test cropped pose fusion on v3 tracks.")
parser.add_argument("--video", type=Path, default=Path("data/raw/clip.mp4"))
parser.add_argument("--baseline", type=Path, default=Path("outputs/tracking_v3.json"))
parser.add_argument("--calibration", type=Path, default=Path("outputs/court_points.json"))
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--model", default="yolo11n-pose.pt")
args = parser.parse_args()

VIDEO = args.video
BASELINE = args.baseline
CALIBRATION = args.calibration

OUT_POSE = (args.output_dir / "tracking_v4_pose.json" if args.output_dir
            else Path("outputs/tracking_v4_pose.json"))
OUT_HYBRID = (args.output_dir / "tracking_v4_hybrid.json" if args.output_dir
              else Path("outputs/tracking_v4_hybrid.json"))
OUT_DIAGNOSTICS = (args.output_dir / "fusion_diagnostics.json" if args.output_dir
                   else Path("outputs/pose/fusion_diagnostics.json"))

MODEL = args.model

CROP_SCALE_X = 1.8
CROP_SCALE_Y = 1.5
MIN_CROP_W = 80
MIN_CROP_H = 100

POSE_CONF = 0.15
MIN_ANKLE_CONF = 0.40
MIN_POSE_IOU = 0.45

COURT_W = 6.10
COURT_L = 13.40
NET_Y = 6.70

# Pose must agree reasonably with the visual detection.
MAX_POSE_BBOX_DISTANCE = 1.50

# This is a fixed experiment weight, not learned.
POSE_WEIGHT = 0.65


# ============================================================
# LOAD INPUTS
# ============================================================

def load_json(path):
    if not path.exists():
        raise SystemExit(f"Missing: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


baseline = load_json(BASELINE)
image_points = np.float32(load_json(CALIBRATION))

court_points = np.float32([
    [0, 0],
    [COURT_W, 0],
    [COURT_W, 4.72],
    [0, 4.72]
])

H = cv2.getPerspectiveTransform(
    image_points,
    court_points
)


def to_court(pixel):
    p = np.float32([[pixel]])
    mapped = cv2.perspectiveTransform(p, H)[0, 0]
    return mapped.astype(float).tolist()


def distance(a, b):
    return math.dist(a, b)


def in_court(point, side):
    x, y = point

    if not (
        math.isfinite(x)
        and math.isfinite(y)
        and 0 <= x <= COURT_W
    ):
        return False

    if side == "Near":
        return 0 <= y <= NET_Y

    return NET_Y <= y <= COURT_L


def make_crop(frame, bbox):
    height, width = frame.shape[:2]
    x, y, w, h = bbox

    cx = x + w / 2
    cy = y + h / 2

    crop_w = max(w * CROP_SCALE_X, MIN_CROP_W)
    crop_h = max(h * CROP_SCALE_Y, MIN_CROP_H)

    x1 = max(0, int(cx - crop_w / 2))
    y1 = max(0, int(cy - crop_h / 2))

    x2 = min(width, int(cx + crop_w / 2))
    y2 = min(height, int(cy + crop_h / 2))

    return frame[y1:y2, x1:x2], x1, y1


def box_iou(a, b):
    # a = xyxy, b = xywh
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bw, bh = map(float, b)

    bx2 = bx1 + bw
    by2 = by1 + bh

    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bw) * max(0, bh)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def fused_position(baseline_xy, pose_xy):
    return [
        (1 - POSE_WEIGHT) * baseline_xy[0]
        + POSE_WEIGHT * pose_xy[0],

        (1 - POSE_WEIGHT) * baseline_xy[1]
        + POSE_WEIGHT * pose_xy[1]
    ]


# ============================================================
# POSE MODEL
# ============================================================

print("Loading YOLO11n-Pose...")
model = YOLO(MODEL)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise SystemExit("Invalid video FPS.")


def detect_ankle_midpoint(frame, tracked):
    bbox = tracked.get("bbox")

    if bbox is None:
        return None, "no_bbox", {}

    crop, ox, oy = make_crop(frame, bbox)

    if crop.size == 0:
        return None, "invalid_crop", {}

    result = model.predict(
        crop,
        imgsz=640,
        conf=POSE_CONF,
        device="cpu",
        verbose=False
    )[0]

    if (
        result.boxes is None
        or result.keypoints is None
        or len(result.boxes) == 0
    ):
        return None, "no_pose", {}

    boxes = result.boxes.xyxy.cpu().numpy()
    points_all = result.keypoints.xy.cpu().numpy()

    if result.keypoints.conf is not None:
        conf_all = result.keypoints.conf.cpu().numpy()
    else:
        conf_all = np.ones(
            points_all.shape[:2],
            dtype=float
        )

    best_idx = None
    best_iou = 0.0

    for idx, box in enumerate(boxes):
        global_box = box + np.array([
            ox, oy, ox, oy
        ])

        overlap = box_iou(global_box, bbox)

        if overlap > best_iou:
            best_iou = overlap
            best_idx = idx

    details = {"pose_iou": float(best_iou)}

    if best_idx is None or best_iou < MIN_POSE_IOU:
        return None, "weak_pose_match", details

    conf = conf_all[best_idx]
    points = points_all[best_idx].copy()

    # COCO indices: left ankle=15, right ankle=16
    left_conf = float(conf[15])
    right_conf = float(conf[16])

    details["left_ankle_conf"] = left_conf
    details["right_ankle_conf"] = right_conf

    if (
        left_conf < MIN_ANKLE_CONF
        or right_conf < MIN_ANKLE_CONF
    ):
        return None, "low_ankle_confidence", details

    points[:, 0] += ox
    points[:, 1] += oy

    midpoint = (
        (points[15] + points[16]) / 2
    ).astype(float).tolist()

    details["ankle_midpoint_pixel"] = midpoint

    return midpoint, "pose_found", details


# ============================================================
# PROCESS THE SAME 30 FRAMES AS V3
# ============================================================

pose_output = []
hybrid_output = []
diagnostics = []

accepted = {"Near": 0, "Far": 0}
rejected = {}

try:
    for source_record in baseline:
        timestamp = float(source_record["time"])
        frame_id = int(source_record["frame"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()

        if not ok:
            raise SystemExit(
                f"Could not read frame {frame_id}"
            )

        # JSON round-trip creates independent copies.
        pose_record = json.loads(
            json.dumps(source_record)
        )
        hybrid_record = json.loads(
            json.dumps(source_record)
        )

        print(f"\n{timestamp:.2f}s")

        for side in ("Near", "Far"):
            tracked = source_record["players"].get(side)

            if tracked is None:
                continue

            baseline_xy = tracked.get("court_m")
            bbox_foot = tracked.get("pixel")

            info = {
                "time": timestamp,
                "side": side,
                "accepted": False,
                "reason": "",
                "pose_court_m": None
            }

            # Crucial safety rule:
            # v3 must have accepted a ground measurement.
            # Do not interpret airborne ankles as floor contact.
            if not tracked.get("ground_updated", False):
                reason = "ground_not_updated"
                midpoint = None
                details = {}

            elif baseline_xy is None or bbox_foot is None:
                reason = "missing_baseline"
                midpoint = None
                details = {}

            else:
                midpoint, reason, details = (
                    detect_ankle_midpoint(frame, tracked)
                )

            info.update(details)

            pose_court = None

            if midpoint is not None:
                pose_court = to_court(midpoint)

                info["pose_court_m"] = pose_court

                if not in_court(pose_court, side):
                    reason = "pose_outside_court"

                elif distance(
                    pose_court,
                    to_court(bbox_foot)
                ) > MAX_POSE_BBOX_DISTANCE:
                    reason = "pose_bbox_disagreement"

                else:
                    reason = "accepted"

                    info["accepted"] = True
                    accepted[side] += 1

                    pose_player = pose_record[
                        "players"
                    ][side]

                    hybrid_player = hybrid_record[
                        "players"
                    ][side]

                    pose_player["court_m"] = pose_court
                    pose_player["position_source"] = (
                        "pose_ankle_midpoint"
                    )

                    hybrid_player["court_m"] = (
                        fused_position(
                            baseline_xy,
                            pose_court
                        )
                    )

                    hybrid_player["position_source"] = (
                        "pose_hybrid"
                    )

            # Retain v3 as the fallback.
            for output_record in (
                pose_record,
                hybrid_record
            ):
                player = output_record["players"][side]

                player["pose_accepted"] = info["accepted"]
                player["pose_court_m"] = pose_court
                player["pose_reason"] = reason

            info["reason"] = reason
            diagnostics.append(info)

            if not info["accepted"]:
                rejected[reason] = (
                    rejected.get(reason, 0) + 1
                )

            print(
                f"  {side}: {reason}"
                f" | pose_used={info['accepted']}"
            )

        pose_output.append(pose_record)
        hybrid_output.append(hybrid_record)

finally:
    cap.release()


# ============================================================
# SAVE RESULTS
# ============================================================

for path, data in (
    (OUT_POSE, pose_output),
    (OUT_HYBRID, hybrid_output),
    (OUT_DIAGNOSTICS, diagnostics)
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


print("\n========== POSE FUSION SUMMARY ==========")

for side in ("Near", "Far"):
    print(
        f"{side}: "
        f"{accepted[side]}/{len(baseline)} "
        "frames accepted"
    )

print("\nRejection reasons:")

for reason, count in sorted(rejected.items()):
    print(f"  {reason}: {count}")

print("\nSaved:")
print(OUT_POSE)
print(OUT_HYBRID)
print(OUT_DIAGNOSTICS)
print("Experiment complete.")
