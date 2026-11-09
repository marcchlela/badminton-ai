"""Lesson 26: inspect temporal body-pose signals without changing a tracker.

Run from the badminton-ai project root:
    python analyze_pose_motion.py

Produces CSVs and plots in outputs/pose_motion/. Manual labels are used only
for plot annotations, never to calculate or accept model features.
"""

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

VIDEO = Path('data/raw/clip.mp4')
MODEL = 'yolo11n-pose.pt'
OUTPUT = Path('outputs/pose_motion')

RALLIES = {
    'rally_20_23': {
        'tracks': Path('outputs/tracking_v3.json'),
        'labels': Path('outputs/manual_evaluation.json'),
    },
    'rally_36_40': {
        'tracks': Path('outputs/holdout_36_40/tracking_v3.json'),
        'labels': Path('outputs/holdout_36_40/manual_evaluation.json'),
    },
}

# Same crop settings as our successful nano pose experiment.
CROP_X = 1.8
CROP_Y = 1.5
POSE_BOX_CONF = 0.15
MATCH_IOU = 0.45
JOINT_CONF = 0.40

# COCO body landmarks: hips 11,12; knees 13,14; ankles 15,16.
JOINT_GROUPS = {'hip': (11, 12), 'knee': (13, 14), 'ankle': (15, 16)}


def read_json(path):
    if not path.exists():
        raise SystemExit(f'Missing {path}')
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def crop_player(frame, box):
    height, width = frame.shape[:2]
    x, y, w, h = map(float, box)  # v3 uses xywh
    cx, cy = x + w / 2, y + h / 2
    cw, ch = max(w * CROP_X, 80), max(h * CROP_Y, 100)
    x1, y1 = max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2))
    x2, y2 = min(width, int(cx + cw / 2)), min(height, int(cy + ch / 2))
    return frame[y1:y2, x1:x2], x1, y1


def intersection_over_union(pose_xyxy, tracked_xywh):
    ax1, ay1, ax2, ay2 = map(float, pose_xyxy)
    x, y, w, h = map(float, tracked_xywh)
    bx1, by1, bx2, by2 = x, y, x + w, y + h
    iw = max(0., min(ax2, bx2) - max(ax1, bx1))
    ih = max(0., min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    union = max(0., ax2 - ax1) * max(0., ay2 - ay1) + max(0., w * h) - intersection
    return intersection / union if union > 0 else 0.


def pose_features(model, frame, bbox):
    """Returns measured joint features or a missing reason; no guessed joints."""
    output = {'pose_iou': None, 'pose_valid': False, 'pose_reason': 'no_pose'}
    for part in JOINT_GROUPS:
        for axis in ('x', 'y'):
            output[f'{part}_{axis}_px'] = None
        output[f'{part}_conf_min'] = None
        output[f'{part}_y_relative_to_box_bottom'] = None

    crop, ox, oy = crop_player(frame, bbox)
    if crop.size == 0:
        output['pose_reason'] = 'empty_crop'
        return output

    result = model.predict(crop, imgsz=640, conf=POSE_BOX_CONF,
                           device='cpu', verbose=False)[0]
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return output

    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = result.keypoints.xy.cpu().numpy()
    if result.keypoints.conf is None:
        confidence = np.ones(keypoints.shape[:2], dtype=float)
    else:
        confidence = result.keypoints.conf.cpu().numpy()

    candidates = [intersection_over_union(box + np.array([ox, oy, ox, oy]), bbox)
                  for box in boxes]
    idx = int(np.argmax(candidates))
    output['pose_iou'] = float(candidates[idx])
    if candidates[idx] < MATCH_IOU:
        output['pose_reason'] = 'weak_box_match'
        return output

    joints = keypoints[idx].copy()
    joints[:, 0] += ox
    joints[:, 1] += oy
    conf = confidence[idx]
    box_height = max(float(bbox[3]), 1.)
    box_bottom = float(bbox[1]) + float(bbox[3])
    for name, (left, right) in JOINT_GROUPS.items():
        output[f'{name}_conf_min'] = float(min(conf[left], conf[right]))
        if min(conf[left], conf[right]) < JOINT_CONF:
            continue
        midpoint = (joints[left] + joints[right]) / 2
        output[f'{name}_x_px'] = float(midpoint[0])
        output[f'{name}_y_px'] = float(midpoint[1])
        output[f'{name}_y_relative_to_box_bottom'] = float(
            (box_bottom - midpoint[1]) / box_height
        )

    output['pose_valid'] = all(output[f'{part}_y_px'] is not None
                               for part in ('hip', 'knee', 'ankle'))
    output['pose_reason'] = 'ok' if output['pose_valid'] else 'low_joint_confidence'
    if output['pose_valid']:
        output['hip_to_ankle_normalized'] = float(
            (output['ankle_y_px'] - output['hip_y_px']) / box_height
        )
        output['knee_to_ankle_normalized'] = float(
            (output['ankle_y_px'] - output['knee_y_px']) / box_height
        )
    else:
        output['hip_to_ankle_normalized'] = None
        output['knee_to_ankle_normalized'] = None
    return output


def make_plot(rows, labels, rally_name, side):
    selected = [r for r in rows if r['side'] == side]
    ts = np.array([r['time'] for r in selected], dtype=float)
    if not len(ts):
        return
    def series(key):
        return [float('nan') if r.get(key) is None else float(r[key])
                for r in selected]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(ts, series('bbox_top_px'), label='Box top (px)', marker='.')
    axes[0].plot(ts, series('bbox_bottom_px'), label='Box bottom (px)', marker='.')
    axes[0].plot(ts, series('ankle_y_px'), label='Ankle midpoint y (px)', marker='.')
    axes[0].invert_yaxis()  # Higher on image is higher in plot.
    axes[0].set_ylabel('Video y (px)')
    axes[0].legend(fontsize=8)

    axes[1].plot(ts, series('hip_to_ankle_normalized'), label='Hip → ankle / box height', marker='.')
    axes[1].plot(ts, series('knee_to_ankle_normalized'), label='Knee → ankle / box height', marker='.')
    axes[1].set_ylabel('Relative body geometry')
    axes[1].legend(fontsize=8)

    axes[2].plot(ts, series('ankle_y_relative_to_box_bottom'),
                 label='Box bottom → ankle / box height', marker='.')
    axes[2].set_ylabel('Relative ankle height')
    axes[2].set_xlabel('Video time (seconds)')
    axes[2].legend(fontsize=8)

    # Labels are shown for visual interpretation ONLY.
    for time, person in labels.items():
        reference = person.get(side, {})
        if reference.get('method') == 'estimated_ground_position':
            for ax in axes:
                ax.axvline(time, linestyle='--', alpha=.5)
        elif reference.get('method') == 'visible_foot_contact':
            for ax in axes:
                ax.axvline(time, linestyle=':', alpha=.2)

    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f'{rally_name} — {side} player | dashed: annotated airborne; dotted: grounded')
    fig.tight_layout()
    filename = OUTPUT / f'{rally_name}_{side.lower()}.png'
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(' Saved plot:', filename)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise SystemExit(f'Unable to open {VIDEO}')

    try:
        for rally_name, paths in RALLIES.items():
            tracks = read_json(paths['tracks'])
            annotations = read_json(paths['labels'])['annotations']
            labels = {round(float(a['time']), 2): a['players'] for a in annotations}
            rows = []
            for frame_record in tracks:
                t = round(float(frame_record['time']), 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_record['frame']))
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Video read failed: {rally_name} at {t}s')
                for side in ('Near', 'Far'):
                    player = frame_record['players'].get(side)
                    if not player or player.get('bbox') is None:
                        continue
                    x, y, w, h = player['bbox']
                    features = pose_features(model, frame, player['bbox'])
                    rows.append({
                        'time': t, 'side': side, 'frame': int(frame_record['frame']),
                        'manual_label': labels.get(t, {}).get(side, {}).get('method', 'not_labeled'),
                        'ground_updated_v3': player.get('ground_updated'),
                        'bbox_top_px': y, 'bbox_bottom_px': y + h,
                        'bbox_height_px': h,
                        **features,
                    })
            if not rows:
                raise RuntimeError(f'No rows in {rally_name}')
            path = OUTPUT / f'{rally_name}_features.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f'\n{rally_name}: {len(rows)} player-frame records')
            for side in ('Near', 'Far'):
                subset = [r for r in rows if r['side'] == side]
                good = sum(r['pose_valid'] for r in subset)
                print(f' {side} usable hip+knee+ankle sets: {good}/{len(subset)}')
                make_plot(rows, labels, rally_name, side)
            print(' Saved features:', path)
    finally:
        cap.release()

    print('\nDone. These plots are diagnostics, NOT a jump classifier.')


if __name__ == '__main__':
    main()
