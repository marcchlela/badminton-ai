"""Experimental, label-free shuttle trajectory linking on a short HD rally."""
import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO = Path('data/raw/high_quality_match.mp4')
LABELS = Path('outputs/shuttle/hd_visibility_audit.json')
OUT = Path('outputs/shuttle/trajectory_v1')
START_SEC, END_SEC = 200.0, 210.0

MOTION_THRESHOLD = 22
MIN_BRIGHTNESS = 115
TOP_CANDIDATES = 20
BEAM_WIDTH = 80
MAX_GAP = 4                 # frames without an observed shuttle candidate
MAX_SPEED = 220.0          # pixels per frame, at original resolution
VIDEO_SCALE = 0.65         # smaller rendered replay; keep analysis full resolution


@dataclass
class Hypothesis:
    score: float
    last: tuple | None       # (frame_number, x, y)
    previous: tuple | None
    gap: int
    path: tuple             # candidate index or None for each processed frame


def detect(gray_frames, max_area, max_side):
    background = np.median(
        np.stack([gray_frames[0], gray_frames[1], gray_frames[3], gray_frames[4]]),
        axis=0,
    ).astype(np.uint8)
    current = gray_frames[2]
    difference = cv2.absdiff(current, background)
    mask = ((difference >= MOTION_THRESHOLD) &
            (current >= MIN_BRIGHTNESS)).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    flat = labels.ravel()
    motion_sums = np.bincount(flat, weights=difference.ravel(), minlength=count)
    bright_sums = np.bincount(flat, weights=current.ravel(), minlength=count)
    found = []
    for i in range(1, count):
        _, _, w, h, area = map(int, stats[i])
        if area < 1 or area > max_area or w > max_side or h > max_side:
            continue
        score = (motion_sums[i] / area) * (bright_sums[i] / area) / 255.0 * math.sqrt(area)
        cx, cy = centers[i]
        found.append((float(cx), float(cy), float(score)))
    found.sort(key=lambda item: item[2], reverse=True)
    return found[:TOP_CANDIDATES]


def extract_candidates(cap, first, stop, frame_count, max_area, max_side):
    # Decode sequentially instead of making hundreds of expensive AV1 seeks.
    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 2)
    window = deque(maxlen=5)
    for _ in range(4):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError('Could not read the first neighboring video frames')
        window.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    all_candidates = []
    for frame_id in range(first, stop):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f'Video ended while decoding frame {frame_id}')
        window.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        all_candidates.append(detect(list(window), max_area, max_side))
        if (frame_id - first + 1) % 25 == 0:
            print(f'  Detected {frame_id - first + 1}/{stop - first} frames', flush=True)
    return all_candidates


def link_candidates(all_candidates):
    # No manual shuttle coordinates are used to initialize or guide the tracker.
    beams = [Hypothesis(0.0, None, None, 0, ())]
    for frame_index, candidates in enumerate(all_candidates):
        next_beams = []
        for h in beams:
            # Missing detection (occlusion or blur)
            next_beams.append(Hypothesis(
                h.score - 2.6, h.last, h.previous, h.gap + 1, h.path + (None,)
            ))
            for rank, (x, y, raw_score) in enumerate(candidates):
                new_point = (frame_index, x, y)
                if h.last is None:
                    link_cost = 2.0  # penalty for starting a track
                else:
                    previous_frame, px, py = h.last
                    dt = frame_index - previous_frame
                    if dt > MAX_GAP + 1:
                        # Track was lost; allow a fresh start with a penalty.
                        link_cost = 5.0
                    else:
                        distance = math.hypot(x - px, y - py)
                        speed = distance / dt
                        if speed > MAX_SPEED:
                            continue
                        link_cost = 0.014 * speed + 0.55 * (dt - 1)
                        if speed < 1.5:
                            link_cost += 1.1  # moving shuttle should rarely be stationary
                        if h.previous is not None:
                            f0, x0, y0 = h.previous
                            old_dt = previous_frame - f0
                            vx = (px - x0) / old_dt
                            vy = (py - y0) / old_dt
                            new_vx = (x - px) / dt
                            new_vy = (y - py) / dt
                            acceleration = math.hypot(new_vx - vx, new_vy - vy)
                            link_cost += 0.010 * min(acceleration, 300)
                # The motion detector's candidate rank is only one source of evidence.
                rank_reward = 4.0 - 1.0 * math.log2(rank + 1)
                next_beams.append(Hypothesis(
                    h.score + rank_reward - link_cost,
                    new_point,
                    h.last if h.last is not None and h.gap <= MAX_GAP else None,
                    0,
                    h.path + (rank,),
                ))
        # Different earlier paths that end in the same state need not all survive.
        best_by_state = {}
        for h in next_beams:
            state = (
                h.last[0] if h.last else -1,
                round(h.last[1]) if h.last else -1,
                round(h.last[2]) if h.last else -1,
                h.previous[0] if h.previous else -1,
                round(h.previous[1]) if h.previous else -1,
                round(h.previous[2]) if h.previous else -1,
                min(h.gap, MAX_GAP + 1),
            )
            if state not in best_by_state or best_by_state[state].score < h.score:
                best_by_state[state] = h
        beams = sorted(best_by_state.values(), key=lambda h: h.score, reverse=True)[:BEAM_WIDTH]
        if (frame_index + 1) % 50 == 0:
            print(f'  Linked {frame_index + 1}/{len(all_candidates)} frames', flush=True)
    return beams[0].path


def evaluate(rows, fps, first):
    if not LABELS.exists():
        print('No labels available: skipping evaluation')
        return {'note': 'No label file available'}
    with LABELS.open(encoding='utf-8') as f:
        labels = json.load(f)['annotations']
    # Keep only manually visible frames. Uncertain frames are NOT negatives.
    visible = [a for a in labels if a['status'] == 'visible' and a['pixel'] is not None]
    hits = {'top1': 0, 'top5': 0, 'track': 0}
    predicted = 0
    tested = 0
    errors = []
    for a in visible:
        frame = int(a['frame'])
        index = frame - first
        if not 0 <= index < len(rows):
            continue
        row = rows[index]
        target = a['pixel']
        radius = 12 * (row['height'] / 720)
        tested += 1
        for field in ('top1', 'top5'):
            candidates = row['candidates'][:1 if field == 'top1' else 5]
            if any(math.hypot(c[0] - target[0], c[1] - target[1]) <= radius
                   for c in candidates):
                hits[field] += 1
        chosen = row['chosen']
        if chosen is not None:
            predicted += 1
            error = math.hypot(chosen[0] - target[0], chosen[1] - target[1])
            errors.append(error)
            if error <= radius:
                hits['track'] += 1
    return {
        'visible_reference_frames': tested,
        'matching_radius_note': '12px at 720p; scales with frame height',
        'top1_hits': hits['top1'],
        'top5_hits': hits['top5'],
        'tracker_hits': hits['track'],
        'tracker_predictions_on_visible_frames': predicted,
        'tracker_misses_on_visible_frames': tested - predicted,
        'mean_error_when_predicted_px': round(float(np.mean(errors)), 1) if errors else None,
        'note': 'Development labels only; uncertain frames excluded; not a false-positive test',
    }


def render_video(rows, fps, first, width, height):
    path = OUT / 'tracking_replay.mp4'
    w = int(round(width * VIDEO_SCALE / 2) * 2)
    h = int(round(height * VIDEO_SCALE / 2) * 2)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError('Could not open MP4 output writer')
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        writer.release()
        raise RuntimeError('Could not reopen the input video for rendering')
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    trail = deque(maxlen=12)
    try:
        for i, row in enumerate(rows):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError('Video ended during replay rendering')
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            def point(p):
                return (int(round(p[0] * w / width)), int(round(p[1] * h / height)))
            if row['candidates']:
                cv2.circle(image, point(row['candidates'][0]), 8, (0, 190, 255), 2)
            chosen = row['chosen']
            if chosen is None:
                trail.clear()
            else:
                trail.append(point(chosen))
                for a, b in zip(list(trail)[:-1], list(trail)[1:]):
                    cv2.line(image, a, b, (60, 240, 80), 2)
                cv2.circle(image, point(chosen), 10, (60, 240, 80), 2)
                cv2.circle(image, point(chosen), 3, (60, 240, 80), -1)
            cv2.rectangle(image, (8, 8), (min(w - 8, 585), 82), (10, 17, 24), -1)
            cv2.putText(image, f'Time {row["time"]:.2f}s | Shuttle tracker v1',
                        (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2)
            cv2.putText(image, 'Yellow = top-1 motion  |  Green = linked track',
                        (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1)
            writer.write(image)
            if (i + 1) % 50 == 0:
                print(f'  Rendered {i + 1}/{len(rows)} frames', flush=True)
    finally:
        cap.release()
        writer.release()
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise SystemExit(f'Video not found: {VIDEO}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = int(round(START_SEC * fps))
    stop = int(round(END_SEC * fps))
    if fps <= 0 or first < 2 or stop + 2 > count or stop <= first:
        cap.release()
        raise SystemExit('Invalid selected time window or FPS')
    scale = height / 720.0
    max_area = round(90 * scale * scale)
    max_side = round(20 * scale)
    print(f'Video: {width}x{height}, {fps:.2f} FPS; {stop - first} frames')
    print(f'Adaptive limits: area={max_area}, side={max_side}')
    print('Step 1/3: sequential frame decoding and motion candidates', flush=True)
    try:
        candidates_by_frame = extract_candidates(cap, first, stop, count, max_area, max_side)
    finally:
        cap.release()
    print('Step 2/3: label-free temporal linking', flush=True)
    choices = link_candidates(candidates_by_frame)
    rows = []
    for index, (candidates, choice) in enumerate(zip(candidates_by_frame, choices)):
        chosen = candidates[choice] if choice is not None else None
        rows.append({'frame': first + index, 'time': (first + index) / fps,
                     'height': height, 'candidates': candidates, 'choice': choice,
                     'chosen': chosen})
    metrics = evaluate(rows, fps, first)
    print('\n========== TRACKING EVALUATION ==========')
    for key, value in metrics.items():
        print(f'{key}: {value}')
    csv_path = OUT / 'tracking_positions.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['frame', 'time', 'x', 'y',
                                                  'candidate_rank', 'top1_x', 'top1_y'])
        writer.writeheader()
        for row in rows:
            chosen = row['chosen']
            top1 = row['candidates'][0] if row['candidates'] else None
            writer.writerow({'frame': row['frame'], 'time': round(row['time'], 4),
                             'x': round(chosen[0], 2) if chosen else '',
                             'y': round(chosen[1], 2) if chosen else '',
                             'candidate_rank': row['choice'] + 1 if chosen else '',
                             'top1_x': round(top1[0], 2) if top1 else '',
                             'top1_y': round(top1[1], 2) if top1 else ''})
    with (OUT / 'evaluation.json').open('w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    print('Step 3/3: rendering replay', flush=True)
    video_path = render_video(rows, fps, first, width, height)
    print('\nSaved:', csv_path, OUT / 'evaluation.json', video_path, sep='\n  ')
    print('Review the whole replay: moving false positives are possible.')


if __name__ == '__main__':
    main()
