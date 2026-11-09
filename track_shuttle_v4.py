import csv
import json
import math
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("outputs/shuttle")

CACHE_PATH = ROOT / "trajectory_v2" / "motion_candidates.json"
LABELS_PATH = ROOT / "hd_visibility_audit.json"
OUT_DIR = ROOT / "trajectory_v4"

MAX_CANDIDATES_PER_FRAME = 20
BEAM_WIDTH = 80

# scoring
RANK_WEIGHT = 2.5
FIRST_JUMP_WEIGHT = 0.08
PREDICTION_WEIGHT = 0.10
JUMP_WEIGHT = 0.015

MISSING_BASE_COST = 10.0
MISSING_STEP_COST = 3.0

# evaluation
BASE_MATCH_RADIUS_720P = 12.0
LARGE_ERROR_THRESHOLD = 50.0


# ============================================================
# HELPERS
# ============================================================

def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def unpack_candidate(raw, default_rank):
    """
    Tries to support multiple cache formats:
    - [x, y]
    - [x, y, score]
    - {"x":..., "y":..., "score":...}
    - {"cx":..., "cy":...}
    """
    if isinstance(raw, dict):
        x = raw.get("x", raw.get("cx", raw.get("center_x")))
        y = raw.get("y", raw.get("cy", raw.get("center_y")))
        if x is None or y is None:
            return None

        score = raw.get("score", raw.get("motion", raw.get("confidence", 0.0)))
        return {
            "x": float(x),
            "y": float(y),
            "rank": int(raw.get("rank", default_rank)),
            "score": safe_float(score, 0.0),
        }

    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        score = 0.0
        if len(raw) >= 3 and isinstance(raw[2], (int, float)):
            score = float(raw[2])

        return {
            "x": float(raw[0]),
            "y": float(raw[1]),
            "rank": default_rank,
            "score": score,
        }

    return None


def load_candidates(cache):
    raw_frames = cache["candidates"]
    frames = []

    first_frame = int(cache.get("first_frame", 0))
    fps = float(cache["fps"])

    for idx, raw_list in enumerate(raw_frames):
        frame_number = first_frame + idx
        time_s = frame_number / fps

        cand_list = []
        for rank, raw in enumerate(raw_list[:MAX_CANDIDATES_PER_FRAME], start=1):
            c = unpack_candidate(raw, rank)
            if c is not None:
                cand_list.append(c)

        frames.append({
            "frame": frame_number,
            "time": time_s,
            "candidates": cand_list
        })

    return frames


def load_visible_labels(path):
    if not path.exists():
        return {}

    data = load_json(path)

    annotations = data["annotations"] if isinstance(data, dict) and "annotations" in data else data

    visible = {}
    for a in annotations:
        if a.get("status") != "visible":
            continue

        pixel = a.get("pixel")
        frame = a.get("frame")

        if pixel is None or frame is None:
            continue

        if len(pixel) < 2:
            continue

        visible[int(frame)] = (float(pixel[0]), float(pixel[1]))

    return visible


def match_radius(frame_height):
    return BASE_MATCH_RADIUS_720P * (frame_height / 720.0)


def rank_cost(rank):
    return RANK_WEIGHT * math.log2(rank + 1.0)


# ============================================================
# TRACKING STATE
# ============================================================

def make_initial_beams(first_frame):
    beams = []

    if first_frame["candidates"]:
        for cand in first_frame["candidates"]:
            beams.append({
                "cost": rank_cost(cand["rank"]),
                "last1": None,  # older visible point
                "last2": (cand["x"], cand["y"]),  # newest visible point
                "miss_streak": 0,
                "path": [{
                    "frame": first_frame["frame"],
                    "time": first_frame["time"],
                    "x": cand["x"],
                    "y": cand["y"],
                    "rank": cand["rank"],
                    "source": "candidate",
                }]
            })

    # also allow starting missing
    beams.append({
        "cost": MISSING_BASE_COST,
        "last1": None,
        "last2": None,
        "miss_streak": 1,
        "path": [{
            "frame": first_frame["frame"],
            "time": first_frame["time"],
            "x": None,
            "y": None,
            "rank": None,
            "source": "missing",
        }]
    })

    beams.sort(key=lambda b: b["cost"])
    return beams[:BEAM_WIDTH]


def transition_cost(beam, cand):
    """
    cand == None means missing
    """
    if cand is None:
        return MISSING_BASE_COST + MISSING_STEP_COST * beam["miss_streak"]

    current = (cand["x"], cand["y"])

    if beam["last2"] is None:
        # no visible history yet
        return rank_cost(cand["rank"])

    if beam["last1"] is None:
        # only one visible point known
        jump = euclidean(current, beam["last2"])
        return rank_cost(cand["rank"]) + FIRST_JUMP_WEIGHT * jump

    # two visible points known -> use velocity prediction
    prev = beam["last1"]
    last = beam["last2"]

    gap = beam["miss_streak"] + 1

    vx = last[0] - prev[0]
    vy = last[1] - prev[1]

    pred = (last[0] + gap * vx, last[1] + gap * vy)

    prediction_error = euclidean(current, pred)
    jump = euclidean(current, last)

    return (
        rank_cost(cand["rank"])
        + PREDICTION_WEIGHT * prediction_error
        + JUMP_WEIGHT * jump
    )


def extend_beam(beam, frame, cand):
    step_cost = transition_cost(beam, cand)

    new_beam = {
        "cost": beam["cost"] + step_cost,
        "last1": beam["last1"],
        "last2": beam["last2"],
        "miss_streak": beam["miss_streak"],
        "path": beam["path"][:]
    }

    if cand is None:
        new_beam["miss_streak"] += 1
        new_beam["path"].append({
            "frame": frame["frame"],
            "time": frame["time"],
            "x": None,
            "y": None,
            "rank": None,
            "source": "missing",
        })
        return new_beam

    current = (cand["x"], cand["y"])

    if new_beam["last2"] is None:
        new_beam["last1"] = None
        new_beam["last2"] = current
    else:
        new_beam["last1"] = new_beam["last2"]
        new_beam["last2"] = current

    new_beam["miss_streak"] = 0

    new_beam["path"].append({
        "frame": frame["frame"],
        "time": frame["time"],
        "x": cand["x"],
        "y": cand["y"],
        "rank": cand["rank"],
        "source": "candidate",
    })

    return new_beam


def track_with_beam_search(frames):
    beams = make_initial_beams(frames[0])

    for i, frame in enumerate(frames[1:], start=2):
        new_beams = []

        candidate_options = frame["candidates"][:] + [None]

        for beam in beams:
            for cand in candidate_options:
                new_beams.append(extend_beam(beam, frame, cand))

        new_beams.sort(key=lambda b: b["cost"])
        beams = new_beams[:BEAM_WIDTH]

        if i % 50 == 0 or i == len(frames):
            print(f"  Linked {i}/{len(frames)} frames")

    return beams[0]


# ============================================================
# EVALUATION
# ============================================================

def evaluate_path(best_path, frames, visible_labels, frame_height):
    radius = match_radius(frame_height)

    by_frame = {row["frame"]: row for row in best_path}
    top1_by_frame = {}

    for frame in frames:
        if frame["candidates"]:
            c = frame["candidates"][0]
            top1_by_frame[frame["frame"]] = (c["x"], c["y"])

    visible_frames = sorted(f for f in visible_labels if f in by_frame)

    tracker_hits = 0
    tracker_predicted = 0
    tracker_errors = []
    tracker_large = 0

    top1_hits = 0
    top1_predicted = 0
    top1_errors = []
    top1_large = 0

    per_frame = []

    for frame in visible_frames:
        truth = visible_labels[frame]

        # tracker
        row = by_frame[frame]
        tracker_xy = None
        if row["x"] is not None and row["y"] is not None:
            tracker_xy = (float(row["x"]), float(row["y"]))
            tracker_predicted += 1
            err = euclidean(truth, tracker_xy)
            tracker_errors.append(err)
            if err <= radius:
                tracker_hits += 1
            if err > LARGE_ERROR_THRESHOLD:
                tracker_large += 1
        else:
            err = None

        # top1
        top1_xy = top1_by_frame.get(frame)
        if top1_xy is not None:
            top1_predicted += 1
            top1_err = euclidean(truth, top1_xy)
            top1_errors.append(top1_err)
            if top1_err <= radius:
                top1_hits += 1
            if top1_err > LARGE_ERROR_THRESHOLD:
                top1_large += 1
        else:
            top1_err = None

        per_frame.append({
            "frame": frame,
            "time": frame / frames[0]["frame"] * 0 + frame / float(cache["fps"]),  # harmless time calc fallback
            "tracker_error_px": None if err is None else round(err, 2),
            "top1_error_px": None if top1_err is None else round(top1_err, 2),
        })

    result = {
        "visible_reference_frames": len(visible_frames),
        "matching_radius_px": round(radius, 2),
        "top1_hits": top1_hits,
        "top1_predictions_on_visible_frames": top1_predicted,
        "top1_mean_error_px": None if not top1_errors else round(sum(top1_errors) / len(top1_errors), 2),
        "top1_large_errors": top1_large,

        "tracker_hits": tracker_hits,
        "tracker_predictions_on_visible_frames": tracker_predicted,
        "tracker_misses_on_visible_frames": len(visible_frames) - tracker_predicted,
        "tracker_mean_error_px": None if not tracker_errors else round(sum(tracker_errors) / len(tracker_errors), 2),
        "tracker_large_errors": tracker_large,

        "note": "Development labels only; uncertain frames excluded; not a false-positive test"
    }

    return result, per_frame


def print_summary(result):
    print("\n========== SHUTTLE TRACKER V4 EVALUATION ==========")
    print(f"visible_reference_frames: {result['visible_reference_frames']}")
    print(f"matching_radius_px: {result['matching_radius_px']}")
    print(f"top1_hits: {result['top1_hits']}")
    print(f"top1_predictions_on_visible_frames: {result['top1_predictions_on_visible_frames']}")
    print(f"top1_mean_error_px: {result['top1_mean_error_px']}")
    print(f"top1_large_errors: {result['top1_large_errors']}")
    print(f"tracker_hits: {result['tracker_hits']}")
    print(f"tracker_predictions_on_visible_frames: {result['tracker_predictions_on_visible_frames']}")
    print(f"tracker_misses_on_visible_frames: {result['tracker_misses_on_visible_frames']}")
    print(f"tracker_mean_error_px: {result['tracker_mean_error_px']}")
    print(f"tracker_large_errors: {result['tracker_large_errors']}")
    print(f"note: {result['note']}")


def save_positions_csv(path, path_rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "x", "y", "rank", "source"])

        for row in path_rows:
            writer.writerow([
                row["frame"],
                f"{row['time']:.4f}",
                "" if row["x"] is None else f"{row['x']:.3f}",
                "" if row["y"] is None else f"{row['y']:.3f}",
                "" if row["rank"] is None else row["rank"],
                row["source"],
            ])


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("========== SHUTTLE TRACKER V4 ==========")

    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing cache file: {CACHE_PATH}")

    cache = load_json(CACHE_PATH)

    frame_width = int(cache.get("frame_width", cache.get("width", 1920)))
    frame_height = int(cache.get("frame_height", cache.get("height", 1080)))

    frames = load_candidates(cache)
    visible_labels = load_visible_labels(LABELS_PATH)

    print(f"Cached frames: {len(frames)}")
    print(f"Frame size: {frame_width}x{frame_height}")
    print("Step 1/2: Global beam-search tracking")
    best_beam = track_with_beam_search(frames)

    print("Step 2/2: Evaluation")
    result, per_frame = evaluate_path(best_beam["path"], frames, visible_labels, frame_height)
    print_summary(result)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    positions_csv = OUT_DIR / "optimized_positions.csv"
    evaluation_json = OUT_DIR / "evaluation.json"
    per_frame_json = OUT_DIR / "per_frame_errors.json"

    save_positions_csv(positions_csv, best_beam["path"])
    save_json(evaluation_json, result)
    save_json(per_frame_json, per_frame)

    print("\nSaved:")
    print(f"  {positions_csv}")
    print(f"  {evaluation_json}")
    print(f"  {per_frame_json}")
    print("\nDone!")