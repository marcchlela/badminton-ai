
"""
Badminton AI — Contact Detection V1

Heuristic racket-contact candidate detector using:
    - TrackNetV3 shuttle observations
    - Incoming/outgoing trajectory regression
    - Direction and velocity changes
    - Player bounding-box proximity
    - Temporal non-maximum suppression

Manual annotations are used ONLY for evaluation.
They are never used to generate candidate scores.

Outputs:
    candidates.csv
    candidates.json
    frame_scores.csv
    evaluation.json
    metadata.json
    diagnostic.png
    optional contact_preview.mp4

Usage:
    python detect_contacts_v1.py --rally coach-36-46 --preview
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

CATALOG = ROOT / "replay_hub" / "rallies.json"

ANNOTATIONS = ROOT / "outputs" / "annotations"

OUTPUT_ROOT = ROOT / "outputs" / "contacts" / "v1"

SIDES = ("Near", "Far")

COLORS = {
    "Near": (243, 207, 104),
    "Far": (103, 188, 246),
    "shuttle": (170, 246, 138),
    "candidate": (194, 207, 104),
    "manual": (250, 139, 167),
    "white": (245, 245, 245),
    "muted": (145, 160, 175),
    "background": (25, 17, 11),
}


# ============================================================
# FILE HELPERS
# ============================================================

def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def resolve_file(value):
    path = (ROOT / value).resolve()

    if not path.is_relative_to(ROOT):
        raise ValueError("Path escapes project")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def valid_vector(value, length):
    if not isinstance(value, (list, tuple)):
        return None

    if len(value) != length:
        return None

    try:
        array = np.asarray(
            value,
            dtype=np.float64,
        )
    except (ValueError, TypeError):
        return None

    if not np.all(np.isfinite(array)):
        return None

    return array


def rounded(value, digits=5):
    if value is None:
        return None

    return round(float(value), digits)


# ============================================================
# LOAD AND VALIDATE RALLY
# ============================================================

def load_rally(rally_id):
    catalog = read_json(CATALOG)

    entry = next(
        (
            item for item in catalog["rallies"]
            if item["id"] == rally_id
        ),
        None,
    )

    if entry is None:
        raise ValueError(
            f"Unknown rally: {rally_id}"
        )

    timeline = read_json(
        resolve_file(entry["timeline"])
    )

    metadata = read_json(
        resolve_file(entry["metadata"])
    )

    video_path = resolve_file(entry["video"])

    fps = float(metadata["fps"])

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid FPS")

    start_frame = int(metadata["start_frame"])

    count = int(metadata["total_frames"])

    if not timeline or len(timeline) != count:
        raise ValueError(
            "Timeline length mismatch"
        )

    for index, row in enumerate(timeline):
        if row["local_frame"] != index:
            raise ValueError(
                f"Local frame mismatch: {index}"
            )

        if row["frame"] != start_frame + index:
            raise ValueError(
                f"Source frame mismatch: {index}"
            )

        expected_time = (
            start_frame + index
        ) / fps

        if (
            not math.isfinite(float(row["time"]))
            or abs(row["time"] - expected_time) > .002
        ):
            raise ValueError(
                f"Timestamp mismatch: {index}"
            )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open source video"
        )

    try:
        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )
        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        video_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        video_fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise ValueError(
            "Invalid source dimensions"
        )

    if video_count != count:
        raise ValueError(
            "Video frame count mismatch"
        )

    if abs(video_fps - fps) > .001:
        raise ValueError(
            "Video FPS mismatch"
        )

    return {
        "id": rally_id,
        "title": entry["title"],
        "timeline": timeline,
        "metadata": metadata,
        "video_path": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "diagonal": math.hypot(width, height),
        "start_frame": start_frame,
        "count": count,
    }


# ============================================================
# SHUTTLE OBSERVATIONS
# ============================================================

def build_shuttle_series(rally):
    timeline = rally["timeline"]

    points = np.full(
        (len(timeline), 2),
        np.nan,
        dtype=np.float64,
    )

    for index, row in enumerate(timeline):
        shuttle = row.get("shuttle") or {}

        if not shuttle.get("visible"):
            continue

        pixel = valid_vector(
            shuttle.get("pixel"),
            2,
        )

        if pixel is None:
            continue

        x, y = pixel

        if not (
            0 <= x < rally["width"]
            and 0 <= y < rally["height"]
        ):
            continue

        points[index] = pixel

    return points


def point_available(points, index):
    if not 0 <= index < len(points):
        return False

    return bool(
        np.all(np.isfinite(points[index]))
    )


# ============================================================
# PLAYER BOUNDING BOXES
# ============================================================

def build_player_boxes(rally):
    result = {
        side: []
        for side in SIDES
    }

    for index, row in enumerate(
        rally["timeline"]
    ):
        if not row.get("players_sampled"):
            continue

        players = row.get("players") or {}

        for side in SIDES:
            player = players.get(side) or {}

            # Prefer actual visual detections.
            if player.get("visual_detection") is not True:
                continue

            box = valid_vector(
                player.get("bbox"),
                4,
            )

            if box is None:
                continue

            if box[2] <= 0 or box[3] <= 0:
                continue

            result[side].append(
                (index, box)
            )

    return result


def nearest_player_box(boxes, side, index,
                       max_gap=3):
    samples = boxes[side]

    if not samples:
        return None

    sample_index, box = min(
        samples,
        key=lambda sample: abs(
            sample[0] - index
        ),
    )

    gap = abs(sample_index - index)

    if gap > max_gap:
        return None

    return {
        "box": box,
        "sample_index": sample_index,
        "frame_gap": gap,
    }


def distance_to_box(point, box):
    px, py = point

    x, y, w, h = box

    dx = max(
        x - px,
        0.0,
        px - (x + w),
    )

    dy = max(
        y - py,
        0.0,
        py - (y + h),
    )

    return math.hypot(dx, dy)


def player_proximity(point, boxes, index):
    if point is None:
        return {
            "player": "Unknown",
            "distance_px": None,
            "distance_relative": None,
            "score": 0.0,
            "available": False,
        }

    candidates = []

    for side in SIDES:
        sample = nearest_player_box(
            boxes,
            side,
            index,
        )

        if sample is None:
            continue

        box = sample["box"]

        distance = distance_to_box(
            point,
            box,
        )

        # Normalize by the detected player's height.
        # This helps account for near/far perspective.
        relative_distance = (
            distance / max(box[3], 1.0)
        )

        score = math.exp(
            -relative_distance / 0.65
        )

        # A slightly older player box is less reliable.
        temporal_weight = (
            1.0 - sample["frame_gap"] * 0.12
        )

        score *= temporal_weight

        candidates.append({
            "player": side,
            "distance_px": distance,
            "distance_relative": relative_distance,
            "score": score,
            "available": True,
        })

    if not candidates:
        return {
            "player": "Unknown",
            "distance_px": None,
            "distance_relative": None,
            "score": 0.0,
            "available": False,
        }

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    best = candidates[0].copy()

    # Avoid assigning a player when both are plausible.
    if len(candidates) > 1:
        difference = (
            best["score"]
            - candidates[1]["score"]
        )

        if difference < 0.12:
            best["player"] = "Unknown"

    if best["score"] < 0.15:
        best["player"] = "Unknown"

    return best


# ============================================================
# TRAJECTORY REGRESSION
# ============================================================

def fit_velocity(points, indices, fps,
                 minimum_points=3):
    valid_indices = [
        index
        for index in indices
        if point_available(points, index)
    ]

    if len(valid_indices) < minimum_points:
        return None

    # Require meaningful temporal support.
    if valid_indices[-1] - valid_indices[0] < 2:
        return None

    times = (
        np.asarray(
            valid_indices,
            dtype=np.float64,
        ) / fps
    )

    # Center times for numerical stability.
    times -= times.mean()

    observations = points[
        valid_indices
    ]

    design = np.column_stack([
        times,
        np.ones(len(times)),
    ])

    coefficients, _, _, _ = (
        np.linalg.lstsq(
            design,
            observations,
            rcond=None,
        )
    )

    velocity = coefficients[0]

    prediction = (
        design @ coefficients
    )

    residuals = (
        observations - prediction
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    residuals ** 2,
                    axis=1,
                )
            )
        )
    )

    return {
        "velocity": velocity,
        "indices": valid_indices,
        "count": len(valid_indices),
        "rmse_px": rmse,
    }


def angle_between(a, b):
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0

    cosine = float(
        np.dot(a, b) / (norm_a * norm_b)
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(cosine)
    )


def estimate_center_point(points, index):
    if point_available(points, index):
        return points[index], "observed"

    previous = next(
        (
            i for i in range(
                index - 1,
                max(-1, index - 3),
                -1,
            )
            if point_available(points, i)
        ),
        None,
    )

    following = next(
        (
            i for i in range(
                index + 1,
                min(len(points), index + 3),
            )
            if point_available(points, i)
        ),
        None,
    )

    if previous is None or following is None:
        return None, "missing"

    if following - previous > 4:
        return None, "missing"

    fraction = (
        (index - previous)
        / (following - previous)
    )

    point = (
        points[previous]
        + fraction * (
            points[following]
            - points[previous]
        )
    )

    return point, "interpolated"


# ============================================================
# CONTACT FEATURE EXTRACTION
# ============================================================

def score_frame(rally, points, boxes,
                index, window):
    before = list(
        range(
            max(0, index - window),
            index,
        )
    )

    after = list(
        range(
            index + 1,
            min(
                len(points),
                index + window + 1,
            ),
        )
    )

    incoming = fit_velocity(
        points,
        before,
        rally["fps"],
    )

    outgoing = fit_velocity(
        points,
        after,
        rally["fps"],
    )

    if incoming is None or outgoing is None:
        return None

    # Don't connect widely separated observations
    # across a large missing-detection gap.
    before_gap = (
        index - incoming["indices"][-1]
    )

    after_gap = (
        outgoing["indices"][0] - index
    )

    if before_gap > 3 or after_gap > 3:
        return None

    center, center_source = (
        estimate_center_point(
            points,
            index,
        )
    )

    if center is None:
        return None

    vin = incoming["velocity"]
    vout = outgoing["velocity"]

    speed_in = float(
        np.linalg.norm(vin)
    )

    speed_out = float(
        np.linalg.norm(vout)
    )

    velocity_change = float(
        np.linalg.norm(vout - vin)
    )

    angle = angle_between(
        vin,
        vout,
    )

    diagonal = rally["diagonal"]

    # Normalize velocity by image diagonal.
    # Units: image diagonals per second.
    speed_in_normalized = (
        speed_in / diagonal
    )

    speed_out_normalized = (
        speed_out / diagonal
    )

    velocity_change_normalized = (
        velocity_change / diagonal
    )

    maximum_speed = max(
        speed_in_normalized,
        speed_out_normalized,
    )

    # Ignore extremely slow observations.
    # This reduces false positives caused by
    # nearly stationary detection jitter.
    if maximum_speed < 0.035:
        return None

    angle_score = float(
        np.clip(angle / 150.0, 0, 1)
    )

    impulse_score = float(
        np.clip(
            velocity_change_normalized / 0.65,
            0,
            1,
        )
    )

    speed_difference = abs(
        speed_out - speed_in
    )

    speed_change_score = float(
        np.clip(
            speed_difference /
            max(speed_in, speed_out, 1e-6),
            0,
            1,
        )
    )

    proximity = player_proximity(
        center,
        boxes,
        index,
    )

    support = (
        incoming["count"] + outgoing["count"]
    ) / (2 * window)

    fit_error = (
        incoming["rmse_px"]
        + outgoing["rmse_px"]
    ) / 2.0

    # A noisy trajectory should contribute
    # less confidence to a candidate.
    fit_scale = max(
        2.0,
        diagonal * 0.006,
    )

    fit_quality = math.exp(
        -fit_error / fit_scale
    )

    # Initial heuristic weights.
    # These are NOT learned or calibrated.
    weighted_score = (
        0.35 * angle_score
        + 0.35 * impulse_score
        + 0.10 * speed_change_score
        + 0.20 * proximity["score"]
    )

    quality_multiplier = (
        0.55
        + 0.25 * support
        + 0.20 * fit_quality
    )

    score = (
        weighted_score
        * quality_multiplier
    )

    if center_source == "interpolated":
        score *= 0.88

    row = rally["timeline"][index]

    return {
        "local_frame": index,
        "source_frame": row["frame"],
        "time_seconds": row["time"],
        "score": rounded(score),
        "angle_degrees": rounded(angle, 2),
        "incoming_speed_px_s": rounded(
            speed_in, 2
        ),
        "outgoing_speed_px_s": rounded(
            speed_out, 2
        ),
        "velocity_change_px_s": rounded(
            velocity_change, 2
        ),
        "velocity_change_normalized": rounded(
            velocity_change_normalized
        ),
        "angle_score": rounded(angle_score),
        "impulse_score": rounded(impulse_score),
        "speed_change_score": rounded(
            speed_change_score
        ),
        "proximity_score": rounded(
            proximity["score"]
        ),
        "predicted_player": proximity["player"],
        "player_distance_px": rounded(
            proximity["distance_px"], 2
        ),
        "player_distance_relative": rounded(
            proximity["distance_relative"]
        ),
        "player_box_available": int(
            proximity["available"]
        ),
        "support": rounded(support),
        "fit_quality": rounded(fit_quality),
        "fit_error_px": rounded(
            fit_error, 2
        ),
        "incoming_observations": incoming["count"],
        "outgoing_observations": outgoing["count"],
        "shuttle_x_px": rounded(center[0], 2),
        "shuttle_y_px": rounded(center[1], 2),
        "center_source": center_source,
        "source": "heuristic_v1",
    }


# ============================================================
# TEMPORAL NON-MAXIMUM SUPPRESSION
# ============================================================

def select_candidates(frame_scores,
                      threshold,
                      minimum_gap):
    eligible = [
        item
        for item in frame_scores
        if item["score"] >= threshold
    ]

    eligible.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = []

    for item in eligible:
        frame = item["local_frame"]

        conflict = any(
            abs(
                other["local_frame"] - frame
            ) <= minimum_gap
            for other in selected
        )

        if conflict:
            continue

        selected.append(item.copy())

    selected.sort(
        key=lambda item: item["local_frame"]
    )

    for number, item in enumerate(
        selected,
        start=1,
    ):
        item["candidate_id"] = (
            f"contact_{number:03d}"
        )

    ranked = sorted(
        selected,
        key=lambda item: item["score"],
        reverse=True,
    )

    for rank, item in enumerate(
        ranked,
        start=1,
    ):
        item["score_rank"] = rank

    return selected


# ============================================================
# MANUAL ANNOTATIONS — EVALUATION ONLY
# ============================================================

def load_confirmed_contacts(rally):
    path = (
        ANNOTATIONS /
        f"{rally['id']}.json"
    )

    if not path.exists():
        return []

    data = read_json(path)

    if data.get("rally_id") != rally["id"]:
        raise ValueError(
            "Annotation rally mismatch"
        )

    events = []

    for item in data.get("events", []):
        if item.get("kind") != "contact":
            continue

        if item.get("confidence") != "confirmed":
            continue

        index = item.get("local_frame")

        if type(index) is not int:
            raise ValueError(
                "Invalid annotation frame"
            )

        if not 0 <= index < rally["count"]:
            raise ValueError(
                "Annotation outside rally"
            )

        if item["source_frame"] != (
            rally["start_frame"] + index
        ):
            raise ValueError(
                "Annotation source frame mismatch"
            )

        events.append(item)

    return sorted(
        events,
        key=lambda item: item["local_frame"],
    )


def evaluate(candidates, manual, tolerance,
             frame_scores, fps):
    pairs = []

    for manual_index, event in enumerate(manual):
        for candidate_index, candidate in enumerate(
            candidates
        ):
            error = abs(
                candidate["local_frame"]
                - event["local_frame"]
            )

            if error <= tolerance:
                pairs.append((
                    error,
                    manual_index,
                    candidate_index,
                ))

    # One-to-one, closest-first matching.
    pairs.sort()

    used_manual = set()
    used_candidates = set()
    matches = []

    for error, mi, ci in pairs:
        if mi in used_manual:
            continue

        if ci in used_candidates:
            continue

        used_manual.add(mi)
        used_candidates.add(ci)

        event = manual[mi]
        candidate = candidates[ci]

        matches.append({
            "event_id": event["id"],
            "manual_source_frame": event["source_frame"],
            "candidate_source_frame": (
                candidate["source_frame"]
            ),
            "absolute_error_frames": error,
            "absolute_error_ms": rounded(
                error * 1000 / fps,
                2,
            ),
            "manual_player": event["player"],
            "predicted_player": (
                candidate["predicted_player"]
            ),
            "manual_shot": event["shot"],
            "candidate_score": candidate["score"],
        })

    score_lookup = {
        row["local_frame"]: row
        for row in frame_scores
    }

    reference_details = []

    for event in manual:
        index = event["local_frame"]

        nearest_candidate = (
            min(
                candidates,
                key=lambda candidate: abs(
                    candidate["local_frame"] - index
                ),
            )
            if candidates else None
        )

        reference_details.append({
            "event_id": event["id"],
            "source_frame": event["source_frame"],
            "player": event["player"],
            "shot": event["shot"],
            "score_at_annotation": (
                score_lookup[index]["score"]
                if index in score_lookup
                else None
            ),
            "nearest_candidate_frame": (
                nearest_candidate["source_frame"]
                if nearest_candidate
                else None
            ),
            "nearest_candidate_error_frames": (
                abs(
                    nearest_candidate["local_frame"]
                    - index
                )
                if nearest_candidate
                else None
            ),
        })

    return {
        "confirmed_manual_contacts": len(manual),
        "predicted_candidates": len(candidates),
        "matched_manual_contacts": len(matches),
        "unmatched_manual_contacts": (
            len(manual) - len(matches)
        ),
        "unreviewed_candidates": (
            len(candidates) - len(matches)
        ),
        "tolerance_frames": tolerance,
        "annotated_contact_recovery": (
            len(matches) / len(manual)
            if manual else None
        ),
        "mean_matched_error_frames": (
            sum(
                match["absolute_error_frames"]
                for match in matches
            ) / len(matches)
            if matches else None
        ),
        "matches": matches,
        "reference_details": reference_details,
        "limitations": [
            "Manual annotations are not exhaustive.",
            "Unmatched candidates are not automatically false positives.",
            "Precision and overall recall cannot yet be estimated reliably.",
            "This is a heuristic baseline, not a trained AI model.",
            "The score is not a probability.",
            "Overlapping clips are not independent evaluation data.",
        ],
    }


# ============================================================
# DIAGNOSTIC PLOT
# ============================================================

def create_diagnostic(rally, frame_scores,
                      candidates, manual,
                      threshold, output):
    width = 1400
    height = 450

    image = np.full(
        (height, width, 3),
        COLORS["background"],
        dtype=np.uint8,
    )

    left = 70
    right = width - 35

    top = 95
    bottom = height - 65

    maximum_frame = max(
        1,
        rally["count"] - 1,
    )

    def x_at(index):
        return int(
            left +
            index / maximum_frame *
            (right - left)
        )

    def y_at(score):
        return int(
            bottom -
            np.clip(score, 0, 1) *
            (bottom - top)
        )

    cv2.putText(
        image,
        "BADMINTON AI / CONTACT DETECTION V1",
        (30, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        .8,
        COLORS["white"],
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"{rally['id']} | "
        f"{len(candidates)} candidates | "
        f"{len(manual)} confirmed contacts",
        (30, 63),
        cv2.FONT_HERSHEY_SIMPLEX,
        .53,
        COLORS["muted"],
        1,
        cv2.LINE_AA,
    )

    for tick in range(6):
        value = tick / 5

        y = y_at(value)

        cv2.line(
            image,
            (left, y),
            (right, y),
            (55, 47, 42),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{value:.1f}",
            (20, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            .42,
            COLORS["muted"],
            1,
            cv2.LINE_AA,
        )

    threshold_y = y_at(threshold)

    cv2.line(
        image,
        (left, threshold_y),
        (right, threshold_y),
        (100, 120, 185),
        1,
        cv2.LINE_AA,
    )

    previous = None

    for row in frame_scores:
        point = (
            x_at(row["local_frame"]),
            y_at(row["score"]),
        )

        if previous is not None:
            previous_row, previous_point = (
                previous
            )

            if (
                row["local_frame"]
                - previous_row["local_frame"]
                == 1
            ):
                cv2.line(
                    image,
                    previous_point,
                    point,
                    COLORS["shuttle"],
                    2,
                    cv2.LINE_AA,
                )

        previous = (row, point)

    for candidate in candidates:
        x = x_at(
            candidate["local_frame"]
        )

        y = y_at(candidate["score"])

        cv2.circle(
            image,
            (x, y),
            6,
            COLORS["candidate"],
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            str(candidate["source_frame"]),
            (x - 17, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            .42,
            COLORS["white"],
            1,
            cv2.LINE_AA,
        )

    for event in manual:
        x = x_at(
            event["local_frame"]
        )

        cv2.line(
            image,
            (x, top),
            (x, bottom),
            COLORS["manual"],
            2,
            cv2.LINE_AA,
        )

    for tick in range(6):
        fraction = tick / 5

        index = round(
            fraction * maximum_frame
        )

        x = x_at(index)

        time_seconds = (
            rally["start_frame"] + index
        ) / rally["fps"]

        cv2.putText(
            image,
            f"{time_seconds:.2f}s",
            (x - 23, bottom + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            .42,
            COLORS["muted"],
            1,
            cv2.LINE_AA,
        )

    legend = [
        ("Score", COLORS["shuttle"]),
        ("Candidate", COLORS["candidate"]),
        ("Manual contact", COLORS["manual"]),
    ]

    for i, (label, color) in enumerate(
        legend
    ):
        x = 70 + i * 230

        cv2.circle(
            image,
            (x, height - 20),
            5,
            color,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            label,
            (x + 12, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            .47,
            COLORS["white"],
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(
        str(output),
        image,
    ):
        raise RuntimeError(
            "Could not save diagnostic image"
        )


# ============================================================
# VIDEO PREVIEW
# ============================================================

def create_preview(rally, points, candidates,
                   manual, output):
    cap = cv2.VideoCapture(
        str(rally["video_path"])
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open preview source"
        )

    width = rally["width"]
    height = rally["height"]

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        rally["fps"],
        (width, height + 100),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Cannot create preview MP4"
        )

    candidate_lookup = {
        item["local_frame"]: item
        for item in candidates
    }

    manual_frames = {
        event["local_frame"]
        for event in manual
    }

    written = 0

    try:
        for index in range(
            rally["count"]
        ):
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Cannot read frame {index}"
                )

            canvas = np.full(
                (height + 100, width, 3),
                COLORS["background"],
                dtype=np.uint8,
            )

            canvas[:height] = frame

            # Shuttle trajectory.
            for j in range(
                max(1, index - 8),
                index + 1,
            ):
                if not (
                    point_available(points, j - 1)
                    and point_available(points, j)
                ):
                    continue

                a = tuple(
                    np.rint(
                        points[j - 1]
                    ).astype(int)
                )

                b = tuple(
                    np.rint(
                        points[j]
                    ).astype(int)
                )

                cv2.line(
                    canvas,
                    a,
                    b,
                    COLORS["shuttle"],
                    2,
                    cv2.LINE_AA,
                )

            if point_available(points, index):
                pixel = tuple(
                    np.rint(
                        points[index]
                    ).astype(int)
                )

                cv2.circle(
                    canvas,
                    pixel,
                    5,
                    COLORS["shuttle"],
                    2,
                    cv2.LINE_AA,
                )

            source_frame = (
                rally["start_frame"] + index
            )

            time_seconds = (
                source_frame / rally["fps"]
            )

            cv2.putText(
                canvas,
                f"Frame {source_frame} | "
                f"{time_seconds:.2f}s",
                (20, height + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                .65,
                COLORS["white"],
                1,
                cv2.LINE_AA,
            )

            nearby = [
                candidate
                for candidate in candidates
                if abs(
                    candidate["local_frame"] - index
                ) <= 4
            ]

            if nearby:
                candidate = min(
                    nearby,
                    key=lambda item: abs(
                        item["local_frame"] - index
                    ),
                )

                message = (
                    f"CANDIDATE "
                    f"{candidate['source_frame']} | "
                    f"{candidate['predicted_player']} | "
                    f"Score {candidate['score']:.3f}"
                )

                cv2.putText(
                    canvas,
                    message,
                    (20, height + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .65,
                    COLORS["candidate"],
                    2,
                    cv2.LINE_AA,
                )

                if index in candidate_lookup:
                    center = (
                        candidate["shuttle_x_px"],
                        candidate["shuttle_y_px"],
                    )

                    cv2.circle(
                        canvas,
                        tuple(
                            int(round(v))
                            for v in center
                        ),
                        22,
                        COLORS["candidate"],
                        2,
                        cv2.LINE_AA,
                    )

            if index in manual_frames:
                cv2.putText(
                    canvas,
                    "CONFIRMED MANUAL CONTACT",
                    (20, height + 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .57,
                    COLORS["manual"],
                    2,
                    cv2.LINE_AA,
                )

            writer.write(canvas)
            written += 1

    finally:
        cap.release()
        writer.release()

    if written != rally["count"]:
        raise RuntimeError(
            "Incomplete preview video"
        )


# ============================================================
# MAIN DETECTION PIPELINE
# ============================================================

def run(args):
    rally = load_rally(args.rally)

    print("========== CONTACT DETECTION V1 ==========")
    print("Rally:", rally["id"])
    print("Frames:", rally["count"])
    print("FPS:", rally["fps"])

    # Convert reference frame settings at 25 FPS
    # into equivalent durations at source FPS.
    window = max(
        3,
        round(
            args.window *
            rally["fps"] / 25.0
        ),
    )

    minimum_gap = max(
        1,
        round(
            args.min_gap *
            rally["fps"] / 25.0
        ),
    )

    tolerance = max(
        1,
        round(
            args.tolerance *
            rally["fps"] / 25.0
        ),
    )

    points = build_shuttle_series(
        rally
    )

    boxes = build_player_boxes(
        rally
    )

    visible_count = int(
        np.sum(
            np.all(
                np.isfinite(points),
                axis=1,
            )
        )
    )

    print()
    print(
        "Valid shuttle observations:",
        visible_count,
    )
    print(
        "Near player visual boxes:",
        len(boxes["Near"]),
    )
    print(
        "Far player visual boxes:",
        len(boxes["Far"]),
    )

    frame_scores = []

    for index in range(rally["count"]):
        row = score_frame(
            rally,
            points,
            boxes,
            index,
            window,
        )

        if row is not None:
            frame_scores.append(row)

    candidates = select_candidates(
        frame_scores,
        args.threshold,
        minimum_gap,
    )

    # Manual data is loaded only AFTER detection.
    manual = load_confirmed_contacts(
        rally
    )

    evaluation = evaluate(
        candidates,
        manual,
        tolerance,
        frame_scores,
        rally["fps"],
    )

    print()
    print(
        "Scorable frames:",
        len(frame_scores),
    )
    print(
        "Candidate threshold:",
        args.threshold,
    )
    print(
        "Contact candidates:",
        len(candidates),
    )

    print()
    print(
        f"{'Frame':>7} "
        f"{'Time':>8} "
        f"{'Player':>9} "
        f"{'Angle':>8} "
        f"{'Score':>8}"
    )

    print("-" * 46)

    for item in candidates:
        print(
            f"{item['source_frame']:>7} "
            f"{item['time_seconds']:>8.2f} "
            f"{item['predicted_player']:>9} "
            f"{item['angle_degrees']:>8.1f} "
            f"{item['score']:>8.3f}"
        )

    print()
    print("========== MANUAL EVALUATION ==========")

    print(
        "Confirmed contacts:",
        evaluation["confirmed_manual_contacts"],
    )

    print(
        "Matched contacts:",
        evaluation["matched_manual_contacts"],
    )

    print(
        "Unreviewed candidates:",
        evaluation["unreviewed_candidates"],
    )

    for item in evaluation["reference_details"]:
        print()
        print(
            "Manual frame:",
            item["source_frame"],
        )
        print(
            "Manual label:",
            item["player"],
            item["shot"],
        )
        print(
            "Score at manual frame:",
            item["score_at_annotation"],
        )
        print(
            "Nearest candidate:",
            item["nearest_candidate_frame"],
        )
        print(
            "Frame error:",
            item["nearest_candidate_error_frames"],
        )

    output = (
        OUTPUT_ROOT / args.rally / args.name
    )

    # Prevent accidental overwriting of experiments.
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    fields = [
        "local_frame",
        "source_frame",
        "time_seconds",
        "score",
        "angle_degrees",
        "incoming_speed_px_s",
        "outgoing_speed_px_s",
        "velocity_change_px_s",
        "velocity_change_normalized",
        "angle_score",
        "impulse_score",
        "speed_change_score",
        "proximity_score",
        "predicted_player",
        "player_distance_px",
        "player_distance_relative",
        "player_box_available",
        "support",
        "fit_quality",
        "fit_error_px",
        "incoming_observations",
        "outgoing_observations",
        "shuttle_x_px",
        "shuttle_y_px",
        "center_source",
        "source",
    ]

    write_csv(
        output / "frame_scores.csv",
        frame_scores,
        fields,
    )

    write_csv(
        output / "candidates.csv",
        candidates,
        [
            "candidate_id",
            "score_rank",
            *fields,
        ],
    )

    write_json(
        output / "candidates.json",
        {
            "schema_version": 1,
            "rally_id": rally["id"],
            "source": "heuristic_v1",
            "candidates": candidates,
        },
    )

    write_json(
        output / "evaluation.json",
        evaluation,
    )

    create_diagnostic(
        rally,
        frame_scores,
        candidates,
        manual,
        args.threshold,
        output / "diagnostic.png",
    )

    if args.preview:
        print()
        print("Rendering contact preview...")

        create_preview(
            rally,
            points,
            candidates,
            manual,
            output / "contact_preview.mp4",
        )

    write_json(
        output / "metadata.json",
        {
            "version": "contact_detection_v1",
            "rally_id": rally["id"],
            "fps": rally["fps"],
            "total_frames": rally["count"],
            "shuttle_observations": visible_count,
            "scorable_frames": len(frame_scores),
            "candidate_count": len(candidates),
            "configuration": {
                "window_frames": window,
                "threshold": args.threshold,
                "minimum_gap_frames": minimum_gap,
                "evaluation_tolerance_frames": tolerance,
            },
            "score_is_probability": False,
            "manual_labels_used_for_detection": False,
        },
    )

    print()
    print("========== COMPLETE ==========")
    print("Output:", output)
    print("Diagnostic: diagnostic.png")

    if args.preview:
        print("Preview: contact_preview.mp4")


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect potential racket contacts "
            "from tracked shuttle trajectories."
        )
    )

    parser.add_argument(
        "--rally",
        default="coach-36-46",
    )

    parser.add_argument(
        "--name",
        default="baseline",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help=(
            "Trajectory fitting window on each side, "
            "in reference frames at 25 FPS."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--min-gap",
        type=int,
        default=8,
        help=(
            "Minimum candidate separation "
            "in reference frames at 25 FPS."
        ),
    )

    parser.add_argument(
        "--tolerance",
        type=int,
        default=3,
        help=(
            "Manual evaluation tolerance "
            "in reference frames at 25 FPS."
        ),
    )

    parser.add_argument(
        "--preview",
        action="store_true",
    )

    args = parser.parse_args()

    if not 3 <= args.window <= 15:
        parser.error(
            "--window must be between 3 and 15"
        )

    if not 0 <= args.threshold <= 1:
        parser.error(
            "--threshold must be between 0 and 1"
        )

    if not 1 <= args.min_gap <= 50:
        parser.error(
            "--min-gap must be between 1 and 50"
        )

    if not 1 <= args.tolerance <= 15:
        parser.error(
            "--tolerance must be between 1 and 15"
        )

    for name in (args.rally, args.name):
        if (
            not name
            or len(name) > 80
            or not all(
                c.isascii()
                and (
                    c.isalnum()
                    or c in "-_"
                )
                for c in name
            )
        ):
            parser.error(
                "Invalid rally or experiment name"
            )

    run(args)


if __name__ == "__main__":
    main()
