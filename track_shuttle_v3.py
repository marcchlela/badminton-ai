
import csv
import json
import math
from pathlib import Path

import track_shuttle_v1 as v1


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("outputs/shuttle")

CACHE_PATH = ROOT / "trajectory_v2/motion_candidates.json"
LABELS_PATH = ROOT / "hd_visibility_audit.json"
OUT = ROOT / "trajectory_v3"

OUT.mkdir(parents=True, exist_ok=True)

# Forward and backward positions must be this close
# for us to call the prediction a consensus.
AGREEMENT_RADIUS = 18.0

# Manual-label matching tolerance at the video's resolution.
BASE_MATCH_RADIUS = 12.0

# Detailed inspection of our known difficult section.
INSPECT_START = 205.9
INSPECT_END = 207.0


# ============================================================
# LOAD EXISTING CANDIDATES
# ============================================================

def load_cache():
    if not CACHE_PATH.exists():
        raise SystemExit(
            "Candidate cache missing. Run track_shuttle_v2.py first."
        )

    with CACHE_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    candidates = [
        [tuple(candidate) for candidate in frame]
        for frame in data["candidates"]
    ]

    return data, candidates


def point_from_choice(candidates, choice):
    if choice is None:
        return None

    if not 0 <= choice < len(candidates):
        return None

    return (
        float(candidates[choice][0]),
        float(candidates[choice][1])
    )


def distance(a, b):
    if a is None or b is None:
        return None

    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def fmt(value):
    if value is None:
        return "---"

    return f"{value:.1f}"


# ============================================================
# FORWARD / BACKWARD TRACKING
# ============================================================

def run_bidirectional(candidates):
    print("\nStep 1/3: Forward tracking")
    forward = list(v1.link_candidates(candidates))

    print("\nStep 2/3: Backward tracking")

    reversed_candidates = list(reversed(candidates))

    backward_reversed = list(
        v1.link_candidates(reversed_candidates)
    )

    backward = list(reversed(backward_reversed))

    if not (
        len(forward) == len(backward) == len(candidates)
    ):
        raise RuntimeError("Tracking result length mismatch")

    return forward, backward


# ============================================================
# BUILD CONSENSUS
# ============================================================

def create_results(data, candidates, forward, backward):
    first_frame = int(data["first_frame"])
    fps = float(data["fps"])

    rows = []

    for index, frame_candidates in enumerate(candidates):
        forward_point = point_from_choice(
            frame_candidates,
            forward[index]
        )

        backward_point = point_from_choice(
            frame_candidates,
            backward[index]
        )

        top1 = (
            tuple(frame_candidates[0][:2])
            if frame_candidates
            else None
        )

        disagreement = distance(
            forward_point,
            backward_point
        )

        if forward_point is None and backward_point is None:
            status = "both_missing"
            consensus = None

        elif forward_point is None or backward_point is None:
            status = "one_missing"
            consensus = None

        elif disagreement <= AGREEMENT_RADIUS:
            status = "agree"
            # Keep an actual observed detection, not an
            # invented midpoint between two candidates.
            consensus = forward_point

        else:
            status = "disagree"
            consensus = None

        rows.append({
            "frame": first_frame + index,
            "time": (first_frame + index) / fps,
            "top1": top1,
            "forward": forward_point,
            "backward": backward_point,
            "consensus": consensus,
            "disagreement": disagreement,
            "status": status,
            "forward_rank": (
                forward[index] + 1
                if forward[index] is not None else None
            ),
            "backward_rank": (
                backward[index] + 1
                if backward[index] is not None else None
            ),
        })

    return rows


# ============================================================
# EVALUATION: MANUAL LABELS ARE USED ONLY HERE
# ============================================================

def evaluate(rows, height):
    with LABELS_PATH.open(encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    by_frame = {
        row["frame"]: row
        for row in rows
    }

    radius = BASE_MATCH_RADIUS * height / 720.0

    labels = [
        annotation
        for annotation in annotations
        if annotation["status"] == "visible"
        and annotation["pixel"] is not None
    ]

    fields = [
        "top1",
        "forward",
        "backward",
        "consensus"
    ]

    results = {
        field: {
            "predictions": 0,
            "hits": 0,
            "errors": [],
            "large_errors": 0
        }
        for field in fields
    }

    status_counts = {}

    print("\n========== LABELED FRAME COMPARISON ==========")

    for annotation in labels:
        frame = int(annotation["frame"])
        row = by_frame.get(frame)

        if row is None:
            continue

        truth = tuple(annotation["pixel"])

        status = row["status"]

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

        frame_errors = {}

        for field in fields:
            prediction = row[field]
            error = distance(prediction, truth)

            frame_errors[field] = error

            if error is None:
                continue

            results[field]["predictions"] += 1
            results[field]["errors"].append(error)

            if error <= radius:
                results[field]["hits"] += 1

            if error > 50:
                results[field]["large_errors"] += 1

        if (
            status != "agree"
            or frame_errors["forward"] is None
            or frame_errors["forward"] > radius
        ):
            print(
                f"{row['time']:.2f}s | "
                f"{status:12s} | "
                f"Top1={fmt(frame_errors['top1']):>6s} | "
                f"Forward={fmt(frame_errors['forward']):>6s} | "
                f"Backward={fmt(frame_errors['backward']):>6s} | "
                f"Consensus={fmt(frame_errors['consensus']):>6s}"
            )

    total_labels = sum(
        int(a["frame"] in by_frame)
        for a in labels
    )

    summary = {
        "visible_reference_frames": total_labels,
        "matching_radius": radius,
        "status_counts": status_counts,
        "methods": {}
    }

    print("\n========== TRACKER V3 EVALUATION ==========")

    for field in fields:
        result = results[field]

        predicted = result["predictions"]
        hits = result["hits"]
        errors = result["errors"]

        precision_on_visible = (
            100.0 * hits / predicted
            if predicted else 0.0
        )

        recall_on_visible = (
            100.0 * hits / total_labels
            if total_labels else 0.0
        )

        mean_error = (
            sum(errors) / len(errors)
            if errors else None
        )

        summary["methods"][field] = {
            "predictions": predicted,
            "hits": hits,
            "precision_on_visible_percent": round(
                precision_on_visible, 1
            ),
            "recall_on_visible_percent": round(
                recall_on_visible, 1
            ),
            "mean_error_px": (
                round(mean_error, 1)
                if mean_error is not None
                else None
            ),
            "errors_over_50px": result["large_errors"]
        }

        print(
            f"{field:10s} | "
            f"hits={hits:2d}/{total_labels} | "
            f"predicted={predicted:2d}/{total_labels} | "
            f"precision={precision_on_visible:5.1f}% | "
            f"recall={recall_on_visible:5.1f}% | "
            f"large_errors={result['large_errors']}"
        )

    print("\nLabel statuses:")
    for key, count in status_counts.items():
        print(f"  {key}: {count}")

    return summary


# ============================================================
# INSPECT THE PROBLEM AREA
# ============================================================

def inspect_window(rows):
    print(
        f"\n========== {INSPECT_START:.1f}s - "
        f"{INSPECT_END:.1f}s =========="
    )

    for row in rows:
        if not INSPECT_START <= row["time"] <= INSPECT_END:
            continue

        print(
            f"{row['time']:6.2f}s | "
            f"{row['status']:12s} | "
            f"forward rank={str(row['forward_rank']):>4s} | "
            f"backward rank={str(row['backward_rank']):>4s} | "
            f"distance={fmt(row['disagreement']):>6s}px"
        )


# ============================================================
# EXPORT
# ============================================================

def save_results(rows, summary):
    csv_path = OUT / "bidirectional_positions.csv"
    json_path = OUT / "evaluation.json"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "frame", "time", "status",
            "top1_x", "top1_y",
            "forward_x", "forward_y",
            "backward_x", "backward_y",
            "consensus_x", "consensus_y",
            "forward_rank", "backward_rank",
            "disagreement_px"
        ]

        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            output = {
                "frame": row["frame"],
                "time": round(row["time"], 4),
                "status": row["status"],
                "forward_rank": row["forward_rank"],
                "backward_rank": row["backward_rank"],
                "disagreement_px": (
                    round(row["disagreement"], 2)
                    if row["disagreement"] is not None
                    else ""
                )
            }

            for field in (
                "top1",
                "forward",
                "backward",
                "consensus"
            ):
                point = row[field]

                output[f"{field}_x"] = (
                    round(point[0], 2)
                    if point is not None else ""
                )

                output[f"{field}_y"] = (
                    round(point[1], 2)
                    if point is not None else ""
                )

            writer.writerow(output)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(csv_path)
    print(json_path)


# ============================================================
# MAIN
# ============================================================

def main():
    print("========== SHUTTLE TRACKER V3 ==========")

    data, candidates = load_cache()

    print("Cached frames:", len(candidates))
    print("No video decoding needed.")

    forward, backward = run_bidirectional(
        candidates
    )

    print("\nStep 3/3: Building consensus")

    rows = create_results(
        data, candidates, forward, backward
    )

    summary = evaluate(
        rows, int(data["height"])
    )

    inspect_window(rows)

    save_results(rows, summary)

    print("\nDone!")


if __name__ == "__main__":
    main()
