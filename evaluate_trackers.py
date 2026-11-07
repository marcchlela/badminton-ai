
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(description="Compare player tracks with manual references.")
parser.add_argument("--reference", type=Path, default=Path("outputs/manual_evaluation.json"))
parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
parser.add_argument("--tracker", action="append", metavar="NAME=PATH",
                    help="Tracker to compare; repeat this option for each tracker")
args = parser.parse_args()

GROUND_TRUTH_PATH = args.reference

DEFAULT_TRACKERS = {
    "Recovered": Path("outputs/tracking_recovered.json"),
    "Kalman v2": Path("outputs/tracking_kalman.json"),
    "Kalman v3": Path("outputs/tracking_v3.json"),
    "v4 Pose": Path("outputs/tracking_v4_pose.json"),
    "v4 Hybrid": Path("outputs/tracking_v4_hybrid.json"),
}

if args.tracker:
    TRACKERS = {}
    for spec in args.tracker:
        name, separator, path = spec.partition("=")
        if not separator or not name.strip() or not path.strip():
            parser.error("each --tracker must use NAME=PATH")
        if name in TRACKERS:
            parser.error(f"duplicate tracker name: {name}")
        TRACKERS[name] = Path(path)
else:
    TRACKERS = DEFAULT_TRACKERS

OUTPUT_DIR = args.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "tracker_errors.csv"
JSON_PATH = OUTPUT_DIR / "evaluation_summary.json"

# A position must be at most this far from the reference
# to count as within our exploratory accuracy tolerance.
TOLERANCE_METRES = 0.50


# ============================================================
# 2. LOAD FILES
# ============================================================

def load_json(path):
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


references = load_json(GROUND_TRUTH_PATH)["annotations"]

tracker_data = {}

for name, path in TRACKERS.items():
    records = load_json(path)

    tracker_data[name] = {
        round(float(record["time"]), 3): record["players"]
        for record in records
    }

print(f"Loaded {len(references)} manually annotated frames")
print(f"Trackers: {', '.join(TRACKERS)}")


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def euclidean_distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def mean(values):
    if not values:
        return None

    return sum(values) / len(values)


def median(values):
    if not values:
        return None

    sorted_values = sorted(values)
    n = len(sorted_values)
    middle = n // 2

    if n % 2 == 1:
        return sorted_values[middle]

    return (
        sorted_values[middle - 1] +
        sorted_values[middle]
    ) / 2


def format_number(value):
    if value is None:
        return "N/A"

    return f"{value:.3f}"


# ============================================================
# 4. COMPARE EACH MANUAL ANNOTATION
# ============================================================

rows = []

for reference in references:

    timestamp = round(float(reference["time"]), 3)

    for side, manual in reference["players"].items():

        true_position = manual.get("court_m")
        method = manual.get("method", "unknown")

        if true_position is None:
            continue

        category = (
            "airborne_estimate"
            if method == "estimated_ground_position"
            else "visible_contact"
        )

        for tracker_name, time_index in tracker_data.items():

            players = time_index.get(timestamp, {})
            prediction = players.get(side)

            if prediction is None:
                predicted_position = None
                error = None
                source = "missing"

            else:
                predicted_position = prediction.get("court_m")
                source = prediction.get(
                    "position_source",
                    "unknown"
                )

                if predicted_position is None:
                    error = None
                else:
                    error = euclidean_distance(
                        true_position,
                        predicted_position
                    )

            rows.append({
                "time": timestamp,
                "player": side,
                "reference_type": category,
                "reference_method": method,
                "tracker": tracker_name,
                "position_source": source,
                "reference_x": true_position[0],
                "reference_y": true_position[1],
                "predicted_x": (
                    predicted_position[0]
                    if predicted_position is not None
                    else None
                ),
                "predicted_y": (
                    predicted_position[1]
                    if predicted_position is not None
                    else None
                ),
                "error_m": error,
                "within_50cm": (
                    error <= TOLERANCE_METRES
                    if error is not None
                    else False
                ),
            })


# ============================================================
# 5. GENERATE SUMMARY STATISTICS
# ============================================================

def summarize(selected_rows):

    valid_errors = [
        row["error_m"]
        for row in selected_rows
        if row["error_m"] is not None
    ]

    within_tolerance = sum(
        row["within_50cm"]
        for row in selected_rows
    )

    count = len(selected_rows)
    coverage = len(valid_errors) / count if count else 0

    return {
        "reference_count": count,
        "predictions_available": len(valid_errors),
        "coverage": coverage,
        "mean_error_m": mean(valid_errors),
        "median_error_m": median(valid_errors),
        "max_error_m": (
            max(valid_errors) if valid_errors else None
        ),
        "within_50cm_rate": (
            within_tolerance / count if count else None
        ),
    }


summary = {
    "tolerance_m": TOLERANCE_METRES,
    "reference_frames": len(references),
    "trackers": {}
}

for name in TRACKERS:

    tracker_rows = [
        row for row in rows
        if row["tracker"] == name
    ]

    visible_rows = [
        row for row in tracker_rows
        if row["reference_type"] == "visible_contact"
    ]

    airborne_rows = [
        row for row in tracker_rows
        if row["reference_type"] == "airborne_estimate"
    ]

    summary["trackers"][name] = {
        "all": summarize(tracker_rows),
        "visible_contacts": summarize(visible_rows),
        "airborne_estimates": summarize(airborne_rows),
        "near_visible": summarize([
            row for row in visible_rows
            if row["player"] == "Near"
        ]),
        "far_visible": summarize([
            row for row in visible_rows
            if row["player"] == "Far"
        ]),
    }


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print("\n========== TRACKER EVALUATION ==========")

for name, groups in summary["trackers"].items():

    print(f"\n{name}")
    print("-" * 45)

    for group_name in (
        "visible_contacts",
        "airborne_estimates",
        "near_visible",
        "far_visible",
    ):

        stats = groups[group_name]

        print(f"\n  {group_name}")
        print(
            "    Mean error: "
            f"{format_number(stats['mean_error_m'])} m"
        )
        print(
            "    Median error: "
            f"{format_number(stats['median_error_m'])} m"
        )
        print(
            "    Coverage: "
            f"{stats['predictions_available']}/"
            f"{stats['reference_count']}"
        )
        print(
            "    Within 50 cm: "
            f"{stats['within_50cm_rate']:.1%}"
            if stats["within_50cm_rate"] is not None
            else "    Within 50 cm: N/A"
        )


print("\n========== FRAME-BY-FRAME ERRORS ==========")

for timestamp in sorted({row["time"] for row in rows}):

    print(f"\n{timestamp:.2f}s")

    for side in ("Near", "Far"):

        matching = [
            row for row in rows
            if row["time"] == timestamp
            and row["player"] == side
        ]

        if not matching:
            continue

        reference_type = matching[0]["reference_type"]

        print(f"  {side} [{reference_type}]")

        for row in matching:
            print(
                f"    {row['tracker']}: "
                f"{format_number(row['error_m'])} m "
                f"({row['position_source']})"
            )


# ============================================================
# 7. EXPORT CSV AND JSON
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "time",
            "player",
            "reference_type",
            "reference_method",
            "tracker",
            "position_source",
            "reference_x",
            "reference_y",
            "predicted_x",
            "predicted_y",
            "error_m",
            "within_50cm",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)


print("\n========== SAVED ==========")
print(CSV_PATH)
print(JSON_PATH)
print("Evaluation complete.")
