
import json
from pathlib import Path

BASE = Path("outputs")

TRACKING = BASE / "tracking_v3.json"
LABELS = BASE / "manual_evaluation.json"

LOOKBACK_SECONDS = 0.30

MIN_TOP_RISE = 20
MIN_HEIGHT_INCREASE = 15


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


tracks = load(TRACKING)
labels = load(LABELS)["annotations"]

track_lookup = {
    round(float(record["time"]), 2): record["players"]
    for record in tracks
}

timestamps = sorted(track_lookup)


def closest_time(target):
    return min(timestamps, key=lambda t: abs(t - target))


def get_box(time, side):
    players = track_lookup.get(time, {})
    player = players.get(side)

    return player.get("bbox") if player else None


def detect_warning(timestamp, side):
    current_time = closest_time(timestamp)
    previous_time = closest_time(
        current_time - LOOKBACK_SECONDS
    )

    # Do not silently shorten the lookback.
    if (
        previous_time >= current_time
        or abs(
            (current_time - previous_time)
            - LOOKBACK_SECONDS
        ) > 0.051
    ):
        return None, None

    current = get_box(current_time, side)
    previous = get_box(previous_time, side)

    if current is None or previous is None:
        return None, None

    _, cy, _, ch = current
    _, py, _, ph = previous

    top_rise = py - cy
    height_increase = ch - ph

    warning = (
        top_rise >= MIN_TOP_RISE
        and height_increase >= MIN_HEIGHT_INCREASE
    )

    features = {
        "top_rise_px": top_rise,
        "height_increase_px": height_increase,
        "previous_time": previous_time
    }

    return warning, features


counts = {
    "TP": 0,
    "FP": 0,
    "TN": 0,
    "FN": 0
}

print("\nAIRBORNE WARNING EXPERIMENT")
print("=" * 75)

for record in labels:
    timestamp = round(float(record["time"]), 2)

    for side in ("Near", "Far"):
        reference = record["players"].get(side)

        if not reference:
            continue

        method = reference.get("method")

        if method not in (
            "visible_foot_contact",
            "estimated_ground_position"
        ):
            continue

        actual_airborne = (
            method == "estimated_ground_position"
        )

        warning, features = detect_warning(
            timestamp, side
        )

        if warning is None:
            print(
                f"{timestamp:5.2f}s {side:4s} "
                "| INSUFFICIENT DATA"
            )
            continue

        if warning and actual_airborne:
            outcome = "TP"
        elif warning and not actual_airborne:
            outcome = "FP"
        elif not warning and actual_airborne:
            outcome = "FN"
        else:
            outcome = "TN"

        counts[outcome] += 1

        print(
            f"{timestamp:5.2f}s {side:4s} | "
            f"top rise={features['top_rise_px']:>4}px | "
            f"height change="
            f"{features['height_increase_px']:>4}px | "
            f"warning={str(warning):5s} | "
            f"label={'AIR' if actual_airborne else 'GROUND':6s} | "
            f"{outcome}"
        )


print("\n========== SUMMARY ==========")

for name, value in counts.items():
    print(f"{name}: {value}")

tp = counts["TP"]
fp = counts["FP"]
fn = counts["FN"]
tn = counts["TN"]

precision = tp / (tp + fp) if tp + fp else None
recall = tp / (tp + fn) if tp + fn else None

print(
    "Warning precision:",
    f"{precision:.1%}" if precision is not None else "N/A"
)

print(
    "Airborne recall:",
    f"{recall:.1%}" if recall is not None else "N/A"
)

print(
    "Grounded false-warning rate:",
    f"{fp / (fp + tn):.1%}"
    if fp + tn else "N/A"
)

print("\nThis is a diagnostic test, not a validated jump classifier.")
print("No warning does NOT prove the player is grounded.")
