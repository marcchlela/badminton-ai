
import json
import numpy as np
from pathlib import Path


# ============================================================
# 1. INPUTS
# ============================================================

TRACKING_PATH = Path("outputs/tracking_sample.json")
DIAGNOSTICS_PATH = Path(
    "outputs/debug_detections/diagnostics.json"
)

OUTPUT_PATH = Path("outputs/tracking_recovered.json")

# Temporary experiment: bridge gaps up to 0.8 seconds
MAX_INTERPOLATION_GAP = 0.8


# ============================================================
# 2. LOAD DATA
# ============================================================

with open(TRACKING_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)

with open(DIAGNOSTICS_PATH, "r", encoding="utf-8") as f:
    diagnostics = json.load(f)

samples.sort(key=lambda item: item["time"])

for sample in samples:
    for player in sample["players"].values():
        player["position_source"] = "measured"


# ============================================================
# 3. RECOVER LOW-CONFIDENCE DETECTION
# ============================================================

recovered_low_conf = 0

for diagnostic in diagnostics:
    timestamp = diagnostic["time"]

    matching = next(
        (
            sample for sample in samples
            if abs(sample["time"] - timestamp) < 0.001
        ),
        None
    )

    if matching is None:
        continue

    # Find person detections inside the correct court half
    for detection in diagnostic["detections"]:
        x, y = detection["court"]

        if not detection["on_court"]:
            continue

        side = "Near" if y < 6.70 else "Far"

        # Never replace an existing valid measurement
        if side in matching["players"]:
            continue

        # Only recover a reasonably confident detection
        if detection["confidence"] < 0.30:
            continue

        matching["players"][side] = {
            "side": side,
            "confidence": detection["confidence"],
            "pixel": detection["foot"],
            "court_m": detection["court"],
            "position_source": "low_confidence"
        }

        recovered_low_conf += 1

        print(
            f"Recovered {side} at {timestamp:.2f}s "
            f"(confidence {detection['confidence']:.2f})"
        )


# ============================================================
# 4. INTERPOLATE SHORT GAPS
# ============================================================

interpolated_count = 0

for side in ("Near", "Far"):

    observed_indices = [
        i for i, sample in enumerate(samples)
        if side in sample["players"]
    ]

    for start_idx, end_idx in zip(
        observed_indices[:-1],
        observed_indices[1:]
    ):

        # No missing samples between these observations
        if end_idx == start_idx + 1:
            continue

        start = samples[start_idx]
        end = samples[end_idx]

        t0 = float(start["time"])
        t1 = float(end["time"])

        if t1 - t0 > MAX_INTERPOLATION_GAP + 1e-6:
            print(
                f"Skipped long {side} gap: "
                f"{t0:.2f}s to {t1:.2f}s"
            )
            continue

        p0 = np.array(
            start["players"][side]["court_m"],
            dtype=float
        )

        p1 = np.array(
            end["players"][side]["court_m"],
            dtype=float
        )

        for i in range(start_idx + 1, end_idx):
            sample = samples[i]
            t = float(sample["time"])

            alpha = (t - t0) / (t1 - t0)

            estimated = (
                (1 - alpha) * p0 + alpha * p1
            )

            sample["players"][side] = {
                "side": side,
                "confidence": None,
                "pixel": None,
                "court_m": estimated.tolist(),
                "position_source": "interpolated",
                "estimated": True
            }

            interpolated_count += 1

        print(
            f"Interpolated {side} gap: "
            f"{t0:.2f}s to {t1:.2f}s"
        )


# ============================================================
# 5. SAVE RECOVERED DATA
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2)

print("\nRecovery complete")
print(f"Low-confidence detections recovered: {recovered_low_conf}")
print(f"Interpolated positions: {interpolated_count}")
print(f"Saved: {OUTPUT_PATH}")
