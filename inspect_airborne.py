
import json
from pathlib import Path

BASE = Path("outputs/holdout_36_40")

def load(filename):
    with open(BASE / filename, "r", encoding="utf-8") as f:
        return json.load(f)

tracks = load("tracking_v3.json")
labels = load("manual_evaluation.json")["annotations"]
diagnostics = load("fusion_diagnostics.json")

for timestamp in (37.2, 39.9):
    print(f"\n========== {timestamp:.1f}s ==========")

    track = next(
        r["players"]["Near"]
        for r in tracks
        if abs(r["time"] - timestamp) < 0.001
    )

    reference = next(
        r["players"]["Near"]
        for r in labels
        if abs(r["time"] - timestamp) < 0.001
    )

    diagnostic = next(
        r for r in diagnostics
        if r["side"] == "Near"
        and abs(r["time"] - timestamp) < 0.001
    )

    print("MANUAL LABEL:")
    print("  Method:", reference["method"])

    print("KALMAN V3:")
    print("  Ground updated:", track["ground_updated"])
    print("  Source:", track["position_source"])
    print("  Position:", track["court_m"])

    print("POSE FUSION:")
    print("  Accepted:", diagnostic["accepted"])
    print("  Reason:", diagnostic["reason"])
    print("  Pose position:", diagnostic["pose_court_m"])
    print("  Pose IoU:", diagnostic.get("pose_iou"))
