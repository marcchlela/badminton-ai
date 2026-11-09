
import json
import math
from pathlib import Path

BASE = Path("outputs/holdout_36_40")

def load(path):
    with open(BASE / path, "r", encoding="utf-8") as f:
        return json.load(f)

tracking = load("tracking_v3.json")
diagnostics = load("fusion_diagnostics.json")
annotations = load("manual_evaluation.json")["annotations"]

diag_lookup = {
    (round(float(d["time"]), 2), d["side"]): d
    for d in diagnostics
}

label_lookup = {
    (round(float(a["time"]), 2), side): p["method"]
    for a in annotations
    for side, p in a["players"].items()
}

print("\nGROUND-CONTACT INVESTIGATION")
print("=" * 95)

for record in tracking:
    t = round(float(record["time"]), 2)

    if not (36.8 <= t <= 37.6 or 39.4 <= t <= 39.9):
        continue

    side = "Near"
    p = record["players"].get(side)

    if not p:
        continue

    bbox = p.get("bbox")
    d = diag_lookup.get((t, side), {})
    method = label_lookup.get((t, side), "not_labeled")

    if bbox is None:
        continue

    x, y, w, h = bbox
    pose = d.get("pose_court_m")
    court = p.get("court_m")

    disagreement = (
        math.dist(pose, court)
        if pose is not None and court is not None
        else None
    )

    distance_text = (
        f"{disagreement:.3f}"
        if disagreement is not None
        else "N/A"
    )

    print(
        f"{t:5.2f}s | "
        f"box_top={y:3d}px | "
        f"box_bottom={y+h:3d}px | "
        f"box_height={h:3d}px | "
        f"ground_updated={str(p['ground_updated']):5s} | "
        f"pose_delta={distance_text:>5s}m | "
        f"label={method}"
    )
