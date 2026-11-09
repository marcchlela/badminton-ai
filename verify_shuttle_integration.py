
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INTEGRATION = (
    ROOT / "outputs/shuttle/tracknet/"
    "integration_test_36_40"
)

NEW_CSV = INTEGRATION / "shuttle_positions.csv"
METADATA = INTEGRATION / "metadata.json"

OLD_CSV = (
    ROOT.parent / "TrackNetV3/prediction/"
    "rally_720p_36_40/rally_720p_36_40_ball.csv"
)

LABELS = ROOT / "outputs/shuttle/visibility_audit.json"

MATCH_RADIUS = 12.0


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def old_point(row):
    if int(float(row["Visibility"])) != 1:
        return None

    return float(row["X"]), float(row["Y"])


def new_point(row):
    if int(row["visible"]) != 1:
        return None

    return float(row["x"]), float(row["y"])


def distance(a, b):
    if a is None or b is None:
        return None

    return math.dist(a, b)


def main():
    print("========== INTEGRATION VERIFICATION ==========")

    new_rows = read_csv(NEW_CSV)
    old_rows = read_csv(OLD_CSV)

    with METADATA.open(encoding="utf-8") as f:
        metadata = json.load(f)

    with LABELS.open(encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    first = int(metadata["first_frame"])
    total = int(metadata["frame_count"])
    fps = float(metadata["fps"])

    new_by_frame = {
        int(row["frame"]): row
        for row in new_rows
    }

    old_by_local = {
        int(row["Frame"]): row
        for row in old_rows
    }

    print("Expected frames:", total)
    print("New prediction rows:", len(new_rows))
    print("Original prediction rows:", len(old_rows))
    print("First original frame:", first)
    print("FPS:", fps)

    # --------------------------------------------------------
    # FRAME ALIGNMENT
    # --------------------------------------------------------

    expected_frames = set(range(first, first + total))

    alignment_ok = (
        len(new_rows) == total
        and len(new_by_frame) == total
        and set(new_by_frame) == expected_frames
        and len(old_rows) == total
        and len(old_by_local) == total
        and set(old_by_local) == set(range(total))
    )

    time_errors = []

    for frame, row in new_by_frame.items():
        expected_time = frame / fps
        actual_time = float(row["time"])

        time_errors.append(
            abs(expected_time - actual_time)
        )

        expected_local = frame - first

        if int(row["local_frame"]) != expected_local:
            alignment_ok = False

    max_time_error = max(time_errors, default=0.0)

    print("\n========== FRAME ALIGNMENT ==========")
    print("Frame numbering valid:", alignment_ok)
    print(
        "Maximum timestamp difference:",
        round(max_time_error, 6),
        "seconds"
    )

    # --------------------------------------------------------
    # COMPARE ORIGINAL AND INTEGRATED INFERENCE
    # --------------------------------------------------------

    coordinate_differences = []
    visibility_disagreements = 0

    for local_frame in range(total):
        old = old_by_local.get(local_frame)
        new = new_by_frame.get(first + local_frame)

        if old is None or new is None:
            continue

        p_old = old_point(old)
        p_new = new_point(new)

        if (p_old is None) != (p_new is None):
            visibility_disagreements += 1

        if p_old is not None and p_new is not None:
            coordinate_differences.append(
                distance(p_old, p_new)
            )

    print("\n========== INFERENCE COMPARISON ==========")
    print(
        "Visibility disagreements:",
        visibility_disagreements
    )
    print(
        "Frames with two visible predictions:",
        len(coordinate_differences)
    )

    if coordinate_differences:
        print(
            "Mean difference:",
            round(
                statistics.mean(coordinate_differences), 2
            ),
            "px"
        )
        print(
            "Maximum difference:",
            round(max(coordinate_differences), 2),
            "px"
        )

    # --------------------------------------------------------
    # EVALUATE AGAINST MANUAL LABELS
    # --------------------------------------------------------

    visible_labels = [
        a for a in annotations
        if a["status"] == "visible"
        and a["pixel"] is not None
        and first <= int(a["frame"]) < first + total
    ]

    correct = 0
    predicted = 0
    errors = []

    print("\n========== MANUAL LABEL CHECK ==========")

    for annotation in visible_labels:
        frame = int(annotation["frame"])
        row = new_by_frame.get(frame)

        point = new_point(row) if row else None
        truth = tuple(map(float, annotation["pixel"]))

        if point is None:
            print(
                f"{float(annotation['time']):.2f}s | MISSING"
            )
            continue

        predicted += 1

        error = distance(point, truth)
        errors.append(error)

        hit = error <= MATCH_RADIUS

        if hit:
            correct += 1

        print(
            f"{float(annotation['time']):.2f}s | "
            f"{error:.2f}px | "
            f"{'HIT' if hit else 'WRONG'}"
        )

    print("\n========== FINAL VERIFICATION ==========")
    print("Frame alignment:", "PASS" if alignment_ok else "FAIL")
    print(
        "Correct predictions:",
        f"{correct}/{len(visible_labels)}"
    )
    print(
        "Predictions on visible frames:",
        f"{predicted}/{len(visible_labels)}"
    )
    print(
        "Mean localization error:",
        (
            f"{statistics.mean(errors):.2f} px"
            if errors else "N/A"
        )
    )

    report = {
        "frame_alignment_pass": alignment_ok,
        "max_timestamp_error_seconds": max_time_error,
        "old_new_visibility_disagreements": visibility_disagreements,
        "mean_old_new_difference_px": (
            statistics.mean(coordinate_differences)
            if coordinate_differences else None
        ),
        "correct_on_visible_labels": correct,
        "visible_reference_frames": len(visible_labels),
        "mean_manual_error_px": (
            statistics.mean(errors) if errors else None
        ),
    }

    output = INTEGRATION / "verification.json"

    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSaved:", output)
    print("Done!")


if __name__ == "__main__":
    main()
