
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def create_variant(source_name, output_name, replacements):
    source = ROOT / source_name
    destination = ROOT / output_name

    if not source.is_file():
        raise FileNotFoundError(source)

    if destination.exists():
        raise FileExistsError(
            f"{destination.name} already exists. "
            "Not overwriting it."
        )

    content = source.read_text(encoding="utf-8")

    for old, new in replacements:
        count = content.count(old)

        if count != 1:
            raise RuntimeError(
                f"{source_name}: expected one occurrence, "
                f"found {count} for {old!r}"
            )

        content = content.replace(old, new, 1)

    destination.write_text(content, encoding="utf-8")
    print("Created:", destination.name)




def main():
    print("========== COACH DEMO SETUP ==========")

    create_variant(
        "build_rally_timeline.py",
        "build_coach_timeline.py",
        [
            (
                "outputs/holdout_36_40/tracking_v4_hybrid.json",
                "outputs/demo_36_46/tracking_v4_hybrid.json",
            ),
            (
                "integration_test_36_40/shuttle_positions.csv",
                "coach_demo_36_46/shuttle_positions.csv",
            ),
            (
                "integration_test_36_40/metadata.json",
                "coach_demo_36_46/metadata.json",
            ),
            (
                "outputs/unified/rally_36_40",
                "outputs/unified/coach_demo_36_46",
            ),
            (
                "END_FRAME = 1000",
                "END_FRAME = 1150",
            ),
        ],
    )

    create_variant(
        "render_unified_replay.py",
        "render_coach_replay.py",
        [
            (
                "outputs/unified/rally_36_40",
                "outputs/unified/coach_demo_36_46",
            ),
            (
                "integration_test_36_40/rally.mp4",
                "coach_demo_36_46/rally.mp4",
            ),
            (
                'RALLY_DIR / "unified_replay_v3.mp4"',
                'RALLY_DIR / "coach_demo_36_46.mp4"',
            ),
        ],
    )

    print("\nOriginal scripts preserved.")
    print("Demo scripts ready!")


if __name__ == "__main__":
    main()
