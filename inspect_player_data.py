
import csv
import json
from pathlib import Path

ROOT = Path("outputs")

KEYWORDS = (
    "player",
    "track",
    "pose",
    "holdout",
    "hybrid",
)

MAX_FILES = 25
MAX_SIZE_MB = 20


def inspect_csv(path):
    with path.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)

        print("Columns:", reader.fieldnames)

        for i, row in enumerate(reader):
            if i >= 2:
                break
            print(f"Example {i + 1}:", row)


def inspect_json(path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        print("Top-level keys:", list(data.keys()))

        for key, value in list(data.items())[:5]:
            if isinstance(value, list):
                print(f"  {key}: list, {len(value)} items")
                if value:
                    print("  First item:", str(value[0])[:350])

            elif isinstance(value, dict):
                print(
                    f"  {key}: dict, keys =",
                    list(value.keys())[:15]
                )

            else:
                print(f"  {key}:", str(value)[:150])

    elif isinstance(data, list):
        print("List length:", len(data))
        if data:
            print("First item:", str(data[0])[:500])


def main():
    if not ROOT.exists():
        raise SystemExit("outputs/ directory not found")

    print("========== PLAYER DATA INVENTORY ==========")

    matches = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in (".csv", ".json"):
            continue

        text = str(path).lower()

        if not any(word in text for word in KEYWORDS):
            continue

        # Exclude the shuttle experiments.
        if "shuttle" in path.parts:
            continue

        size_mb = path.stat().st_size / (1024 ** 2)

        if size_mb > MAX_SIZE_MB:
            continue

        matches.append(path)

    matches.sort(key=lambda p: str(p).lower())

    print("Matching data files:", len(matches))

    for path in matches[:MAX_FILES]:
        print("\n" + "=" * 55)
        print("FILE:", path)

        try:
            if path.suffix.lower() == ".csv":
                inspect_csv(path)
            else:
                inspect_json(path)

        except Exception as exc:
            print("Could not inspect:", exc)

    if len(matches) > MAX_FILES:
        print(
            f"\n{len(matches) - MAX_FILES} "
            "additional files not displayed."
        )

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
