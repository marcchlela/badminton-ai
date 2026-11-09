
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TRACKNET_ROOT = PROJECT_ROOT.parent / "TrackNetV3"

TRACKNET_PYTHON = (
    TRACKNET_ROOT / ".venv" / "Scripts" / "python.exe"
)

TRACKNET_SCRIPT = TRACKNET_ROOT / "predict.py"

TRACKNET_CHECKPOINT = (
    TRACKNET_ROOT / "ckpts" / "TrackNet_best.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "shuttle" / "tracknet"
)


# ============================================================
# VIDEO EXTRACTION
# ============================================================

def extract_clip(source, output, start_sec, end_sec):
    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS")

    first_frame = round(start_sec * fps)
    end_frame = round(end_sec * fps)

    if (
        first_frame < 0
        or end_frame <= first_frame
        or end_frame > total
    ):
        cap.release()
        raise ValueError(
            "Invalid rally interval for this video"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create rally video")

    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)

    written = 0

    try:
        for frame_id in range(first_frame, end_frame):
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Cannot read source frame {frame_id}"
                )

            writer.write(frame)
            written += 1

    finally:
        writer.release()
        cap.release()

    check = cv2.VideoCapture(str(output))
    verified = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()

    if written != end_frame - first_frame:
        raise RuntimeError("Incomplete rally extraction")

    if verified != written:
        raise RuntimeError("Output video frame count mismatch")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "first_frame": first_frame,
        "end_frame_exclusive": end_frame,
        "frame_count": written,
    }


# ============================================================
# RUN PRETRAINED TRACKNET
# ============================================================

def run_tracknet(video_path, prediction_dir):
    prediction_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(TRACKNET_PYTHON),
        str(TRACKNET_SCRIPT),
        "--video_file",
        video_path.as_posix(),
        "--tracknet_file",
        TRACKNET_CHECKPOINT.as_posix(),
        "--save_dir",
        prediction_dir.as_posix(),
        "--batch_size",
        "1",
        "--eval_mode",
        "nonoverlap",
    ]

    print("\nStep 2/3: Running pretrained TrackNetV3")
    print("Using TrackNet's separate Python environment.")
    print("Model: TrackNet only, no inpainting.\n")

    subprocess.run(
        command,
        cwd=str(TRACKNET_ROOT),
        check=True,
    )


# ============================================================
# NORMALIZE TRACKING DATA
# ============================================================

def convert_predictions(
    raw_csv,
    output_csv,
    video_info,
):
    expected = video_info["frame_count"]
    first = video_info["first_frame"]
    fps = video_info["fps"]

    predictions = {}

    with raw_csv.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        reader = csv.DictReader(f)

        required = {"Frame", "Visibility", "X", "Y"}

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Unexpected TrackNet columns: {reader.fieldnames}"
            )

        for row in reader:
            local_frame = int(row["Frame"])

            if not 0 <= local_frame < expected:
                raise ValueError(
                    f"Unexpected local frame: {local_frame}"
                )

            if local_frame in predictions:
                raise ValueError(
                    f"Duplicate local frame: {local_frame}"
                )

            visible = int(float(row["Visibility"])) == 1

            x = float(row["X"]) if visible else None
            y = float(row["Y"]) if visible else None

            predictions[local_frame] = {
                "visible": visible,
                "x": x,
                "y": y,
            }

    if len(predictions) != expected:
        raise RuntimeError(
            f"Expected {expected} prediction rows, "
            f"got {len(predictions)}"
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    visible_count = 0

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "time",
                "local_frame",
                "visible",
                "x",
                "y",
            ],
        )

        writer.writeheader()

        for local_frame in range(expected):
            result = predictions[local_frame]

            original_frame = first + local_frame
            original_time = original_frame / fps

            if result["visible"]:
                visible_count += 1

            writer.writerow({
                "frame": original_frame,
                "time": round(original_time, 4),
                "local_frame": local_frame,
                "visible": int(result["visible"]),
                "x": (
                    round(result["x"], 2)
                    if result["visible"] else ""
                ),
                "y": (
                    round(result["y"], 2)
                    if result["visible"] else ""
                ),
            })

    return visible_count


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run pretrained TrackNetV3 from the "
            "main badminton-ai project."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--start",
        type=float,
        required=True,
        help="Rally start time in seconds",
    )

    parser.add_argument(
        "--end",
        type=float,
        required=True,
        help="Rally end time in seconds",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Output folder name",
    )

    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        raise ValueError(
            "Name must contain only letters, numbers, "
            "underscores, or hyphens."
        )

    source = Path(args.video)

    if not source.is_absolute():
        source = PROJECT_ROOT / source

    source = source.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    required_files = [
        TRACKNET_PYTHON,
        TRACKNET_SCRIPT,
        TRACKNET_CHECKPOINT,
    ]

    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(
                f"TrackNet dependency missing: {path}"
            )

    if args.start < 0 or args.end <= args.start:
        raise ValueError("Invalid start/end times")

    output_dir = OUTPUT_ROOT / args.name

    clip_path = output_dir / "rally.mp4"
    prediction_dir = output_dir / "raw_predictions"
    output_csv = output_dir / "shuttle_positions.csv"
    metadata_path = output_dir / "metadata.json"

    # Avoid accidentally mixing old prediction data.
    if (
        output_csv.exists()
        or metadata_path.exists()
        or prediction_dir.exists()
        or clip_path.exists()
    ):
        raise FileExistsError(
            f"Output folder already contains results: {output_dir}\n"
            "Choose a new --name to preserve previous runs."
        )

    print("========== BADMINTON AI / TRACKNET ==========")
    print("Source:", source)
    print("Interval:", args.start, "to", args.end)
    print("Output:", output_dir)

    print("\nStep 1/3: Extracting rally")

    video_info = extract_clip(
        source,
        clip_path,
        args.start,
        args.end,
    )

    print(
        f"Clip: {video_info['frame_count']} frames, "
        f"{video_info['width']}x{video_info['height']}"
    )

    run_tracknet(
        clip_path,
        prediction_dir,
    )

    raw_csv = prediction_dir / "rally_ball.csv"

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"TrackNet did not create {raw_csv}"
        )

    print("\nStep 3/3: Normalizing coordinates")

    visible_count = convert_predictions(
        raw_csv,
        output_csv,
        video_info,
    )

    metadata = {
        "source_video": str(source),
        "rally_start_sec": args.start,
        "rally_end_sec": args.end,
        "model": "TrackNetV3 TrackNet-only",
        "inference_mode": "nonoverlap",
        "coordinates": "original-video pixel coordinates",
        "frame_mapping": "original source frame numbers",
        **video_info,
        "predicted_visible_frames": visible_count,
        "predicted_missing_frames": (
            video_info["frame_count"] - visible_count
        ),
    }

    with metadata_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n========== INTEGRATION COMPLETE ==========")
    print("Frames:", video_info["frame_count"])
    print("Predicted visible:", visible_count)
    print(
        "Predicted missing:",
        video_info["frame_count"] - visible_count
    )

    print("\nSaved:")
    print(output_csv)
    print(metadata_path)
    print(clip_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
