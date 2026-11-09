
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent

SOURCE = (
    ROOT / "outputs/unified/rally_36_40/"
    "unified_replay_v3.mp4"
)

OUTPUT = (
    ROOT / "outputs/unified/rally_36_40/"
    "coach_demo_10s.mp4"
)

SPEED = 0.4  # 40% of original speed


def main():
    cap = cv2.VideoCapture(str(SOURCE))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or count == 0:
        cap.release()
        raise RuntimeError("Invalid input video")

    output_count = round(count / SPEED)

    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video")

    frames = []

    try:
        for _ in range(count):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Incomplete input video")
            frames.append(frame)

        for i in range(output_count):
            # Duplicate frames to slow playback without
            # changing tracking coordinates or timestamps.
            source_index = min(
                int(i * SPEED),
                count - 1
            )

            frame = frames[source_index].copy()

            cv2.rectangle(
                frame,
                (width - 245, height - 47),
                (width - 12, height - 12),
                (34, 20, 12),
                -1
            )

            cv2.putText(
                frame,
                "SLOW MOTION 0.4x",
                (width - 226, height - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (245, 245, 245),
                1,
                cv2.LINE_AA
            )

            writer.write(frame)

    finally:
        writer.release()
        cap.release()

    check = cv2.VideoCapture(str(OUTPUT))
    verified = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
    check.release()

    print("========== COACH DEMO ==========")
    print("Original frames:", count)
    print("Output frames:", verified)
    print("FPS:", fps)
    print("Duration:", round(verified / fps, 2), "seconds")
    print("Playback speed:", SPEED)
    print("Resolution:", f"{width}x{height}")
    print("Saved:", OUTPUT)

    if verified != output_count:
        raise RuntimeError("Output verification failed")

    print("Done!")


if __name__ == "__main__":
    main()
