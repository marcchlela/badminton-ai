
import cv2
import numpy as np
from pathlib import Path

# ============================================================
# 1. CONFIGURATION
# ============================================================

VIDEO_PATH = Path("data/raw/clip.mp4")

RECOVERED_PATH = Path(
    "outputs/recovered_tactical_replay.mp4"
)

KALMAN_PATH = Path(
    "outputs/kalman_tactical_replay.mp4"
)

OUTPUT_PATH = Path(
    "outputs/tracker_comparison.mp4"
)

START_SECONDS = 20.0
OUTPUT_FPS = 25

WIDTH = 1600
HEIGHT = 760

BG = (34, 20, 12)
PANEL = (53, 34, 20)
WHITE = (251, 247, 243)
MUTED = (183, 160, 145)
ACCENT = (142, 93, 232)


# ============================================================
# 2. OPEN VIDEOS
# ============================================================

paths = [
    VIDEO_PATH,
    RECOVERED_PATH,
    KALMAN_PATH
]

for path in paths:
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

caps = [
    cv2.VideoCapture(str(path))
    for path in paths
]

if not all(cap.isOpened() for cap in caps):
    raise SystemExit("Could not open all three videos.")

source, recovered, kalman = caps

source_fps = source.get(cv2.CAP_PROP_FPS)

frame_counts = [
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for cap in (recovered, kalman)
]

total_frames = min(frame_counts)

# Start original video at the same moment as the replays
source.set(
    cv2.CAP_PROP_POS_FRAMES,
    round(START_SECONDS * source_fps)
)


# ============================================================
# 3. IMAGE PLACEMENT
# ============================================================

def draw_panel(canvas, image, x, y, w, h):
    # Background
    cv2.rectangle(
        canvas,
        (x, y),
        (x + w, y + h),
        PANEL,
        -1
    )

    ih, iw = image.shape[:2]

    scale = min(w / iw, h / ih)

    nw = round(iw * scale)
    nh = round(ih * scale)

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )

    px = x + (w - nw) // 2
    py = y + (h - nh) // 2

    canvas[
        py:py + nh,
        px:px + nw
    ] = resized


def draw_text(canvas, text, x, y, size=0.55):
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        WHITE,
        1,
        cv2.LINE_AA
    )


# ============================================================
# 4. CREATE OUTPUT
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    OUTPUT_FPS,
    (WIDTH, HEIGHT)
)

if not writer.isOpened():
    raise SystemExit("Could not create output video.")


# ============================================================
# 5. RENDER COMPARISON
# ============================================================

rendered = 0

try:
    for i in range(total_frames):

        timestamp = START_SECONDS + i / OUTPUT_FPS

        # Select the correct original video frame
        source_frame = round(timestamp * source_fps)

        source.set(
            cv2.CAP_PROP_POS_FRAMES,
            source_frame
        )

        ok1, real = source.read()
        ok2, old = recovered.read()
        ok3, new = kalman.read()

        if not (ok1 and ok2 and ok3):
            print(f"Stopped at {timestamp:.2f}s")
            break

        canvas = np.full(
            (HEIGHT, WIDTH, 3),
            BG,
            dtype=np.uint8
        )

        # Heading
        cv2.putText(
            canvas,
            "BADMINTON AI  /  TRACKER COMPARISON",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.83,
            WHITE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            f"MATCH TIME  {timestamp:.2f}s",
            (25, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            ACCENT,
            2,
            cv2.LINE_AA
        )

        # Column labels
        draw_text(canvas, "ORIGINAL MATCH", 25, 122)
        draw_text(canvas, "RECOVERED", 845, 122)
        draw_text(canvas, "KALMAN V2", 1225, 122)

        # Three synchronized panels
        draw_panel(canvas, real, 20, 140, 800, 550)
        draw_panel(canvas, old, 840, 140, 360, 550)
        draw_panel(canvas, new, 1220, 140, 360, 550)

        draw_text(
            canvas,
            "Visual comparison only - predicted positions are not ground truth",
            25,
            730,
            0.45
        )

        writer.write(canvas)
        rendered += 1

finally:
    for cap in caps:
        cap.release()

    writer.release()

print(f"Saved: {OUTPUT_PATH}")
print(f"Rendered {rendered} comparison frames")
