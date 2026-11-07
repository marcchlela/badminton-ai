
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

SOURCE_PATH = Path("data/raw/clip.mp4")

TACTICAL_PATH = Path(
    "outputs/smooth_tactical_replay.mp4"
)

OUTPUT_PATH = Path(
    "outputs/synchronized_review.mp4"
)

START_SECONDS = 20.0

OUTPUT_FPS = 25
OUTPUT_WIDTH = 1400
OUTPUT_HEIGHT = 800


# ============================================================
# 2. COLORS
# ============================================================

def bgr(hex_color):
    rgb = bytes.fromhex(hex_color.lstrip("#"))
    return tuple(reversed(rgb))


BG = bgr("#0C1422")
PANEL = bgr("#142235")
TEXT = bgr("#F3F7FB")
MUTED = bgr("#91A1B6")
ACCENT = bgr("#E85D8E")


# ============================================================
# 3. LOAD BOTH VIDEOS
# ============================================================

if not SOURCE_PATH.exists():
    raise SystemExit(
        f"Source video missing: {SOURCE_PATH}"
    )

if not TACTICAL_PATH.exists():
    raise SystemExit(
        f"Tactical replay missing: {TACTICAL_PATH}"
    )

source = cv2.VideoCapture(str(SOURCE_PATH))
tactical = cv2.VideoCapture(str(TACTICAL_PATH))

if not source.isOpened() or not tactical.isOpened():
    raise SystemExit("Unable to open video files.")

source_fps = source.get(cv2.CAP_PROP_FPS)
tactical_fps = tactical.get(cv2.CAP_PROP_FPS)

tactical_frames = int(
    tactical.get(cv2.CAP_PROP_FRAME_COUNT)
)

if source_fps <= 0 or tactical_fps <= 0:
    raise SystemExit("Invalid input video frame rate.")

duration = tactical_frames / tactical_fps
output_frames = round(duration * OUTPUT_FPS)

print(f"Source FPS: {source_fps:.2f}")
print(f"Tactical FPS: {tactical_fps:.2f}")
print(f"Review duration: {duration:.2f}s")
print(f"Output frames: {output_frames}")


# ============================================================
# 4. IMAGE PLACEMENT
# ============================================================

def fit_inside(image, max_w, max_h):
    """
    Resize an image to fit inside a target area,
    preserving its aspect ratio.
    """

    h, w = image.shape[:2]

    scale = min(
        max_w / w,
        max_h / h
    )

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


def place_centered(canvas, image, x, y, w, h):
    """
    Place image centered inside a rectangular area.
    """

    resized = fit_inside(image, w, h)

    rh, rw = resized.shape[:2]

    px = x + (w - rw) // 2
    py = y + (h - rh) // 2

    canvas[
        py:py + rh,
        px:px + rw
    ] = resized


# ============================================================
# 5. VIDEO WRITER
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    OUTPUT_FPS,
    (OUTPUT_WIDTH, OUTPUT_HEIGHT)
)

if not writer.isOpened():
    raise SystemExit("Could not create output video.")


# ============================================================
# 6. SYNCHRONIZE AND RENDER
# ============================================================

for i in range(output_frames):

    elapsed = i / OUTPUT_FPS
    source_time = START_SECONDS + elapsed

    # Seek to the matching source-video frame.
    source_frame_number = round(
        source_time * source_fps
    )

    source.set(
        cv2.CAP_PROP_POS_FRAMES,
        source_frame_number
    )

    ok_real, real_frame = source.read()

    # Read tactical frame for the same elapsed time.
    tactical_frame_number = min(
        round(elapsed * tactical_fps),
        tactical_frames - 1
    )

    tactical.set(
        cv2.CAP_PROP_POS_FRAMES,
        tactical_frame_number
    )

    ok_ai, ai_frame = tactical.read()

    if not ok_real or not ok_ai:
        print(
            f"Stopped: missing frame at {source_time:.2f}s"
        )
        break

    # Main canvas
    canvas = np.full(
        (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3),
        BG,
        dtype=np.uint8
    )

    # Panel backgrounds
    cv2.rectangle(
        canvas,
        (20, 100),
        (1010, 720),
        PANEL,
        -1
    )

    cv2.rectangle(
        canvas,
        (1020, 100),
        (1380, 720),
        PANEL,
        -1
    )

    # Main heading
    cv2.putText(
        canvas,
        "BADMINTON AI  /  SYNCHRONIZED REVIEW",
        (30, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        TEXT,
        2,
        cv2.LINE_AA
    )

    # Timestamp
    cv2.putText(
        canvas,
        f"MATCH TIME: {source_time:.2f}s",
        (30, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        ACCENT,
        1,
        cv2.LINE_AA
    )

    # Panel titles
    cv2.putText(
        canvas,
        "ORIGINAL MATCH",
        (35, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        TEXT,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "AI TACTICAL REPLAY",
        (1035, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT,
        1,
        cv2.LINE_AA
    )

    # Original video, landscape
    place_centered(
        canvas,
        real_frame,
        30, 150, 970, 550
    )

    # Tactical court, portrait
    place_centered(
        canvas,
        ai_frame,
        1030, 145, 340, 560
    )

    # Footer
    cv2.putText(
        canvas,
        "AI POSITIONS ARE ESTIMATES - REVIEW AGAINST VIDEO",
        (30, 758),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        MUTED,
        1,
        cv2.LINE_AA
    )

    writer.write(canvas)

    if (i + 1) % 25 == 0:
        print(
            f"Rendered {i + 1}/{output_frames} frames"
        )


# ============================================================
# 7. CLEANUP
# ============================================================

source.release()
tactical.release()
writer.release()

print(f"\nSaved: {OUTPUT_PATH}")
print("Synchronized review complete.")