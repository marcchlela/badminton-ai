
import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_PATH = Path("outputs/tracking_kalman.json")
OUTPUT_PATH = Path("outputs/kalman_tactical_replay.mp4")

COURT_W = 6.10
COURT_L = 13.40

PPM = 75
MARGIN = 45
HEADER_H = 85
FOOTER_H = 75

OUTPUT_FPS = 25
DURATION = 3.0

# Filtering configuration
GAUSSIAN_SIGMA = 1.0

# Only interpolate across short gaps
MAX_GAP_SECONDS = 0.31

# Briefly fade out a player after tracking is lost
FADE_SECONDS = 0.25


# ============================================================
# 2. COLORS
# ============================================================

def bgr(color):
    value = bytes.fromhex(color.lstrip("#"))
    return tuple(reversed(value))


BG = bgr("#0C1422")
COURT = bgr("#176B61")
LINES = bgr("#E3F5F0")
NET = bgr("#E85D8E")
TEXT = bgr("#F3F7FB")
MUTED = bgr("#91A1B6")

NEAR_COLOR = bgr("#68CFF3")
FAR_COLOR = bgr("#F6BC67")


# ============================================================
# 3. LOAD TRACKING DATA
# ============================================================

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)

samples.sort(key=lambda s: s["time"])

if len(samples) < 2:
    raise SystemExit("Not enough tracking samples.")

START_TIME = samples[0]["time"]

print(f"Loaded {len(samples)} tracking samples")


# ============================================================
# 4. FILTER TRACKING POSITIONS
# ============================================================

def smooth_segment(points):
    """
    Apply median filtering followed by Gaussian smoothing.

    Input:
        N x 2 array of player court coordinates

    Output:
        Smoothed N x 2 array
    """

    points = np.asarray(points, dtype=float)

    if len(points) < 3:
        return points.copy()

    # Step A: median filter to reduce isolated spikes
    median = points.copy()

    for i in range(1, len(points) - 1):
        window = points[i - 1:i + 2]
        median[i] = np.median(window, axis=0)

    # Step B: Gaussian smoothing
    smoothed = median.copy()

    for i in range(len(median)):
        start = max(0, i - 2)
        end = min(len(median), i + 3)

        indices = np.arange(start, end)

        weights = np.exp(
            -0.5 * ((indices - i) / GAUSSIAN_SIGMA) ** 2
        )

        weights /= weights.sum()

        smoothed[i] = np.sum(
            median[start:end] * weights[:, None],
            axis=0
        )

    return smoothed


def prepare_player(side):
    """
    Build independent continuous tracking segments.

    Long detection gaps split the trajectory so we
    never draw an invented path across them.
    """

    observations = []

    for sample in samples:
        player = sample["players"].get(side)

        if player is None:
            continue

        observations.append((
            float(sample["time"]),
            np.array(player["court_m"], dtype=float)
        ))

    groups = []
    current = []

    for time_s, point in observations:
        if (
            current
            and time_s - current[-1][0] > MAX_GAP_SECONDS
        ):
            groups.append(current)
            current = []

        current.append((time_s, point))

    if current:
        groups.append(current)

    segments = []

    for group in groups:
        times = np.array(
            [item[0] for item in group],
            dtype=float
        )

        positions = np.array(
            [item[1] for item in group],
            dtype=float
        )

        filtered = smooth_segment(positions)

        segments.append({
            "times": times,
            "positions": filtered
        })

    return segments


TRACKS = {
    "Near": prepare_player("Near"),
    "Far": prepare_player("Far")
}

for side, segments in TRACKS.items():
    print(f"{side}: {len(segments)} trajectory segment(s)")


# ============================================================
# 5. INTERPOLATE FILTERED POSITIONS
# ============================================================

def get_position(side, time_s):
    """
    Return (position, opacity, status).

    opacity = 1.0 -> tracked / interpolated
    opacity < 1.0 -> fading after last observation

    Long gaps remain unobserved.
    """

    segments = TRACKS[side]

    for segment in segments:
        times = segment["times"]
        positions = segment["positions"]

        first = times[0]
        last = times[-1]

        if first <= time_s <= last:

            x = np.interp(
                time_s,
                times,
                positions[:, 0]
            )

            y = np.interp(
                time_s,
                times,
                positions[:, 1]
            )

            return np.array([x, y]), 1.0, "FILTERED"

        # After a segment ends, briefly show the last
        # known location as a fading marker.
        if last < time_s < last + FADE_SECONDS:
            # Do not fade through the start of a new segment.
            if any(
                first < other["times"][0] <= time_s
                for other in segments
            ):
                continue

            opacity = 1.0 - (
                (time_s - last) / FADE_SECONDS
            )

            return positions[-1], opacity, "LOST"

    return None, 0.0, "MISSING"


# ============================================================
# 6. CANVAS / COORDINATES
# ============================================================

W = round(COURT_W * PPM) + 2 * MARGIN
COURT_PIXEL_H = round(COURT_L * PPM)

H = (
    HEADER_H
    + COURT_PIXEL_H
    + 2 * MARGIN
    + FOOTER_H
)


def to_pixel(x, y):
    return (
        round(MARGIN + float(x) * PPM),
        round(
            HEADER_H
            + MARGIN
            + (COURT_L - float(y)) * PPM
        )
    )


def line(frame, a, b, color=LINES, thickness=2):
    cv2.line(
        frame,
        to_pixel(*a),
        to_pixel(*b),
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 7. DRAW BADMINTON COURT
# ============================================================

def make_court():
    frame = np.full((H, W, 3), BG, dtype=np.uint8)

    cv2.rectangle(
        frame,
        to_pixel(0, COURT_L),
        to_pixel(COURT_W, 0),
        COURT,
        -1
    )

    # Doubles outer boundaries
    line(frame, (0, 0), (COURT_W, 0))
    line(frame, (0, COURT_L), (COURT_W, COURT_L))
    line(frame, (0, 0), (0, COURT_L))
    line(frame, (COURT_W, 0), (COURT_W, COURT_L))

    # Singles sidelines
    for x in (0.46, COURT_W - 0.46):
        line(frame, (x, 0), (x, COURT_L))

    # Short service lines
    for y in (4.72, 8.68):
        line(frame, (0, y), (COURT_W, y))

    # Doubles long service lines
    for y in (0.76, COURT_L - 0.76):
        line(frame, (0, y), (COURT_W, y))

    # Center service lines
    line(frame, (COURT_W / 2, 0),
         (COURT_W / 2, 4.72))

    line(frame, (COURT_W / 2, 8.68),
         (COURT_W / 2, COURT_L))

    # Pink net with dark backing
    line(frame, (0, 6.70),
         (COURT_W, 6.70), BG, 7)

    line(frame, (0, 6.70),
         (COURT_W, 6.70), NET, 5)

    cv2.putText(
        frame,
        "BADMINTON AI / SMOOTHED REPLAY",
        (MARGIN, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        TEXT,
        1,
        cv2.LINE_AA
    )

    return frame


BASE_FRAME = make_court()


# ============================================================
# 8. DRAW PLAYER AND MOVEMENT HISTORY
# ============================================================

def blend_color(color, opacity):
    return tuple(
        int(COURT[i] * (1 - opacity) + color[i] * opacity)
        for i in range(3)
    )


def draw_player(frame, position, color, name, opacity):
    if position is None:
        return

    px, py = to_pixel(*position)

    marker_color = blend_color(color, opacity)

    cv2.circle(
        frame,
        (px, py),
        12,
        BG,
        -1,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (px, py),
        10,
        marker_color,
        -1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        name,
        (px + 16, py - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT if opacity > 0.5 else MUTED,
        1,
        cv2.LINE_AA
    )


def draw_trail(frame, side, time_s, color):
    """
    Draw filtered movement history without
    connecting across long detection gaps.
    """

    for segment in TRACKS[side]:
        times = segment["times"]
        positions = segment["positions"]

        visible = np.where(times <= time_s)[0]

        if len(visible) < 2:
            continue

        points = [
            to_pixel(*positions[i])
            for i in visible
        ]

        for a, b in zip(points[:-1], points[1:]):
            cv2.line(
                frame,
                a,
                b,
                color,
                2,
                cv2.LINE_AA
            )


# ============================================================
# 9. RENDER VIDEO
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    OUTPUT_FPS,
    (W, H)
)

if not writer.isOpened():
    raise SystemExit("Failed to open video writer.")

total_frames = round(DURATION * OUTPUT_FPS)

for i in range(total_frames):

    elapsed = i / OUTPUT_FPS
    time_s = START_TIME + elapsed

    frame = BASE_FRAME.copy()

    near, near_opacity, near_status = get_position(
        "Near", time_s
    )

    far, far_opacity, far_status = get_position(
        "Far", time_s
    )

    draw_trail(frame, "Near", time_s, NEAR_COLOR)
    draw_trail(frame, "Far", time_s, FAR_COLOR)

    draw_player(
        frame, near, NEAR_COLOR, "NEAR", near_opacity
    )

    draw_player(
        frame, far, FAR_COLOR, "FAR", far_opacity
    )

    cv2.putText(
        frame,
        f"{time_s:.2f}s",
        (MARGIN, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT,
        2,
        cv2.LINE_AA
    )

    footer_y = (
        HEADER_H + COURT_PIXEL_H + MARGIN + 35
    )

    cv2.putText(
        frame,
        "MEDIAN + GAUSSIAN TEMPORAL FILTER",
        (MARGIN, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        MUTED,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"NEAR: {near_status}    FAR: {far_status}",
        (MARGIN, footer_y + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        TEXT,
        1,
        cv2.LINE_AA
    )

    writer.write(frame)

writer.release()

print(f"Saved {OUTPUT_PATH}")
print(f"Rendered {total_frames} frames at {OUTPUT_FPS} FPS")
