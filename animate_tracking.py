
import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("outputs/tracking_sample.json")
OUTPUT_PATH = Path("outputs/first_tactical_replay.mp4")

COURT_W = 6.10
COURT_L = 13.40
NET_Y = 6.70

PPM = 75
MARGIN = 45
HEADER_H = 85
FOOTER_H = 75

OUTPUT_FPS = 25

# Display the whole 3-second analysis window.
# No new positions are inferred after the final sample.
DURATION = 3.0

NEAR_COLOR = "#68CFF3"
FAR_COLOR = "#F6BC67"


# ============================================================
# 2. COLORS
# ============================================================

def bgr(hex_color):
    value = hex_color.lstrip("#")
    return tuple(reversed(bytes.fromhex(value)))


BG = bgr("#0C1422")
COURT = bgr("#176B61")
LINES = bgr("#E3F5F0")
NET = bgr("#E85D8E")
TEXT = bgr("#F3F7FB")
MUTED = bgr("#91A1B6")

NEAR = bgr(NEAR_COLOR)
FAR = bgr(FAR_COLOR)


# ============================================================
# 3. READ TRACKING DATA
# ============================================================

if not DATA_PATH.exists():
    raise SystemExit(
        "Tracking data missing. Run track_video.py first."
    )

with open(DATA_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)

samples.sort(key=lambda s: s["time"])

if len(samples) < 2:
    raise SystemExit("At least two samples are required.")

START_TIME = samples[0]["time"]
LAST_TIME = samples[-1]["time"]

print(f"Loaded {len(samples)} samples")
print(f"First sample: {START_TIME:.2f}s")
print(f"Last sample: {LAST_TIME:.2f}s")


# ============================================================
# 4. CANVAS AND COORDINATES
# ============================================================

W = round(COURT_W * PPM) + 2 * MARGIN
COURT_PIXEL_H = round(COURT_L * PPM)

H = (
    HEADER_H
    + COURT_PIXEL_H
    + 2 * MARGIN
    + FOOTER_H
)


def court_pixel(x, y):
    px = round(MARGIN + x * PPM)

    py = round(
        HEADER_H
        + MARGIN
        + (COURT_L - y) * PPM
    )

    return (px, py)


def draw_line(frame, a, b, color=LINES, thickness=2):
    cv2.line(
        frame,
        court_pixel(*a),
        court_pixel(*b),
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 5. DRAW STATIC COURT
# ============================================================

def draw_court():
    frame = np.full((H, W, 3), BG, dtype=np.uint8)

    cv2.rectangle(
        frame,
        court_pixel(0, COURT_L),
        court_pixel(COURT_W, 0),
        COURT,
        -1
    )

    # Doubles outer boundaries
    draw_line(frame, (0, 0), (COURT_W, 0))
    draw_line(frame, (0, COURT_L), (COURT_W, COURT_L))
    draw_line(frame, (0, 0), (0, COURT_L))
    draw_line(frame, (COURT_W, 0), (COURT_W, COURT_L))

    # Singles sidelines
    inset = 0.46

    for x in [inset, COURT_W - inset]:
        draw_line(frame, (x, 0), (x, COURT_L))

    # Short service lines
    for y in [4.72, 8.68]:
        draw_line(frame, (0, y), (COURT_W, y))

    # Doubles long service lines
    for y in [0.76, COURT_L - 0.76]:
        draw_line(frame, (0, y), (COURT_W, y))

    # Center service lines
    draw_line(frame, (COURT_W / 2, 0),
              (COURT_W / 2, 4.72))

    draw_line(frame, (COURT_W / 2, 8.68),
              (COURT_W / 2, COURT_L))

    # Net backing and pink net
    draw_line(
        frame,
        (0, NET_Y),
        (COURT_W, NET_Y),
        BG,
        7
    )

    draw_line(
        frame,
        (0, NET_Y),
        (COURT_W, NET_Y),
        NET,
        5
    )

    cv2.putText(
        frame,
        "BADMINTON AI  /  TACTICAL REPLAY",
        (MARGIN, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        TEXT,
        1,
        cv2.LINE_AA
    )

    return frame


STATIC_COURT = draw_court()


# ============================================================
# 6. INTERPOLATE BETWEEN OBSERVED POSITIONS
# ============================================================

def player_position(time_s, side):
    """
    Linear interpolation between two actual detections.

    If either endpoint is missing, return None.
    Never extrapolate an unseen trajectory.
    """

    # After last sample, only show the last observation.
    if time_s >= LAST_TIME:
        player = samples[-1]["players"].get(side)

        if player is None:
            return None

        return np.array(player["court_m"], dtype=float)

    for i in range(len(samples) - 1):
        a = samples[i]
        b = samples[i + 1]

        if not (a["time"] <= time_s < b["time"]):
            continue

        pa = a["players"].get(side)
        pb = b["players"].get(side)

        # Missing data means no interpolated trajectory.
        if pa is None or pb is None:
            return None

        t0 = a["time"]
        t1 = b["time"]

        alpha = (time_s - t0) / (t1 - t0)

        p0 = np.array(pa["court_m"], dtype=float)
        p1 = np.array(pb["court_m"], dtype=float)

        return (1 - alpha) * p0 + alpha * p1

    return None


# ============================================================
# 7. DRAW SAMPLED MOVEMENT TRAILS
# ============================================================

def draw_trail(frame, time_s, side, color):
    """
    Connect previously sampled detections only.

    Missing frames interrupt the trail.
    """

    previous = None

    for sample in samples:
        if sample["time"] > time_s:
            break

        player = sample["players"].get(side)

        if player is None:
            previous = None
            continue

        current = court_pixel(*player["court_m"])

        if previous is not None:
            cv2.line(
                frame,
                previous,
                current,
                color,
                2,
                cv2.LINE_AA
            )

        cv2.circle(
            frame,
            current,
            3,
            color,
            -1,
            cv2.LINE_AA
        )

        previous = current


# ============================================================
# 8. DRAW CURRENT PLAYER POSITIONS
# ============================================================

def draw_player(frame, position, color, label):
    if position is None:
        return False

    x, y = position

    px, py = court_pixel(float(x), float(y))

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
        color,
        -1,
        cv2.LINE_AA
    )

    # Put labels in a consistent position
    cv2.putText(
        frame,
        label,
        (px + 15, py - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT,
        1,
        cv2.LINE_AA
    )

    return True


# ============================================================
# 9. RENDER ANIMATION
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    OUTPUT_FPS,
    (W, H)
)

if not writer.isOpened():
    raise SystemExit("Could not create video writer.")

total_frames = round(DURATION * OUTPUT_FPS)

for frame_index in range(total_frames):

    elapsed = frame_index / OUTPUT_FPS
    time_s = START_TIME + elapsed

    frame = STATIC_COURT.copy()

    near_pos = player_position(time_s, "Near")
    far_pos = player_position(time_s, "Far")

    draw_trail(frame, time_s, "Near", NEAR)
    draw_trail(frame, time_s, "Far", FAR)

    draw_player(frame, near_pos, NEAR, "NEAR")
    draw_player(frame, far_pos, FAR, "FAR")

    # Timestamp
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

    # Status information
    footer_y = HEADER_H + COURT_PIXEL_H + MARGIN + 33

    if time_s >= LAST_TIME:
        status = "LAST SAMPLE HELD - NO EXTRAPOLATION"
    else:
        status = "INTERPOLATED BETWEEN 2 FPS DETECTIONS"

    cv2.putText(
        frame,
        status,
        (MARGIN, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        MUTED,
        1,
        cv2.LINE_AA
    )

    if far_pos is None:
        cv2.putText(
            frame,
            "FAR PLAYER: MISSING DETECTION",
            (MARGIN, footer_y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            FAR,
            1,
            cv2.LINE_AA
        )

    writer.write(frame)

writer.release()

print(f"Saved {OUTPUT_PATH}")
print(f"Rendered {total_frames} frames at {OUTPUT_FPS} FPS")
print(f"Duration: {DURATION:.2f} seconds")
