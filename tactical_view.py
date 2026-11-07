
import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# 1. THEME / COLORS
# ============================================================

def hex_to_bgr(value):
    """Convert hexadecimal RGB color to OpenCV BGR."""
    value = value.lstrip("#")
    r, g, b = bytes.fromhex(value)
    return (b, g, r)


BG_COLOR = hex_to_bgr("#0C1422")
COURT_COLOR = hex_to_bgr("#176B61")
LINE_COLOR = hex_to_bgr("#E3F5F0")
NET_COLOR = hex_to_bgr("#E85D8E")

NEAR_COLOR = hex_to_bgr("#68CFF3")
FAR_COLOR = hex_to_bgr("#F6BC67")
TEXT_COLOR = hex_to_bgr("#F3F7FB")


# ============================================================
# 2. COURT DIMENSIONS (METRES)
# ============================================================

COURT_WIDTH = 6.10
COURT_LENGTH = 13.40

SINGLES_WIDTH = 5.18
SINGLE_INSET = (COURT_WIDTH - SINGLES_WIDTH) / 2.0

NET_Y = COURT_LENGTH / 2.0

SHORT_SERVICE_DISTANCE = 1.98
NEAR_SHORT_SERVICE_Y = NET_Y - SHORT_SERVICE_DISTANCE
FAR_SHORT_SERVICE_Y = NET_Y + SHORT_SERVICE_DISTANCE

DOUBLES_LONG_SERVICE_INSET = 0.76


# ============================================================
# 3. LOAD AUTOMATIC PLAYER DETECTIONS
# ============================================================

DETECTIONS_PATH = Path("outputs/auto_players.json")

if not DETECTIONS_PATH.exists():
    raise SystemExit(
        "Automatic player data not found. "
        "Run detect_players.py first."
    )

with open(DETECTIONS_PATH, "r", encoding="utf-8") as f:
    detection_data = json.load(f)

# Organize players by their court side
players = {
    player["name"]: player
    for player in detection_data["players"]
}

# Make sure both players were detected
if "Near" not in players or "Far" not in players:
    raise SystemExit(
        "Cannot render tactical view: "
        "one or both players are missing."
    )

# Read the court coordinates calculated by our AI pipeline
near_court = np.array(
    players["Near"]["court_m"],
    dtype=np.float32
)

far_court = np.array(
    players["Far"]["court_m"],
    dtype=np.float32
)

print("AI Near player position (m):", near_court)
print("AI Far player position (m):", far_court)


# ============================================================
# 4. CREATE 2D CANVAS
# ============================================================

PPM = 90
MARGIN = 50

img_w = round(COURT_WIDTH * PPM) + 2 * MARGIN
img_h = round(COURT_LENGTH * PPM) + 2 * MARGIN

canvas = np.zeros(
    (img_h, img_w, 3),
    dtype=np.uint8
)

canvas[:] = BG_COLOR


# ============================================================
# 5. COORDINATE HELPERS
# ============================================================

def to_img_coords(x_m, y_m):
    """
    Convert court coordinates in metres to canvas pixels.

    Court origin (0, 0) = near baseline, left corner.
    Y increases toward the far baseline.
    """

    x = round(MARGIN + x_m * PPM)

    # Flip Y for image coordinates
    y = round(
        MARGIN + (COURT_LENGTH - y_m) * PPM
    )

    return (x, y)


def line_m(p1, p2, color=LINE_COLOR, thickness=2):
    """Draw a line using court coordinates in metres."""

    cv2.line(
        canvas,
        to_img_coords(*p1),
        to_img_coords(*p2),
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 6. DRAW COURT SURFACE
# ============================================================

court_top_left = to_img_coords(
    0,
    COURT_LENGTH
)

court_bottom_right = to_img_coords(
    COURT_WIDTH,
    0
)

cv2.rectangle(
    canvas,
    court_top_left,
    court_bottom_right,
    COURT_COLOR,
    -1
)


# ============================================================
# 7. DRAW BADMINTON COURT LINES
# ============================================================

# Outer doubles boundaries
line_m((0, 0), (COURT_WIDTH, 0))
line_m((COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH))
line_m((COURT_WIDTH, COURT_LENGTH), (0, COURT_LENGTH))
line_m((0, COURT_LENGTH), (0, 0))

# Singles sidelines
line_m(
    (SINGLE_INSET, 0),
    (SINGLE_INSET, COURT_LENGTH)
)

line_m(
    (COURT_WIDTH - SINGLE_INSET, 0),
    (COURT_WIDTH - SINGLE_INSET, COURT_LENGTH)
)

# Short service lines
line_m(
    (0, NEAR_SHORT_SERVICE_Y),
    (COURT_WIDTH, NEAR_SHORT_SERVICE_Y)
)

line_m(
    (0, FAR_SHORT_SERVICE_Y),
    (COURT_WIDTH, FAR_SHORT_SERVICE_Y)
)

# Doubles long service lines
line_m(
    (0, DOUBLES_LONG_SERVICE_INSET),
    (COURT_WIDTH, DOUBLES_LONG_SERVICE_INSET)
)

line_m(
    (0, COURT_LENGTH - DOUBLES_LONG_SERVICE_INSET),
    (COURT_WIDTH, COURT_LENGTH - DOUBLES_LONG_SERVICE_INSET)
)

# Center service lines
line_m(
    (COURT_WIDTH / 2.0, 0),
    (COURT_WIDTH / 2.0, NEAR_SHORT_SERVICE_Y)
)

line_m(
    (COURT_WIDTH / 2.0, FAR_SHORT_SERVICE_Y),
    (COURT_WIDTH / 2.0, COURT_LENGTH)
)


# ============================================================
# 8. DRAW NET
# ============================================================

# Dark backing for contrast
line_m(
    (0, NET_Y),
    (COURT_WIDTH, NET_Y),
    color=BG_COLOR,
    thickness=7
)

# Pink net
line_m(
    (0, NET_Y),
    (COURT_WIDTH, NET_Y),
    color=NET_COLOR,
    thickness=5
)


# ============================================================
# 9. DRAW PLAYER MARKERS
# ============================================================

MARKER_RADIUS = 10


def draw_player(court_position, color, label):
    """Draw a player at their real-world court position."""

    x_m, y_m = court_position

    px, py = to_img_coords(
        float(x_m),
        float(y_m)
    )

    # Dark marker outline
    cv2.circle(
        canvas,
        (px, py),
        MARKER_RADIUS + 2,
        BG_COLOR,
        -1,
        cv2.LINE_AA
    )

    # Player circle
    cv2.circle(
        canvas,
        (px, py),
        MARKER_RADIUS,
        color,
        -1,
        cv2.LINE_AA
    )

    # Player name
    cv2.putText(
        canvas,
        label,
        (px + 16, py - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA
    )


draw_player(
    near_court,
    NEAR_COLOR,
    "Near"
)

draw_player(
    far_court,
    FAR_COLOR,
    "Far"
)


# ============================================================
# 10. SAVE OUTPUT
# ============================================================

OUTPUT_PATH = Path("outputs/tactical_view.jpg")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

success = cv2.imwrite(
    str(OUTPUT_PATH),
    canvas
)

if not success:
    raise SystemExit(
        "Failed to save tactical view."
    )

print(f"Saved {OUTPUT_PATH}")
