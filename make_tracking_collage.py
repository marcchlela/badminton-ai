import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# 1. CONFIG
# ============================================================

VIDEO_PATH = Path("data/raw/clip.mp4")
TRACKING_PATH = Path("outputs/tracking_recovered.json")
COURT_POINTS_PATH = Path("outputs/court_points.json")

OUTPUT_DIR = Path("outputs/collages")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "tracking_collage.jpg"

COURT_W = 6.10
COURT_L = 13.40
NET_Y = COURT_L / 2.0
NEAR_SHORT_SERVICE_Y = NET_Y - 1.98

THUMB_W = 640
THUMB_H = 360
COLS = 5

# ============================================================
# 2. COLORS
# ============================================================

def bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    rgb = bytes.fromhex(hex_color)
    return tuple(reversed(rgb))

NEAR_COLOR = bgr("#68CFF3")       # cyan
FAR_COLOR = bgr("#F6BC67")        # gold
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (18, 18, 18)

MEASURED_TAG = bgr("#22C55E")     # green
LOW_CONF_TAG = bgr("#EAB308")     # yellow
INTERP_TAG = bgr("#E85D8E")       # pink
OUTLINE = (10, 10, 10)

# ============================================================
# 3. LOAD DATA
# ============================================================

with open(TRACKING_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)

samples.sort(key=lambda x: x["time"])

with open(COURT_POINTS_PATH, "r", encoding="utf-8") as f:
    image_points = np.float32(json.load(f))

court_points = np.float32([
    [0.0, 0.0],
    [COURT_W, 0.0],
    [COURT_W, NEAR_SHORT_SERVICE_Y],
    [0.0, NEAR_SHORT_SERVICE_Y],
])

# court metres -> image pixels
H_inv = cv2.getPerspectiveTransform(court_points, image_points)

def court_to_image(x_m, y_m):
    pt = np.float32([[[x_m, y_m]]])
    projected = cv2.perspectiveTransform(pt, H_inv)[0, 0]
    return float(projected[0]), float(projected[1])

# ============================================================
# 4. DRAW HELPERS
# ============================================================

def draw_marker(frame, px, py, player_color, source, label):
    px = int(round(px))
    py = int(round(py))

    if source == "measured":
        tag_color = MEASURED_TAG
        tag_text = "MEASURED"
    elif source == "low_confidence":
        tag_color = LOW_CONF_TAG
        tag_text = "LOW CONF"
    elif source == "interpolated":
        tag_color = INTERP_TAG
        tag_text = "INTERP"
    else:
        tag_color = (180, 180, 180)
        tag_text = source.upper()

    # outer ring
    cv2.circle(frame, (px, py), 11, OUTLINE, -1, cv2.LINE_AA)
    cv2.circle(frame, (px, py), 8, player_color, -1, cv2.LINE_AA)

    # label box
    text = f"{label} | {tag_text}"
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )

    box_x1 = px + 12
    box_y1 = py - 24
    box_x2 = box_x1 + tw + 10
    box_y2 = box_y1 + th + 10

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), OUTLINE, -1)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), tag_color, 2)

    cv2.putText(
        frame,
        text,
        (box_x1 + 5, box_y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
        cv2.LINE_AA
    )

def annotate_frame(frame, sample):
    time_s = sample["time"]
    frame_id = sample["frame"]

    result = frame.copy()

    # Header
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        result,
        f"TIME: {time_s:.2f}s   FRAME: {frame_id}",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        TEXT_COLOR,
        2,
        cv2.LINE_AA
    )

    players = sample.get("players", {})

    for side in ("Near", "Far"):
        if side not in players:
            continue

        player = players[side]
        source = player.get("position_source", "measured")
        label = side.upper()

        # Use original pixel if we have it.
        if player.get("pixel") is not None:
            px, py = player["pixel"]
        else:
            # Interpolated case -> project court position back to image
            x_m, y_m = player["court_m"]
            px, py = court_to_image(x_m, y_m)

        color = NEAR_COLOR if side == "Near" else FAR_COLOR

        draw_marker(
            result,
            px, py,
            color,
            source,
            label
        )

    return result

# ============================================================
# 5. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise SystemExit("Could not open video.")

# ============================================================
# 6. GENERATE THUMBNAILS
# ============================================================

thumbs = []

try:
    for sample in samples:
        frame_id = int(sample["frame"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()

        if not ok:
            print(f"Failed to read frame {frame_id}")
            continue

        annotated = annotate_frame(frame, sample)
        thumb = cv2.resize(annotated, (THUMB_W, THUMB_H))
        thumbs.append(thumb)

finally:
    cap.release()

if not thumbs:
    raise SystemExit("No thumbnails generated.")

# ============================================================
# 7. BUILD CONTACT SHEET
# ============================================================

rows = (len(thumbs) + COLS - 1) // COLS
sheet_h = rows * THUMB_H
sheet_w = COLS * THUMB_W

sheet = np.full((sheet_h, sheet_w, 3), BG_COLOR, dtype=np.uint8)

for i, thumb in enumerate(thumbs):
    row = i // COLS
    col = i % COLS

    y1 = row * THUMB_H
    y2 = y1 + THUMB_H
    x1 = col * THUMB_W
    x2 = x1 + THUMB_W

    sheet[y1:y2, x1:x2] = thumb

# ============================================================
# 8. SAVE
# ============================================================

success = cv2.imwrite(str(OUTPUT_PATH), sheet)

if not success:
    raise SystemExit("Failed to save collage.")

print(f"Saved collage to: {OUTPUT_PATH}")
print(f"Frames included: {len(thumbs)}")