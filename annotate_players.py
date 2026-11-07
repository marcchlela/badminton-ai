
import argparse
import cv2
import json
import math
import numpy as np
from pathlib import Path


# ============================================================
# 1. CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(description="Annotate player court positions.")
parser.add_argument("--video", type=Path, default=Path("data/raw/clip.mp4"))
parser.add_argument("--calibration", type=Path, default=Path("outputs/court_points.json"))
parser.add_argument("--output", type=Path, default=Path("outputs/manual_evaluation.json"))
parser.add_argument("--start", type=float)
parser.add_argument("--end", type=float)
parser.add_argument("--step", type=float, default=0.3)
args = parser.parse_args()

VIDEO_PATH = args.video
CALIBRATION_PATH = args.calibration
OUTPUT_PATH = args.output


TIMES = [
    # Normal movement
    20.0, 20.2, 20.4, 20.6, 20.8,
    21.0, 21.2, 21.4,

    # Fast movement and lunge
    21.5, 21.6, 21.7, 21.8,

    # Recovery and jump sequence
    22.0, 22.1, 22.2, 22.3,
    22.4, 22.5, 22.6, 22.7,
    22.8, 22.9
]

if args.start is not None or args.end is not None:
    if args.start is None or args.end is None or args.end <= args.start or args.step <= 0:
        parser.error("--start and --end must define a positive interval and --step must be positive")
    count = math.ceil((args.end - args.start) / args.step - 1e-9)
    TIMES = [round(args.start + i * args.step, 3) for i in range(count)]


COURT_W = 6.10
COURT_L = 13.40

WINDOW_W = 1320
WINDOW_H = 730

VIDEO_X = 20
VIDEO_Y = 95
VIDEO_W = 896
VIDEO_H = 504

COURT_X = 1000
COURT_Y = 100
COURT_SCALE = 40

COURT_PX_W = round(COURT_W * COURT_SCALE)
COURT_PX_H = round(COURT_L * COURT_SCALE)

WINDOW_NAME = "Badminton AI - Manual Evaluation"


# ============================================================
# 2. COLORS
# ============================================================

BG = (34, 20, 12)
COURT_COLOR = (97, 107, 23)
WHITE = (240, 245, 245)
MUTED = (175, 175, 175)

NEAR_COLOR = (243, 207, 104)
FAR_COLOR = (103, 188, 246)
PINK = (142, 93, 232)


# ============================================================
# 3. LOAD VIDEO AND COURT CALIBRATION
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise SystemExit("Could not open badminton video.")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    raise SystemExit("Invalid video frame rate.")

with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
    image_points = np.float32(json.load(f))

court_points = np.float32([
    [0.0, 0.0],
    [COURT_W, 0.0],
    [COURT_W, 4.72],
    [0.0, 4.72]
])

H = cv2.getPerspectiveTransform(
    image_points,
    court_points
)


def image_to_court(x, y):
    point = np.float32([[[x, y]]])
    result = cv2.perspectiveTransform(point, H)[0, 0]

    return [float(result[0]), float(result[1])]


def court_to_canvas(x, y):
    px = round(COURT_X + x * COURT_SCALE)
    py = round(COURT_Y + (COURT_L - y) * COURT_SCALE)

    return px, py


def canvas_to_court(px, py):
    x = (px - COURT_X) / COURT_SCALE
    y = COURT_L - (py - COURT_Y) / COURT_SCALE

    return [float(x), float(y)]


# ============================================================
# 4. LOAD FRAMES
# ============================================================

frames = []

for timestamp in TIMES:
    frame_id = round(timestamp * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()

    if not success:
        raise SystemExit(
            f"Could not load video frame at {timestamp}s"
        )

    frames.append({
        "time": timestamp,
        "frame": frame_id,
        "image": frame
    })

cap.release()


# ============================================================
# 5. ANNOTATION STATE
# ============================================================

annotations = {}

if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)

    for item in existing.get("annotations", []):
        annotations[str(item["time"])] = item["players"]

current_index = 0
selected_side = "Near"

notice = "Select a player, then click the video or court."


def current_key():
    return str(TIMES[current_index])


def current_players():
    return annotations.setdefault(current_key(), {})


def save_annotations():
    records = []

    for item in frames:
        key = str(item["time"])

        if key not in annotations:
            continue

        records.append({
            "time": item["time"],
            "frame": item["frame"],
            "players": annotations[key]
        })

    output = {
        "description": (
            "Manual badminton player position references. "
            "Video clicks project visible floor-contact "
            "points; court clicks are subjective estimates."
        ),
        "annotations": records
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


# ============================================================
# 6. DRAW HELPERS
# ============================================================

def put(frame, message, x, y, scale=0.55,
        color=WHITE, thickness=1):
    cv2.putText(
        frame, message, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


def court_line(canvas, a, b, color=WHITE, width=2):
    cv2.line(
        canvas,
        court_to_canvas(*a),
        court_to_canvas(*b),
        color, width, cv2.LINE_AA
    )


def draw_court(canvas):
    top_left = court_to_canvas(0, COURT_L)
    bottom_right = court_to_canvas(COURT_W, 0)

    cv2.rectangle(
        canvas, top_left, bottom_right,
        COURT_COLOR, -1
    )

    # Outer boundaries
    court_line(canvas, (0, 0), (COURT_W, 0))
    court_line(canvas, (0, COURT_L),
               (COURT_W, COURT_L))
    court_line(canvas, (0, 0), (0, COURT_L))
    court_line(canvas, (COURT_W, 0),
               (COURT_W, COURT_L))

    # Singles sidelines
    for x in (0.46, 5.64):
        court_line(canvas, (x, 0), (x, COURT_L))

    # Service lines
    for y in (0.76, 4.72, 8.68, 12.64):
        court_line(canvas, (0, y), (COURT_W, y))

    # Center service lines
    court_line(canvas, (3.05, 0), (3.05, 4.72))
    court_line(canvas, (3.05, 8.68), (3.05, 13.40))

    # Net
    court_line(canvas, (0, 6.70),
               (COURT_W, 6.70), PINK, 3)


def draw_marker(canvas, point, color, label, radius=6):
    x, y = point

    cv2.circle(
        canvas, (round(x), round(y)),
        radius + 2, (0, 0, 0), -1, cv2.LINE_AA
    )

    cv2.circle(
        canvas, (round(x), round(y)),
        radius, color, -1, cv2.LINE_AA
    )

    put(
        canvas, label,
        round(x) + radius + 4,
        round(y) - radius - 3,
        0.4, color
    )


# ============================================================
# 7. DRAW ANNOTATION INTERFACE
# ============================================================

def render():
    canvas = np.full(
        (WINDOW_H, WINDOW_W, 3),
        BG, dtype=np.uint8
    )

    frame_info = frames[current_index]
    image = frame_info["image"]

    resized = cv2.resize(
        image, (VIDEO_W, VIDEO_H)
    )

    canvas[
        VIDEO_Y:VIDEO_Y + VIDEO_H,
        VIDEO_X:VIDEO_X + VIDEO_W
    ] = resized

    draw_court(canvas)

    put(
        canvas,
        "BADMINTON AI / MANUAL POSITION EVALUATION",
        20, 37, 0.78, WHITE, 2
    )

    put(
        canvas,
        f"FRAME {current_index + 1}/{len(frames)}"
        f"   |   MATCH TIME {frame_info['time']:.2f}s",
        20, 72, 0.63, PINK
    )

    put(canvas, "REFERENCE COURT", COURT_X, 79, 0.53)

    players = current_players()

    for side in ("Near", "Far"):
        record = players.get(side)

        if record is None or record["court_m"] is None:
            continue

        color = NEAR_COLOR if side == "Near" else FAR_COLOR

        point = court_to_canvas(*record["court_m"])

        draw_marker(
            canvas, point, color,
            "N" if side == "Near" else "F"
        )

        pixel = record.get("pixel")

        if pixel is not None:
            video_point = (
                VIDEO_X + pixel[0] * VIDEO_W / image.shape[1],
                VIDEO_Y + pixel[1] * VIDEO_H / image.shape[0]
            )

            draw_marker(
                canvas, video_point, color, side, radius=3
            )

    near_done = "OK" if "Near" in players else "-"
    far_done = "OK" if "Far" in players else "-"

    put(
        canvas,
        f"Selected: {selected_side.upper()}   "
        f"|  Near: {near_done}   Far: {far_done}",
        20, 629, 0.62
    )

    put(
        canvas,
        "1 = Near    2 = Far    "
        "Click VIDEO = visible foot contact",
        20, 657, 0.49
    )

    put(
        canvas,
        "Click COURT = estimated floor position "
        "(e.g. jumping)",
        20, 680, 0.49
    )

    put(
        canvas,
        "X = uncertain/skip    R = reset frame    "
        "N = next    B = back    Q = save/quit",
        20, 705, 0.45, MUTED
    )

    put(
        canvas,
        notice[:50],
        970, 669, 0.36, MUTED
    )

    return canvas


# ============================================================
# 8. MOUSE INPUT
# ============================================================

def on_mouse(event, x, y, flags, userdata):
    global notice

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    image = frames[current_index]["image"]

    color_name = selected_side

    # Click original video where the shoe touches the floor
    if (
        VIDEO_X <= x < VIDEO_X + VIDEO_W
        and VIDEO_Y <= y < VIDEO_Y + VIDEO_H
    ):

        original_x = (
            (x - VIDEO_X) * image.shape[1] / VIDEO_W
        )

        original_y = (
            (y - VIDEO_Y) * image.shape[0] / VIDEO_H
        )

        point = image_to_court(
            original_x, original_y
        )

        if not (
            0 <= point[0] <= COURT_W
            and 0 <= point[1] <= COURT_L
        ):
            notice = "Clicked foot projects outside court."
            return

        current_players()[color_name] = {
            "court_m": point,
            "pixel": [original_x, original_y],
            "method": "visible_foot_contact"
        }

        notice = f"{color_name}: visible floor contact saved."

    # Click tactical court for estimated ground position
    elif (
        COURT_X <= x <= COURT_X + COURT_PX_W
        and COURT_Y <= y <= COURT_Y + COURT_PX_H
    ):

        point = canvas_to_court(x, y)

        current_players()[color_name] = {
            "court_m": point,
            "pixel": None,
            "method": "estimated_ground_position"
        }

        notice = f"{color_name}: estimated ground position saved."


# ============================================================
# 9. INTERACTIVE LOOP
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_AUTOSIZE
)

cv2.setMouseCallback(WINDOW_NAME, on_mouse)

print("Annotation tool started.")
print("1/2: player, N/B: navigation, X: skip")
print("Click video for visible foot contact.")
print("Click court for estimated floor position.")
print("Q: save and quit")

try:
    while True:
        cv2.imshow(WINDOW_NAME, render())

        key = cv2.waitKey(30) & 0xFF

        if key == ord("1"):
            selected_side = "Near"

        elif key == ord("2"):
            selected_side = "Far"

        elif key == ord("x"):
            current_players()[selected_side] = {
                "court_m": None,
                "pixel": None,
                "method": "uncertain_skipped"
            }
            notice = f"{selected_side}: marked uncertain."

        elif key == ord("r"):
            annotations[current_key()] = {}
            notice = "Current frame reset."

        elif key == ord("n"):
            save_annotations()

            if current_index < len(frames) - 1:
                current_index += 1
                notice = "Next frame."
            else:
                print("Reached the final frame.")
                break

        elif key == ord("b"):
            save_annotations()

            if current_index > 0:
                current_index -= 1
                notice = "Previous frame."

        elif key == ord("q") or key == 27:
            break

finally:
    save_annotations()
    cv2.destroyAllWindows()

print(f"Saved annotations: {OUTPUT_PATH}")
