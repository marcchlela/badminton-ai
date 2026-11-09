
import json
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# FILES
# ============================================================

ROOT = Path(__file__).resolve().parent
RALLY_DIR = ROOT / "outputs/unified/coach_demo_36_46"

TIMELINE_FILE = RALLY_DIR / "timeline.json"
METADATA_FILE = RALLY_DIR / "metadata.json"

VIDEO_FILE = (
    ROOT / "outputs/shuttle/tracknet/"
    "coach_demo_36_46/rally.mp4"
)

OUTPUT_FILE = RALLY_DIR / "coach_demo_36_46_v31.mp4"

# ============================================================
# COURT + DESIGN
# ============================================================

COURT_W = 6.10
COURT_L = 13.40
PANEL_W = 470

# All colors are BGR for OpenCV
BG = (34, 20, 12)
COURT = (97, 107, 23)
LINES = (240, 245, 227)
NET = (142, 93, 232)
NEAR = (243, 207, 104)
FAR = (103, 188, 246)
WHITE = (255, 255, 255)
MUTED = (165, 160, 150)
GREEN = (110, 245, 125)

FONT = cv2.FONT_HERSHEY_SIMPLEX

PLAYER_TRAIL_FRAMES = 13
SHUTTLE_TRAIL_FRAMES = 9

# Player samples are roughly 10 FPS,
# while the video is 25 FPS.
MAX_PLAYER_GAP_FRAMES = 5
EDGE_HOLD_FRAMES = 4

# Smoothing is for DISPLAY only.
PLAYER_SMOOTH_RADIUS = 2

# The court shuttle is illustrative.
# It does NOT represent a measured ground position.
SHUTTLE_MARGIN_X = 0.65
SHUTTLE_MARGIN_Y = 1.00


# ============================================================
# HELPERS
# ============================================================

def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def valid_point(value):
    if not isinstance(value, (list, tuple)):
        return None

    if len(value) < 2:
        return None

    if value[0] is None or value[1] is None:
        return None

    x = float(value[0])
    y = float(value[1])

    if not (math.isfinite(x) and math.isfinite(y)):
        return None

    return np.array([x, y], dtype=np.float64)


def valid_box(value):
    if not isinstance(value, (list, tuple)):
        return None

    if len(value) != 4:
        return None

    if any(v is None for v in value):
        return None

    box = np.array(value, dtype=np.float64)

    if not np.isfinite(box).all():
        return None

    if box[2] <= 0 or box[3] <= 0:
        return None

    return box


def put_text(
    img, message, x, y,
    color=WHITE, scale=0.5, thickness=1
):
    cv2.putText(
        img,
        str(message),
        (int(x), int(y)),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def dim_color(color, amount):
    return tuple(
        int(channel * amount)
        for channel in color
    )


def screen_point(value):
    return tuple(
        int(round(v))
        for v in value
    )


# ============================================================
# PLAYER SERIES
# ============================================================

def interpolate_samples(samples, count):
    """
    Create frame-by-frame positions from sparse samples.

    Returns:
        values: one entry per video frame
        measured: whether that frame has a real sample

    Estimated values never overwrite original tracking.
    """

    values = [None] * count
    measured = [False] * count

    for i, value in samples.items():
        values[i] = value.copy()
        measured[i] = True

    indices = sorted(samples)

    if not indices:
        return values, measured

    # Interpolate short gaps between samples.
    for a, b in zip(indices, indices[1:]):
        gap = b - a

        if gap <= 1:
            continue

        if gap > MAX_PLAYER_GAP_FRAMES:
            continue

        for i in range(a + 1, b):
            t = (i - a) / gap

            values[i] = (
                samples[a] * (1 - t)
                + samples[b] * t
            )

    # Hold for short gaps at either end.
    first = indices[0]
    last = indices[-1]

    for i in range(
        max(0, first - EDGE_HOLD_FRAMES),
        first
    ):
        values[i] = samples[first].copy()

    for i in range(
        last + 1,
        min(count, last + EDGE_HOLD_FRAMES + 1)
    ):
        values[i] = samples[last].copy()

    return values, measured


def smooth_series(values, radius):
    """
    Symmetric moving average.
    Only smooth within uninterrupted valid segments.
    """

    result = [None] * len(values)

    for i, value in enumerate(values):
        if value is None:
            continue

        weighted = np.zeros_like(value)
        total_weight = 0.0

        for offset in range(-radius, radius + 1):
            j = i + offset

            if not (0 <= j < len(values)):
                continue

            if values[j] is None:
                continue

            lo = min(i, j)
            hi = max(i, j)

            if any(
                values[k] is None
                for k in range(lo, hi + 1)
            ):
                continue

            weight = radius + 1 - abs(offset)

            weighted += values[j] * weight
            total_weight += weight

        if total_weight:
            result[i] = weighted / total_weight

    return result


def build_player_data(timeline):
    count = len(timeline)

    result = {}

    for side in ("Near", "Far"):
        courts = {}
        boxes = {}

        for i, row in enumerate(timeline):
            if not row.get("players_sampled"):
                continue

            player = row.get(
                "players", {}
            ).get(side)

            if not isinstance(player, dict):
                continue

            court = valid_point(
                player.get("court_m")
            )

            if court is not None:
                if (
                    0 <= court[0] <= COURT_W
                    and 0 <= court[1] <= COURT_L
                ):
                    courts[i] = court

            box = valid_box(
                player.get("bbox")
            )

            # Only use visually detected boxes.
            if (
                box is not None
                and player.get("visual_detection") is True
            ):
                boxes[i] = box

        court_series, court_measured = (
            interpolate_samples(courts, count)
        )

        box_series, box_measured = (
            interpolate_samples(boxes, count)
        )

        court_series = smooth_series(
            court_series,
            PLAYER_SMOOTH_RADIUS
        )

        box_series = smooth_series(
            box_series,
            PLAYER_SMOOTH_RADIUS
        )

        result[side] = {
            "court": court_series,
            "court_measured": court_measured,
            "boxes": box_series,
            "box_measured": box_measured,
            "court_samples": len(courts),
            "box_samples": len(boxes),
        }

    return result


# ============================================================
# SHUTTLE SCHEMATIC COORDINATES
# ============================================================

def build_shuttle_display(timeline):
    """
    IMPORTANT:
    This is a motion-preserving schematic view.

    It uses the recorded range of shuttle IMAGE
    positions to distribute the path inside the
    court illustration.

    It does not estimate the shuttle's 3D position,
    altitude, landing point, or true ground projection.

    Unlike V2, there is no ground-plane homography
    and no frame-by-frame boundary clipping.
    """

    image_points = []

    for row in timeline:
        shuttle = row["shuttle"]

        if not shuttle.get("visible"):
            continue

        p = valid_point(shuttle.get("pixel"))

        if p is not None:
            image_points.append(p)

    if len(image_points) < 2:
        return [None] * len(timeline)

    points = np.array(image_points)

    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())

    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())

    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)

    display = []

    for row in timeline:
        shuttle = row["shuttle"]

        if not shuttle.get("visible"):
            display.append(None)
            continue

        point = valid_point(
            shuttle.get("pixel")
        )

        if point is None:
            display.append(None)
            continue

        # Normalize the actual image movement.
        tx = (point[0] - x_min) / x_range
        ty = (point[1] - y_min) / y_range

        # Within the illustrated court bounds,
        # retaining variation in both image axes.
        court_x = (
            SHUTTLE_MARGIN_X
            + tx * (COURT_W - 2 * SHUTTLE_MARGIN_X)
        )

        court_y = (
            SHUTTLE_MARGIN_Y
            + (1 - ty)
            * (COURT_L - 2 * SHUTTLE_MARGIN_Y)
        )

        display.append(
            np.array(
                [court_x, court_y],
                dtype=np.float64
            )
        )

    return display


# ============================================================
# COURT DRAWING
# ============================================================

def make_court_panel(height):
    panel = np.full(
        (height, PANEL_W, 3),
        BG,
        dtype=np.uint8
    )

    top_area = 94
    bottom_area = height - 116

    ppm = min(
        (PANEL_W - 112) / COURT_W,
        (bottom_area - top_area) / COURT_L
    )

    court_width = round(COURT_W * ppm)
    court_height = round(COURT_L * ppm)

    left = (PANEL_W - court_width) // 2

    top = (
        top_area
        + (bottom_area - top_area - court_height) // 2
    )

    bottom = top + court_height

    def to_screen(point):
        return (
            round(left + point[0] * ppm),
            round(bottom - point[1] * ppm)
        )

    cv2.rectangle(
        panel,
        to_screen((0, COURT_L)),
        to_screen((COURT_W, 0)),
        COURT,
        -1
    )

    def line(a, b, color=LINES, width=1):
        cv2.line(
            panel,
            to_screen(a),
            to_screen(b),
            color,
            width,
            cv2.LINE_AA
        )

    # Doubles boundaries
    line((0, 0), (COURT_W, 0), width=2)
    line((0, COURT_L), (COURT_W, COURT_L), width=2)
    line((0, 0), (0, COURT_L), width=2)
    line((COURT_W, 0), (COURT_W, COURT_L), width=2)

    # Singles sidelines
    side = (COURT_W - 5.18) / 2

    line((side, 0), (side, COURT_L))
    line(
        (COURT_W - side, 0),
        (COURT_W - side, COURT_L)
    )

    # Doubles long-service lines
    line((0, 0.76), (COURT_W, 0.76))
    line(
        (0, COURT_L - 0.76),
        (COURT_W, COURT_L - 0.76)
    )

    # Short-service lines
    short_near = 6.70 - 1.98
    short_far = 6.70 + 1.98

    line((0, short_near), (COURT_W, short_near))
    line((0, short_far), (COURT_W, short_far))

    # Center-service lines
    line((3.05, 0), (3.05, short_near))
    line((3.05, short_far), (3.05, COURT_L))

    # Thin pink net
    line((0, 6.70), (COURT_W, 6.70), NET, 2)

    return panel, to_screen


# ============================================================
# PLAYER TRAILS
# ============================================================

def draw_player_trail(
    panel, series, index, to_screen, color
):
    start = max(
        0, index - PLAYER_TRAIL_FRAMES
    )

    # Draw oldest segments faintest.
    for i in range(start + 1, index + 1):
        a = series[i - 1]
        b = series[i]

        if a is None or b is None:
            continue

        # Skip very large jumps.
        if np.linalg.norm(a - b) > 1.1:
            continue

        age = index - i

        strength = (
            0.22
            + 0.60 * (
                1 - age / PLAYER_TRAIL_FRAMES
            )
        )

        cv2.line(
            panel,
            to_screen(a),
            to_screen(b),
            dim_color(color, strength),
            3,
            cv2.LINE_AA
        )


def draw_player_marker(
    panel, position, to_screen, color
):
    center = to_screen(position)

    cv2.circle(
        panel, center,
        9, BG, -1, cv2.LINE_AA
    )

    cv2.circle(
        panel, center,
        7, color, -1, cv2.LINE_AA
    )

    cv2.circle(
        panel, center,
        7, WHITE, 1, cv2.LINE_AA
    )


# ============================================================
# WHITE SHUTTLECOCK ICON
# ============================================================

def draw_shuttle_icon(
    panel, center, direction, scale=1.0
):
    """
    Draw a compact white shuttlecock icon.

    Direction describes its 2D schematic movement.
    No 3D orientation is inferred.
    """

    norm = np.linalg.norm(direction)

    if norm < 0.001:
        direction = np.array(
            [0.0, -1.0], dtype=np.float64
        )
    else:
        direction = direction / norm

    perpendicular = np.array([
        -direction[1],
        direction[0]
    ])

    c = np.array(center, dtype=np.float64)

    # Cork is the leading end.
    cork = c + direction * (5 * scale)

    feather_base = c - direction * (2 * scale)
    feather_end = c - direction * (12 * scale)

    feather_left = (
        feather_end + perpendicular * (6 * scale)
    )

    feather_right = (
        feather_end - perpendicular * (6 * scale)
    )

    polygon = np.array([
        feather_base,
        feather_left,
        feather_right
    ], dtype=np.int32)

    cv2.fillConvexPoly(
        panel,
        polygon,
        WHITE,
        cv2.LINE_AA
    )

    cv2.line(
        panel,
        screen_point(feather_left),
        screen_point(feather_right),
        (205, 205, 205),
        1,
        cv2.LINE_AA
    )

    cv2.circle(
        panel,
        screen_point(cork),
        max(2, round(3 * scale)),
        WHITE,
        -1,
        cv2.LINE_AA
    )


def shuttle_direction(series, index):
    current = series[index]

    if current is None:
        return np.array(
            [0.0, -1.0]
        )

    for earlier in range(
        index - 1,
        max(-1, index - 5),
        -1
    ):
        previous = series[earlier]

        if previous is None:
            continue

        delta = current - previous

        if np.linalg.norm(delta) > 0.03:
            return np.array([
                delta[0],
                -delta[1]
            ])

    return np.array(
        [0.0, -1.0]
    )


def draw_shuttle_trail(
    panel, series, index, to_screen
):
    start = max(
        0, index - SHUTTLE_TRAIL_FRAMES
    )

    for i in range(start + 1, index + 1):
        a = series[i - 1]
        b = series[i]

        if a is None or b is None:
            continue

        age = index - i

        strength = (
            0.18
            + 0.55 * (
                1 - age / SHUTTLE_TRAIL_FRAMES
            )
        )

        cv2.line(
            panel,
            to_screen(a),
            to_screen(b),
            dim_color(WHITE, strength),
            2,
            cv2.LINE_AA
        )


# ============================================================
# PLAYER BOXES ON VIDEO
# ============================================================

def dashed_line(img, a, b, color):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    delta = b - a
    distance = np.linalg.norm(delta)

    if distance < 1:
        return

    direction = delta / distance
    step = 13
    dash = 7

    for offset in range(0, int(distance), step):
        end = min(offset + dash, distance)

        p1 = a + direction * offset
        p2 = a + direction * end

        cv2.line(
            img,
            screen_point(p1),
            screen_point(p2),
            color,
            2,
            cv2.LINE_AA
        )


def draw_player_box(
    frame, box, color, side, estimated
):
    height, width = frame.shape[:2]

    x, y, w, h = box

    x1 = max(0, min(width - 1, round(x)))
    y1 = max(0, min(height - 1, round(y)))

    x2 = max(0, min(width - 1, round(x + w)))
    y2 = max(0, min(height - 1, round(y + h)))

    if x2 <= x1 or y2 <= y1:
        return

    if estimated:
        dashed_line(
            frame, (x1, y1), (x2, y1), color
        )
        dashed_line(
            frame, (x2, y1), (x2, y2), color
        )
        dashed_line(
            frame, (x2, y2), (x1, y2), color
        )
        dashed_line(
            frame, (x1, y2), (x1, y1), color
        )
    else:
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
            cv2.LINE_AA
        )

    label = (
        side.upper()
        + (" ~" if estimated else "")
    )

    label_y = max(68, y1 - 8)

    put_text(
        frame,
        label,
        x1,
        label_y,
        color,
        0.47,
        2
    )


# ============================================================
# VIDEO OVERLAY
# ============================================================

def draw_video(
    frame, row, index, players, shuttle_trail
):
    # Player boxes first, shuttle after.
    for side, color in (
        ("Near", NEAR),
        ("Far", FAR)
    ):
        data = players[side]
        box = data["boxes"][index]

        if box is None:
            continue

        # A smoothed box is a display estimate
        # even at sampled timestamps.
        estimated = not data["box_measured"][index]

        draw_player_box(
            frame,
            box,
            color,
            side,
            estimated
        )

    shuttle_data = row["shuttle"]

    shuttle = (
        valid_point(
            shuttle_data.get("pixel")
        )
        if shuttle_data.get("visible")
        else None
    )

    shuttle_trail.append(shuttle)

    trail_points = list(shuttle_trail)

    for a, b in zip(
        trail_points,
        trail_points[1:]
    ):
        if a is None or b is None:
            continue

        if np.linalg.norm(a - b) > 140:
            continue

        cv2.line(
            frame,
            screen_point(a),
            screen_point(b),
            GREEN,
            2,
            cv2.LINE_AA
        )

    if shuttle is not None:
        p = screen_point(shuttle)

        cv2.circle(
            frame, p,
            11, (10, 12, 14),
            4, cv2.LINE_AA
        )

        cv2.circle(
            frame, p,
            8, GREEN,
            2, cv2.LINE_AA
        )

        cv2.circle(
            frame, p,
            2, WHITE,
            -1, cv2.LINE_AA
        )

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], 52),
        BG,
        -1
    )

    put_text(
        frame,
        f"FRAME {row['frame']}  |  "
        f"{row['time']:.2f}s",
        18, 32,
        WHITE, 0.65, 2
    )

    status = (
        "SHUTTLE DETECTED"
        if shuttle is not None
        else "SHUTTLE MISSING"
    )

    put_text(
        frame,
        status,
        frame.shape[1] - 250,
        32,
        GREEN if shuttle is not None else MUTED,
        0.48,
        2
    )


# ============================================================
# TACTICAL PANEL
# ============================================================

def draw_tactical_panel(
    height, index, players, shuttle_display,
    shuttle_image, video_height
):
    panel, to_screen = make_court_panel(height)

    put_text(
        panel,
        "TACTICAL COURT",
        24, 36,
        WHITE, 0.72, 2
    )

    put_text(
        panel,
        "Player positions + schematic shuttle",
        24, 61,
        MUTED, 0.41
    )

    # Player trails underneath markers.
    for side, color in (
        ("Near", NEAR),
        ("Far", FAR)
    ):
        draw_player_trail(
            panel,
            players[side]["court"],
            index,
            to_screen,
            color
        )

    # Shuttle trail
    draw_shuttle_trail(
        panel,
        shuttle_display,
        index,
        to_screen
    )

    # Player markers
    shown = 0

    for side, color in (
        ("Near", NEAR),
        ("Far", FAR)
    ):
        p = players[side]["court"][index]

        if p is None:
            continue

        shown += 1

        draw_player_marker(
            panel,
            p,
            to_screen,
            color
        )

    # White shuttlecock
    shuttle = shuttle_display[index]

    if shuttle is not None:
        center = to_screen(shuttle)

        direction = shuttle_direction(
            shuttle_display, index
        )

        # Perspective-style size variation only.
        # This is NOT a measured altitude.
        if shuttle_image is not None:
            relative_y = (
                shuttle_image[1]
                / max(video_height, 1)
            )
            relative_y = float(
                np.clip(relative_y, 0, 1)
            )
        else:
            relative_y = 0.5

        scale = 0.75 + 0.35 * relative_y

        draw_shuttle_icon(
            panel, center, direction, scale
        )

    put_text(
        panel,
        "NEAR",
        24, height - 90,
        NEAR, 0.48, 2
    )

    put_text(
        panel,
        "FAR",
        125, height - 90,
        FAR, 0.48, 2
    )

    put_text(
        panel,
        "White shuttle: illustrative motion",
        24, height - 62,
        MUTED, 0.40
    )

    put_text(
        panel,
        "Not measured height or landing position",
        24, height - 40,
        MUTED, 0.36
    )

    put_text(
        panel,
        f"Players shown: {shown}/2",
        24, height - 18,
        WHITE, 0.42
    )

    return panel


def smooth_schematic_shuttle(positions, radius=2):
    """
    Smooth the illustrated 2D shuttle path only.

    - Keeps missing detections missing.
    - Never changes raw TrackNet predictions.
    - Does not interpolate through missing frames.
    - Does not estimate shuttle height or court position.
    """
    output = [None] * len(positions)

    for i, current in enumerate(positions):
        if current is None:
            continue

        total = np.zeros(2, dtype=np.float64)
        total_weight = 0.0

        for offset in range(-radius, radius + 1):
            j = i + offset

            if not 0 <= j < len(positions):
                continue

            if positions[j] is None:
                continue

            # Don't smooth across missing predictions.
            lo, hi = sorted((i, j))

            if any(
                positions[k] is None
                for k in range(lo, hi + 1)
            ):
                continue

            weight = radius + 1 - abs(offset)
            total += positions[j] * weight
            total_weight += weight

        output[i] = (
            total / total_weight
            if total_weight else current.copy()
        )

    return output



# ============================================================
# MAIN
# ============================================================

def main():
    timeline = read_json(TIMELINE_FILE)
    metadata = read_json(METADATA_FILE)

    if not timeline:
        raise RuntimeError(
            "Unified timeline is empty"
        )

    expected = int(metadata["total_frames"])
    first = int(metadata["start_frame"])
    fps = float(metadata["fps"])

    if len(timeline) != expected:
        raise RuntimeError(
            "Timeline length mismatch"
        )

    for i, row in enumerate(timeline):
        if int(row["frame"]) != first + i:
            raise RuntimeError(
                f"Frame mismatch at index {i}"
            )

    players = build_player_data(timeline)
    shuttle_display = smooth_schematic_shuttle(
        build_shuttle_display(timeline),
        radius=2
    )

    cap = cv2.VideoCapture(str(VIDEO_FILE))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {VIDEO_FILE}"
        )

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if (
        video_frames != expected
        or abs(video_fps - fps) > 0.001
    ):
        cap.release()
        raise RuntimeError(
            "Video and timeline are not aligned"
        )

    writer = cv2.VideoWriter(
        str(OUTPUT_FILE),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width + PANEL_W, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Cannot create output video"
        )

    shuttle_trail = deque(
        maxlen=SHUTTLE_TRAIL_FRAMES
    )

    written = 0

    print("========== UNIFIED REPLAY V3 ==========")

    for side in ("Near", "Far"):
        data = players[side]

        print(
            f"{side}: "
            f"{data['court_samples']} player samples, "
            f"{data['box_samples']} detected boxes"
        )

        print(
            "  Display frames:",
            sum(
                p is not None
                for p in data["court"]
            )
        )

    print(
        "Shuttle schematic positions:",
        sum(
            p is not None
            for p in shuttle_display
        )
    )

    try:
        for i, row in enumerate(timeline):
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    f"Cannot read video frame {i}"
                )

            draw_video(
                frame,
                row,
                i,
                players,
                shuttle_trail
            )

            image_point = (
                valid_point(
                    row["shuttle"].get("pixel")
                )
                if row["shuttle"].get("visible")
                else None
            )

            panel = draw_tactical_panel(
                height,
                i,
                players,
                shuttle_display,
                image_point,
                height
            )

            combined = np.hstack(
                (frame, panel)
            )

            writer.write(combined)
            written += 1

            if written % 25 == 0:
                print(
                    f"Rendered {written}/{expected}"
                )

    finally:
        cap.release()
        writer.release()

    check = cv2.VideoCapture(
        str(OUTPUT_FILE)
    )

    verified = int(
        check.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    check.release()

    print("\n========== V3 SUMMARY ==========")
    print("Expected frames:", expected)
    print("Written frames:", written)
    print("Verified frames:", verified)
    print("Output:", OUTPUT_FILE)

    if written != expected or verified != expected:
        raise RuntimeError(
            "Output verification failed"
        )

    print("\nReplay V3 complete!")


if __name__ == "__main__":
    main()
