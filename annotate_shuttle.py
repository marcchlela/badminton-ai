
import cv2
import json
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

VIDEO = Path("data/raw/high_quality_match.mp4")
OUTPUT = Path("outputs/shuttle/hd_visibility_audit.json")

RALLIES = [
    (200.0, 210.0),
]

SAMPLE_FPS = 5
HEADER_H = 52
ZOOM_RADIUS = 35
ZOOM_SCALE = 6

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise SystemExit(f"Cannot open {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    raise SystemExit("Invalid FPS")

print(f"Video FPS: {fps}")
print(f"Total frames: {total_frames}")

# Generate sample frames without duplication.
sample_frames = []

for start, end in RALLIES:
    count = int(round((end - start) * SAMPLE_FPS))

    for k in range(count):
        t = start + k / SAMPLE_FPS
        frame_id = round(t * fps)

        if 0 <= frame_id < total_frames:
            sample_frames.append(frame_id)

sample_frames = sorted(set(sample_frames))

# ============================================================
# LOAD OR CREATE ANNOTATIONS
# ============================================================

data = {
    "video": str(VIDEO),
    "video_fps": fps,
    "sample_fps": SAMPLE_FPS,
    "annotations": [
        {
            "frame": frame_id,
            "time": round(frame_id / fps, 3),
            "status": "unreviewed",
            "pixel": None
        }
        for frame_id in sample_frames
    ]
}

if OUTPUT.exists():
    with open(OUTPUT, "r", encoding="utf-8") as f:
        previous = json.load(f)

    old = {
        a["frame"]: a
        for a in previous.get("annotations", [])
    }

    for record in data["annotations"]:
        if record["frame"] in old:
            saved = old[record["frame"]]
            record["status"] = saved["status"]
            record["pixel"] = saved.get("pixel")

records = data["annotations"]

def save():
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ============================================================
# INTERACTION STATE
# ============================================================

state = {
    "index": 0,
    "mouse_x": 640,
    "mouse_y": 360,
    "frame": None,
    "loaded_index": -1,
}

# Resume at the first unfinished frame.
for i, record in enumerate(records):
    if record["status"] == "unreviewed":
        state["index"] = i
        break

def record_answer(status, pixel=None):
    i = state["index"]
    records[i]["status"] = status
    records[i]["pixel"] = pixel

    save()

    if i < len(records) - 1:
        state["index"] += 1
    else:
        state["index"] = len(records)

def on_mouse(event, x, y, flags, userdata):
    if state["index"] >= len(records):
        return

    # Video starts below the header.
    video_y = y - HEADER_H
    frame = state["frame"]

    if frame is None:
        return

    height, width = frame.shape[:2]

    if 0 <= x < width and 0 <= video_y < height:
        state["mouse_x"] = x
        state["mouse_y"] = video_y

        if event == cv2.EVENT_LBUTTONDOWN:
            record_answer(
                "visible",
                [int(x), int(video_y)]
            )

# ============================================================
# WINDOWS
# ============================================================

cv2.namedWindow(
    "Shuttle Audit", cv2.WINDOW_AUTOSIZE
)
cv2.namedWindow(
    "Shuttle Zoom", cv2.WINDOW_AUTOSIZE
)

cv2.setMouseCallback("Shuttle Audit", on_mouse)

def make_zoom(frame, x, y):
    h, w = frame.shape[:2]
    r = ZOOM_RADIUS

    x1 = max(0, x - r)
    x2 = min(w, x + r)
    y1 = max(0, y - r)
    y2 = min(h, y + r)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    zoom = cv2.resize(
        crop,
        None,
        fx=ZOOM_SCALE,
        fy=ZOOM_SCALE,
        interpolation=cv2.INTER_NEAREST
    )

    cx = (x - x1) * ZOOM_SCALE
    cy = (y - y1) * ZOOM_SCALE

    cv2.drawMarker(
        zoom, (cx, cy),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        25, 2
    )

    return zoom


# ============================================================
# MAIN LOOP
# ============================================================

print("\nSHUTTLE VISIBILITY AUDIT")
print("-----------------------------")
print("Left click : mark shuttle position")
print("X          : shuttle not visible")
print("U          : uncertain / cannot tell")
print("B          : previous frame")
print("Q          : save and quit")
print("-----------------------------")

try:
    while state["index"] < len(records):
        i = state["index"]
        record = records[i]

        if state["loaded_index"] != i:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                record["frame"]
            )

            ok, frame = cap.read()

            if not ok:
                print(
                    "Failed to read frame:",
                    record["frame"]
                )
                break

            state["frame"] = frame
            state["loaded_index"] = i

        frame = state["frame"]
        h, w = frame.shape[:2]

        canvas = cv2.copyMakeBorder(
            frame,
            HEADER_H, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(22, 22, 22)
        )

        title = (
            f"Frame {i + 1}/{len(records)} | "
            f"Time {record['time']:.2f}s | "
            f"{record['status']}"
        )

        cv2.putText(
            canvas, title,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )

        cv2.putText(
            canvas,
            "CLICK: visible | X: missing | "
            "U: uncertain | B: back | Q: quit",
            (10, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (180, 230, 255), 1
        )

        if record["pixel"] is not None:
            px, py = record["pixel"]

            cv2.circle(
                canvas,
                (int(px), int(py) + HEADER_H),
                7, (0, 255, 255), 2
            )

        cv2.imshow("Shuttle Audit", canvas)

        zoom = make_zoom(
            frame,
            state["mouse_x"],
            state["mouse_y"]
        )

        if zoom is not None:
            cv2.imshow("Shuttle Zoom", zoom)

        key = cv2.waitKey(40) & 0xFF

        if key == ord("q"):
            save()
            print("Saved and exited.")
            break

        elif key == ord("x"):
            record_answer("not_visible")

        elif key == ord("u"):
            record_answer("uncertain")

        elif key == ord("b"):
            if state["index"] > 0:
                state["index"] -= 1

finally:
    save()
    cap.release()
    cv2.destroyAllWindows()

# ============================================================
# SUMMARY
# ============================================================

counts = {}

for record in records:
    status = record["status"]
    counts[status] = counts.get(status, 0) + 1

print("\n========== AUDIT SUMMARY ==========")
print("Total sampled frames:", len(records))

for status, count in sorted(counts.items()):
    print(f"{status}: {count}")

print("Saved:", OUTPUT)
