
import cv2
import json
import numpy as np

# Load original frame and calibration
image = cv2.imread("outputs/calibration_frame.jpg")

if image is None:
    raise SystemExit("Calibration image not found")

with open("outputs/court_points.json") as f:
    image_points = np.float32(json.load(f))

# Coordinates in real-world metres
court_points = np.float32([
    [0, 0],
    [6.10, 0],
    [6.10, 4.72],
    [0, 4.72]
])

H = cv2.getPerspectiveTransform(image_points, court_points)

players = []
labels = ["Near Player", "Far Player"]


def on_mouse(event, x, y, flags, userdata):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(players) >= 2:
        return

    point = np.float32([[[x, y]]])
    court_xy = cv2.perspectiveTransform(point, H)[0, 0]

    player = {
        "name": labels[len(players)],
        "pixel": [x, y],
        "court_m": [float(court_xy[0]), float(court_xy[1])]
    }

    players.append(player)

    print(f"{player['name']}:")
    print(f"  Image: ({x}, {y})")
    print(f"  Court: ({court_xy[0]:.2f}, {court_xy[1]:.2f}) m")


cv2.namedWindow("Player Annotation")
cv2.setMouseCallback("Player Annotation", on_mouse)

while True:
    preview = image.copy()

    for i, player in enumerate(players):
        x, y = player["pixel"]
        color = (255, 200, 70) if i == 0 else (70, 190, 255)

        cv2.circle(preview, (x, y), 7, color, -1)

        cv2.putText(
            preview,
            labels[i],
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Player Annotation", preview)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):
        players.clear()
        print("Reset player positions")

    elif key == ord("s") and len(players) == 2:
        with open("outputs/player_positions.json", "w") as f:
            json.dump(players, f, indent=2)

        cv2.imwrite("outputs/player_positions_preview.jpg", preview)

        print("Saved player positions!")
        break

    elif key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()
