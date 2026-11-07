
import cv2
import json

image = cv2.imread("outputs/calibration_frame.jpg")

if image is None:
    raise SystemExit("Could not load calibration image")

points = []

labels = [
    "Near baseline LEFT",
    "Near baseline RIGHT",
    "Near short-service RIGHT",
    "Near short-service LEFT",
]

def on_mouse(event, x, y, flags, userdata):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(points) >= 4:
        return

    points.append([x, y])
    i = len(points) - 1

    print(f"Point {i + 1} ({labels[i]}): ({x}, {y})")


cv2.namedWindow("Court Calibration", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Court Calibration", on_mouse)

while True:
    preview = image.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(preview, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            preview,
            str(i + 1),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow("Court Calibration", preview)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):
        points.clear()
        print("Points reset")

    elif key == ord("s") and len(points) == 4:
        with open("outputs/court_points.json", "w") as f:
            json.dump(points, f, indent=2)

        cv2.imwrite("outputs/court_points_preview.jpg", preview)
        print("Saved calibration points!")
        break

    elif key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()
