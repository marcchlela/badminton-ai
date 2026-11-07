
import cv2
import numpy as np
import json

# Load our calibration frame and selected points
image = cv2.imread("outputs/calibration_frame.jpg")

if image is None:
    raise SystemExit("Calibration image not found")

with open("outputs/court_points.json") as file:
    points = json.load(file)

# Pixel coordinates from the original image
image_points = np.float32(points)

# Known court coordinates, in metres
court_width = 6.10
court_length = 13.40
short_service_y = 6.70 - 1.98

court_points = np.float32([
    [0, 0],
    [court_width, 0],
    [court_width, short_service_y],
    [0, short_service_y],
])

# Calculate image-pixels -> court-metres transformation
H = cv2.getPerspectiveTransform(image_points, court_points)

print("Homography matrix:")
print(H)

# Test an additional court line not used for calibration
test_pixel = np.float32([[[640, 310]]])
test_court = cv2.perspectiveTransform(test_pixel, H)

x, y = test_court[0, 0]
print(f"\nTest point maps to: ({x:.2f}, {y:.2f}) metres")
print("Expected far short-service Y: 8.68 metres")

# Create an overhead court image
pixels_per_metre = 75
margin = 30

output_points = court_points.copy()

# Convert metres into output-image pixels
output_points[:, 0] = margin + court_points[:, 0] * pixels_per_metre

# Flip Y so the near baseline appears at the bottom
output_points[:, 1] = margin + (
    court_length - court_points[:, 1]
) * pixels_per_metre

# Image pixels -> overhead image pixels
H_image = cv2.getPerspectiveTransform(
    image_points, output_points
)

output_width = round(court_width * pixels_per_metre) + 2 * margin
output_height = round(court_length * pixels_per_metre) + 2 * margin

topdown = cv2.warpPerspective(
    image,
    H_image,
    (output_width, output_height)
)

cv2.imwrite("outputs/court_topdown.jpg", topdown)

print("\nSaved outputs/court_topdown.jpg")
