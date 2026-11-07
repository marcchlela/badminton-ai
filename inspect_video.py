
import cv2
from pathlib import Path

# Path to our badminton footage
video_path = Path("data/raw/clip.mp4")

if not video_path.exists():
    print(f"Video not found: {video_path}")
    raise SystemExit(1)

# Open the video
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("OpenCV could not open the video.")
    raise SystemExit(1)

# Read video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")

if fps > 0:
    print(f"Approximate duration: {total_frames / fps:.2f} seconds")

# Read our first frame
success, frame = cap.read()

if success:
    print(f"Frame array shape: {frame.shape}")

    # Save the frame as an image
    output = Path("outputs/first_frame.jpg")
    cv2.imwrite(str(output), frame)
    print(f"Saved first frame to: {output}")
else:
    print("Could not read the first frame.")

# Release the video resource
cap.release()
