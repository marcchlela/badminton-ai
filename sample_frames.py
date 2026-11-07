
import cv2
import numpy as np

video_path = "data/raw/clip.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise SystemExit("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)

# The timestamps we want to inspect
timestamps = [0, 5, 10, 15, 20, 25]

samples = []

for seconds in timestamps:
    frame_number = int(seconds * fps)

    # Jump to the requested frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        continue

    if seconds == 20:
        cv2.imwrite("outputs/calibration_frame.jpg", frame)

    # Make a smaller preview
    frame = cv2.resize(frame, (480, 270))

    # Label it with its timestamp
    cv2.putText(
        frame,
        f"{seconds} seconds",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    samples.append(frame)

cap.release()

if len(samples) != 6:
    raise SystemExit("Could not read all six frames")

# Combine into two rows of three frames
top = np.hstack(samples[:3])
bottom = np.hstack(samples[3:])

contact_sheet = np.vstack([top, bottom])

cv2.imwrite("outputs/sample_frames.jpg", contact_sheet)

print("Saved outputs/sample_frames.jpg")
