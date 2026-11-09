
import cv2
import sys
from pathlib import Path

video_path = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "data/raw/clip.mp4"
)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise SystemExit(f"Could not open {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = frames / fps if fps > 0 else 0

size_mb = video_path.stat().st_size / (1024 * 1024)

codec_id = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join(
    chr((codec_id >> (8 * i)) & 255)
    for i in range(4)
)

cap.release()

print("\n========== VIDEO QUALITY ==========")
print("File:", video_path)
print(f"Resolution: {width} x {height}")
print(f"Frame rate: {fps:.2f} FPS")
print(f"Duration: {duration:.2f} seconds")
print(f"Frames: {frames}")
print(f"File size: {size_mb:.1f} MB")
print(f"Codec: {codec}")

if duration > 0:
    bitrate = size_mb * 8 / duration
    print(f"Approx file bitrate: {bitrate:.2f} Mbps")

print("\nNote: bitrate includes any audio and container overhead.")
print("Resolution and bitrate alone do not prove visual quality.")
