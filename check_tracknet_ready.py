
import sys
import shutil
import platform
import importlib.metadata
from pathlib import Path

print("========== TRACKNET READINESS CHECK ==========")

print("\nSYSTEM")
print("OS:", platform.platform())
print("Python:", sys.version.split()[0])
print("Executable:", sys.executable)

print("\nINSTALLED PACKAGES")

packages = [
    "torch",
    "torchvision",
    "opencv-python",
    "numpy",
    "Pillow",
    "scipy",
    "matplotlib",
]

for name in packages:
    try:
        version = importlib.metadata.version(name)
        print(f"{name}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")

print("\nPYTORCH DEVICE")

try:
    import torch

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)

            print(f"GPU {i}: {props.name}")
            print(f"GPU memory: {memory_gb:.2f} GB")
    else:
        print("Inference device: CPU")

except ImportError:
    print("PyTorch is not installed.")
except Exception as exc:
    print("PyTorch check failed:", type(exc).__name__, str(exc))

print("\nSTORAGE")

drive = Path.cwd().anchor
total, used, free = shutil.disk_usage(drive)

print("Drive:", drive)
print(f"Free: {free / 1024**3:.2f} GB")
print(f"Total: {total / 1024**3:.2f} GB")

print("\nVIDEO")

video = Path("data/raw/high_quality_match.mp4")

if video.exists():
    print("HD test video: FOUND")
else:
    print("HD test video: MISSING")

print("\n========== CHECK COMPLETE ==========")
