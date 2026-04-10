import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import torch
import numpy as np
import pandas
from ultralytics import YOLO
from utils.paths import PathResolver
from utils.device import get_device

paths = PathResolver()
device = get_device()
_src = Path(__file__).resolve().parent

def main():
    # Load the model (migrated from torch.hub.load local YOLOv5 to ultralytics)
    model_path = paths.models("fire_and_smoke_detection") / "best.pt"
    model = YOLO(str(model_path))

    # Images
    imgs = str(_src / '26.jpg')  # batch of images

    # Inference
    results = model(imgs, device=device)
    for r in results:
        r.show()
        print(r.boxes)


if __name__ == "__main__":
    main()
