"""Entry point for Live PoseDetector (Project 10).

Launches the pose detector using MediaPipe Holistic via the main module.
Usage:
    python main.py
"""

import runpy
import sys
from pathlib import Path

# Ensure project directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "poseDetector-I.py"),
        run_name="__main__",
    )
