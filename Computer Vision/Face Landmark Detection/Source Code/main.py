"""Entry point for Face Landmark Detection.

Usage:
    python main.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    print("[INFO] Starting Face Landmark Detection (webcam)...")
    import faceLandmark
    faceLandmark.main()


if __name__ == "__main__":
    main()
