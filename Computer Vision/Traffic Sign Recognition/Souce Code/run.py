"""Entry point for Traffic Sign Recognition (Flask app).

Usage:
    python run.py
"""
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent
sys.path.insert(0, str(_src.parents[1]))
sys.path.insert(0, str(_src))


def main():
    from Traffic_app import app
    print("[INFO] Starting Traffic Sign Recognition Flask server...")
    app.run(debug=False, port=5000)


if __name__ == "__main__":
    main()
