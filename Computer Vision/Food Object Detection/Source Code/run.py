"""Entry point for Food Object Detection (Streamlit app).

Usage:
    streamlit run run.py
    OR
    python run.py   (launches streamlit programmatically)
"""
import sys
import os
from pathlib import Path

_src = Path(__file__).resolve().parent
sys.path.insert(0, str(_src.parents[1]))


def main():
    os.system(f'streamlit run "{_src / "food_app.py"}"')


if __name__ == "__main__":
    main()
