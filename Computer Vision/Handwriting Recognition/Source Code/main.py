"""Entry point for Handwriting Recognition.

Usage:
    python main.py                    # infer on sample image
    python main.py --train            # train the model
    python main.py --validate         # validate the model
"""
import sys
import os
from pathlib import Path

_src = Path(__file__).resolve().parent
# Add src/ to path so submodules can be imported
sys.path.insert(0, str(_src / "src"))
sys.path.insert(0, str(_src.parents[1]))

if __name__ == "__main__":
    # Delegate to src/main.py
    os.chdir(str(_src / "src"))
    from src.main import main  # noqa: E402
    # main() is called inside src/main.py's __main__ block via argparse
    import src.main as _m
    _m.main()
