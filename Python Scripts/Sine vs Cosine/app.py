"""Plot sine and cosine waves.

Usage:
    uv run python app.py [--output sine_vs_cosine.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_plot() -> plt.Figure:
    """Create and return a labeled sine and cosine figure."""
    time = np.linspace(-2 * np.pi, 2 * np.pi, 256, endpoint=True)
    figure, axis = plt.subplots()
    axis.plot(time, np.sin(time), label="Sine")
    axis.plot(time, np.cos(time), label="Cosine")
    axis.set(title="Sine and cosine waves", xlabel="Time", ylabel="Amplitude")
    axis.grid(True, which="both")
    axis.axhline(y=0, color="black", linewidth=0.8)
    axis.legend()
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sine and cosine waves.")
    parser.add_argument("--output", type=Path, help="Optional image output path")
    args = parser.parse_args()

    figure = create_plot()
    if args.output:
        figure.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
