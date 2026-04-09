# Sine vs Cosine

> A matplotlib script that plots sine and cosine waves on the same axes.

## Overview

This script generates and displays a combined sine and cosine wave plot using NumPy for computation and Matplotlib for visualization. The waves are plotted over the range $-2\pi$ to $2\pi$.

## Features

- Plots a sine wave and a cosine wave on the same axes
- Uses 256 evenly spaced sample points across the range $[-2\pi, 2\pi]$
- Displays a grid, axis labels, title, and a horizontal line at $y = 0$
- Interactive Matplotlib window (`plot.show()`)

## Project Structure

```
SINE_VS_COSINE/
└── app.py
```

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`

## Installation

```bash
cd "SINE_VS_COSINE"
pip install numpy matplotlib
```

## Usage

```bash
python app.py
```

A Matplotlib window will open displaying the sine and cosine waves. Close the window to exit.

## How It Works

1. `np.linspace(-2*np.pi, 2*np.pi, 256, endpoint=True)` generates 256 evenly spaced time values
2. `np.sin(time)` and `np.cos(time)` compute the amplitude arrays
3. Both curves are plotted with `plot.plot()`
4. Title is set to "Sine & Cos wave", X-axis to "Time", Y-axis to "Amplitude"
5. `plot.grid(True, which='both')` enables full grid lines
6. `plot.axhline(y=0, color='k')` draws a black horizontal reference line at zero
7. `plot.show()` displays the interactive plot window

## Configuration

- **Sample count**: Change `256` in `np.linspace()` to adjust resolution
- **Range**: Modify the `-2*np.pi` / `2*np.pi` bounds in `np.linspace()` to change the x-axis range

## Limitations

- No legend to distinguish the sine and cosine curves
- No command-line arguments for customization
- Does not save the plot to a file — display only

## Security Notes

No security concerns.

## License

Not specified.
