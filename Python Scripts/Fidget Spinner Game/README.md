# Fidget Spinner Game in Python

> A fidget spinner simulation built with Python's `turtle` graphics module.

## Overview

This script creates an animated fidget spinner with three colored dots (red, green, blue) that the user can spin by pressing the spacebar. The spinner gradually slows down due to simulated friction, creating a realistic spinning effect.

## Features

- Three-armed fidget spinner with red, green, and blue dots
- Press **spacebar** to add spin momentum
- Gradual deceleration (friction) animation
- Smooth 50 FPS animation loop (20ms interval)
- Interactive `turtle` graphics window (420×420 pixels)

## Project Structure

```
Fidget Spinner Game in Python/
└── Fidget Spinner Game in Python.py   # Main game script
```

## Requirements

- Python 3.x
- `turtle` (included with standard Python)

## Installation

```bash
cd "Fidget Spinner Game in Python"
```

No package installation needed.

## Usage

```bash
python "Fidget Spinner Game in Python.py"
```

- Press **Space** to flick the spinner (adds 10 units of angular momentum each press)
- The spinner will gradually slow down and stop
- Press **Space** again to spin it more — momentum stacks

## How It Works

1. **State Management**: A dictionary `state = {'turn': 0}` tracks the current angular momentum.
2. **`flick()`**: Called on spacebar press, adds 10 to the `turn` value.
3. **`spinner()`**: Clears the canvas, calculates the rotation angle as `turn / 10`, then draws three arms at 120° intervals, each ending with a colored dot (radius 120). The canvas is updated without tracer for flicker-free rendering.
4. **`animate()`**: Decrements `turn` by 1 each frame (if > 0), calls `spinner()`, and schedules itself again after 20ms using `ontimer()`.
5. **Window Setup**: Creates a 420×420 window, hides the cursor, disables automatic screen updates (`tracer(False)`), sets pen width to 20, and binds the spacebar key.

## Configuration

No configuration files. The following values are hardcoded:

| Parameter      | Value      | Description                    |
|----------------|------------|--------------------------------|
| Window size    | 420 × 420  | Turtle window dimensions       |
| Arm length     | 100 pixels | Distance from center to dot    |
| Dot size       | 120 pixels | Diameter of each colored dot   |
| Pen width      | 20 pixels  | Width of the spinner arms      |
| Flick momentum | +10        | Momentum added per spacebar    |
| Friction       | -1/frame   | Deceleration per frame         |
| Frame interval | 20ms       | Animation timer interval       |

## Limitations

- No quit button — close the window manually to exit
- The spinner only moves in one direction (clockwise)
- No score or speed display
- The `done()` call at the end blocks the main thread

## Security Notes

No security concerns.

## License

Not specified.
