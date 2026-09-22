# Digital Clock

A small Tkinter desktop clock that shows the local system time in 12-hour format and lets you switch between light and dark themes.

## Requirements

- Python 3.13+ with Tkinter available
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run python digital_clock.py
```

Use the **Theme** menu to switch between light and dark modes. Close the window normally to stop the clock.

## Behavior

- Updates the display every second using the local system time.
- Uses a single frame, label, and scheduled callback for both themes.
- Does not collect, save, or send data.

## Project files

```text
digital_clock.py   # Tkinter application
Digital Clock.PNG  # Existing project screenshot
pyproject.toml     # uv project definition
uv.lock            # Resolved Python environment
```

## Verification

```powershell
uv run python -m py_compile digital_clock.py
uv lock --check
```
