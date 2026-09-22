# Digital Clock With GUI

A resizable Tkinter clock that displays local system time in 24-hour `HH:MM:SS` format.

## Requirements

- Python 3.13+ with Tkinter available
- [uv](https://docs.astral.sh/uv/)

## Run

From this directory:

```powershell
uv sync
uv run python "Digital Clock Gui.py"
```

The clock refreshes every 200 milliseconds. Resize the window as needed and close it normally to exit.

## Appearance

The default UI preserves the original yellow background, dark text, bold `Boulder` font setting, and thick label border. If `Boulder` is unavailable on the system, Tkinter selects a fallback font.

## Project files

```text
Digital Clock Gui.py  # Tkinter application
pyproject.toml        # uv project definition
uv.lock               # Resolved Python environment
```

## Verification

```powershell
uv run python -m py_compile "Digital Clock Gui.py"
uv lock --check
```
