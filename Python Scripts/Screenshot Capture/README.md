# Screenshot Capture

Local command-line screenshot utility with configurable capture timing.

## Setup

Requirements: Python 3.13+ and a desktop session that permits screen capture.

```powershell
cd "Screenshot Capture"
uv sync
```

## Usage

Capture one screenshot to `images/`:

```powershell
uv run python screenshot.py
```

Capture five screenshots per minute:

```powershell
uv run python screenshot.py --unit m --frequency 5 --count 5 --path screenshots
```

Run until interrupted:

```powershell
uv run python screenshot.py --unit s --frequency 1 --continuous
```

Options:

- `--path PATH`: Output directory. Default: `images`.
- `--unit`: `h`, `m`, or `s`. Default: `h`.
- `--frequency`: Captures per selected unit. Default: `1`.
- `--count`: Number of captures. Default: `1`.
- `--continuous`: Capture until `Ctrl+C`.

## Behavior

Each capture is saved as a timestamped PNG, including microseconds to avoid filename collisions. The output directory is created when needed. Intervals under one second are clamped to one second.

Screenshots can contain sensitive information. Capture only screens and accounts you are authorized to record, and protect or delete image files when they are no longer needed.

## Project files

```text
Screenshot Capture/
├── screenshot.py
├── pyproject.toml
└── uv.lock
```
