# Pomodoro Timer GUI

Single-file Tkinter Pomodoro timer with configurable work, short-break, and long-break intervals.

## Setup

Requirements: Python 3.13+. This project uses only the Python standard library.

```powershell
cd "Pomodoro Timer GUI"
uv sync
```

## Usage

```powershell
uv run python main.py
```

Controls:

- **Start**: Start or resume the current phase.
- **Pause**: Pause the countdown.
- **Reset**: Return to the first work phase and clear completed sessions.
- **Skip**: Finish the current phase immediately.

Set work, short-break, and long-break durations before starting. The default schedule is 25 minutes of work, 5 minutes of short break, and a 15-minute long break after every four completed work sessions.

## Behavior

The timer runs in a background thread and schedules display updates on Tkinter's event loop, so the window stays responsive. At phase completion it uses Windows' standard notification beep where available and selects the next work or break phase.

Session counts live only in memory and reset when the application closes. This maintained version intentionally does not modify the Windows hosts file or block websites.

## Project files

```text
Pomodoro Timer GUI/
├── main.py
├── pyproject.toml
└── uv.lock
```

Legacy image and sound assets remain in the folder but are not used by the maintained timer.
