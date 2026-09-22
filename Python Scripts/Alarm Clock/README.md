# Alarm Clock

`main.py` is a small terminal alarm clock. Set one alarm from the command line,
or open an interactive prompt to add, list, and delete several alarms.

## Install and run

```powershell
cd "Python Scripts/Alarm Clock"
uv sync
uv run python main.py 07:30 "Wake up"
```

The alarm accepts 24-hour `HH:MM` and `HH:MM:SS` times. A time that has already
passed is scheduled for the following day. Keep the terminal open until the
alarm rings; press `Ctrl+C` to cancel a command-line alarm.

Run without arguments for interactive mode:

```powershell
uv run python main.py
```

Interactive commands:

- `add HH:MM [label]` adds an alarm.
- `list` shows alarms and their remaining time.
- `del N` cancels and removes alarm number `N`.
- `quit` exits the prompt.

On Windows, the active entrypoint uses the built-in system notification sound.
Other platforms fall back to the terminal bell when supported.

`alarm.py` and the `musics/` folder are preserved legacy files. They are not used
by `main.py`; run `main.py` for the supported workflow.
