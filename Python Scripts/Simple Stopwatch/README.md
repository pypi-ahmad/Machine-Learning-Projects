# Simple Stopwatch

A terminal stopwatch with start/stop, lap, reset, and lap-history controls.
The supported entrypoint is `main.py`.

## Run

```powershell
uv run python main.py
```

On Windows, use Enter or Space to start and stop, `l` to record a lap, `r` to
reset, and `q` to quit. The timer uses `time.perf_counter()` and displays
centiseconds.

On terminals without raw-key support, the app falls back to line-based input:
press Enter to start or stop, then enter `l`, `r`, or `q` on a line.

## Files

- `main.py` is the maintained CLI stopwatch.
- `stopwatch.py` is a separate legacy Tkinter example and is not the supported
  entrypoint for this project.

## Dependencies

The CLI uses only Python's standard library. uv records the Python requirement
and supplies the reproducible project environment.
