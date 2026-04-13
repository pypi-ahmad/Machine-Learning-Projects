"""Stopwatch — CLI tool.

Start/stop/lap/reset a terminal stopwatch with lap history.

Usage:
    python main.py
"""

import time


def format_time(elapsed: float) -> str:
    centis = int(elapsed * 100) % 100
    secs   = int(elapsed) % 60
    mins   = int(elapsed) // 60 % 60
    hrs    = int(elapsed) // 3600
    if hrs:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}.{centis:02d}"
    return f"{mins:02d}:{secs:02d}.{centis:02d}"


def main():
    print("Stopwatch")
    print("─────────────────────────────")
    print("  Enter  → start / stop")
    print("  l      → lap")
    print("  r      → reset")
    print("  q      → quit")
    print("─────────────────────────────\n")

    running    = False
    start_time = 0.0
    accumulated = 0.0
    laps: list[float] = []

    import sys, select, os

    def _kbhit() -> bool:
        """Non-blocking stdin check (Unix only; falls back to blocking on Windows)."""
        if sys.platform == "win32":
            import msvcrt
            return msvcrt.kbhit()
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(rlist)

    def _getkey() -> str:
        if sys.platform == "win32":
            import msvcrt
            return msvcrt.getwch()
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def elapsed() -> float:
        if running:
            return accumulated + (time.perf_counter() - start_time)
        return accumulated

    # Use simpler blocking input on Windows if raw mode unavailable
    try:
        import msvcrt as _m
        RAW = True
    except ImportError:
        try:
            import tty as _t, termios as _tm
            RAW = True
        except ImportError:
            RAW = False

    if RAW:
        import threading

        stop_event = threading.Event()

        def display_loop():
            while not stop_event.is_set():
                t = elapsed()
                lap_str = f"  Lap {len(laps)+1}: {format_time(t)}"
                print(f"\r  {format_time(t)}   ", end="", flush=True)
                time.sleep(0.05)

        thread = threading.Thread(target=display_loop, daemon=True)
        thread.start()

        while True:
            key = _getkey().lower()
            t   = elapsed()

            if key in ("\r", "\n", " "):
                if running:
                    accumulated += time.perf_counter() - start_time
                    running = False
                    print(f"\r  {format_time(accumulated)}  [STOPPED]", flush=True)
                else:
                    start_time = time.perf_counter()
                    running = True
                    print(f"\r  [RUNNING…]              ", flush=True)

            elif key == "l":
                lap_time = elapsed()
                laps.append(lap_time)
                prev = laps[-2] if len(laps) >= 2 else 0.0
                split = lap_time - prev
                print(f"\n  Lap {len(laps)}: {format_time(lap_time)}  (split {format_time(split)})")

            elif key == "r":
                running = False
                accumulated = 0.0
                laps.clear()
                print(f"\r  00:00.00  [RESET]         ", flush=True)

            elif key in ("q", "\x03"):
                stop_event.set()
                print(f"\r  Final: {format_time(elapsed())}        ")
                if laps:
                    print("\n  Lap history:")
                    prev = 0.0
                    for i, lap in enumerate(laps, 1):
                        print(f"    Lap {i}: {format_time(lap)}  (split {format_time(lap - prev)})")
                        prev = lap
                break
    else:
        # Fallback: blocking input
        print("  (Enter commands: start/stop=Enter, l=lap, r=reset, q=quit)")
        while True:
            key = input().strip().lower() or "\r"
            t   = elapsed()
            if key in ("\r", ""):
                if running:
                    accumulated += time.perf_counter() - start_time
                    running = False
                    print(f"  Stopped at {format_time(accumulated)}")
                else:
                    start_time = time.perf_counter()
                    running = True
                    print("  Running…")
            elif key == "l":
                lap_time = elapsed()
                laps.append(lap_time)
                prev = laps[-2] if len(laps) >= 2 else 0.0
                print(f"  Lap {len(laps)}: {format_time(lap_time)}  (split {format_time(lap_time - prev)})")
            elif key == "r":
                running = False
                accumulated = 0.0
                laps.clear()
                print("  Reset.")
            elif key == "q":
                print(f"  Final: {format_time(elapsed())}")
                break


if __name__ == "__main__":
    main()
