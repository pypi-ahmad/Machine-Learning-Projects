"""Set one terminal alarm or manage several alarms interactively."""

from __future__ import annotations

import argparse
import threading
import time
from datetime import datetime, timedelta


def beep(count: int = 3) -> None:
    """Play a short system alert, with a terminal-bell fallback."""
    for _ in range(count):
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except (ImportError, OSError):
            print("\a", end="", flush=True)
        time.sleep(0.5)


def parse_time(value: str, now: datetime | None = None) -> datetime:
    """Return the next occurrence of an HH:MM or HH:MM:SS time."""
    current_time = now or datetime.now()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            parsed_time = datetime.strptime(value.strip(), fmt)
            alarm_dt = current_time.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=parsed_time.second,
                microsecond=0,
            )
            if alarm_dt <= current_time:
                alarm_dt += timedelta(days=1)
            return alarm_dt
        except ValueError:
            continue
    raise ValueError(f"Invalid time: {value!r}. Use HH:MM or HH:MM:SS.")


class Alarm:
    """A background alarm that can be cancelled before it rings."""

    def __init__(self, alarm_dt: datetime, label: str) -> None:
        self.alarm_dt = alarm_dt
        self.label = label
        self.fired = False
        self._cancelled = threading.Event()
        self._thread = threading.Thread(target=self._wait, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Prevent this alarm from ringing if it has not already fired."""
        self._cancelled.set()

    def wait(self) -> None:
        """Wait until this alarm fires or is cancelled."""
        self._thread.join()

    def _wait(self) -> None:
        delay = max((self.alarm_dt - datetime.now()).total_seconds(), 0)
        if self._cancelled.wait(delay):
            return
        if self._cancelled.is_set():
            return
        self.fired = True
        print(f"\n\n  ALARM: {self.label}  [{self.alarm_dt.strftime('%H:%M:%S')}]\n")
        beep(4)


def format_delta(alarm_dt: datetime, now: datetime | None = None) -> str:
    """Format the remaining time until an alarm."""
    diff = (alarm_dt - (now or datetime.now())).total_seconds()
    if diff < 0:
        return "fired"
    h = int(diff // 3600)
    m = int((diff % 3600) // 60)
    s = int(diff % 60)
    if h:
        return f"in {h}h {m}m"
    if m:
        return f"in {m}m {s}s"
    return f"in {s}s"


def interactive_mode() -> None:
    """Run the terminal interface for adding, listing, and deleting alarms."""
    alarms: list[Alarm] = []
    print("Alarm Clock  (type 'help' for commands)")

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=2)
        verb = parts[0].lower()

        if verb in ("q", "quit", "exit"):
            print("Bye!")
            break

        elif verb == "help":
            print("  add HH:MM [label]   Set an alarm")
            print("  list                Show all alarms")
            print("  del N               Delete alarm #N")
            print("  quit                Exit")

        elif verb == "add":
            if len(parts) < 2:
                print("  Usage: add HH:MM [label]")
                continue
            try:
                alarm_dt = parse_time(parts[1])
                label    = parts[2] if len(parts) > 2 else f"Alarm {len(alarms)+1}"
                alarms.append(Alarm(alarm_dt, label))
                print(f"  Alarm set: {label} @ {alarm_dt.strftime('%H:%M:%S')} ({format_delta(alarm_dt)})")
            except ValueError as e:
                print(f"  Error: {e}")

        elif verb == "list":
            if not alarms:
                print("  No alarms set.")
            else:
                for i, a in enumerate(alarms, 1):
                    status = "✓ fired" if a.fired else format_delta(a.alarm_dt)
                    print(f"  {i}. {a.label:20s}  {a.alarm_dt.strftime('%H:%M:%S')}  {status}")

        elif verb == "del":
            try:
                idx = int(parts[1]) - 1
                removed = alarms.pop(idx)
                removed.cancel()
                print(f"  Deleted: {removed.label}")
            except (IndexError, ValueError):
                print("  Invalid alarm number.")

        else:
            # Try treating the whole command as a time
            try:
                alarm_dt = parse_time(cmd.split()[0])
                label    = " ".join(cmd.split()[1:]) or f"Alarm {len(alarms)+1}"
                alarms.append(Alarm(alarm_dt, label))
                print(f"  Alarm set: {label} @ {alarm_dt.strftime('%H:%M:%S')} ({format_delta(alarm_dt)})")
            except ValueError:
                print(f"  Unknown command: '{verb}'. Type 'help'.")


def main() -> None:
    """Run one alarm from the command line or start interactive mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("time", nargs="?", help="alarm time in HH:MM or HH:MM:SS")
    parser.add_argument("label", nargs="?", default="Alarm", help="optional alarm message")
    args = parser.parse_args()

    if args.time is None:
        interactive_mode()
        return

    try:
        alarm_dt = parse_time(args.time)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    alarm = Alarm(alarm_dt, args.label)
    print(f"Alarm set for {alarm_dt.strftime('%H:%M:%S')} ({format_delta(alarm_dt)})")
    try:
        alarm.wait()
    except KeyboardInterrupt:
        alarm.cancel()
        print("\nAlarm cancelled.")


if __name__ == "__main__":
    main()
