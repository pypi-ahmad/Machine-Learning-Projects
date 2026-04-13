"""Alarm Clock — CLI tool.

Set one or more alarms by time (HH:MM or HH:MM:SS).
Plays a beep and shows a message when each alarm fires.
Runs until all alarms trigger or user quits.

Usage:
    python main.py
    python main.py 07:30 "Wake up!"
"""

import sys
import threading
import time
from datetime import datetime, timedelta


def beep(n: int = 3):
    for _ in range(n):
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            print("\a", end="", flush=True)
        time.sleep(0.5)


def parse_time(s: str) -> datetime:
    now = datetime.now()
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(s.strip(), fmt)
            alarm_dt = now.replace(hour=t.hour, minute=t.minute,
                                   second=t.second, microsecond=0)
            if alarm_dt <= now:
                alarm_dt += timedelta(days=1)
            return alarm_dt
        except ValueError:
            pass
    raise ValueError(f"Invalid time: '{s}'. Use HH:MM or HH:MM:SS.")


class Alarm:
    def __init__(self, alarm_dt: datetime, label: str):
        self.alarm_dt = alarm_dt
        self.label    = label
        self.fired    = False
        self._thread  = threading.Thread(target=self._wait, daemon=True)
        self._thread.start()

    def _wait(self):
        now = datetime.now()
        delay = (self.alarm_dt - now).total_seconds()
        if delay > 0:
            time.sleep(delay)
        self.fired = True
        print(f"\n\n  🔔  ALARM: {self.label}  [{self.alarm_dt.strftime('%H:%M:%S')}]\n")
        beep(4)


def format_delta(dt: datetime) -> str:
    diff = (dt - datetime.now()).total_seconds()
    if diff < 0:
        return "fired"
    h = int(diff // 3600)
    m = int((diff % 3600) // 60)
    s = int(diff % 60)
    if h:
        return f"in {h}h {m}m"
    elif m:
        return f"in {m}m {s}s"
    return f"in {s}s"


def main():
    alarms: list[Alarm] = []

    # Quick mode: python main.py HH:MM "label"
    if len(sys.argv) >= 2:
        try:
            alarm_dt = parse_time(sys.argv[1])
            label    = sys.argv[2] if len(sys.argv) > 2 else "Alarm"
            alarms.append(Alarm(alarm_dt, label))
            print(f"  Alarm set for {alarm_dt.strftime('%H:%M:%S')} ({format_delta(alarm_dt)})")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    print("Alarm Clock  (type 'help' for commands)")

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not cmd:
            continue

        parts = cmd.split(None, 2)
        verb  = parts[0].lower()

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


if __name__ == "__main__":
    main()
