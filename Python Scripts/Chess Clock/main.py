"""Chess Clock — CLI tool.

Two-player chess clock with configurable time controls.
Supports classical, rapid, blitz, bullet, and increment modes.

Usage:
    python main.py
    python main.py --minutes 10
    python main.py --minutes 5 --increment 3
    python main.py --preset blitz
"""

import argparse
import sys
import threading
import time


PRESETS = {
    "bullet":    (1,   0,  "Bullet"),
    "blitz":     (5,   0,  "Blitz"),
    "rapid":     (10,  0,  "Rapid"),
    "classical": (60,  0,  "Classical"),
    "fischer":   (5,   3,  "Fischer Rapid"),
    "bronstein": (10,  2,  "Bronstein"),
}


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    m = s // 60
    s = s % 60
    if m >= 60:
        h = m // 60
        m = m % 60
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class ChessClock:
    def __init__(self, minutes: float, increment: float):
        self.initial   = minutes * 60.0
        self.increment = increment
        self.times     = [self.initial, self.initial]   # player 0, player 1
        self.current   = None        # None = not started; 0 or 1 = whose clock is running
        self.running   = False
        self.paused    = False
        self._lock     = threading.Lock()
        self._last_tick = None

    def start(self, player: int):
        with self._lock:
            self.current    = player
            self.running    = True
            self.paused     = False
            self._last_tick = time.perf_counter()

    def switch(self) -> bool:
        """Switch active player. Returns False if flag fallen."""
        with self._lock:
            if self.current is None: return True
            now = time.perf_counter()
            elapsed = now - self._last_tick
            self.times[self.current] -= elapsed
            if self.times[self.current] <= 0:
                self.times[self.current] = 0
                self.running = False
                return False
            self.times[self.current] += self.increment
            self.current    = 1 - self.current
            self._last_tick = now
        return True

    def tick(self) -> float:
        """Return remaining time for active player, applying elapsed."""
        with self._lock:
            if self.current is None or not self.running:
                return self.times[self.current or 0]
            now     = time.perf_counter()
            elapsed = now - self._last_tick
            return max(0, self.times[self.current] - elapsed)

    def other_time(self) -> float:
        with self._lock:
            other = 1 - (self.current or 0)
            return self.times[other]

    def pause(self):
        with self._lock:
            if self.running and not self.paused:
                now     = time.perf_counter()
                elapsed = now - self._last_tick
                self.times[self.current] -= elapsed
                self.times[self.current]  = max(0, self.times[self.current])
                self.paused  = True
                self.running = False

    def resume(self):
        with self._lock:
            if self.paused:
                self.paused     = False
                self.running    = True
                self._last_tick = time.perf_counter()


def clear_line():
    print("\r" + " " * 70 + "\r", end="", flush=True)


def display_status(clock: ChessClock, player_names: list[str], move_num: int) -> None:
    p = clock.current
    if p is None: return
    active  = clock.tick()
    passive = clock.other_time()
    other   = 1 - p

    low_warn = "⚠️ " if active < 30 else "   "
    status = (
        f"  Move {move_num:>3}  │  "
        f"{player_names[p]:>10}: {fmt_time(active):>8} {low_warn} │  "
        f"{player_names[other]:>10}: {fmt_time(passive):>8}"
    )
    print(f"\r{status}", end="", flush=True)


def play(minutes: float, increment: float, names: list[str]) -> None:
    clock    = ChessClock(minutes, increment)
    move_num = 1

    print(f"\n  Time control: {minutes:.0f}+{increment:.0f}  "
          f"({fmt_time(minutes*60)} per player)")
    print(f"  Players: {names[0]} (White) vs {names[1]} (Black)")
    print(f"  Controls: ENTER = move played | P = pause | Q = quit\n")
    print("  Press ENTER to start White's clock…")

    try:
        input()
    except (EOFError, KeyboardInterrupt):
        return

    clock.start(0)
    stop_event = threading.Event()

    # Background thread: flag fallen check
    def watcher():
        while not stop_event.is_set():
            if clock.running:
                t = clock.tick()
                if t <= 0:
                    stop_event.set()
            time.sleep(0.1)

    t = threading.Thread(target=watcher, daemon=True)
    t.start()

    while not stop_event.is_set():
        display_status(clock, names, move_num)
        try:
            cmd = input().strip().upper()
        except (EOFError, KeyboardInterrupt):
            clock.pause()
            break

        if stop_event.is_set(): break

        if cmd == "":       # move played
            ok = clock.switch()
            if not ok:
                break
            if clock.current == 0:
                move_num += 1
        elif cmd == "P":
            if clock.paused:
                print("\n  Resuming…")
                clock.resume()
            else:
                clock.pause()
                print("\n  Paused. Press ENTER to resume…")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    break
                clock.resume()
        elif cmd == "Q":
            print("\n  Game aborted.")
            stop_event.set()
            break

    stop_event.set()
    t.join(timeout=1)

    # Final state
    print()
    for i, name in enumerate(names):
        t_left = clock.times[i]
        flagged = "🏳️ FLAG!" if t_left <= 0 else fmt_time(t_left)
        print(f"  {name:>12}: {flagged}")

    if clock.times[0] <= 0:
        print(f"\n  🏆 {names[1]} wins on time!")
    elif clock.times[1] <= 0:
        print(f"\n  🏆 {names[0]} wins on time!")
    else:
        print(f"\n  Game ended at move {move_num}.")


def main():
    parser = argparse.ArgumentParser(description="Chess Clock")
    parser.add_argument("--minutes",   type=float, default=None, help="Minutes per player")
    parser.add_argument("--increment", type=float, default=0,    help="Increment per move (seconds)")
    parser.add_argument("--preset",    choices=list(PRESETS.keys()), default=None)
    parser.add_argument("--white",     default="White", help="White player name")
    parser.add_argument("--black",     default="Black", help="Black player name")
    args = parser.parse_args()

    if args.preset:
        minutes, increment, label = PRESETS[args.preset]
        print(f"  Preset: {label} ({minutes}+{increment})")
    elif args.minutes:
        minutes, increment = args.minutes, args.increment
    else:
        print("=== Chess Clock ===")
        print("Presets: " + ", ".join(PRESETS.keys()))
        preset = input("Preset (or press ENTER for custom): ").strip().lower()
        if preset in PRESETS:
            minutes, increment, label = PRESETS[preset]
            print(f"  {label} ({minutes}+{increment})")
        else:
            minutes   = float(input("Minutes per player [5]: ").strip() or 5)
            increment = float(input("Increment seconds   [0]: ").strip() or 0)

    play(minutes, increment, [args.white, args.black])


if __name__ == "__main__":
    main()
