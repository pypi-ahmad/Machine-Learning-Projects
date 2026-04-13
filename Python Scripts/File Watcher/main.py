"""File Watcher — CLI tool.

Monitor a directory for file changes (create, modify, delete, rename)
using periodic polling.  Logs events to console and optionally to a file.

Usage:
    python main.py
"""

import os
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Watcher logic
# ---------------------------------------------------------------------------

def snapshot(directory: Path, recursive: bool = True) -> dict[Path, tuple[float, int]]:
    """Return {path: (mtime, size)} for all files in directory."""
    state: dict[Path, tuple[float, int]] = {}
    glob = directory.rglob if recursive else directory.glob
    try:
        for f in glob("*"):
            if f.is_file():
                try:
                    st = f.stat()
                    state[f] = (st.st_mtime, st.st_size)
                except OSError:
                    pass
    except PermissionError:
        pass
    return state


def diff_snapshots(
    old: dict[Path, tuple[float, int]],
    new: dict[Path, tuple[float, int]],
) -> list[tuple[str, Path]]:
    """Return list of (event_type, path)."""
    events = []
    for path, (mtime, size) in new.items():
        if path not in old:
            events.append(("CREATED ", path))
        elif old[path] != (mtime, size):
            events.append(("MODIFIED", path))
    for path in old:
        if path not in new:
            events.append(("DELETED ", path))
    return events


def watch(
    directory: Path,
    interval: float = 1.0,
    recursive: bool = True,
    pattern: str | None = None,
    log_file: Path | None = None,
    max_events: int = 0,
) -> None:
    """Poll for changes until interrupted or max_events reached."""
    print(f"\n  Watching: {directory.resolve()}")
    print(f"  Interval: {interval}s  |  Recursive: {recursive}")
    print(f"  Press Ctrl+C to stop.\n")

    log_fh = open(log_file, "a", encoding="utf-8") if log_file else None

    def emit(event: str, path: Path) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rel = path.relative_to(directory) if directory in path.parents or path.parent == directory else path
        line = f"  [{ts}]  {event}  {rel}"
        print(line)
        if log_fh:
            log_fh.write(line.strip() + "\n")
            log_fh.flush()

    old = snapshot(directory, recursive)
    event_count = 0

    try:
        while True:
            time.sleep(interval)
            new = snapshot(directory, recursive)
            events = diff_snapshots(old, new)

            for etype, path in events:
                if pattern and pattern not in path.name:
                    continue
                emit(etype, path)
                event_count += 1
                if max_events and event_count >= max_events:
                    print(f"\n  Reached {max_events} events. Stopping.")
                    return

            old = new
    except KeyboardInterrupt:
        print(f"\n  Stopped. Detected {event_count} event(s).")
    finally:
        if log_fh:
            log_fh.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
File Watcher
------------
1. Watch a directory
2. Take a directory snapshot
3. Compare two snapshots (manual diff)
0. Quit
"""


def get_dir(prompt: str = "  Directory to watch: ") -> Path | None:
    path_str = input(prompt).strip().strip('"')
    p = Path(path_str) if path_str else Path(".")
    if not p.is_dir():
        print(f"  Not a directory: {p}")
        return None
    return p


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    print("File Watcher")
    saved_snapshots: dict[str, dict] = {}

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            root = get_dir()
            if not root:
                continue
            interval_s = input("  Poll interval in seconds (default 1.0): ").strip()
            try:
                interval = float(interval_s) if interval_s else 1.0
            except ValueError:
                interval = 1.0
            rec     = input("  Recursive? (y/n, default y): ").strip().lower() != "n"
            pattern = input("  Filename filter (blank=all): ").strip() or None
            log_s   = input("  Log file path (blank=none): ").strip().strip('"')
            log_p   = Path(log_s) if log_s else None
            max_s   = input("  Max events before stopping (0=infinite): ").strip()
            max_ev  = int(max_s) if max_s.isdigit() else 0

            watch(root, interval, rec, pattern, log_p, max_ev)

        elif choice == "2":
            root = get_dir("  Directory to snapshot: ")
            if not root:
                continue
            rec  = input("  Recursive? (y/n, default y): ").strip().lower() != "n"
            snap = snapshot(root, rec)
            name = input("  Save snapshot as (default A): ").strip() or "A"
            saved_snapshots[name] = snap
            print(f"  Snapshot '{name}': {len(snap)} files")

        elif choice == "3":
            if len(saved_snapshots) < 2:
                print("  Need at least 2 snapshots. Use option 2 to create them.")
                continue
            print(f"  Available snapshots: {', '.join(saved_snapshots)}")
            a = input("  First snapshot name: ").strip()
            b = input("  Second snapshot name: ").strip()
            if a not in saved_snapshots or b not in saved_snapshots:
                print("  Snapshot not found.")
                continue
            events = diff_snapshots(saved_snapshots[a], saved_snapshots[b])
            if not events:
                print("  No differences found.")
            else:
                print(f"\n  {len(events)} difference(s):")
                for etype, path in events:
                    print(f"  {etype}  {path}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
