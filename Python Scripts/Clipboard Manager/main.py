"""Clipboard Manager — CLI tool.

Maintains a history of clipboard entries.
Copy, paste, search, pin, and export clipboard items.
Uses pyperclip for cross-platform clipboard access.

Usage:
    python main.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

DATA_FILE = Path(__file__).with_name("clipboard_history.json")
EXPORT_FILE = Path(__file__).with_name("clipboard_export.json")
MAX_HISTORY = 100


def load_history() -> list[dict]:
    try:
        history = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    return history if isinstance(history, list) else []


def save_history(history: list[dict]):
    DATA_FILE.write_text(
        json.dumps(history[-MAX_HISTORY:], indent=2), encoding="utf-8"
    )


def entry_at(history: list[dict], position: int) -> dict:
    """Return an entry for a one-based displayed position."""
    if position < 1 or position > len(history):
        raise IndexError(position)
    return history[position - 1]


def add_entry(history: list[dict], text: str, pinned: bool = False) -> bool:
    if not text.strip():
        return False
    if history and history[-1]["text"] == text:
        return False   # no duplicate
    history.append({
        "id":      len(history) + 1,
        "text":    text,
        "preview": text[:80].replace("\n", "↵"),
        "time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pinned":  pinned,
        "chars":   len(text),
    })
    return True


def print_entry(e: dict, index: int):
    pin = "📌" if e["pinned"] else "  "
    print(f"  {pin} {index:>3}. [{e['time']}]  {e['preview']}")


MENU = """
Clipboard Manager
──────────────────────────────
  list [n]        Show last n entries (default 20)
  copy  <id>      Copy entry to clipboard
  del   <id>      Delete entry
  pin   <id>      Toggle pin
  find  <text>    Search entries
  add   <text>    Manually add entry
  clear           Clear unpinned history
  watch           Auto-watch clipboard (Ctrl-C to stop)
  export          Export history to JSON
  q               Quit
──────────────────────────────"""


def main():
    if not HAS_CLIPBOARD:
        print("  pyperclip not installed. Run: pip install pyperclip")
        print("  Running in manual-entry mode only.\n")

    history = load_history()
    print(f"  Loaded {len(history)} clipboard entries.")

    while True:
        print(MENU)
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue
        parts = raw.split(None, 1)
        cmd   = parts[0].lower()
        arg   = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("q", "quit"):
            print("Bye!")
            break

        elif cmd == "list":
            n = int(arg) if arg.isdigit() else 20
            shown = history[-n:]
            if not shown:
                print("  (empty)")
            else:
                for i, e in enumerate(reversed(shown), 1):
                    print_entry(e, len(history) - i + 1)

        elif cmd == "copy":
            if not HAS_CLIPBOARD:
                print("  pyperclip not available.")
                continue
            try:
                idx = int(arg) - 1
                entry = entry_at(history, idx + 1)
                pyperclip.copy(entry["text"])
                print(f"  Copied: {entry['preview']}")
            except (ValueError, IndexError):
                print("  Invalid entry number.")

        elif cmd == "del":
            try:
                idx = int(arg) - 1
                entry_at(history, idx + 1)
                removed = history.pop(idx)
                save_history(history)
                print(f"  Deleted: {removed['preview']}")
            except (ValueError, IndexError):
                print("  Invalid entry number.")

        elif cmd == "pin":
            try:
                idx = int(arg) - 1
                entry = entry_at(history, idx + 1)
                entry["pinned"] = not entry["pinned"]
                save_history(history)
                state = "Pinned" if entry["pinned"] else "Unpinned"
                print(f"  {state}: {entry['preview']}")
            except (ValueError, IndexError):
                print("  Invalid entry number.")

        elif cmd == "find":
            if not arg:
                print("  Usage: find <text>")
                continue
            results = [e for e in history if arg.lower() in e["text"].lower()]
            if not results:
                print("  No matches.")
            else:
                for i, e in enumerate(results, 1):
                    print_entry(e, i)

        elif cmd == "add":
            if add_entry(history, arg):
                save_history(history)
                print(f"  Added: {arg[:60]}")
            else:
                print("  Empty or duplicate.")

        elif cmd == "clear":
            before = len(history)
            history = [e for e in history if e["pinned"]]
            save_history(history)
            print(f"  Cleared {before - len(history)} unpinned entries.")

        elif cmd == "watch":
            if not HAS_CLIPBOARD:
                print("  pyperclip not available.")
                continue
            print("  Watching clipboard… (Ctrl-C to stop)")
            last = pyperclip.paste()
            try:
                while True:
                    time.sleep(1)
                    current = pyperclip.paste()
                    if current != last:
                        if add_entry(history, current):
                            save_history(history)
                            print(f"  Captured: {current[:60]}")
                        last = current
            except KeyboardInterrupt:
                print("\n  Stopped watching.")

        elif cmd == "export":
            EXPORT_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"  Exported {len(history)} entries to {EXPORT_FILE}")

        else:
            print(f"  Unknown command: {cmd}")


if __name__ == "__main__":
    main()
