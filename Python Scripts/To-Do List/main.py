"""To-Do List — CLI tool.

Add, complete, delete, and search tasks with priorities and due dates.
Data persisted to JSON.

Usage:
    python main.py
"""

import json
from datetime import date, datetime
from pathlib import Path

DATA_FILE = Path("todos.json")
PRIORITIES = {"h": "High", "m": "Medium", "l": "Low"}


def load() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save(todos: list[dict]):
    DATA_FILE.write_text(json.dumps(todos, indent=2))


def next_id(todos: list[dict]) -> int:
    return max((t["id"] for t in todos), default=0) + 1


def format_task(t: dict, idx: int | None = None) -> str:
    check  = "✓" if t["done"] else "○"
    pri    = {"High": "!", "Medium": "-", "Low": " "}.get(t["priority"], " ")
    due    = f"  due:{t['due']}" if t.get("due") else ""
    tags   = f"  [{','.join(t['tags'])}]" if t.get("tags") else ""
    num    = f"{t['id']:>3}." if idx is None else f"{idx:>3}."
    return f"  {num} [{check}] {pri} {t['title']}{due}{tags}"


MENU = """
To-Do List
──────────────────────────────
  add   <title>             Add task
  done  <id>                Mark complete
  del   <id>                Delete task
  list  [all|done|todo]     List tasks
  pri   <id> <h|m|l>        Set priority
  due   <id> <YYYY-MM-DD>   Set due date
  tag   <id> <tag>          Add tag
  find  <keyword>           Search tasks
  clear                     Delete all done tasks
  q                         Quit
──────────────────────────────"""


def main():
    todos = load()

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

        elif cmd == "add":
            if not arg:
                print("  Usage: add <title>")
                continue
            todos.append({
                "id": next_id(todos), "title": arg,
                "done": False, "priority": "Medium",
                "due": None, "tags": [],
                "created": str(date.today()),
            })
            save(todos)
            print(f"  Added: {arg}")

        elif cmd == "done":
            try:
                tid = int(arg)
                t   = next((x for x in todos if x["id"] == tid), None)
                if t:
                    t["done"] = not t["done"]
                    save(todos)
                    print(f"  {'Completed' if t['done'] else 'Reopened'}: {t['title']}")
                else:
                    print(f"  Task {tid} not found.")
            except ValueError:
                print("  Usage: done <id>")

        elif cmd == "del":
            try:
                tid   = int(arg)
                before = len(todos)
                todos  = [x for x in todos if x["id"] != tid]
                save(todos)
                print(f"  {'Deleted.' if len(todos) < before else 'Not found.'}")
            except ValueError:
                print("  Usage: del <id>")

        elif cmd == "list":
            mode = arg.lower() if arg else "todo"
            if mode == "all":
                shown = todos
            elif mode == "done":
                shown = [t for t in todos if t["done"]]
            else:
                shown = [t for t in todos if not t["done"]]
            if not shown:
                print("  (empty)")
            else:
                # Sort: high priority first, then by due date
                priority_order = {"High": 0, "Medium": 1, "Low": 2}
                shown.sort(key=lambda t: (priority_order.get(t["priority"], 1), t.get("due") or "9"))
                for t in shown:
                    print(format_task(t))

        elif cmd == "pri":
            sub = arg.split()
            if len(sub) != 2 or sub[1].lower() not in PRIORITIES:
                print("  Usage: pri <id> <h|m|l>")
                continue
            tid = int(sub[0])
            t   = next((x for x in todos if x["id"] == tid), None)
            if t:
                t["priority"] = PRIORITIES[sub[1].lower()]
                save(todos)
                print(f"  Priority set to {t['priority']}: {t['title']}")
            else:
                print(f"  Task {tid} not found.")

        elif cmd == "due":
            sub = arg.split()
            if len(sub) != 2:
                print("  Usage: due <id> <YYYY-MM-DD>")
                continue
            try:
                tid = int(sub[0])
                datetime.strptime(sub[1], "%Y-%m-%d")  # validate
                t   = next((x for x in todos if x["id"] == tid), None)
                if t:
                    t["due"] = sub[1]
                    save(todos)
                    print(f"  Due date set: {t['title']} → {sub[1]}")
                else:
                    print(f"  Task {tid} not found.")
            except ValueError:
                print("  Invalid date. Use YYYY-MM-DD.")

        elif cmd == "tag":
            sub = arg.split(None, 1)
            if len(sub) != 2:
                print("  Usage: tag <id> <tag>")
                continue
            tid = int(sub[0])
            t   = next((x for x in todos if x["id"] == tid), None)
            if t:
                if sub[1] not in t["tags"]:
                    t["tags"].append(sub[1])
                    save(todos)
                print(f"  Tagged: {t['title']} → {t['tags']}")
            else:
                print(f"  Task {tid} not found.")

        elif cmd == "find":
            if not arg:
                print("  Usage: find <keyword>")
                continue
            found = [t for t in todos if arg.lower() in t["title"].lower()
                     or arg.lower() in " ".join(t.get("tags", []))]
            if not found:
                print("  No matching tasks.")
            else:
                for t in found:
                    print(format_task(t))

        elif cmd == "clear":
            before = len(todos)
            todos  = [t for t in todos if not t["done"]]
            save(todos)
            print(f"  Cleared {before - len(todos)} completed task(s).")

        else:
            print(f"  Unknown command: '{cmd}'")


if __name__ == "__main__":
    main()
