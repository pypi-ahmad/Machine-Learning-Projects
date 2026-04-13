"""Notes App — CLI tool.

A simple note-taking app that saves notes as individual text files.
Supports adding, viewing, editing, searching, and deleting notes.

Usage:
    python main.py
"""

import os
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOTES_DIR = Path(__file__).parent / "notes"


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def ensure_notes_dir() -> None:
    NOTES_DIR.mkdir(exist_ok=True)


def list_notes() -> list[Path]:
    ensure_notes_dir()
    return sorted(NOTES_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)


def note_filename(title: str) -> str:
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
    safe = safe.strip().replace(" ", "_")
    return f"{safe}.txt"


def save_note(title: str, content: str) -> Path:
    ensure_notes_dir()
    path = NOTES_DIR / note_filename(title)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(f"Title: {title}\nSaved: {timestamp}\n\n{content}", encoding="utf-8")
    return path


def read_note(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def delete_note(path: Path) -> None:
    path.unlink()


def search_notes(query: str) -> list[Path]:
    query_lower = query.lower()
    results = []
    for note in list_notes():
        content = note.read_text(encoding="utf-8").lower()
        if query_lower in content:
            results.append(note)
    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def note_preview(path: Path, max_chars: int = 80) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    title_line = lines[0] if lines else path.stem
    # get content after blank line
    body = ""
    for i, line in enumerate(lines):
        if i > 2 and line.strip():
            body = line[:max_chars]
            break
    return f"{title_line}  |  {body}..."


def pick_note(notes: list[Path]) -> Path | None:
    if not notes:
        print("  No notes found.")
        return None
    print()
    for i, note in enumerate(notes, 1):
        mtime = datetime.datetime.fromtimestamp(note.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        title = note.stem.replace("_", " ")
        print(f"  {i:>3}. [{mtime}] {title}")
    choice = input("\n  Choose note number (0 to cancel): ").strip()
    if choice == "0" or not choice.isdigit():
        return None
    idx = int(choice) - 1
    if 0 <= idx < len(notes):
        return notes[idx]
    print("  Invalid number.")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Notes App
---------
1. Add new note
2. View all notes
3. Read a note
4. Edit a note
5. Delete a note
6. Search notes
0. Quit
"""


def add_note() -> None:
    title = input("  Note title: ").strip()
    if not title:
        print("  Title cannot be empty.")
        return
    print("  Enter note content (type '###' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line == "###":
            break
        lines.append(line)
    content = "\n".join(lines)
    path = save_note(title, content)
    print(f"  Saved: {path.name}")


def view_all() -> None:
    notes = list_notes()
    if not notes:
        print("  No notes yet.")
        return
    print(f"\n  {len(notes)} note(s):\n")
    for i, note in enumerate(notes, 1):
        mtime = datetime.datetime.fromtimestamp(note.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        title = note.stem.replace("_", " ")
        size = note.stat().st_size
        print(f"  {i:>3}. {title:<35} {mtime}  {size}b")


def read_note_cmd() -> None:
    notes = list_notes()
    path = pick_note(notes)
    if path:
        print(f"\n  {'─' * 50}")
        print(read_note(path))
        print(f"  {'─' * 50}")


def edit_note_cmd() -> None:
    notes = list_notes()
    path = pick_note(notes)
    if not path:
        return
    existing = read_note(path)
    print(f"\n  Current content:\n  {'─' * 40}")
    print(existing)
    print(f"  {'─' * 40}")
    print("  Enter new content (type '###' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line == "###":
            break
        lines.append(line)
    # Extract original title
    first_line = existing.splitlines()[0]
    title = first_line.replace("Title: ", "").strip()
    save_note(title, "\n".join(lines))
    path.unlink(missing_ok=True)
    print(f"  Updated: {title}")


def delete_note_cmd() -> None:
    notes = list_notes()
    path = pick_note(notes)
    if not path:
        return
    confirm = input(f"  Delete '{path.stem}'? (y/n): ").strip().lower()
    if confirm == "y":
        delete_note(path)
        print("  Deleted.")
    else:
        print("  Cancelled.")


def search_notes_cmd() -> None:
    query = input("  Search query: ").strip()
    if not query:
        return
    results = search_notes(query)
    if not results:
        print(f"  No notes matching '{query}'.")
        return
    print(f"\n  Found {len(results)} note(s):")
    path = pick_note(results)
    if path:
        print(f"\n  {'─' * 50}")
        print(read_note(path))
        print(f"  {'─' * 50}")


def main() -> None:
    ensure_notes_dir()
    while True:
        print(MENU)
        choice = input("Choice: ").strip()
        if choice == "0":
            print("Bye!")
            break
        elif choice == "1":
            add_note()
        elif choice == "2":
            view_all()
        elif choice == "3":
            read_note_cmd()
        elif choice == "4":
            edit_note_cmd()
        elif choice == "5":
            delete_note_cmd()
        elif choice == "6":
            search_notes_cmd()
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
