"""Sticky Notes — Tkinter app.

Create resizable, coloured sticky notes that persist between sessions.
Drag notes anywhere on screen, double-click to edit, right-click for options.

Usage:
    python main.py
"""

import json
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, font as tkfont

DATA_FILE = Path("sticky_notes.json")

NOTE_COLORS = ["#fff176", "#a5d6a7", "#ef9a9a", "#90caf9", "#ffcc80", "#ce93d8", "#80deea"]

DEFAULT_W, DEFAULT_H = 220, 180


def load_notes() -> list[dict]:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return []


def save_notes(notes: list[dict]) -> None:
    DATA_FILE.write_text(json.dumps(notes, indent=2))


# ---------------------------------------------------------------------------
# StickyNote window
# ---------------------------------------------------------------------------

class StickyNote(tk.Toplevel):
    def __init__(self, master, note_id: int, x: int, y: int,
                 w: int, h: int, color: str, text: str, manager):
        super().__init__(master)
        self.note_id = note_id
        self.manager = manager
        self.color   = color

        self.overrideredirect(True)   # no title bar
        self.attributes("-topmost", True)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.configure(bg=color)

        self._build(text)
        self._bind_drag()

    def _build(self, text: str):
        # Title bar
        self.bar = tk.Frame(self, bg=self._darker(), height=22, cursor="fleur")
        self.bar.pack(fill="x")
        self.bar.pack_propagate(False)

        tk.Button(self.bar, text="✕", bg=self._darker(), fg="#555", bd=0,
                  font=("Segoe UI", 9), command=self.close_note).pack(side="right", padx=4)
        tk.Button(self.bar, text="⊕", bg=self._darker(), fg="#555", bd=0,
                  font=("Segoe UI", 9), command=self.manager.new_note).pack(side="right")

        # Text area
        note_font = tkfont.Font(family="Segoe UI", size=11)
        self.text = tk.Text(self, bg=self.color, bd=0, wrap="word",
                            font=note_font, relief="flat", padx=6, pady=4)
        self.text.insert("1.0", text)
        self.text.pack(fill="both", expand=True)
        self.text.bind("<KeyRelease>", lambda _: self.manager.save())
        self.text.bind("<Button-3>", self._context_menu)

    def _darker(self) -> str:
        r = int(self.color[1:3], 16)
        g = int(self.color[3:5], 16)
        b = int(self.color[5:7], 16)
        factor = 0.85
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

    def _bind_drag(self):
        self.bar.bind("<ButtonPress-1>",   self._drag_start)
        self.bar.bind("<B1-Motion>",       self._drag_motion)
        self.bar.bind("<ButtonRelease-1>", lambda _: self.manager.save())

    def _drag_start(self, event):
        self._drag_x = event.x_root - self.winfo_x()
        self._drag_y = event.y_root - self.winfo_y()

    def _drag_motion(self, event):
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self.geometry(f"+{x}+{y}")

    def _context_menu(self, event):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Change color…", command=self._change_color)
        menu.add_command(label="New note",       command=self.manager.new_note)
        menu.add_separator()
        menu.add_command(label="Delete note",    command=self.close_note)
        menu.tk_popup(event.x_root, event.y_root)

    def _change_color(self):
        color = colorchooser.askcolor(title="Choose note color",
                                       initialcolor=self.color, parent=self)
        if color and color[1]:
            self.color = color[1]
            self.configure(bg=self.color)
            self.text.configure(bg=self.color)
            self.bar.configure(bg=self._darker())
            self.manager.save()

    def get_data(self) -> dict:
        return {
            "id":    self.note_id,
            "x":     self.winfo_x(),
            "y":     self.winfo_y(),
            "w":     self.winfo_width(),
            "h":     self.winfo_height(),
            "color": self.color,
            "text":  self.text.get("1.0", "end-1c"),
        }

    def close_note(self):
        self.manager.remove_note(self.note_id)
        self.destroy()


# ---------------------------------------------------------------------------
# Manager (hidden root window)
# ---------------------------------------------------------------------------

class NoteManager:
    def __init__(self):
        self.root  = tk.Tk()
        self.root.withdraw()              # hide the root
        self.root.title("Sticky Notes")
        self.notes: list[StickyNote] = []
        self._next_id = 0

        existing = load_notes()
        if existing:
            for n in existing:
                self._spawn(n["x"], n["y"], n["w"], n["h"],
                             n["color"], n["text"])
        else:
            self.new_note()              # start with one blank note

        # System tray via task bar icon approach: just a small control window
        self._control_win()
        self.root.mainloop()

    def _control_win(self):
        ctrl = tk.Toplevel(self.root)
        ctrl.title("Sticky Notes")
        ctrl.geometry("200x80+20+20")
        ctrl.resizable(False, False)
        ctrl.protocol("WM_DELETE_WINDOW", self._quit)
        tk.Button(ctrl, text="➕  New Note", command=self.new_note,
                  font=("Segoe UI", 11), bg="#fff176",
                  relief="flat", padx=10, pady=6).pack(fill="x", padx=12, pady=(12,4))
        tk.Button(ctrl, text="✕  Quit", command=self._quit,
                  font=("Segoe UI", 9), relief="flat").pack(fill="x", padx=12)

    def _spawn(self, x, y, w, h, color, text):
        note = StickyNote(self.root, self._next_id, x, y, w, h, color, text, self)
        self.notes.append(note)
        self._next_id += 1
        return note

    def new_note(self):
        import random
        color = NOTE_COLORS[len(self.notes) % len(NOTE_COLORS)]
        x = 100 + len(self.notes) * 30
        y = 100 + len(self.notes) * 30
        self._spawn(x, y, DEFAULT_W, DEFAULT_H, color, "")
        self.save()

    def remove_note(self, note_id: int):
        self.notes = [n for n in self.notes if n.note_id != note_id]
        self.save()

    def save(self):
        data = [n.get_data() for n in self.notes if n.winfo_exists()]
        save_notes(data)

    def _quit(self):
        self.save()
        self.root.quit()


def main():
    NoteManager()


if __name__ == "__main__":
    main()
