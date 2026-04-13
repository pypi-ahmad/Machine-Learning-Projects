"""Dice Simulator — Tkinter GUI.

Visual dice roller with animated roll, multiple dice types,
roll history, and statistics panel.

Usage:
    python main.py
"""

import random
import tkinter as tk
from tkinter import ttk
from collections import Counter


# Dot positions (normalized 0–1 on a unit square) for d6 faces 1–6
D6_DOTS = {
    1: [(0.5, 0.5)],
    2: [(0.25, 0.25), (0.75, 0.75)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
    6: [(0.25, 0.2), (0.75, 0.2), (0.25, 0.5), (0.75, 0.5), (0.25, 0.8), (0.75, 0.8)],
}

DIE_SIZE  = 90
DOT_R     = 7
PAD       = 8
BG        = "#1e293b"
DIE_BG    = "#f8fafc"
DOT_CLR   = "#1e293b"
ACCENT    = "#6366f1"


class DiceFace(tk.Canvas):
    """Canvas that renders a single d6 face."""

    def __init__(self, parent, size=DIE_SIZE, **kwargs):
        super().__init__(parent, width=size, height=size,
                         bg=BG, highlightthickness=0, **kwargs)
        self.size  = size
        self.value = 1
        self._anim_step = 0
        self._anim_id   = None
        self.draw(1)

    def draw(self, value: int):
        self.value = value
        self.delete("all")
        s = self.size
        # Die body
        self.create_rectangle(PAD, PAD, s-PAD, s-PAD,
                               fill=DIE_BG, outline=ACCENT, width=3,
                               tags="die")
        # Dots
        for nx, ny in D6_DOTS.get(value, []):
            cx = PAD + (s - 2*PAD) * nx
            cy = PAD + (s - 2*PAD) * ny
            self.create_oval(cx-DOT_R, cy-DOT_R, cx+DOT_R, cy+DOT_R,
                             fill=DOT_CLR, outline="", tags="dot")

    def animate_roll(self, final: int, steps: int = 8, delay: int = 60):
        if self._anim_id: self.after_cancel(self._anim_id)
        self._anim_step = 0
        self._final     = final
        self._steps     = steps
        self._delay     = delay

        def step():
            self._anim_step += 1
            if self._anim_step < self._steps:
                self.draw(random.randint(1, 6))
                self._anim_id = self.after(self._delay, step)
            else:
                self.draw(self._final)
                self._anim_id = None
        step()


class DiceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Dice Simulator")
        root.configure(bg=BG)
        root.resizable(False, False)

        self.history: list[tuple] = []
        self._build_ui()

    def _build_ui(self):
        # Controls row
        ctrl = tk.Frame(self.root, bg=BG)
        ctrl.pack(padx=15, pady=12, fill=tk.X)

        tk.Label(ctrl, text="Number of dice:", fg="white", bg=BG,
                 font=("Segoe UI", 11)).pack(side=tk.LEFT)
        self.n_dice_var = tk.IntVar(value=2)
        ttk.Spinbox(ctrl, from_=1, to=6, textvariable=self.n_dice_var, width=4,
                    font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=6)

        tk.Button(ctrl, text="🎲  Roll", command=self.roll,
                  bg=ACCENT, fg="white", font=("Segoe UI", 12, "bold"),
                  relief=tk.FLAT, padx=14, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=12)

        self.total_var = tk.StringVar(value="Total: —")
        tk.Label(ctrl, textvariable=self.total_var, fg="#94a3b8", bg=BG,
                 font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=10)

        # Dice canvas area
        self.dice_frame = tk.Frame(self.root, bg=BG)
        self.dice_frame.pack(padx=15, pady=6)
        self.dice_faces: list[DiceFace] = []
        self._rebuild_dice(2)

        # Modifier
        mod_row = tk.Frame(self.root, bg=BG)
        mod_row.pack(pady=4)
        tk.Label(mod_row, text="Modifier:", fg="white", bg=BG, font=("Segoe UI", 11)).pack(side=tk.LEFT)
        self.modifier_var = tk.IntVar(value=0)
        ttk.Spinbox(mod_row, from_=-20, to=20, textvariable=self.modifier_var,
                    width=5, font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=6)

        # Stats and history
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(padx=15, pady=8, fill=tk.BOTH)

        # History
        hist_frame = tk.LabelFrame(bottom, text=" Roll History ", fg=ACCENT,
                                   bg=BG, font=("Segoe UI", 10, "bold"))
        hist_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,8))
        self.hist_box = tk.Listbox(hist_frame, width=22, height=12,
                                   bg="#0f172a", fg="#e2e8f0",
                                   font=("Courier", 10), selectmode=tk.SINGLE,
                                   highlightthickness=0, relief=tk.FLAT)
        self.hist_box.pack(padx=4, pady=4, fill=tk.BOTH, expand=True)
        tk.Button(hist_frame, text="Clear", command=self.clear_history,
                  bg="#334155", fg="white", relief=tk.FLAT,
                  font=("Segoe UI", 9)).pack(pady=(0,4))

        # Stats
        stat_frame = tk.LabelFrame(bottom, text=" Statistics ", fg=ACCENT,
                                   bg=BG, font=("Segoe UI", 10, "bold"))
        stat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.stat_var = tk.StringVar(value="No rolls yet.")
        tk.Label(stat_frame, textvariable=self.stat_var, fg="#94a3b8", bg=BG,
                 font=("Courier", 10), justify=tk.LEFT).pack(padx=8, pady=8, anchor=tk.W)

        # Watch n_dice changes
        self.n_dice_var.trace_add("write", lambda *_: self._rebuild_dice(self.n_dice_var.get()))

    def _rebuild_dice(self, n: int):
        for w in self.dice_frame.winfo_children():
            w.destroy()
        self.dice_faces = []
        for _ in range(n):
            df = DiceFace(self.dice_frame)
            df.pack(side=tk.LEFT, padx=6, pady=6)
            self.dice_faces.append(df)

    def roll(self):
        n       = self.n_dice_var.get()
        mod     = self.modifier_var.get()
        self._rebuild_dice(n)
        results = [random.randint(1, 6) for _ in range(n)]
        total   = sum(results) + mod

        for face, val in zip(self.dice_faces, results):
            face.animate_roll(val)

        mod_str = f"{mod:+d}" if mod else ""
        self.total_var.set(f"Total: {total}" + (f"  ({'+'.join(map(str,results))}{mod_str})" if n>1 or mod else ""))
        self.history.append(tuple(results))
        self._update_history(results, total, mod_str)
        self._update_stats()

    def _update_history(self, results, total, mod_str):
        entry = f"[{', '.join(map(str,results))}]{mod_str} = {total}"
        self.hist_box.insert(0, entry)
        if self.hist_box.size() > 50:
            self.hist_box.delete(50, tk.END)

    def _update_stats(self):
        all_vals = [v for roll in self.history for v in roll]
        if not all_vals:
            self.stat_var.set("No rolls yet.")
            return
        counts  = Counter(all_vals)
        total   = len(all_vals)
        avg     = sum(all_vals) / total
        mn, mx  = min(all_vals), max(all_vals)
        lines   = [f"Total rolls: {len(self.history)}",
                   f"Total dice:  {total}",
                   f"Average:     {avg:.2f}",
                   f"Min / Max:   {mn} / {mx}",
                   "",
                   "Face  Count  Freq"]
        for face in range(1, 7):
            c = counts.get(face, 0)
            bar = "█" * int(c / total * 15)
            lines.append(f"  {face}     {c:>4}   {bar}")
        self.stat_var.set("\n".join(lines))

    def clear_history(self):
        self.history.clear()
        self.hist_box.delete(0, tk.END)
        self.stat_var.set("No rolls yet.")
        self.total_var.set("Total: —")


def main():
    root = tk.Tk()
    DiceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
