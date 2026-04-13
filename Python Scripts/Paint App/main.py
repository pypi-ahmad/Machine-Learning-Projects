"""Paint App — Tkinter desktop app.

A simple drawing application with pencil, eraser, shapes,
fill bucket, color picker, and brush size control.

Usage:
    python main.py
"""

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox


class PaintApp(tk.Tk):
    TOOLS = ["pencil", "eraser", "line", "rect", "oval", "fill", "text"]

    def __init__(self):
        super().__init__()
        self.title("Paint App")
        self.geometry("1000x680")
        self.configure(bg="#2b2d3a")

        self._color     = "#ffffff"
        self._bg_color  = "#000000"
        self._tool      = tk.StringVar(value="pencil")
        self._size      = tk.IntVar(value=3)
        self._last_x    = None
        self._last_y    = None
        self._start_xy  = None
        self._preview   = None

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        tb = tk.Frame(self, bg="#1e2030", height=48)
        tb.pack(fill="x")

        for tool in self.TOOLS:
            tk.Radiobutton(
                tb, text=tool.title(), variable=self._tool, value=tool,
                bg="#1e2030", fg="#cdd6f4", selectcolor="#363a4f",
                activebackground="#1e2030", activeforeground="#cba6f7",
                font=("Consolas", 10), indicatoron=False, width=7,
            ).pack(side="left", padx=2, pady=6)

        tk.Label(tb, text="Size:", bg="#1e2030", fg="#888").pack(side="left", padx=(12, 2))
        tk.Scale(tb, from_=1, to=40, orient="horizontal", variable=self._size,
                 bg="#1e2030", fg="#cdd6f4", troughcolor="#363a4f",
                 highlightthickness=0, length=100).pack(side="left")

        # Color swatches
        self._fg_swatch = tk.Label(tb, bg=self._color, width=3, relief="ridge",
                                    cursor="hand2")
        self._fg_swatch.pack(side="left", padx=(14, 2), pady=8, ipady=8)
        self._fg_swatch.bind("<Button-1>", self._pick_fg)

        self._bg_swatch = tk.Label(tb, bg=self._bg_color, width=3, relief="ridge",
                                    cursor="hand2")
        self._bg_swatch.pack(side="left", padx=2, pady=8, ipady=8)
        self._bg_swatch.bind("<Button-1>", self._pick_bg)

        tk.Label(tb, text="FG / BG", bg="#1e2030", fg="#888", font=("Consolas", 8)).pack(side="left")

        # Action buttons
        for lbl, cmd in [("Clear", self._clear), ("Save", self._save), ("New", self._new)]:
            tk.Button(tb, text=lbl, command=cmd, bg="#363a4f", fg="#cdd6f4",
                      relief="flat", padx=8).pack(side="right", pady=8, padx=4)

        # Palette row
        pal = tk.Frame(self, bg="#1e2030")
        pal.pack(fill="x")
        PALETTE = [
            "#ffffff","#c0c0c0","#808080","#000000",
            "#ff0000","#ff7f00","#ffff00","#00ff00",
            "#00ffff","#0000ff","#8b00ff","#ff00ff",
            "#ffb3ba","#ffdfba","#ffffba","#baffc9",
            "#bae1ff","#f0e68c","#dda0dd","#98fb98",
        ]
        for color in PALETTE:
            b = tk.Label(pal, bg=color, width=2, cursor="hand2", relief="ridge")
            b.pack(side="left", padx=1, pady=2, ipady=6)
            b.bind("<Button-1>", lambda e, c=color: self._set_color(c))
            b.bind("<Button-3>", lambda e, c=color: self._set_bg(c))

        # Canvas
        self._canvas = tk.Canvas(self, bg=self._bg_color, cursor="crosshair",
                                  highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

        # Status bar
        self._status = tk.StringVar(value="x=0, y=0")
        tk.Label(self, textvariable=self._status, bg="#1e2030", fg="#555",
                 anchor="w", padx=8).pack(fill="x", side="bottom")
        self._canvas.bind("<Motion>", self._on_motion)

    # ── Color pickers ────────────────────────────────────────────────────────

    def _pick_fg(self, _=None):
        c = colorchooser.askcolor(color=self._color, title="Foreground Color")
        if c and c[1]:
            self._color = c[1]
            self._fg_swatch.config(bg=self._color)

    def _pick_bg(self, _=None):
        c = colorchooser.askcolor(color=self._bg_color, title="Background Color")
        if c and c[1]:
            self._bg_color = c[1]
            self._bg_swatch.config(bg=self._bg_color)

    def _set_color(self, c):
        self._color = c
        self._fg_swatch.config(bg=c)

    def _set_bg(self, c):
        self._bg_color = c
        self._bg_swatch.config(bg=c)

    # ── Mouse handlers ────────────────────────────────────────────────────────

    def _on_press(self, e):
        self._last_x, self._last_y = e.x, e.y
        self._start_xy = (e.x, e.y)
        tool = self._tool.get()
        if tool == "fill":
            self._flood_fill(e.x, e.y)
        elif tool == "text":
            self._place_text(e.x, e.y)

    def _on_drag(self, e):
        tool = self._tool.get()
        sz   = self._size.get()
        if tool == "pencil":
            self._canvas.create_line(self._last_x, self._last_y, e.x, e.y,
                                      fill=self._color, width=sz,
                                      capstyle="round", smooth=True)
            self._last_x, self._last_y = e.x, e.y
        elif tool == "eraser":
            half = sz * 2
            self._canvas.create_rectangle(e.x - half, e.y - half,
                                           e.x + half, e.y + half,
                                           fill=self._bg_color, outline="")
            self._last_x, self._last_y = e.x, e.y
        elif tool in ("line", "rect", "oval"):
            # Remove preview
            if self._preview:
                self._canvas.delete(self._preview)
            x0, y0 = self._start_xy
            if tool == "line":
                self._preview = self._canvas.create_line(x0, y0, e.x, e.y,
                                                          fill=self._color, width=sz)
            elif tool == "rect":
                self._preview = self._canvas.create_rectangle(x0, y0, e.x, e.y,
                                                               outline=self._color, width=sz)
            elif tool == "oval":
                self._preview = self._canvas.create_oval(x0, y0, e.x, e.y,
                                                         outline=self._color, width=sz)

    def _on_release(self, e):
        self._preview = None

    def _on_motion(self, e):
        self._status.set(f"x={e.x}, y={e.y}  |  tool={self._tool.get()}  |  size={self._size.get()}")

    # ── Special tools ────────────────────────────────────────────────────────

    def _flood_fill(self, x: int, y: int):
        # Simple bounding-box fill for shapes — real flood fill needs image data
        items = self._canvas.find_closest(x, y)
        if items:
            self._canvas.itemconfig(items[0], fill=self._color)

    def _place_text(self, x: int, y: int):
        t = tk.simpledialog_text = tk.simpledialog.askstring if hasattr(tk, "simpledialog") else None
        try:
            from tkinter import simpledialog
            txt = simpledialog.askstring("Text", "Enter text:", parent=self)
            if txt:
                self._canvas.create_text(x, y, text=txt, fill=self._color,
                                          font=("Consolas", self._size.get() * 3 + 8))
        except Exception:
            pass

    # ── File operations ────────────────────────────────────────────────────────

    def _clear(self):
        if messagebox.askyesno("Clear", "Clear the canvas?"):
            self._canvas.delete("all")

    def _new(self):
        if messagebox.askyesno("New", "Start a new canvas?"):
            self._canvas.delete("all")
            self._canvas.config(bg=self._bg_color)

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".ps",
                                             filetypes=[("PostScript", "*.ps"),
                                                        ("All files", "*.*")])
        if path:
            self._canvas.postscript(file=path)
            messagebox.showinfo("Saved", f"Saved to {path} (PostScript format).")


if __name__ == "__main__":
    from tkinter import simpledialog
    app = PaintApp()
    app.mainloop()
