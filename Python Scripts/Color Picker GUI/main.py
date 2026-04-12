"""Color Picker GUI — Tkinter app.

Pick colors via RGB sliders or hex input.
Displays HEX, RGB, HSL, HSV, and CMYK values.
Save a palette of favourite colors.

Usage:
    python main.py
"""

import colorsys
import tkinter as tk
from tkinter import font as tkfont, messagebox


# ---------------------------------------------------------------------------
# Color conversion helpers
# ---------------------------------------------------------------------------

def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int] | None:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return None


def rgb_to_hsl(r: int, g: int, b: int) -> tuple[int, int, int]:
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    return int(h * 360), int(s * 100), int(l * 100)


def rgb_to_hsv(r: int, g: int, b: int) -> tuple[int, int, int]:
    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    return int(h * 360), int(s * 100), int(v * 100)


def rgb_to_cmyk(r: int, g: int, b: int) -> tuple[int, int, int, int]:
    if r == g == b == 0:
        return 0, 0, 0, 100
    r_, g_, b_ = r/255, g/255, b/255
    k = 1 - max(r_, g_, b_)
    c = (1 - r_ - k) / (1 - k)
    m = (1 - g_ - k) / (1 - k)
    y = (1 - b_ - k) / (1 - k)
    return int(c*100), int(m*100), int(y*100), int(k*100)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class ColorPickerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Color Picker")
        self.resizable(False, False)
        self.configure(bg="#f5f5f5")
        self._palette: list[str] = []
        self._build_ui()
        self._update_from_rgb()

    def _build_ui(self):
        lf = tkfont.Font(family="Segoe UI", size=10)
        mf = tkfont.Font(family="Consolas", size=11)

        # Color preview
        self.preview = tk.Frame(self, width=320, height=80, bg="#ff0000")
        self.preview.grid(row=0, column=0, columnspan=2, padx=20, pady=(16, 8), sticky="ew")
        self.hex_display = tk.Label(self.preview, text="#ff0000", font=tkfont.Font(family="Consolas", size=16, weight="bold"),
                                     bg="#ff0000", fg="white")
        self.hex_display.place(relx=0.5, rely=0.5, anchor="center")

        # RGB sliders
        slider_frame = tk.LabelFrame(self, text="RGB Sliders", bg="#f5f5f5", font=lf)
        slider_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=4, sticky="ew")

        self.r_var = tk.IntVar(value=255)
        self.g_var = tk.IntVar(value=0)
        self.b_var = tk.IntVar(value=0)

        for row, (label, var, color) in enumerate([("R", self.r_var, "#e74c3c"),
                                                    ("G", self.g_var, "#2ecc71"),
                                                    ("B", self.b_var, "#3498db")]):
            tk.Label(slider_frame, text=label, width=2, font=lf, bg="#f5f5f5",
                     fg=color).grid(row=row, column=0, padx=(8,0), pady=2)
            sl = tk.Scale(slider_frame, from_=0, to=255, orient="horizontal",
                          variable=var, length=220, bg="#f5f5f5",
                          troughcolor=color, command=lambda _: self._update_from_rgb(),
                          showvalue=True, resolution=1)
            sl.grid(row=row, column=1, padx=4, pady=2)

        # Hex input
        hex_frame = tk.Frame(self, bg="#f5f5f5")
        hex_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=4, sticky="ew")
        tk.Label(hex_frame, text="HEX:", font=lf, bg="#f5f5f5").pack(side="left")
        self.hex_var = tk.StringVar(value="#ff0000")
        hex_entry = tk.Entry(hex_frame, textvariable=self.hex_var, font=mf, width=10)
        hex_entry.pack(side="left", padx=4)
        tk.Button(hex_frame, text="Apply", font=lf, command=self._apply_hex,
                  bg="#3498db", fg="white", relief="flat", padx=8).pack(side="left")
        hex_entry.bind("<Return>", lambda _: self._apply_hex())

        # Color values display
        vals_frame = tk.LabelFrame(self, text="Color Values", bg="#f5f5f5", font=lf)
        vals_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=4, sticky="ew")
        self.val_labels: dict[str, tk.Label] = {}
        for i, key in enumerate(["HEX", "RGB", "HSL", "HSV", "CMYK"]):
            tk.Label(vals_frame, text=f"{key}:", font=lf, bg="#f5f5f5", anchor="w",
                     width=6).grid(row=i, column=0, padx=8, pady=2, sticky="w")
            lbl = tk.Label(vals_frame, text="—", font=mf, bg="#f5f5f5", anchor="w",
                           width=30)
            lbl.grid(row=i, column=1, sticky="w")
            self.val_labels[key] = lbl

        # Copy buttons
        copy_frame = tk.Frame(self, bg="#f5f5f5")
        copy_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=4, sticky="ew")
        for key in ["HEX", "RGB", "HSL"]:
            tk.Button(copy_frame, text=f"Copy {key}", font=lf, command=lambda k=key: self._copy(k),
                      bg="#ecf0f1", relief="flat", padx=8, pady=4).pack(side="left", padx=4)

        # Palette
        pal_frame = tk.LabelFrame(self, text="Saved Palette", bg="#f5f5f5", font=lf)
        pal_frame.grid(row=5, column=0, columnspan=2, padx=20, pady=4, sticky="ew")
        tk.Button(pal_frame, text="➕ Save color", font=lf, command=self._save_color,
                  bg="#2ecc71", fg="white", relief="flat", padx=8, pady=4).pack(pady=6)
        self.palette_canvas = tk.Canvas(pal_frame, height=40, bg="#f5f5f5", highlightthickness=0)
        self.palette_canvas.pack(fill="x", padx=8, pady=(0, 8))
        self.palette_canvas.bind("<Button-1>", self._palette_click)

    # ------------------------------------------------------------------

    def _current_hex(self) -> str:
        return rgb_to_hex(self.r_var.get(), self.g_var.get(), self.b_var.get())

    def _update_from_rgb(self, *_):
        r, g, b = self.r_var.get(), self.g_var.get(), self.b_var.get()
        hex_color = rgb_to_hex(r, g, b)
        # Determine text color for contrast
        luminance = 0.299*r + 0.587*g + 0.114*b
        fg = "white" if luminance < 128 else "#222"

        self.preview.configure(bg=hex_color)
        self.hex_display.configure(text=hex_color, bg=hex_color, fg=fg)
        self.hex_var.set(hex_color)

        h_sl, s_sl, l_sl = rgb_to_hsl(r, g, b)
        h_sv, s_sv, v_sv = rgb_to_hsv(r, g, b)
        c, m, y, k = rgb_to_cmyk(r, g, b)

        self.val_labels["HEX"].config(text=hex_color.upper())
        self.val_labels["RGB"].config(text=f"rgb({r}, {g}, {b})")
        self.val_labels["HSL"].config(text=f"hsl({h_sl}°, {s_sl}%, {l_sl}%)")
        self.val_labels["HSV"].config(text=f"hsv({h_sv}°, {s_sv}%, {v_sv}%)")
        self.val_labels["CMYK"].config(text=f"cmyk({c}%, {m}%, {y}%, {k}%)")

    def _apply_hex(self):
        rgb = hex_to_rgb(self.hex_var.get())
        if rgb:
            self.r_var.set(rgb[0])
            self.g_var.set(rgb[1])
            self.b_var.set(rgb[2])
            self._update_from_rgb()
        else:
            messagebox.showwarning("Invalid HEX", "Please enter a valid 6-digit HEX color (e.g. #ff0000).")

    def _copy(self, key: str):
        text = self.val_labels[key].cget("text")
        self.clipboard_clear()
        self.clipboard_append(text)

    def _save_color(self):
        color = self._current_hex()
        if color not in self._palette:
            self._palette.append(color)
            self._draw_palette()

    def _draw_palette(self):
        self.palette_canvas.delete("all")
        for i, color in enumerate(self._palette):
            x = i * 44 + 4
            self.palette_canvas.create_rectangle(x, 4, x+36, 36, fill=color,
                                                  outline="#ccc", tags=f"swatch_{i}")

    def _palette_click(self, event):
        i = (event.x - 4) // 44
        if 0 <= i < len(self._palette):
            rgb = hex_to_rgb(self._palette[i])
            if rgb:
                self.r_var.set(rgb[0])
                self.g_var.set(rgb[1])
                self.b_var.set(rgb[2])
                self._update_from_rgb()


def main():
    ColorPickerApp().mainloop()


if __name__ == "__main__":
    main()
