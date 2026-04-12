"""Image Viewer GUI — Tkinter app.

Browse and view images with zoom, rotate, and fit-to-window.
Supports JPG, PNG, GIF, BMP, TIFF via Pillow (falls back to
tkinter's built-in PhotoImage for PNG/GIF if Pillow is absent).

Usage:
    python main.py [image_path]
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    from PIL import Image, ImageTk
    PILLOW = True
except ImportError:
    PILLOW = False

SUPPORTED_EXTS = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"
) if PILLOW else (".png", ".gif")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class ImageViewer(tk.Tk):
    def __init__(self, start_path: str | None = None):
        super().__init__()
        self.title("Image Viewer")
        self.geometry("900x650")
        self.configure(bg="#1e1e1e")

        self._image_list: list[Path] = []
        self._index = 0
        self._zoom  = 1.0
        self._rotation = 0
        self._photo = None    # keep reference
        self._pil_img = None  # original PIL image

        self._build_ui()

        if start_path:
            self._load_path(Path(start_path))
        else:
            self._show_placeholder()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Toolbar
        toolbar = tk.Frame(self, bg="#2d2d2d")
        toolbar.pack(fill="x")

        btn_cfg = dict(bg="#2d2d2d", fg="white", bd=0, padx=10, pady=6,
                       font=("Segoe UI", 10), activebackground="#444")
        tk.Button(toolbar, text="📂 Open", command=self._open_file, **btn_cfg).pack(side="left")
        tk.Button(toolbar, text="📁 Folder", command=self._open_folder, **btn_cfg).pack(side="left")
        tk.Button(toolbar, text="◀", command=self._prev, **btn_cfg).pack(side="left", padx=(8,0))
        tk.Button(toolbar, text="▶", command=self._next, **btn_cfg).pack(side="left")
        tk.Button(toolbar, text="🔍+", command=self._zoom_in,  **btn_cfg).pack(side="left", padx=(8,0))
        tk.Button(toolbar, text="🔍-", command=self._zoom_out, **btn_cfg).pack(side="left")
        tk.Button(toolbar, text="⊡ Fit",   command=self._fit,    **btn_cfg).pack(side="left")
        tk.Button(toolbar, text="↺ Rotate", command=self._rotate, **btn_cfg).pack(side="left")
        self.info_var = tk.StringVar(value="No image")
        tk.Label(toolbar, textvariable=self.info_var, bg="#2d2d2d", fg="#aaa",
                 font=("Segoe UI", 9)).pack(side="right", padx=10)

        # Canvas
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self._redraw())
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.bind("<Left>",  lambda _: self._prev())
        self.bind("<Right>", lambda _: self._next())
        self.bind("<plus>",  lambda _: self._zoom_in())
        self.bind("<minus>", lambda _: self._zoom_out())

    def _show_placeholder(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()  or 900
        h = self.canvas.winfo_height() or 600
        self.canvas.create_text(w//2, h//2, text="Open an image or folder to start",
                                 fill="#555", font=("Segoe UI", 14))

    # ------------------------------------------------------------------
    # File / folder loading
    # ------------------------------------------------------------------

    def _open_file(self):
        ext_str = " ".join(f"*{e}" for e in SUPPORTED_EXTS)
        path = filedialog.askopenfilename(
            filetypes=[("Images", ext_str), ("All files", "*.*")])
        if path:
            self._load_path(Path(path))

    def _open_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self._image_list = sorted(
                p for p in Path(folder).iterdir()
                if p.suffix.lower() in SUPPORTED_EXTS
            )
            if self._image_list:
                self._index = 0
                self._show_current()
            else:
                messagebox.showinfo("No images", "No supported images found in folder.")

    def _load_path(self, path: Path):
        folder = path.parent
        self._image_list = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTS
        )
        try:
            self._index = self._image_list.index(path)
        except ValueError:
            self._image_list = [path]
            self._index = 0
        self._show_current()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _prev(self):
        if self._image_list:
            self._index = (self._index - 1) % len(self._image_list)
            self._show_current()

    def _next(self):
        if self._image_list:
            self._index = (self._index + 1) % len(self._image_list)
            self._show_current()

    def _show_current(self):
        if not self._image_list:
            return
        self._zoom     = 1.0
        self._rotation = 0
        path = self._image_list[self._index]
        try:
            if PILLOW:
                self._pil_img = Image.open(path)
            else:
                self._pil_img = None
                self._tk_img = tk.PhotoImage(file=str(path))
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")
            return
        self.title(f"Image Viewer — {path.name}")
        self.info_var.set(
            f"{path.name}  [{self._index+1}/{len(self._image_list)}]"
            + (f"  {self._pil_img.size[0]}×{self._pil_img.size[1]}" if PILLOW and self._pil_img else "")
        )
        self._fit()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _fit(self):
        if self._pil_img is None and not PILLOW:
            self._zoom = 1.0
            self._redraw()
            return
        if PILLOW and self._pil_img:
            cw = self.canvas.winfo_width()  or 900
            ch = self.canvas.winfo_height() or 600
            iw, ih = self._pil_img.size
            self._zoom = min(cw / iw, ch / ih, 1.0)
            self._redraw()

    def _zoom_in(self):
        self._zoom = min(self._zoom * 1.2, 10.0)
        self._redraw()

    def _zoom_out(self):
        self._zoom = max(self._zoom / 1.2, 0.05)
        self._redraw()

    def _rotate(self):
        self._rotation = (self._rotation + 90) % 360
        self._redraw()

    def _on_scroll(self, event):
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _redraw(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()  or 900
        ch = self.canvas.winfo_height() or 600

        if PILLOW and self._pil_img:
            img = self._pil_img
            if self._rotation:
                img = img.rotate(self._rotation, expand=True)
            nw = max(1, int(img.width  * self._zoom))
            nh = max(1, int(img.height * self._zoom))
            img = img.resize((nw, nh), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(cw//2, ch//2, anchor="center", image=self._photo)
        elif hasattr(self, "_tk_img"):
            self.canvas.create_image(cw//2, ch//2, anchor="center", image=self._tk_img)
        else:
            self._show_placeholder()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    ImageViewer(path).mainloop()


if __name__ == "__main__":
    main()
