"""Video Launcher — Tkinter desktop app.

Browse local video files, organize a playlist, and launch them
with the system default player. Supports thumbnails via folder art.

Usage:
    python main.py
"""

import json
import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

DATA_FILE  = os.path.join(os.path.dirname(__file__), "video_library.json")
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts"}


def load() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"library": [], "playlists": {}}


def save(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def open_file(path: str):
    """Launch the video with the OS default player."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class VideoLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Launcher")
        self.geometry("900x600")
        self.configure(bg="#1e1e2e")

        self._data       = load()
        self._library    = self._data.get("library", [])
        self._playlists  = self._data.get("playlists", {})
        self._pl_var     = tk.StringVar(value="")

        self._build_ui()
        self._refresh()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        tb = tk.Frame(self, bg="#181825")
        tb.pack(fill="x")
        for lbl, cmd in [("+ Add Files", self._add_files),
                          ("+ Add Folder", self._add_folder),
                          ("🗑 Remove", self._remove),
                          ("▶ Play", self._play_selected),
                          ("▶ Play All", self._play_all)]:
            tk.Button(tb, text=lbl, command=cmd, bg="#313244", fg="#cdd6f4",
                      relief="flat", padx=8).pack(side="left", padx=2, pady=6)

        # Playlist controls
        tk.Label(tb, text="Playlist:", bg="#181825", fg="#888").pack(side="left", padx=(12, 2))
        self._pl_menu_var = tk.StringVar(value="(none)")
        self._pl_menu = tk.OptionMenu(tb, self._pl_menu_var, "(none)")
        self._pl_menu.config(bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                              relief="flat", font=("Consolas", 9))
        self._pl_menu.pack(side="left")
        tk.Button(tb, text="+ PL", command=self._new_playlist,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)
        tk.Button(tb, text="Add→PL", command=self._add_to_playlist,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)

        # Search
        sf = tk.Frame(self, bg="#1e1e2e")
        sf.pack(fill="x", padx=8, pady=4)
        tk.Label(sf, text="🔍", bg="#1e1e2e", fg="#888").pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh())
        tk.Entry(sf, textvariable=self._search_var, bg="#313244", fg="#cdd6f4",
                 insertbackground="#cba6f7", font=("Consolas", 11),
                 relief="flat", width=32).pack(side="left", padx=4)
        self._count_var = tk.StringVar(value="0 videos")
        tk.Label(sf, textvariable=self._count_var, bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).pack(side="right")

        # Table
        cols = ("name", "size", "ext", "path")
        self._tree = ttk.Treeview(self, columns=cols, show="headings")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        for col, w, anchor in [("name", 320, "w"), ("size", 80, "e"),
                                 ("ext", 60, "w"), ("path", 340, "w")]:
            self._tree.heading(col, text=col.upper())
            self._tree.column(col, width=w, anchor=anchor)
        self._tree.pack(fill="both", expand=True, padx=8, pady=4)
        self._tree.bind("<Double-Button-1>", lambda _: self._play_selected())
        sb = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # Status
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self._status_var, bg="#181825", fg="#555",
                 anchor="w", padx=8).pack(fill="x", side="bottom")

    # ── Library management ────────────────────────────────────────────────────

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Video files", " ".join(f"*{e}" for e in VIDEO_EXTS)),
                       ("All files", "*.*")])
        added = 0
        for p in paths:
            if not any(e["path"] == p for e in self._library):
                self._library.append(self._make_entry(p))
                added += 1
        if added:
            self._persist()
            self._refresh()
            self._status_var.set(f"Added {added} file(s).")

    def _add_folder(self):
        d = filedialog.askdirectory()
        if not d:
            return
        added = 0
        for root, _, files in os.walk(d):
            for fn in sorted(files):
                ext = os.path.splitext(fn)[1].lower()
                if ext in VIDEO_EXTS:
                    p = os.path.join(root, fn)
                    if not any(e["path"] == p for e in self._library):
                        self._library.append(self._make_entry(p))
                        added += 1
        if added:
            self._persist()
            self._refresh()
            self._status_var.set(f"Added {added} video(s) from folder.")

    def _make_entry(self, path: str) -> dict:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        return {
            "name": os.path.splitext(os.path.basename(path))[0],
            "path": path,
            "ext":  os.path.splitext(path)[1].lower(),
            "size": size,
        }

    def _remove(self):
        sel = self._tree.selection()
        if not sel:
            return
        paths = {self._tree.item(s, "values")[3] for s in sel}
        self._library = [e for e in self._library if e["path"] not in paths]
        self._persist()
        self._refresh()

    def _play_selected(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a video to play.")
            return
        path = self._tree.item(sel[0], "values")[3]
        if not os.path.exists(path):
            messagebox.showerror("Not Found", f"File not found:\n{path}")
            return
        open_file(path)
        self._status_var.set(f"Playing: {os.path.basename(path)}")

    def _play_all(self):
        q = self._search_var.get().lower()
        for e in self._library:
            if q and q not in e["name"].lower():
                continue
            if os.path.exists(e["path"]):
                open_file(e["path"])

    # ── Playlists ─────────────────────────────────────────────────────────────

    def _new_playlist(self):
        from tkinter import simpledialog
        name = simpledialog.askstring("New Playlist", "Name:", parent=self)
        if name and name not in self._playlists:
            self._playlists[name] = []
            self._persist()
            self._rebuild_pl_menu()

    def _add_to_playlist(self):
        pl = self._pl_menu_var.get()
        if pl == "(none)" or pl not in self._playlists:
            messagebox.showinfo("Select PL", "Create/select a playlist first.")
            return
        for s in self._tree.selection():
            path = self._tree.item(s, "values")[3]
            if path not in self._playlists[pl]:
                self._playlists[pl].append(path)
        self._persist()
        self._status_var.set(f"Added to playlist '{pl}'.")

    def _rebuild_pl_menu(self):
        menu = self._pl_menu["menu"]
        menu.delete(0, "end")
        menu.add_command(label="(none)", command=lambda: self._pl_menu_var.set("(none)"))
        for pl in self._playlists:
            menu.add_command(label=pl, command=lambda v=pl: self._pl_menu_var.set(v))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _refresh(self):
        self._tree.delete(*self._tree.get_children())
        q = self._search_var.get().lower()
        shown = 0
        for e in self._library:
            if q and q not in e["name"].lower():
                continue
            self._tree.insert("", "end", values=(
                e["name"], human_size(e["size"]), e["ext"], e["path"]))
            shown += 1
        self._count_var.set(f"{shown} / {len(self._library)} videos")
        self._rebuild_pl_menu()

    def _persist(self):
        self._data["library"]   = self._library
        self._data["playlists"] = self._playlists
        save(self._data)


if __name__ == "__main__":
    app = VideoLauncher()
    app.mainloop()
