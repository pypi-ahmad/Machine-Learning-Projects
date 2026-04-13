"""Music Player — Tkinter desktop app.

Browse and play audio files (MP3, WAV, OGG) using pygame.mixer.
Includes playlist management, progress bar, and volume control.

Usage:
    python main.py
"""

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import pygame
    pygame.mixer.init()
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False


class MusicPlayer(tk.Tk):
    SUPPORTED = (".mp3", ".wav", ".ogg", ".flac")

    def __init__(self):
        super().__init__()
        self.title("Music Player")
        self.geometry("500x560")
        self.configure(bg="#1e1e2e")
        self.resizable(False, False)

        self._playlist   = []        # list of file paths
        self._current    = -1
        self._playing    = False
        self._paused     = False
        self._duration   = 0
        self._seek_var   = tk.DoubleVar(value=0)
        self._vol_var    = tk.DoubleVar(value=0.7)
        self._now_var    = tk.StringVar(value="No track loaded")
        self._time_var   = tk.StringVar(value="0:00 / 0:00")

        self._build_ui()
        if PYGAME_OK:
            pygame.mixer.music.set_volume(0.7)
        self._poll()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Album art placeholder
        art = tk.Frame(self, bg="#313244", width=200, height=200)
        art.pack(pady=(20, 8))
        art.pack_propagate(False)
        tk.Label(art, text="♪", bg="#313244", fg="#cba6f7",
                 font=("Arial", 72)).pack(expand=True)

        # Track info
        tk.Label(self, textvariable=self._now_var, bg="#1e1e2e", fg="#cdd6f4",
                 font=("Consolas", 12, "bold"), wraplength=440).pack()
        tk.Label(self, textvariable=self._time_var, bg="#1e1e2e", fg="#888",
                 font=("Consolas", 10)).pack()

        # Progress bar
        tk.Scale(self, variable=self._seek_var, from_=0, to=100,
                 orient="horizontal", length=440, showvalue=False,
                 bg="#1e1e2e", troughcolor="#313244", fg="#cba6f7",
                 highlightthickness=0, command=self._on_seek).pack(pady=4)

        # Controls
        ctrl = tk.Frame(self, bg="#1e1e2e")
        ctrl.pack()
        for lbl, cmd in [("⏮", self._prev), ("⏸/▶", self._play_pause), ("⏹", self._stop), ("⏭", self._next)]:
            tk.Button(ctrl, text=lbl, command=cmd, bg="#313244", fg="#cdd6f4",
                      activebackground="#45475a", font=("Arial", 16), width=3,
                      relief="flat").pack(side="left", padx=4)

        # Volume
        vol_fr = tk.Frame(self, bg="#1e1e2e")
        vol_fr.pack(pady=4)
        tk.Label(vol_fr, text="🔊", bg="#1e1e2e", fg="#888").pack(side="left")
        tk.Scale(vol_fr, variable=self._vol_var, from_=0, to=1, resolution=0.05,
                 orient="horizontal", length=200, showvalue=False,
                 bg="#1e1e2e", troughcolor="#313244", highlightthickness=0,
                 command=self._on_volume).pack(side="left")

        # Playlist
        pl_fr = tk.Frame(self, bg="#1e1e2e")
        pl_fr.pack(fill="both", expand=True, padx=12, pady=(8, 4))
        tk.Label(pl_fr, text="Playlist", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 10)).pack(anchor="w")
        sb = tk.Scrollbar(pl_fr)
        sb.pack(side="right", fill="y")
        self._lb = tk.Listbox(pl_fr, bg="#313244", fg="#cdd6f4",
                               selectbackground="#45475a", font=("Consolas", 10),
                               yscrollcommand=sb.set, activestyle="none")
        self._lb.pack(fill="both", expand=True)
        sb.config(command=self._lb.yview)
        self._lb.bind("<Double-Button-1>", self._on_double_click)

        # Buttons
        btn_fr = tk.Frame(self, bg="#1e1e2e")
        btn_fr.pack(fill="x", padx=12, pady=(0, 8))
        for lbl, cmd in [("+ Add Files", self._add_files), ("+ Add Folder", self._add_folder),
                          ("✕ Remove", self._remove_selected), ("🗑 Clear All", self._clear)]:
            tk.Button(btn_fr, text=lbl, command=cmd, bg="#313244", fg="#cdd6f4",
                      relief="flat", padx=6).pack(side="left", padx=2)

        if not PYGAME_OK:
            tk.Label(self, text="⚠ pygame not installed — playback disabled",
                     bg="#1e1e2e", fg="#f38ba8", font=("Consolas", 9)).pack()

    # ── Playlist management ───────────────────────────────────────────────────

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Audio files", " ".join(f"*{e}" for e in self.SUPPORTED)),
                       ("All files", "*.*")])
        for p in paths:
            if p not in self._playlist:
                self._playlist.append(p)
                self._lb.insert("end", os.path.basename(p))

    def _add_folder(self):
        d = filedialog.askdirectory()
        if d:
            for fn in sorted(os.listdir(d)):
                if any(fn.lower().endswith(e) for e in self.SUPPORTED):
                    p = os.path.join(d, fn)
                    if p not in self._playlist:
                        self._playlist.append(p)
                        self._lb.insert("end", fn)

    def _remove_selected(self):
        for idx in reversed(self._lb.curselection()):
            self._lb.delete(idx)
            self._playlist.pop(idx)

    def _clear(self):
        self._stop()
        self._playlist.clear()
        self._lb.delete(0, "end")

    # ── Playback ──────────────────────────────────────────────────────────────

    def _play_at(self, idx: int):
        if not (0 <= idx < len(self._playlist)):
            return
        if not PYGAME_OK:
            messagebox.showwarning("pygame missing", "Install pygame to enable playback.")
            return
        self._current = idx
        path = self._playlist[idx]
        self._now_var.set(os.path.basename(path))
        self._lb.selection_clear(0, "end")
        self._lb.selection_set(idx)
        self._lb.see(idx)
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self._playing = True
            self._paused  = False
            # Estimate duration (pygame doesn't always give it)
            self._duration = 0
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))

    def _play_pause(self):
        if not self._playlist:
            return
        if self._current == -1:
            self._play_at(0)
            return
        if not PYGAME_OK:
            return
        if self._playing and not self._paused:
            pygame.mixer.music.pause()
            self._paused = True
        elif self._paused:
            pygame.mixer.music.unpause()
            self._paused = False
        else:
            self._play_at(self._current)

    def _stop(self):
        if PYGAME_OK:
            pygame.mixer.music.stop()
        self._playing = False
        self._paused  = False
        self._seek_var.set(0)
        self._time_var.set("0:00 / 0:00")

    def _next(self):
        if self._playlist:
            self._play_at((self._current + 1) % len(self._playlist))

    def _prev(self):
        if self._playlist:
            self._play_at((self._current - 1) % len(self._playlist))

    def _on_double_click(self, _):
        sel = self._lb.curselection()
        if sel:
            self._play_at(sel[0])

    def _on_seek(self, val):
        pass  # Seeking via pygame requires audio length info; placeholder

    def _on_volume(self, val):
        if PYGAME_OK:
            pygame.mixer.music.set_volume(float(val))

    # ── Poll ──────────────────────────────────────────────────────────────────

    def _poll(self):
        if PYGAME_OK and self._playing and not self._paused:
            if not pygame.mixer.music.get_busy():
                # Track ended — auto-next
                self._next()
        self.after(500, self._poll)


if __name__ == "__main__":
    app = MusicPlayer()
    app.mainloop()
