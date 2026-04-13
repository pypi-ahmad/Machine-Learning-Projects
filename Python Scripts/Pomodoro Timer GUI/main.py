"""Pomodoro Timer GUI — Tkinter app.

Work / short-break / long-break timer using the Pomodoro technique.
Configurable intervals, sound alert, session history.

Usage:
    python main.py
"""

import threading
import time
import tkinter as tk
from tkinter import font as tkfont


WORK_MIN  = 25
SHORT_MIN = 5
LONG_MIN  = 15
LONG_AFTER = 4  # long break every N pomodoros


class PomodoroApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pomodoro Timer")
        self.resizable(False, False)
        self.configure(bg="#2c2c2c")

        self._work_min  = tk.IntVar(value=WORK_MIN)
        self._short_min = tk.IntVar(value=SHORT_MIN)
        self._long_min  = tk.IntVar(value=LONG_MIN)

        self._seconds_left = 0
        self._running      = False
        self._paused       = False
        self._phase        = "Work"
        self._pomodoros    = 0
        self._thread: threading.Thread | None = None
        self._stop_event   = threading.Event()

        self._build_ui()

    def _build_ui(self):
        title_f = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        time_f  = tkfont.Font(family="Segoe UI", size=52, weight="bold")
        btn_f   = tkfont.Font(family="Segoe UI", size=11)
        lbl_f   = tkfont.Font(family="Segoe UI", size=10)

        # Phase label
        self._phase_var = tk.StringVar(value="Work")
        tk.Label(self, textvariable=self._phase_var, font=title_f,
                 bg="#2c2c2c", fg="#e74c3c").pack(pady=(20, 0))

        # Time display
        self._time_var = tk.StringVar(value="25:00")
        tk.Label(self, textvariable=self._time_var, font=time_f,
                 bg="#2c2c2c", fg="white").pack(pady=10)

        # Pomodoro dots
        self._dots_var = tk.StringVar(value="○ ○ ○ ○")
        tk.Label(self, textvariable=self._dots_var, font=lbl_f,
                 bg="#2c2c2c", fg="#aaa").pack()

        # Buttons
        btn_frame = tk.Frame(self, bg="#2c2c2c")
        btn_frame.pack(pady=12)
        cfg = dict(font=btn_f, relief="flat", padx=14, pady=6)
        self._start_btn = tk.Button(btn_frame, text="▶ Start", bg="#27ae60", fg="white",
                                     command=self._start, **cfg)
        self._start_btn.grid(row=0, column=0, padx=4)
        tk.Button(btn_frame, text="⏸ Pause",  bg="#f39c12", fg="white",
                  command=self._pause, **cfg).grid(row=0, column=1, padx=4)
        tk.Button(btn_frame, text="⏹ Reset",  bg="#e74c3c", fg="white",
                  command=self._reset, **cfg).grid(row=0, column=2, padx=4)
        tk.Button(btn_frame, text="⏭ Skip",   bg="#3498db", fg="white",
                  command=self._skip, **cfg).grid(row=0, column=3, padx=4)

        # Settings
        settings = tk.LabelFrame(self, text="Settings (min)", bg="#2c2c2c",
                                  fg="#aaa", font=lbl_f, bd=1)
        settings.pack(padx=20, pady=8, fill="x")
        for col, (label, var) in enumerate([("Work", self._work_min),
                                             ("Short break", self._short_min),
                                             ("Long break", self._long_min)]):
            tk.Label(settings, text=label, bg="#2c2c2c", fg="#aaa", font=lbl_f).grid(
                row=0, column=col*2, padx=(10, 2), pady=6)
            tk.Spinbox(settings, from_=1, to=60, textvariable=var, width=4,
                       font=lbl_f, bg="#3c3c3c", fg="white", buttonbackground="#555").grid(
                row=0, column=col*2+1, padx=(0, 10))

        # Session count
        self._session_var = tk.StringVar(value="Sessions: 0")
        tk.Label(self, textvariable=self._session_var, font=lbl_f,
                 bg="#2c2c2c", fg="#aaa").pack(pady=(0, 16))

    def _set_display(self):
        mins = self._seconds_left // 60
        secs = self._seconds_left % 60
        self._time_var.set(f"{mins:02d}:{secs:02d}")
        dots = ""
        for i in range(LONG_AFTER):
            dots += "● " if i < self._pomodoros % LONG_AFTER else "○ "
        self._dots_var.set(dots.strip())
        self._session_var.set(f"Sessions: {self._pomodoros}")

    def _get_phase_seconds(self) -> int:
        if self._phase == "Work":
            return self._work_min.get() * 60
        elif self._phase == "Short Break":
            return self._short_min.get() * 60
        else:
            return self._long_min.get() * 60

    def _start(self):
        if self._running:
            return
        if not self._paused:
            self._seconds_left = self._get_phase_seconds()
        self._running  = True
        self._paused   = False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        colors = {"Work": "#e74c3c", "Short Break": "#27ae60", "Long Break": "#3498db"}
        self._phase_var.set(self._phase)
        self.configure(bg=colors.get(self._phase, "#2c2c2c"))

    def _tick(self):
        while self._seconds_left > 0 and not self._stop_event.is_set():
            time.sleep(1)
            if not self._stop_event.is_set():
                self._seconds_left -= 1
                self.after(0, self._set_display)
        if not self._stop_event.is_set():
            self.after(0, self._phase_complete)

    def _phase_complete(self):
        self._running = False
        self._alert()
        if self._phase == "Work":
            self._pomodoros += 1
            if self._pomodoros % LONG_AFTER == 0:
                self._phase = "Long Break"
            else:
                self._phase = "Short Break"
        else:
            self._phase = "Work"
        self._phase_var.set(self._phase)
        self._seconds_left = self._get_phase_seconds()
        self._set_display()

    def _alert(self):
        try:
            import winsound
            winsound.MessageBeep()
        except Exception:
            print("\a")

    def _pause(self):
        if self._running:
            self._stop_event.set()
            self._running = False
            self._paused  = True

    def _reset(self):
        self._stop_event.set()
        self._running  = False
        self._paused   = False
        self._phase    = "Work"
        self._pomodoros = 0
        self._phase_var.set("Work")
        self.configure(bg="#2c2c2c")
        self._seconds_left = self._work_min.get() * 60
        self._set_display()

    def _skip(self):
        self._stop_event.set()
        self._running = False
        self._paused  = False
        self._phase_complete()


def main():
    PomodoroApp().mainloop()


if __name__ == "__main__":
    main()
