"""Display the local system time in a resizable Tkinter window."""

import time
import tkinter as tk

TEXT_FONT = ("Boulder", 68, "bold")
BACKGROUND = "#f2e750"
FOREGROUND = "#363529"
BORDER_WIDTH = 25
UPDATE_INTERVAL_MS = 200


def formatted_time() -> str:
    """Return the local time in 24-hour format."""
    return time.strftime("%H:%M:%S")


def main() -> None:
    """Create and run the digital-clock window."""
    app_window = tk.Tk()
    app_window.title("Digital Clock")
    app_window.geometry("420x150")
    app_window.configure(background=BACKGROUND)

    label = tk.Label(
        app_window,
        font=TEXT_FONT,
        background=BACKGROUND,
        foreground=FOREGROUND,
        borderwidth=BORDER_WIDTH,
    )
    label.pack(fill=tk.BOTH, expand=True)

    def update_clock() -> None:
        label.config(text=formatted_time())
        app_window.after(UPDATE_INTERVAL_MS, update_clock)

    update_clock()
    app_window.mainloop()


if __name__ == "__main__":
    main()
