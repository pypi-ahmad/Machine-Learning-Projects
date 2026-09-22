"""Display the local system time in a themed Tkinter window."""

import tkinter as tk
from time import strftime

THEMES = {
    "light": {"background": "white", "foreground": "black"},
    "dark": {"background": "#22478a", "foreground": "black"},
}


def formatted_time() -> str:
    """Return the current local time in 12-hour format."""
    return strftime("%I:%M:%S %p")


class DigitalClock:
    """Own the clock widgets, theme, and recurring display update."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Digital Clock")
        root.resizable(False, False)

        canvas = tk.Canvas(root, height=140, width=400, highlightthickness=0)
        canvas.pack()
        self.frame = tk.Frame(root)
        self.frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)
        self.label = tk.Label(self.frame, font=("Calibri", 40, "bold"))
        self.label.pack(anchor="center", expand=True)

        menu_bar = tk.Menu(root)
        theme_menu = tk.Menu(menu_bar, tearoff=0)
        theme_menu.add_command(label="Light", command=lambda: self.apply_theme("light"))
        theme_menu.add_command(label="Dark", command=lambda: self.apply_theme("dark"))
        menu_bar.add_cascade(label="Theme", menu=theme_menu)
        root.config(menu=menu_bar)

        self.apply_theme("dark")
        self.update_clock()

    def apply_theme(self, name: str) -> None:
        """Apply a named theme to the existing widgets."""
        theme = THEMES[name]
        self.root.configure(background=theme["background"])
        self.frame.configure(background=theme["background"])
        self.label.configure(background=theme["background"], foreground=theme["foreground"])

    def update_clock(self) -> None:
        """Refresh the display once per second."""
        self.label.configure(text=formatted_time())
        self.root.after(1000, self.update_clock)


def main() -> None:
    """Create and run the clock window."""
    root = tk.Tk()
    DigitalClock(root)
    root.mainloop()


if __name__ == "__main__":
    main()
