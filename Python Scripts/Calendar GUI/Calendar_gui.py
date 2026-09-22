import calendar
import tkinter as tk
from tkinter import messagebox


def calendar_text(year):
    """Return the full text calendar for a valid Gregorian year."""
    if year < 1:
        raise ValueError("Enter a year from 1 onward.")
    return calendar.calendar(year)


def show_calendar(root, year_field):
    try:
        year = int(year_field.get().strip())
        cal_data = calendar_text(year)
    except ValueError:
        messagebox.showerror("Calendar", "Enter a whole year from 1 onward.")
        return

    box = tk.Toplevel(root)
    box.title("Calendar For The Year")
    box.geometry("550x600")
    box.configure(background="white")

    first_label = tk.Label(box, text="CALENDAR", bg="dark grey", font=("times", 28, "bold"))
    first_label.grid(row=1, column=1)

    cal_year = tk.Label(box, text=cal_data, font="consolas 10 bold", justify=tk.LEFT)
    cal_year.grid(row=2, column=1, padx=20)


def main():
    gui = tk.Tk()
    gui.configure(background="misty rose")
    gui.title("CALENDAR")
    gui.geometry("250x250")

    cal = tk.Label(gui, text="CALENDAR", bg="lavender", font=("Helvetica", 28, "bold", "underline"))
    year = tk.Label(gui, text="Enter Year", bg="peach puff", padx=10, pady=10)
    year_field = tk.Entry(gui)

    show_button = tk.Button(
        gui,
        text="Show Calendar",
        fg="Black",
        bg="lavender",
        command=lambda: show_calendar(gui, year_field),
    )
    exit_button = tk.Button(gui, text="CLOSE", bg="peach puff", command=gui.destroy)

    cal.grid(row=1, column=1)
    year.grid(row=3, column=1)
    year_field.grid(row=4, column=1)
    show_button.grid(row=5, column=1)
    exit_button.grid(row=7, column=1)

    gui.mainloop()


if __name__ == "__main__":
    main()
