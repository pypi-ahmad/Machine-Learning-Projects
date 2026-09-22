"""Calculate a person's age in whole years from their birth date."""

from __future__ import annotations

import tkinter as tk
from datetime import date
from tkinter import messagebox


def calculate_age(birth_date: date, today: date | None = None) -> int:
    """Return completed years, rejecting birth dates that are in the future."""
    current_date = today or date.today()
    if birth_date > current_date:
        raise ValueError("Birth date cannot be in the future.")
    return current_date.year - birth_date.year - ((current_date.month, current_date.day) < (birth_date.month, birth_date.day))


class AgeCalculatorApp:
    """Minimal Tkinter form for validated age calculation."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Age Calculator")
        self.root.resizable(False, False)
        frame = tk.Frame(root, padx=12, pady=12)
        frame.pack()
        self.name_var = tk.StringVar()
        self.year_var = tk.StringVar()
        self.month_var = tk.StringVar()
        self.day_var = tk.StringVar()
        self.result_var = tk.StringVar(value="Enter a name and birth date.")
        self.add_field(frame, "Name", self.name_var, 0)
        self.add_field(frame, "Year", self.year_var, 1)
        self.add_field(frame, "Month", self.month_var, 2)
        self.add_field(frame, "Day", self.day_var, 3)
        tk.Button(frame, text="Calculate age", command=self.show_age).grid(row=4, column=1, pady=(4, 8), sticky="e")
        tk.Label(frame, textvariable=self.result_var, wraplength=260).grid(row=5, column=0, columnspan=2, sticky="w")

    @staticmethod
    def add_field(frame: tk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        """Add one labeled text input to the form."""
        tk.Label(frame, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=4)
        tk.Entry(frame, textvariable=variable, width=24).grid(row=row, column=1, padx=(8, 0), pady=4)

    def show_age(self) -> None:
        """Validate the form and display the calculated whole-year age."""
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Age Calculator", "Enter a name.")
            return
        try:
            birth_date = date(int(self.year_var.get()), int(self.month_var.get()), int(self.day_var.get()))
            age = calculate_age(birth_date)
        except ValueError as error:
            messagebox.showerror("Age Calculator", str(error))
            return
        self.result_var.set(f"{name}'s age is {age}.")


def main() -> None:
    """Launch the local desktop calculator."""
    root = tk.Tk()
    AgeCalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
