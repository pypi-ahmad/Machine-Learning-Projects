"""BMI Calculator GUI — Tkinter app.

Calculate Body Mass Index from height and weight.
Supports metric (kg/cm) and imperial (lb/in) units.
Displays BMI category with color-coded result.

Usage:
    python main.py
"""

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk


# ---------------------------------------------------------------------------
# BMI logic
# ---------------------------------------------------------------------------

def calc_bmi(weight: float, height_m: float) -> float:
    if height_m <= 0:
        return 0.0
    return weight / (height_m ** 2)


def bmi_category(bmi: float) -> tuple[str, str]:
    """Return (category, hex_color)."""
    if bmi < 18.5:
        return "Underweight", "#3498db"
    elif bmi < 25.0:
        return "Normal weight", "#2ecc71"
    elif bmi < 30.0:
        return "Overweight", "#f39c12"
    else:
        return "Obese", "#e74c3c"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class BMIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BMI Calculator")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")
        self._build_ui()

    def _build_ui(self):
        PAD = {"padx": 12, "pady": 6}

        title_font  = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        label_font  = tkfont.Font(family="Segoe UI", size=11)
        result_font = tkfont.Font(family="Segoe UI", size=26, weight="bold")
        cat_font    = tkfont.Font(family="Segoe UI", size=13)

        # Title
        tk.Label(self, text="BMI Calculator", font=title_font,
                 bg="#f0f0f0").grid(row=0, column=0, columnspan=2, pady=(16, 8))

        # Unit selector
        self.unit_var = tk.StringVar(value="metric")
        frame_unit = tk.Frame(self, bg="#f0f0f0")
        frame_unit.grid(row=1, column=0, columnspan=2, **PAD)
        tk.Label(frame_unit, text="Units:", font=label_font, bg="#f0f0f0").pack(side="left")
        tk.Radiobutton(frame_unit, text="Metric (kg / cm)", variable=self.unit_var,
                       value="metric", bg="#f0f0f0", font=label_font,
                       command=self._on_unit_change).pack(side="left", padx=8)
        tk.Radiobutton(frame_unit, text="Imperial (lb / in)", variable=self.unit_var,
                       value="imperial", bg="#f0f0f0", font=label_font,
                       command=self._on_unit_change).pack(side="left")

        # Weight
        self.weight_label = tk.Label(self, text="Weight (kg):", font=label_font, bg="#f0f0f0")
        self.weight_label.grid(row=2, column=0, sticky="e", **PAD)
        self.weight_var = tk.StringVar()
        tk.Entry(self, textvariable=self.weight_var, font=label_font, width=12).grid(
            row=2, column=1, sticky="w", **PAD)

        # Height
        self.height_label = tk.Label(self, text="Height (cm):", font=label_font, bg="#f0f0f0")
        self.height_label.grid(row=3, column=0, sticky="e", **PAD)
        self.height_var = tk.StringVar()
        tk.Entry(self, textvariable=self.height_var, font=label_font, width=12).grid(
            row=3, column=1, sticky="w", **PAD)

        # Age (optional, for display)
        tk.Label(self, text="Age (optional):", font=label_font, bg="#f0f0f0").grid(
            row=4, column=0, sticky="e", **PAD)
        self.age_var = tk.StringVar()
        tk.Entry(self, textvariable=self.age_var, font=label_font, width=12).grid(
            row=4, column=1, sticky="w", **PAD)

        # Calculate button
        tk.Button(self, text="Calculate BMI", font=label_font, bg="#3498db", fg="white",
                  relief="flat", padx=12, pady=6, command=self._calculate).grid(
            row=5, column=0, columnspan=2, pady=12)

        # Result
        self.result_frame = tk.Frame(self, bg="#ffffff", bd=1, relief="solid")
        self.result_frame.grid(row=6, column=0, columnspan=2, padx=20, pady=8, sticky="ew")
        self.result_frame.columnconfigure(0, weight=1)

        self.bmi_label = tk.Label(self.result_frame, text="—", font=result_font,
                                   bg="#ffffff", fg="#333333")
        self.bmi_label.grid(row=0, column=0, pady=(12, 4))

        self.cat_label = tk.Label(self.result_frame, text="Enter values above",
                                   font=cat_font, bg="#ffffff", fg="#888888")
        self.cat_label.grid(row=1, column=0, pady=(0, 8))

        self.tip_label = tk.Label(self.result_frame, text="", font=tkfont.Font(family="Segoe UI", size=9),
                                   bg="#ffffff", fg="#aaaaaa", wraplength=260, justify="center")
        self.tip_label.grid(row=2, column=0, pady=(0, 12), padx=12)

        # Scale reference
        self._draw_scale()

    def _draw_scale(self):
        tk.Label(self, text="BMI Scale", font=tkfont.Font(family="Segoe UI", size=9),
                 bg="#f0f0f0", fg="#888888").grid(row=7, column=0, columnspan=2)
        canvas = tk.Canvas(self, width=280, height=24, bg="#f0f0f0", highlightthickness=0)
        canvas.grid(row=8, column=0, columnspan=2, padx=20, pady=(0, 16))
        segments = [
            (0,    18.5, "#3498db", "< 18.5"),
            (18.5, 25.0, "#2ecc71", "18.5–25"),
            (25.0, 30.0, "#f39c12", "25–30"),
            (30.0, 40.0, "#e74c3c", "> 30"),
        ]
        total = 40.0
        x = 0
        for lo, hi, color, label in segments:
            w = int((hi - lo) / total * 280)
            canvas.create_rectangle(x, 0, x + w, 20, fill=color, outline="")
            canvas.create_text(x + w // 2, 10, text=label,
                               fill="white", font=("Segoe UI", 7))
            x += w

    def _on_unit_change(self):
        unit = self.unit_var.get()
        self.weight_label.config(text="Weight (lb):" if unit == "imperial" else "Weight (kg):")
        self.height_label.config(text="Height (in):" if unit == "imperial" else "Height (cm):")

    def _calculate(self):
        try:
            w = float(self.weight_var.get())
            h = float(self.height_var.get())
        except ValueError:
            self.bmi_label.config(text="—", fg="#333333")
            self.cat_label.config(text="Please enter valid numbers.", fg="#e74c3c")
            self.tip_label.config(text="")
            return

        if self.unit_var.get() == "imperial":
            w = w * 0.453592
            h = h * 2.54
        h_m = h / 100

        bmi = calc_bmi(w, h_m)
        category, color = bmi_category(bmi)

        self.bmi_label.config(text=f"{bmi:.1f}", fg=color)
        self.cat_label.config(text=category, fg=color)

        tips = {
            "Underweight": "Consider a balanced, calorie-rich diet and consult a doctor.",
            "Normal weight": "Great! Maintain your healthy lifestyle.",
            "Overweight": "Try regular exercise and a balanced diet.",
            "Obese": "Consult a healthcare professional for personalised advice.",
        }
        self.tip_label.config(text=tips.get(category, ""))

        age_text = ""
        try:
            age = int(self.age_var.get())
            age_text = f"  (Age: {age})"
        except ValueError:
            pass


def main():
    app = BMIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
