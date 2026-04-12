"""Unit Converter GUI — Tkinter app.

Convert between common units across multiple categories:
Length, Weight, Temperature, Volume, Area, Speed, Time, Data.

Usage:
    python main.py
"""

import tkinter as tk
from tkinter import font as tkfont, ttk


# ---------------------------------------------------------------------------
# Conversion tables
# ---------------------------------------------------------------------------

# All values are conversion factors TO the base unit (first in list).
# Temperature is handled separately.

CATEGORIES: dict[str, dict[str, float]] = {
    "Length (m)": {
        "Meter":      1,
        "Kilometer":  1000,
        "Centimeter": 0.01,
        "Millimeter": 0.001,
        "Mile":       1609.344,
        "Yard":       0.9144,
        "Foot":       0.3048,
        "Inch":       0.0254,
        "Nautical mile": 1852,
    },
    "Weight (kg)": {
        "Kilogram":    1,
        "Gram":        0.001,
        "Milligram":   1e-6,
        "Pound":       0.453592,
        "Ounce":       0.0283495,
        "Tonne":       1000,
        "Stone":       6.35029,
    },
    "Volume (L)": {
        "Liter":        1,
        "Milliliter":   0.001,
        "Cubic meter":  1000,
        "Gallon (US)":  3.78541,
        "Quart (US)":   0.946353,
        "Pint (US)":    0.473176,
        "Cup (US)":     0.236588,
        "Fluid oz (US)": 0.0295735,
    },
    "Area (m²)": {
        "Square meter":      1,
        "Square kilometer":  1e6,
        "Square mile":       2589988.11,
        "Hectare":           10000,
        "Acre":              4046.86,
        "Square foot":       0.092903,
        "Square inch":       0.00064516,
        "Square yard":       0.836127,
    },
    "Speed (m/s)": {
        "Meter/second":    1,
        "Kilometer/hour":  1/3.6,
        "Mile/hour":       0.44704,
        "Knot":            0.514444,
        "Foot/second":     0.3048,
    },
    "Time (s)": {
        "Second":  1,
        "Minute":  60,
        "Hour":    3600,
        "Day":     86400,
        "Week":    604800,
        "Month":   2629800,
        "Year":    31557600,
        "Millisecond": 0.001,
        "Microsecond": 1e-6,
    },
    "Data (bytes)": {
        "Byte":     1,
        "Kilobyte": 1024,
        "Megabyte": 1024**2,
        "Gigabyte": 1024**3,
        "Terabyte": 1024**4,
        "Bit":      0.125,
        "Kibibyte": 1024,
        "Mebibyte": 1024**2,
    },
    "Temperature": {
        "Celsius":    0,
        "Fahrenheit": 0,
        "Kelvin":     0,
    },
}


def convert(value: float, from_unit: str, to_unit: str, category: str) -> float:
    if category == "Temperature":
        # Convert from_unit → Celsius first
        if from_unit == "Celsius":
            c = value
        elif from_unit == "Fahrenheit":
            c = (value - 32) * 5 / 9
        else:  # Kelvin
            c = value - 273.15

        # Celsius → to_unit
        if to_unit == "Celsius":
            return c
        elif to_unit == "Fahrenheit":
            return c * 9/5 + 32
        else:  # Kelvin
            return c + 273.15
    else:
        units = CATEGORIES[category]
        base = value * units[from_unit]
        return base / units[to_unit]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class UnitConverterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Unit Converter")
        self.resizable(False, False)
        self.configure(bg="#f5f5f5")
        self._build_ui()

    def _build_ui(self):
        lf = tkfont.Font(family="Segoe UI", size=11)
        bf = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        rf = tkfont.Font(family="Consolas", size=14, weight="bold")

        tk.Label(self, text="Unit Converter", font=tkfont.Font(family="Segoe UI", size=16, weight="bold"),
                 bg="#f5f5f5").grid(row=0, column=0, columnspan=2, pady=(16,8))

        # Category
        tk.Label(self, text="Category:", font=lf, bg="#f5f5f5").grid(row=1, column=0, sticky="e", padx=12, pady=4)
        self.cat_var = tk.StringVar(value="Length (m)")
        cat_cb = ttk.Combobox(self, textvariable=self.cat_var, values=list(CATEGORIES.keys()),
                               state="readonly", font=lf, width=20)
        cat_cb.grid(row=1, column=1, sticky="w", padx=12, pady=4)
        cat_cb.bind("<<ComboboxSelected>>", self._on_cat_change)

        # From
        tk.Label(self, text="From:", font=lf, bg="#f5f5f5").grid(row=2, column=0, sticky="e", padx=12, pady=4)
        self.from_var = tk.StringVar()
        self.from_cb  = ttk.Combobox(self, textvariable=self.from_var, state="readonly", font=lf, width=20)
        self.from_cb.grid(row=2, column=1, sticky="w", padx=12, pady=4)
        self.from_cb.bind("<<ComboboxSelected>>", lambda _: self._calc())

        # Value
        tk.Label(self, text="Value:", font=lf, bg="#f5f5f5").grid(row=3, column=0, sticky="e", padx=12, pady=4)
        self.val_var = tk.StringVar(value="1")
        val_entry = tk.Entry(self, textvariable=self.val_var, font=lf, width=22)
        val_entry.grid(row=3, column=1, sticky="w", padx=12, pady=4)
        val_entry.bind("<KeyRelease>", lambda _: self._calc())

        # Swap button
        tk.Button(self, text="⇅ Swap", font=lf, bg="#ecf0f1", relief="flat",
                  command=self._swap, padx=8).grid(row=4, column=0, columnspan=2, pady=4)

        # To
        tk.Label(self, text="To:", font=lf, bg="#f5f5f5").grid(row=5, column=0, sticky="e", padx=12, pady=4)
        self.to_var = tk.StringVar()
        self.to_cb  = ttk.Combobox(self, textvariable=self.to_var, state="readonly", font=lf, width=20)
        self.to_cb.grid(row=5, column=1, sticky="w", padx=12, pady=4)
        self.to_cb.bind("<<ComboboxSelected>>", lambda _: self._calc())

        # Result
        result_frame = tk.Frame(self, bg="#3498db", bd=0)
        result_frame.grid(row=6, column=0, columnspan=2, padx=20, pady=12, sticky="ew")
        self.result_var = tk.StringVar(value="—")
        tk.Label(result_frame, textvariable=self.result_var, font=rf, bg="#3498db",
                 fg="white", pady=12, padx=16).pack()

        # Quick reference: all units
        tk.Label(self, text="All conversions from input value:", font=lf,
                 bg="#f5f5f5").grid(row=7, column=0, columnspan=2, pady=(4,0))
        self.table_text = tk.Text(self, font=tkfont.Font(family="Consolas", size=9),
                                   height=10, width=44, bg="#f8f8f8", relief="flat", state="disabled")
        self.table_text.grid(row=8, column=0, columnspan=2, padx=20, pady=(0,16))

        self._on_cat_change()

    def _on_cat_change(self, event=None):
        cat   = self.cat_var.get()
        units = list(CATEGORIES[cat].keys())
        self.from_cb["values"] = units
        self.to_cb["values"]   = units
        self.from_var.set(units[0])
        self.to_var.set(units[1] if len(units) > 1 else units[0])
        self._calc()

    def _calc(self, *_):
        try:
            value = float(self.val_var.get())
        except ValueError:
            self.result_var.set("—")
            return
        cat      = self.cat_var.get()
        from_u   = self.from_var.get()
        to_u     = self.to_var.get()
        if not from_u or not to_u:
            return
        result = convert(value, from_u, to_u, cat)
        self.result_var.set(f"{result:,.6g} {to_u.split()[0]}")

        # Table of all conversions
        self.table_text.configure(state="normal")
        self.table_text.delete("1.0", "end")
        for unit in CATEGORIES[cat]:
            r = convert(value, from_u, unit, cat)
            self.table_text.insert("end", f"  {unit:<22} {r:,.6g}\n")
        self.table_text.configure(state="disabled")

    def _swap(self):
        a, b = self.from_var.get(), self.to_var.get()
        self.from_var.set(b)
        self.to_var.set(a)
        self._calc()


def main():
    UnitConverterApp().mainloop()


if __name__ == "__main__":
    main()
