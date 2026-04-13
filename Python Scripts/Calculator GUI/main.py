"""Calculator GUI — Tkinter desktop app.

Full-featured calculator with standard and scientific modes.

Usage:
    python main.py
"""

import math
import tkinter as tk
from tkinter import ttk


class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calculator")
        self.resizable(False, False)
        self.configure(bg="#1e1e2e")

        self._expr   = ""
        self._result = tk.StringVar(value="0")
        self._expr_var = tk.StringVar(value="")
        self._sci_mode = tk.BooleanVar(value=False)

        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Display
        frm = tk.Frame(self, bg="#1e1e2e", padx=8, pady=8)
        frm.pack(fill="x")
        tk.Label(frm, textvariable=self._expr_var, bg="#1e1e2e", fg="#888",
                 font=("Consolas", 12), anchor="e").pack(fill="x")
        tk.Label(frm, textvariable=self._result, bg="#1e1e2e", fg="#cdd6f4",
                 font=("Consolas", 28, "bold"), anchor="e").pack(fill="x")

        # Mode toggle
        tog = tk.Frame(self, bg="#1e1e2e")
        tog.pack(fill="x", padx=8)
        tk.Checkbutton(tog, text="Scientific mode", variable=self._sci_mode,
                       command=self._rebuild_buttons,
                       bg="#1e1e2e", fg="#888", selectcolor="#313244",
                       activebackground="#1e1e2e", activeforeground="#cdd6f4").pack(anchor="w")

        self._btn_frame = tk.Frame(self, bg="#1e1e2e", padx=8, pady=4)
        self._btn_frame.pack()
        self._rebuild_buttons()

    def _rebuild_buttons(self):
        for w in self._btn_frame.winfo_children():
            w.destroy()

        if self._sci_mode.get():
            layout = [
                ["sin", "cos", "tan", "√",  "x²", "π",  "e",  "C"],
                ["7",   "8",   "9",   "/",  "log","ln", "(",  ")"],
                ["4",   "5",   "6",   "*",  "1/x","x^y","±",  "%"],
                ["1",   "2",   "3",   "-",  "CE", "⌫",  "",   ""],
                ["0",   ".",   "=",   "+",  "",   "",   "",   ""],
            ]
        else:
            layout = [
                ["C",  "⌫", "%",  "/"],
                ["7",  "8",  "9",  "*"],
                ["4",  "5",  "6",  "-"],
                ["1",  "2",  "3",  "+"],
                ["±",  "0",  ".",  "="],
            ]

        COLORS = {
            "=":  ("#cba6f7", "#1e1e2e"),
            "+":  ("#89b4fa", "#1e1e2e"),
            "-":  ("#89b4fa", "#1e1e2e"),
            "*":  ("#89b4fa", "#1e1e2e"),
            "/":  ("#89b4fa", "#1e1e2e"),
            "C":  ("#f38ba8", "#1e1e2e"),
            "CE": ("#f38ba8", "#1e1e2e"),
        }

        for r, row in enumerate(layout):
            for c, lbl in enumerate(row):
                if not lbl:
                    continue
                bg, fg = COLORS.get(lbl, ("#313244", "#cdd6f4"))
                tk.Button(
                    self._btn_frame, text=lbl, width=5, height=2,
                    bg=bg, fg=fg, activebackground="#45475a",
                    relief="flat", font=("Consolas", 13),
                    command=lambda l=lbl: self._press(l),
                ).grid(row=r, column=c, padx=2, pady=2, sticky="nsew")

    # ── Logic ───────────────────────────────────────────────────────────────

    def _press(self, key: str):
        if key == "C":
            self._expr = ""
            self._result.set("0")
            self._expr_var.set("")
        elif key == "CE":
            self._expr = ""
            self._expr_var.set("")
        elif key == "⌫":
            self._expr = self._expr[:-1]
            self._expr_var.set(self._expr)
        elif key == "=":
            self._evaluate()
        elif key == "±":
            try:
                val = float(self._result.get())
                self._result.set(str(-val))
                self._expr = str(-val)
            except ValueError:
                pass
        elif key == "%":
            try:
                val = float(self._result.get()) / 100
                self._result.set(str(val))
                self._expr = str(val)
            except ValueError:
                pass
        elif key == "π":
            self._expr += str(math.pi)
            self._expr_var.set(self._expr)
        elif key == "e":
            self._expr += str(math.e)
            self._expr_var.set(self._expr)
        elif key == "√":
            self._apply_unary(math.sqrt)
        elif key == "x²":
            self._apply_unary(lambda x: x ** 2)
        elif key == "1/x":
            self._apply_unary(lambda x: 1 / x if x != 0 else float("inf"))
        elif key == "sin":
            self._apply_unary(math.sin)
        elif key == "cos":
            self._apply_unary(math.cos)
        elif key == "tan":
            self._apply_unary(math.tan)
        elif key == "log":
            self._apply_unary(math.log10)
        elif key == "ln":
            self._apply_unary(math.log)
        elif key == "x^y":
            self._expr += "**"
            self._expr_var.set(self._expr)
        else:
            self._expr += key
            self._expr_var.set(self._expr)

    def _apply_unary(self, fn):
        try:
            val = float(self._result.get())
            res = fn(val)
            res_str = f"{res:.10g}"
            self._result.set(res_str)
            self._expr = res_str
            self._expr_var.set("")
        except (ValueError, ZeroDivisionError):
            self._result.set("Error")
            self._expr = ""

    def _evaluate(self):
        try:
            self._expr_var.set(self._expr + " =")
            result = eval(self._expr, {"__builtins__": {}})  # nosec — no user code
            result_str = f"{result:.10g}"
            self._result.set(result_str)
            self._expr = result_str
        except Exception:
            self._result.set("Error")
            self._expr = ""


if __name__ == "__main__":
    app = Calculator()
    app.mainloop()
