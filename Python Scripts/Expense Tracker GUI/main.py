"""Expense Tracker GUI — Tkinter desktop app.

Track personal expenses with categories, date filtering,
charts via ASCII bars, and JSON persistence.

Usage:
    python main.py
"""

import json
import os
import tkinter as tk
from datetime import date, datetime
from tkinter import messagebox, ttk

DATA_FILE = os.path.join(os.path.dirname(__file__), "expenses.json")

CATEGORIES = ["Food", "Transport", "Housing", "Entertainment",
              "Health", "Shopping", "Education", "Utilities", "Other"]


def load_data() -> list[dict]:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []


def save_data(data: list[dict]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class ExpenseTracker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Expense Tracker")
        self.geometry("860x620")
        self.configure(bg="#1e1e2e")

        self._data = load_data()
        self._build_ui()
        self._refresh_table()
        self._update_summary()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Left: form
        left = tk.Frame(self, bg="#1e1e2e", width=260)
        left.pack(side="left", fill="y", padx=8, pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="Add Expense", bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 13, "bold")).pack(anchor="w", pady=(0, 8))

        fields = [("Date (YYYY-MM-DD)", str(date.today())),
                  ("Amount ($)", ""),
                  ("Description", "")]
        self._entries = {}
        for label, default in fields:
            tk.Label(left, text=label, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9)).pack(anchor="w")
            e = tk.Entry(left, bg="#313244", fg="#cdd6f4", insertbackground="#cba6f7",
                         font=("Consolas", 11), relief="flat")
            e.insert(0, default)
            e.pack(fill="x", pady=(0, 6))
            self._entries[label] = e

        tk.Label(left, text="Category", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).pack(anchor="w")
        self._cat_var = tk.StringVar(value=CATEGORIES[0])
        tk.OptionMenu(left, self._cat_var, *CATEGORIES).configure(
            bg="#313244", fg="#cdd6f4", activebackground="#45475a",
            relief="flat", font=("Consolas", 10),
        )
        tk.OptionMenu(left, self._cat_var, *CATEGORIES).pack(fill="x", pady=(0, 10))

        tk.Button(left, text="+ Add Expense", command=self._add,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 11, "bold")).pack(fill="x", pady=4)
        tk.Button(left, text="✕ Delete Selected", command=self._delete,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10)).pack(fill="x", pady=4)

        tk.Separator(left, orient="horizontal").pack(fill="x", pady=12)

        # Summary
        tk.Label(left, text="Summary", bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 11, "bold")).pack(anchor="w")
        self._summary_text = tk.Text(left, bg="#181825", fg="#cdd6f4",
                                      font=("Consolas", 9), height=14,
                                      state="disabled", relief="flat")
        self._summary_text.pack(fill="both", expand=True)

        # Right: table + filter
        right = tk.Frame(self, bg="#1e1e2e")
        right.pack(side="right", fill="both", expand=True, padx=(0, 8), pady=8)

        # Filter row
        flt = tk.Frame(right, bg="#1e1e2e")
        flt.pack(fill="x", pady=(0, 6))
        tk.Label(flt, text="Filter category:", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).pack(side="left")
        self._filter_var = tk.StringVar(value="All")
        tk.OptionMenu(flt, self._filter_var, "All", *CATEGORIES,
                      command=lambda _: self._refresh_table()).pack(side="left", padx=4)
        tk.Label(flt, text="Month (YYYY-MM):", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).pack(side="left", padx=(12, 0))
        self._month_e = tk.Entry(flt, bg="#313244", fg="#cdd6f4", width=8,
                                  font=("Consolas", 10), relief="flat")
        self._month_e.pack(side="left", padx=4)
        tk.Button(flt, text="Apply", command=self._refresh_table,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left")

        # Treeview
        cols = ("date", "amount", "category", "description")
        self._tree = ttk.Treeview(right, columns=cols, show="headings", height=18)
        self._tree.pack(fill="both", expand=True)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")

        widths = [100, 90, 110, 260]
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col.title())
            self._tree.column(col, width=w, anchor="w")

        sb = ttk.Scrollbar(right, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def _add(self):
        date_str = self._entries["Date (YYYY-MM-DD)"].get().strip()
        amt_str  = self._entries["Amount ($)"].get().strip()
        desc     = self._entries["Description"].get().strip()
        cat      = self._cat_var.get()

        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid Date", "Use YYYY-MM-DD format.")
            return
        try:
            amt = float(amt_str)
            if amt <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Amount", "Enter a positive number.")
            return
        if not desc:
            messagebox.showerror("Missing", "Enter a description.")
            return

        self._data.append({"date": date_str, "amount": round(amt, 2),
                            "category": cat, "description": desc})
        save_data(self._data)
        self._entries["Amount ($)"].delete(0, "end")
        self._entries["Description"].delete(0, "end")
        self._refresh_table()
        self._update_summary()

    def _delete(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        if messagebox.askyesno("Delete", "Delete this expense?"):
            self._data.pop(idx)
            save_data(self._data)
            self._refresh_table()
            self._update_summary()

    def _refresh_table(self):
        self._tree.delete(*self._tree.get_children())
        cat_f   = self._filter_var.get()
        month_f = self._month_e.get().strip()
        for i, e in enumerate(self._data):
            if cat_f != "All" and e["category"] != cat_f:
                continue
            if month_f and not e["date"].startswith(month_f):
                continue
            self._tree.insert("", "end", tags=(str(i),),
                              values=(e["date"], f"${e['amount']:.2f}",
                                      e["category"], e["description"]))

    def _update_summary(self):
        total = sum(e["amount"] for e in self._data)
        by_cat = {}
        for e in self._data:
            by_cat[e["category"]] = by_cat.get(e["category"], 0) + e["amount"]

        lines = [f"Total: ${total:.2f}", f"Entries: {len(self._data)}", ""]
        if by_cat:
            lines.append("By Category:")
            max_val = max(by_cat.values()) or 1
            for cat in CATEGORIES:
                if cat in by_cat:
                    v  = by_cat[cat]
                    bar = "█" * int(v / max_val * 14)
                    lines.append(f" {cat[:11]:<11} ${v:7.2f}")
                    lines.append(f"  {bar}")

        self._summary_text.config(state="normal")
        self._summary_text.delete("1.0", "end")
        self._summary_text.insert("1.0", "\n".join(lines))
        self._summary_text.config(state="disabled")


if __name__ == "__main__":
    app = ExpenseTracker()
    app.mainloop()
