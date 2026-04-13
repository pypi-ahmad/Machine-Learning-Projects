"""Attendance App — Tkinter desktop app.

Track student or employee attendance with daily records,
reports, and percentage calculations. JSON persistence.

Usage:
    python main.py
"""

import json
import os
import tkinter as tk
from datetime import date, datetime, timedelta
from tkinter import messagebox, ttk

DATA_FILE = os.path.join(os.path.dirname(__file__), "attendance.json")
STATUSES  = ["Present", "Absent", "Late", "Excused"]
STATUS_COLORS = {"Present": "#a6e3a1", "Absent": "#f38ba8",
                 "Late": "#fab387", "Excused": "#89b4fa"}


def load() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"members": [], "records": {}}


def save(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attendance Tracker")
        self.geometry("960x600")
        self.configure(bg="#1e1e2e")

        self._data    = load()
        self._members = self._data.get("members", [])  # list of names
        self._records = self._data.get("records", {})  # {date: {name: status}}
        self._sel_date = str(date.today())

        self._build_ui()
        self._refresh_member_list()
        self._load_date_records()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1e1e2e")
        style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4",
                         padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#45475a")])

        self._mark_tab   = tk.Frame(nb, bg="#1e1e2e")
        self._report_tab = tk.Frame(nb, bg="#1e1e2e")
        self._member_tab = tk.Frame(nb, bg="#1e1e2e")
        nb.add(self._mark_tab,   text=" Mark Attendance ")
        nb.add(self._report_tab, text=" Reports ")
        nb.add(self._member_tab, text=" Members ")

        self._build_mark_tab()
        self._build_report_tab()
        self._build_member_tab()

    # ── Mark tab ──────────────────────────────────────────────────────────────

    def _build_mark_tab(self):
        top = tk.Frame(self._mark_tab, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)

        tk.Label(top, text="Date:", bg="#1e1e2e", fg="#888").pack(side="left")
        self._date_var = tk.StringVar(value=self._sel_date)
        tk.Entry(top, textvariable=self._date_var, bg="#313244", fg="#cdd6f4", width=12,
                 insertbackground="#cba6f7", font=("Consolas", 11), relief="flat").pack(side="left", padx=4)
        tk.Button(top, text="Load", command=self._load_date_records,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)
        tk.Button(top, text="◀ Prev Day", command=lambda: self._shift_date(-1),
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)
        tk.Button(top, text="▶ Next Day", command=lambda: self._shift_date(1),
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)
        tk.Button(top, text="Today", command=self._goto_today,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat").pack(side="left", padx=4)

        # Mark all buttons
        mf = tk.Frame(self._mark_tab, bg="#1e1e2e")
        mf.pack(fill="x", padx=8, pady=2)
        tk.Label(mf, text="Mark all:", bg="#1e1e2e", fg="#888").pack(side="left")
        for s in STATUSES:
            tk.Button(mf, text=s, bg=STATUS_COLORS[s], fg="#1e1e2e", relief="flat",
                      command=lambda st=s: self._mark_all(st)).pack(side="left", padx=2)
        tk.Button(mf, text="💾 Save", command=self._save_records,
                  bg="#a6e3a1", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="right")

        # Scrollable list of members with radio buttons per status
        canvas_frame = tk.Frame(self._mark_tab, bg="#1e1e2e")
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self._att_canvas = tk.Canvas(canvas_frame, bg="#1e1e2e", highlightthickness=0)
        sb = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._att_canvas.yview)
        self._att_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._att_canvas.pack(fill="both", expand=True)

        self._att_inner = tk.Frame(self._att_canvas, bg="#1e1e2e")
        self._att_canvas.create_window((0, 0), window=self._att_inner, anchor="nw")
        self._att_inner.bind("<Configure>",
                              lambda e: self._att_canvas.configure(
                                  scrollregion=self._att_canvas.bbox("all")))
        self._status_vars = {}   # name → StringVar

    # ── Report tab ────────────────────────────────────────────────────────────

    def _build_report_tab(self):
        top = tk.Frame(self._report_tab, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="From:", bg="#1e1e2e", fg="#888").pack(side="left")
        self._from_var = tk.StringVar(value=str(date.today() - timedelta(days=30)))
        tk.Entry(top, textvariable=self._from_var, bg="#313244", fg="#cdd6f4", width=12,
                 font=("Consolas", 10), relief="flat").pack(side="left", padx=4)
        tk.Label(top, text="To:", bg="#1e1e2e", fg="#888").pack(side="left")
        self._to_var = tk.StringVar(value=str(date.today()))
        tk.Entry(top, textvariable=self._to_var, bg="#313244", fg="#cdd6f4", width=12,
                 font=("Consolas", 10), relief="flat").pack(side="left", padx=4)
        tk.Button(top, text="Generate Report", command=self._generate_report,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left", padx=8)

        self._report_text = tk.Text(self._report_tab, bg="#313244", fg="#cdd6f4",
                                     font=("Consolas", 10), relief="flat",
                                     state="disabled")
        self._report_text.pack(fill="both", expand=True, padx=8, pady=4)

    # ── Members tab ───────────────────────────────────────────────────────────

    def _build_member_tab(self):
        top = tk.Frame(self._member_tab, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="Name:", bg="#1e1e2e", fg="#888").pack(side="left")
        self._new_member_var = tk.StringVar()
        tk.Entry(top, textvariable=self._new_member_var, bg="#313244", fg="#cdd6f4",
                 width=24, insertbackground="#cba6f7", font=("Consolas", 11),
                 relief="flat").pack(side="left", padx=4)
        tk.Button(top, text="+ Add Member", command=self._add_member,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left")
        tk.Button(top, text="✕ Remove", command=self._remove_member,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(side="left", padx=4)

        self._member_lb = tk.Listbox(self._member_tab, bg="#313244", fg="#cdd6f4",
                                      font=("Consolas", 11), selectbackground="#45475a",
                                      activestyle="none")
        self._member_lb.pack(fill="both", expand=True, padx=8, pady=4)

    # ── Logic ──────────────────────────────────────────────────────────────────

    def _load_date_records(self):
        self._sel_date = self._date_var.get().strip()
        day_records    = self._records.get(self._sel_date, {})

        for w in self._att_inner.winfo_children():
            w.destroy()
        self._status_vars = {}

        # Header
        hdr = tk.Frame(self._att_inner, bg="#313244")
        hdr.pack(fill="x", pady=2)
        tk.Label(hdr, text=f"{'#':<4}{'Name':<25}", bg="#313244", fg="#888",
                 font=("Consolas", 10)).pack(side="left", padx=4)
        for s in STATUSES:
            tk.Label(hdr, text=f"{s:^10}", bg="#313244", fg=STATUS_COLORS[s],
                     font=("Consolas", 10), width=10).pack(side="left")

        for i, name in enumerate(self._members, 1):
            row    = tk.Frame(self._att_inner, bg="#1e1e2e" if i % 2 else "#181825")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{i:<4}{name:<25}", bg=row.cget("bg"), fg="#cdd6f4",
                     font=("Consolas", 10), width=28).pack(side="left", padx=4)
            var    = tk.StringVar(value=day_records.get(name, "Present"))
            self._status_vars[name] = var
            for s in STATUSES:
                tk.Radiobutton(row, variable=var, value=s, bg=row.cget("bg"),
                               fg=STATUS_COLORS[s], selectcolor=row.cget("bg"),
                               activebackground=row.cget("bg"), text=s,
                               font=("Consolas", 9)).pack(side="left", padx=2)

    def _mark_all(self, status: str):
        for var in self._status_vars.values():
            var.set(status)

    def _save_records(self):
        self._records[self._sel_date] = {name: var.get()
                                         for name, var in self._status_vars.items()}
        self._data["records"] = self._records
        save(self._data)
        messagebox.showinfo("Saved", f"Attendance for {self._sel_date} saved.")

    def _shift_date(self, delta: int):
        try:
            d = datetime.strptime(self._date_var.get().strip(), "%Y-%m-%d").date()
            self._date_var.set(str(d + timedelta(days=delta)))
            self._load_date_records()
        except ValueError:
            pass

    def _goto_today(self):
        self._date_var.set(str(date.today()))
        self._load_date_records()

    def _generate_report(self):
        try:
            frm = datetime.strptime(self._from_var.get(), "%Y-%m-%d").date()
            to  = datetime.strptime(self._to_var.get(), "%Y-%m-%d").date()
        except ValueError:
            messagebox.showerror("Invalid Date", "Use YYYY-MM-DD format.")
            return

        lines = [f"Attendance Report: {frm} → {to}\n"]
        for name in self._members:
            counts = {s: 0 for s in STATUSES}
            total  = 0
            d = frm
            while d <= to:
                ds = str(d)
                if ds in self._records and name in self._records[ds]:
                    counts[self._records[ds][name]] += 1
                    total += 1
                d += timedelta(days=1)
            pct = counts["Present"] / total * 100 if total else 0
            lines.append(f"{name:<24}  P={counts['Present']} A={counts['Absent']} "
                         f"L={counts['Late']} E={counts['Excused']}  "
                         f"Attendance: {pct:.1f}%")

        self._report_text.config(state="normal")
        self._report_text.delete("1.0", "end")
        self._report_text.insert("1.0", "\n".join(lines))
        self._report_text.config(state="disabled")

    def _add_member(self):
        name = self._new_member_var.get().strip()
        if name and name not in self._members:
            self._members.append(name)
            self._data["members"] = self._members
            save(self._data)
            self._refresh_member_list()
            self._load_date_records()
        self._new_member_var.set("")

    def _remove_member(self):
        sel = self._member_lb.curselection()
        if sel:
            name = self._member_lb.get(sel[0])
            if messagebox.askyesno("Remove", f"Remove member '{name}'?"):
                self._members.remove(name)
                self._data["members"] = self._members
                save(self._data)
                self._refresh_member_list()
                self._load_date_records()

    def _refresh_member_list(self):
        self._member_lb.delete(0, "end")
        for n in self._members:
            self._member_lb.insert("end", n)


if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
