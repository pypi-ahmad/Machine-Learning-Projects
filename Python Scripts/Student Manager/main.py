"""Student Manager — Tkinter desktop app.

Manage student records including grades, subjects, and GPA.
Supports add/edit/delete, grade calculator, and JSON persistence.

Usage:
    python main.py
"""

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk

DATA_FILE = os.path.join(os.path.dirname(__file__), "students.json")
GRADES    = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "F"]
GRADE_PTS = {"A+": 4.0, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
             "C+": 2.3, "C": 2.0, "C-": 1.7, "D": 1.0, "F": 0.0}
SUBJECTS  = ["Math", "Science", "English", "History", "Computer Science",
             "Physics", "Chemistry", "Biology", "Literature", "Economics"]


def load() -> list[dict]:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def save(data: list[dict]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def calc_gpa(grades: dict) -> float:
    pts = [GRADE_PTS.get(g, 0) for g in grades.values()]
    return round(sum(pts) / len(pts), 2) if pts else 0.0


class StudentManager(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Manager")
        self.geometry("1000x640")
        self.configure(bg="#1e1e2e")

        self._data     = load()
        self._edit_idx = None
        self._build_ui()
        self._refresh()

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

        self._list_tab  = tk.Frame(nb, bg="#1e1e2e")
        self._form_tab  = tk.Frame(nb, bg="#1e1e2e")
        self._stats_tab = tk.Frame(nb, bg="#1e1e2e")
        nb.add(self._list_tab,  text=" Student List ")
        nb.add(self._form_tab,  text=" Add / Edit ")
        nb.add(self._stats_tab, text=" Statistics ")

        self._build_list_tab()
        self._build_form_tab()
        self._build_stats_tab()

    # ── List tab ──────────────────────────────────────────────────────────────

    def _build_list_tab(self):
        top = tk.Frame(self._list_tab, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="🔍", bg="#1e1e2e", fg="#888").pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh())
        tk.Entry(top, textvariable=self._search_var, bg="#313244", fg="#cdd6f4",
                 insertbackground="#cba6f7", font=("Consolas", 11),
                 relief="flat", width=24).pack(side="left", padx=4)
        tk.Button(top, text="✏ Edit", command=self._edit_selected,
                  bg="#89b4fa", fg="#1e1e2e", relief="flat").pack(side="right", padx=2)
        tk.Button(top, text="🗑 Delete", command=self._delete_selected,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(side="right", padx=2)
        tk.Button(top, text="+ New", command=self._new_student,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="right", padx=2)

        cols = ("id", "name", "grade", "gpa", "email", "age")
        self._tree = ttk.Treeview(self._list_tab, columns=cols, show="headings")
        style = ttk.Style()
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        for col, w in zip(cols, [60, 200, 80, 70, 200, 50]):
            self._tree.heading(col, text=col.upper())
            self._tree.column(col, width=w, anchor="w")
        self._tree.pack(fill="both", expand=True, padx=8, pady=4)
        self._tree.bind("<Double-Button-1>", lambda _: self._edit_selected())
        sb = ttk.Scrollbar(self._list_tab, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self._count_var = tk.StringVar(value="0 students")
        tk.Label(self._list_tab, textvariable=self._count_var,
                 bg="#1e1e2e", fg="#888", font=("Consolas", 9)).pack(anchor="w", padx=8)

    # ── Form tab ──────────────────────────────────────────────────────────────

    def _build_form_tab(self):
        self._form_title_var = tk.StringVar(value="Add New Student")
        tk.Label(self._form_tab, textvariable=self._form_title_var, bg="#1e1e2e",
                 fg="#cba6f7", font=("Consolas", 13, "bold")).pack(anchor="w", padx=12, pady=8)

        # Basic info
        info = tk.Frame(self._form_tab, bg="#1e1e2e")
        info.pack(fill="x", padx=12)
        self._form_fields = {}
        for r, (lbl, default) in enumerate([("Student ID", ""),
                                              ("Full Name", ""),
                                              ("Email", ""),
                                              ("Age", "18"),
                                              ("Class/Grade", "10th")]):
            tk.Label(info, text=lbl, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9)).grid(row=r, column=0, sticky="w", pady=3)
            e = tk.Entry(info, bg="#313244", fg="#cdd6f4", insertbackground="#cba6f7",
                         font=("Consolas", 11), relief="flat", width=30)
            e.insert(0, default)
            e.grid(row=r, column=1, padx=8, pady=3, sticky="w")
            self._form_fields[lbl] = e

        # Grades per subject
        tk.Label(self._form_tab, text="Subject Grades", bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        grades_frame = tk.Frame(self._form_tab, bg="#1e1e2e")
        grades_frame.pack(fill="x", padx=12)
        self._grade_vars = {}
        for i, subj in enumerate(SUBJECTS):
            r, c = divmod(i, 3)
            tk.Label(grades_frame, text=subj, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9), width=16, anchor="w").grid(row=r, column=c*2, pady=2)
            var = tk.StringVar(value="B")
            ttk.Combobox(grades_frame, textvariable=var, values=GRADES, width=5,
                         state="readonly").grid(row=r, column=c*2+1, padx=4, pady=2)
            self._grade_vars[subj] = var

        btn = tk.Frame(self._form_tab, bg="#1e1e2e")
        btn.pack(fill="x", padx=12, pady=12)
        tk.Button(btn, text="💾 Save Student", command=self._save_student,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 11, "bold")).pack(side="left", padx=4)
        tk.Button(btn, text="✕ Cancel", command=self._cancel_edit,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=4)

    # ── Stats tab ─────────────────────────────────────────────────────────────

    def _build_stats_tab(self):
        self._stats_text = tk.Text(self._stats_tab, bg="#313244", fg="#cdd6f4",
                                    font=("Consolas", 10), relief="flat", state="disabled")
        self._stats_text.pack(fill="both", expand=True, padx=8, pady=8)
        tk.Button(self._stats_tab, text="🔄 Refresh Stats", command=self._update_stats,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(pady=4)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def _new_student(self):
        self._edit_idx = None
        for e in self._form_fields.values():
            e.delete(0, "end")
        for v in self._grade_vars.values():
            v.set("B")
        self._form_title_var.set("Add New Student")

    def _save_student(self):
        fields = {k: e.get().strip() for k, e in self._form_fields.items()}
        if not fields["Full Name"]:
            messagebox.showerror("Missing", "Name is required.")
            return
        grades = {subj: var.get() for subj, var in self._grade_vars.items()}
        entry  = {**fields, "grades": grades, "gpa": calc_gpa(grades)}

        if self._edit_idx is not None:
            self._data[self._edit_idx] = entry
        else:
            self._data.append(entry)
        save(self._data)
        self._cancel_edit()
        self._refresh()
        self._update_stats()

    def _edit_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        s   = self._data[idx]
        self._edit_idx = idx
        self._form_title_var.set(f"Editing: {s['Full Name']}")
        for lbl, e in self._form_fields.items():
            e.delete(0, "end")
            e.insert(0, s.get(lbl, ""))
        for subj, var in self._grade_vars.items():
            var.set(s.get("grades", {}).get(subj, "B"))

    def _cancel_edit(self):
        self._edit_idx = None
        self._form_title_var.set("Add New Student")

    def _delete_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx  = int(self._tree.item(sel[0], "tags")[0])
        name = self._data[idx]["Full Name"]
        if messagebox.askyesno("Delete", f"Delete student '{name}'?"):
            self._data.pop(idx)
            save(self._data)
            self._refresh()

    def _refresh(self):
        self._tree.delete(*self._tree.get_children())
        q = self._search_var.get().lower()
        shown = 0
        for i, s in enumerate(self._data):
            if q and q not in s.get("Full Name", "").lower():
                continue
            self._tree.insert("", "end", tags=(str(i),),
                              values=(s.get("Student ID", ""), s.get("Full Name", ""),
                                      s.get("Class/Grade", ""), s.get("gpa", ""),
                                      s.get("Email", ""), s.get("Age", "")))
            shown += 1
        self._count_var.set(f"{shown} / {len(self._data)} students")

    def _update_stats(self):
        if not self._data:
            return
        gpas    = [s.get("gpa", 0) for s in self._data]
        avg_gpa = sum(gpas) / len(gpas)
        honor   = sum(1 for g in gpas if g >= 3.5)
        fail    = sum(1 for g in gpas if g < 1.0)

        lines = [f"Total students : {len(self._data)}",
                 f"Average GPA    : {avg_gpa:.2f}",
                 f"Honor roll (≥3.5) : {honor}",
                 f"At risk (GPA<1.0) : {fail}", ""]

        # GPA distribution
        buckets = {"4.0": 0, "3.0-3.9": 0, "2.0-2.9": 0, "1.0-1.9": 0, "<1.0": 0}
        for g in gpas:
            if g >= 4.0:   buckets["4.0"] += 1
            elif g >= 3.0: buckets["3.0-3.9"] += 1
            elif g >= 2.0: buckets["2.0-2.9"] += 1
            elif g >= 1.0: buckets["1.0-1.9"] += 1
            else:          buckets["<1.0"] += 1
        lines.append("GPA Distribution:")
        mx = max(buckets.values()) or 1
        for k, v in buckets.items():
            bar = "█" * int(v / mx * 20)
            lines.append(f"  {k:>8}: {bar} {v}")

        self._stats_text.config(state="normal")
        self._stats_text.delete("1.0", "end")
        self._stats_text.insert("1.0", "\n".join(lines))
        self._stats_text.config(state="disabled")


if __name__ == "__main__":
    app = StudentManager()
    app.mainloop()
