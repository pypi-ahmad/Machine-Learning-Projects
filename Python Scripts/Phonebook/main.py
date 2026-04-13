"""Phonebook — Tkinter desktop app.

Manage contacts with name, phone, email, and group.
Supports search, edit, delete, and JSON persistence.

Usage:
    python main.py
"""

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk

DATA_FILE = os.path.join(os.path.dirname(__file__), "contacts.json")
GROUPS    = ["Personal", "Work", "Family", "Friends", "Other"]


def load() -> list[dict]:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def save(data: list[dict]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class Phonebook(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phonebook")
        self.geometry("820x540")
        self.configure(bg="#1e1e2e")

        self._data      = load()
        self._edit_idx  = None
        self._build_ui()
        self._refresh()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Left panel — form
        left = tk.Frame(self, bg="#1e1e2e", width=240)
        left.pack(side="left", fill="y", padx=8, pady=8)
        left.pack_propagate(False)

        tk.Label(left, text="Contact Details", bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 12, "bold")).pack(anchor="w", pady=(0, 8))

        self._vars = {}
        for label in ["Name *", "Phone *", "Email", "Address"]:
            tk.Label(left, text=label, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9)).pack(anchor="w")
            e = tk.Entry(left, bg="#313244", fg="#cdd6f4", insertbackground="#cba6f7",
                         font=("Consolas", 11), relief="flat")
            e.pack(fill="x", pady=(0, 5))
            self._vars[label] = e

        tk.Label(left, text="Group", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).pack(anchor="w")
        self._group_var = tk.StringVar(value=GROUPS[0])
        grp_menu = tk.OptionMenu(left, self._group_var, *GROUPS)
        grp_menu.config(bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                        relief="flat", font=("Consolas", 10))
        grp_menu.pack(fill="x", pady=(0, 10))

        self._save_btn = tk.Button(left, text="💾 Save Contact", command=self._save_contact,
                                    bg="#cba6f7", fg="#1e1e2e", relief="flat",
                                    font=("Consolas", 11, "bold"))
        self._save_btn.pack(fill="x", pady=4)
        tk.Button(left, text="✕ Cancel Edit", command=self._cancel_edit,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(fill="x", pady=2)
        tk.Button(left, text="🗑 Delete", command=self._delete,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(fill="x", pady=2)

        tk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # Stats
        self._stats_var = tk.StringVar()
        tk.Label(left, textvariable=self._stats_var, bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9), justify="left").pack(anchor="w")

        # Right panel — list + search
        right = tk.Frame(self, bg="#1e1e2e")
        right.pack(fill="both", expand=True, padx=(0, 8), pady=8)

        # Search
        sf = tk.Frame(right, bg="#1e1e2e")
        sf.pack(fill="x", pady=(0, 6))
        tk.Label(sf, text="🔍", bg="#1e1e2e", fg="#888").pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh())
        tk.Entry(sf, textvariable=self._search_var, bg="#313244", fg="#cdd6f4",
                 insertbackground="#cba6f7", font=("Consolas", 11),
                 relief="flat").pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(sf, text="Group:", bg="#1e1e2e", fg="#888").pack(side="left", padx=(8, 2))
        self._grp_filter = tk.StringVar(value="All")
        tk.OptionMenu(sf, self._grp_filter, "All", *GROUPS,
                      command=lambda _: self._refresh()).pack(side="left")

        # Table
        cols = ("name", "phone", "email", "group")
        self._tree = ttk.Treeview(right, columns=cols, show="headings", height=22)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        for col, w in zip(cols, [180, 130, 200, 100]):
            self._tree.heading(col, text=col.title())
            self._tree.column(col, width=w, anchor="w")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<Double-Button-1>", self._on_select)
        sb = ttk.Scrollbar(right, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def _save_contact(self):
        name  = self._vars["Name *"].get().strip()
        phone = self._vars["Phone *"].get().strip()
        email = self._vars["Email"].get().strip()
        addr  = self._vars["Address"].get().strip()
        grp   = self._group_var.get()

        if not name or not phone:
            messagebox.showerror("Missing", "Name and Phone are required.")
            return

        entry = {"name": name, "phone": phone, "email": email,
                 "address": addr, "group": grp}

        if self._edit_idx is not None:
            self._data[self._edit_idx] = entry
        else:
            self._data.append(entry)

        save(self._data)
        self._cancel_edit()
        self._refresh()

    def _delete(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select a contact first.")
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        if messagebox.askyesno("Delete", f"Delete {self._data[idx]['name']}?"):
            self._data.pop(idx)
            save(self._data)
            self._cancel_edit()
            self._refresh()

    def _on_select(self, _):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        c   = self._data[idx]
        self._edit_idx = idx
        for label, key in [("Name *", "name"), ("Phone *", "phone"),
                            ("Email", "email"), ("Address", "address")]:
            self._vars[label].delete(0, "end")
            self._vars[label].insert(0, c.get(key, ""))
        self._group_var.set(c.get("group", GROUPS[0]))
        self._save_btn.config(text="✏ Update Contact")

    def _cancel_edit(self):
        self._edit_idx = None
        for e in self._vars.values():
            e.delete(0, "end")
        self._group_var.set(GROUPS[0])
        self._save_btn.config(text="💾 Save Contact")

    def _refresh(self):
        self._tree.delete(*self._tree.get_children())
        q   = self._search_var.get().lower()
        gf  = self._grp_filter.get()
        shown = 0
        for i, c in enumerate(self._data):
            if gf != "All" and c.get("group") != gf:
                continue
            if q and q not in (c.get("name", "") + c.get("phone", "") +
                                c.get("email", "")).lower():
                continue
            self._tree.insert("", "end", tags=(str(i),),
                              values=(c["name"], c["phone"],
                                      c.get("email", ""), c.get("group", "")))
            shown += 1

        by_grp = {}
        for c in self._data:
            by_grp[c.get("group", "Other")] = by_grp.get(c.get("group", "Other"), 0) + 1
        stats = f"Total: {len(self._data)} contacts\nShowing: {shown}\n"
        stats += "\n".join(f"  {g}: {n}" for g, n in by_grp.items())
        self._stats_var.set(stats)


if __name__ == "__main__":
    app = Phonebook()
    app.mainloop()
