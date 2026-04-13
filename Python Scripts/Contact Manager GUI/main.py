"""Contact Manager GUI — Tkinter desktop app.

Full-featured contact manager with groups, favorites,
search, import/export (CSV), and JSON persistence.

Usage:
    python main.py
"""

import csv
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

DATA_FILE = os.path.join(os.path.dirname(__file__), "contacts_manager.json")
GROUPS    = ["Personal", "Work", "Family", "Friends", "Business", "Other"]


def load() -> list[dict]:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def save(data: list[dict]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class ContactManager(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Contact Manager")
        self.geometry("1000x620")
        self.configure(bg="#1e1e2e")

        self._data     = load()
        self._edit_idx = None
        self._build_ui()
        self._refresh()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        tb = tk.Frame(self, bg="#181825")
        tb.pack(fill="x")
        for lbl, cmd in [("+ New", self._new), ("✏ Edit", self._edit_selected),
                          ("🗑 Delete", self._delete), ("★ Favorite", self._toggle_favorite),
                          ("⬆ Import CSV", self._import_csv), ("⬇ Export CSV", self._export_csv)]:
            bg = "#cba6f7" if lbl == "+ New" else "#f38ba8" if "Delete" in lbl else "#313244"
            fg = "#1e1e2e" if bg != "#313244" else "#cdd6f4"
            tk.Button(tb, text=lbl, command=cmd, bg=bg, fg=fg,
                      relief="flat", padx=6).pack(side="left", padx=2, pady=6)

        # Filter bar
        flt = tk.Frame(self, bg="#1e1e2e")
        flt.pack(fill="x", padx=8, pady=4)
        tk.Label(flt, text="🔍", bg="#1e1e2e", fg="#888").pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh())
        tk.Entry(flt, textvariable=self._search_var, bg="#313244", fg="#cdd6f4",
                 insertbackground="#cba6f7", font=("Consolas", 11),
                 relief="flat", width=24).pack(side="left", padx=4)
        tk.Label(flt, text="Group:", bg="#1e1e2e", fg="#888").pack(side="left", padx=(8, 2))
        self._grp_var = tk.StringVar(value="All")
        tk.OptionMenu(flt, self._grp_var, "All", "Favorites", *GROUPS,
                      command=lambda _: self._refresh()).pack(side="left")
        self._count_var = tk.StringVar(value="0 contacts")
        tk.Label(flt, textvariable=self._count_var, bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).pack(side="right")

        # Paned
        paned = tk.PanedWindow(self, orient="horizontal", bg="#1e1e2e",
                                sashwidth=6, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        # Left: contact list
        lf = tk.Frame(paned, bg="#1e1e2e")
        paned.add(lf, minsize=500)
        cols = ("star", "name", "phone", "email", "company", "group")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        for col, w in zip(cols, [30, 180, 120, 190, 130, 90]):
            self._tree.heading(col, text=col.upper())
            self._tree.column(col, width=w, anchor="w")
        self._tree.tag_configure("fav", foreground="#fab387")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        self._tree.bind("<Double-Button-1>", lambda _: self._edit_selected())
        sb = ttk.Scrollbar(lf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # Right: detail / form
        rf = tk.Frame(paned, bg="#1e1e2e")
        paned.add(rf, minsize=300)

        self._form_title_var = tk.StringVar(value="Contact Details")
        tk.Label(rf, textvariable=self._form_title_var, bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 12, "bold")).pack(anchor="w", padx=8, pady=4)

        self._form_fields = {}
        for lbl in ["First Name", "Last Name", "Phone", "Mobile", "Email",
                     "Company", "Job Title", "Address", "City", "Country", "Birthday", "Notes"]:
            tk.Label(rf, text=lbl, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 8)).pack(anchor="w", padx=8)
            if lbl == "Notes":
                t = tk.Text(rf, bg="#313244", fg="#cdd6f4", height=3, width=32,
                            font=("Consolas", 9), insertbackground="#cba6f7", relief="flat")
                t.pack(fill="x", padx=8, pady=(0, 3))
                self._form_fields[lbl] = t
            else:
                e = tk.Entry(rf, bg="#313244", fg="#cdd6f4",
                             insertbackground="#cba6f7", font=("Consolas", 9),
                             relief="flat")
                e.pack(fill="x", padx=8, pady=(0, 3))
                self._form_fields[lbl] = e

        tk.Label(rf, text="Group", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 8)).pack(anchor="w", padx=8)
        self._form_grp = tk.StringVar(value=GROUPS[0])
        tk.OptionMenu(rf, self._form_grp, *GROUPS).pack(fill="x", padx=8, pady=(0, 4))

        btn = tk.Frame(rf, bg="#1e1e2e")
        btn.pack(fill="x", padx=8, pady=4)
        tk.Button(btn, text="💾 Save", command=self._save,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left", padx=2)
        tk.Button(btn, text="✕ Clear", command=self._clear_form,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def _new(self):
        self._edit_idx = None
        self._clear_form()
        self._form_title_var.set("New Contact")

    def _save(self):
        f = {}
        for lbl, w in self._form_fields.items():
            if isinstance(w, tk.Text):
                f[lbl] = w.get("1.0", "end-1c").strip()
            else:
                f[lbl] = w.get().strip()
        name = f"{f.get('First Name', '')} {f.get('Last Name', '')}".strip()
        if not name:
            messagebox.showerror("Missing", "At least a first name is required.")
            return
        entry = {**f, "group": self._form_grp.get(),
                 "favorite": self._data[self._edit_idx].get("favorite", False)
                             if self._edit_idx is not None else False}
        if self._edit_idx is not None:
            self._data[self._edit_idx] = entry
        else:
            self._data.append(entry)
        save(self._data)
        self._refresh()
        self._form_title_var.set(f"Saved: {name}")

    def _delete(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        name = self._data[idx].get("First Name", "")
        if messagebox.askyesno("Delete", f"Delete {name}?"):
            self._data.pop(idx)
            save(self._data)
            self._clear_form()
            self._refresh()

    def _edit_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        self._edit_idx = idx
        c = self._data[idx]
        self._form_title_var.set(f"Editing: {c.get('First Name','')} {c.get('Last Name','')}")
        for lbl, w in self._form_fields.items():
            val = c.get(lbl, "")
            if isinstance(w, tk.Text):
                w.delete("1.0", "end"); w.insert("1.0", val)
            else:
                w.delete(0, "end"); w.insert(0, val)
        self._form_grp.set(c.get("group", GROUPS[0]))

    def _on_select(self, _):
        self._edit_selected()

    def _toggle_favorite(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        self._data[idx]["favorite"] = not self._data[idx].get("favorite", False)
        save(self._data)
        self._refresh()

    def _clear_form(self):
        self._edit_idx = None
        self._form_title_var.set("Contact Details")
        for w in self._form_fields.values():
            if isinstance(w, tk.Text):
                w.delete("1.0", "end")
            else:
                w.delete(0, "end")
        self._form_grp.set(GROUPS[0])

    def _refresh(self):
        self._tree.delete(*self._tree.get_children())
        q   = self._search_var.get().lower()
        gf  = self._grp_var.get()
        shown = 0
        for i, c in enumerate(self._data):
            if gf == "Favorites" and not c.get("favorite"):
                continue
            if gf not in ("All", "Favorites") and c.get("group") != gf:
                continue
            full = f"{c.get('First Name','')} {c.get('Last Name','')}"
            if q and q not in full.lower() and q not in c.get("phone", "").lower():
                continue
            tags = (str(i),)
            if c.get("favorite"):
                tags = tags + ("fav",)
            self._tree.insert("", "end", tags=tags,
                              values=("★" if c.get("favorite") else " ",
                                      full, c.get("Phone", ""), c.get("Email", ""),
                                      c.get("Company", ""), c.get("group", "")))
            shown += 1
        self._count_var.set(f"{shown} / {len(self._data)} contacts")

    # ── Import / Export ───────────────────────────────────────────────────────

    def _import_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        added = 0
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._data.append(dict(row))
                added += 1
        save(self._data)
        self._refresh()
        messagebox.showinfo("Imported", f"Imported {added} contacts.")

    def _export_csv(self):
        if not self._data:
            messagebox.showinfo("Empty", "No contacts to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
        if path:
            keys = list(self._data[0].keys())
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self._data)
            messagebox.showinfo("Exported", f"Exported {len(self._data)} contacts to {path}.")


if __name__ == "__main__":
    app = ContactManager()
    app.mainloop()
