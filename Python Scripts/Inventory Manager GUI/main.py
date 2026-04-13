"""Inventory Manager GUI — Tkinter desktop app.

Track products, stock levels, suppliers, and reorder alerts.
Full CRUD with search, low-stock alerts, and JSON persistence.

Usage:
    python main.py
"""

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk

DATA_FILE   = os.path.join(os.path.dirname(__file__), "inventory.json")
CATEGORIES  = ["Electronics", "Clothing", "Food", "Furniture",
               "Sports", "Books", "Health", "Tools", "Other"]


def load() -> list[dict]:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def save(data: list[dict]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class InventoryManager(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Inventory Manager")
        self.geometry("1020x620")
        self.configure(bg="#1e1e2e")

        self._data     = load()
        self._edit_idx = None
        self._build_ui()
        self._refresh()
        self._check_alerts()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Toolbar
        tb = tk.Frame(self, bg="#181825")
        tb.pack(fill="x")
        tk.Button(tb, text="+ Add Item", command=self._new,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left", padx=4, pady=6)
        tk.Button(tb, text="✏ Edit", command=self._edit_selected,
                  bg="#89b4fa", fg="#1e1e2e", relief="flat").pack(side="left", padx=2, pady=6)
        tk.Button(tb, text="🗑 Delete", command=self._delete,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(side="left", padx=2, pady=6)
        tk.Button(tb, text="⚠ Low Stock", command=self._show_alerts,
                  bg="#fab387", fg="#1e1e2e", relief="flat").pack(side="left", padx=6, pady=6)

        # Search + filter
        sf = tk.Frame(self, bg="#1e1e2e")
        sf.pack(fill="x", padx=8, pady=4)
        tk.Label(sf, text="🔍", bg="#1e1e2e", fg="#888").pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh())
        tk.Entry(sf, textvariable=self._search_var, bg="#313244", fg="#cdd6f4",
                 insertbackground="#cba6f7", font=("Consolas", 11),
                 relief="flat", width=24).pack(side="left", padx=4)
        tk.Label(sf, text="Category:", bg="#1e1e2e", fg="#888").pack(side="left", padx=(8, 2))
        self._cat_var = tk.StringVar(value="All")
        tk.OptionMenu(sf, self._cat_var, "All", *CATEGORIES,
                      command=lambda _: self._refresh()).pack(side="left")
        self._count_var = tk.StringVar(value="0 items")
        tk.Label(sf, textvariable=self._count_var, bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).pack(side="right")

        # Main paned window
        paned = tk.PanedWindow(self, orient="horizontal", bg="#1e1e2e",
                                sashwidth=6, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        # Left: table
        lf = tk.Frame(paned, bg="#1e1e2e")
        paned.add(lf, minsize=560)
        cols = ("sku", "name", "category", "qty", "price", "reorder", "supplier")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        widths = [70, 180, 100, 60, 80, 70, 140]
        for col, w in zip(cols, widths):
            self._tree.heading(col, text=col.upper())
            self._tree.column(col, width=w, anchor="w")
        self._tree.pack(fill="both", expand=True)
        self._tree.tag_configure("low", foreground="#f38ba8")
        self._tree.bind("<Double-Button-1>", lambda _: self._edit_selected())
        sb = ttk.Scrollbar(lf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # Right: form
        rf = tk.Frame(paned, bg="#1e1e2e")
        paned.add(rf, minsize=280)

        self._form_title_var = tk.StringVar(value="Item Details")
        tk.Label(rf, textvariable=self._form_title_var, bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 12, "bold")).pack(anchor="w", padx=8, pady=4)

        self._form_fields = {}
        for lbl, default in [("SKU", ""), ("Name", ""), ("Quantity", "0"),
                               ("Unit Price ($)", "0.00"), ("Reorder Level", "10"),
                               ("Supplier", ""), ("Location", "")]:
            tk.Label(rf, text=lbl, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9)).pack(anchor="w", padx=8)
            e = tk.Entry(rf, bg="#313244", fg="#cdd6f4", insertbackground="#cba6f7",
                         font=("Consolas", 11), relief="flat")
            e.insert(0, default)
            e.pack(fill="x", padx=8, pady=(0, 4))
            self._form_fields[lbl] = e

        tk.Label(rf, text="Category", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).pack(anchor="w", padx=8)
        self._form_cat = tk.StringVar(value=CATEGORIES[0])
        cm = tk.OptionMenu(rf, self._form_cat, *CATEGORIES)
        cm.config(bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                  relief="flat", font=("Consolas", 10))
        cm.pack(fill="x", padx=8, pady=(0, 6))

        tk.Label(rf, text="Notes", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).pack(anchor="w", padx=8)
        self._notes_text = tk.Text(rf, bg="#313244", fg="#cdd6f4", height=4,
                                    font=("Consolas", 10), insertbackground="#cba6f7",
                                    relief="flat", padx=4)
        self._notes_text.pack(fill="x", padx=8, pady=(0, 8))

        btn = tk.Frame(rf, bg="#1e1e2e")
        btn.pack(fill="x", padx=8)
        tk.Button(btn, text="💾 Save", command=self._save,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left", padx=2)
        tk.Button(btn, text="✕ Clear", command=self._clear_form,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)

        # Summary
        tk.Separator(rf, orient="horizontal").pack(fill="x", padx=8, pady=8)
        self._summary_var = tk.StringVar()
        tk.Label(rf, textvariable=self._summary_var, bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9), justify="left").pack(anchor="w", padx=8)

        # Status
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self._status_var, bg="#181825", fg="#555",
                 anchor="w", padx=8).pack(fill="x", side="bottom")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def _new(self):
        self._edit_idx = None
        self._clear_form()
        self._form_title_var.set("New Item")

    def _save(self):
        f     = {k: e.get().strip() for k, e in self._form_fields.items()}
        notes = self._notes_text.get("1.0", "end-1c").strip()
        if not f["Name"]:
            messagebox.showerror("Missing", "Name is required.")
            return
        try:
            qty   = int(f["Quantity"])
            price = float(f["Unit Price ($)"])
            reord = int(f["Reorder Level"])
        except ValueError:
            messagebox.showerror("Invalid", "Quantity, Price, and Reorder Level must be numbers.")
            return
        entry = {"sku": f["SKU"], "name": f["Name"], "category": self._form_cat.get(),
                 "qty": qty, "price": price, "reorder": reord,
                 "supplier": f["Supplier"], "location": f["Location"], "notes": notes}
        if self._edit_idx is not None:
            self._data[self._edit_idx] = entry
        else:
            self._data.append(entry)
        save(self._data)
        self._refresh()
        self._check_alerts()
        self._form_title_var.set(f"Saved: {entry['name']}")

    def _delete(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        if messagebox.askyesno("Delete", f"Delete '{self._data[idx]['name']}'?"):
            self._data.pop(idx)
            save(self._data)
            self._clear_form()
            self._refresh()

    def _edit_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        e   = self._data[idx]
        self._edit_idx = idx
        self._form_title_var.set(f"Editing: {e['name']}")
        for lbl, key in [("SKU", "sku"), ("Name", "name"), ("Quantity", "qty"),
                          ("Unit Price ($)", "price"), ("Reorder Level", "reorder"),
                          ("Supplier", "supplier"), ("Location", "location")]:
            self._form_fields[lbl].delete(0, "end")
            self._form_fields[lbl].insert(0, str(e.get(key, "")))
        self._form_cat.set(e.get("category", CATEGORIES[0]))
        self._notes_text.delete("1.0", "end")
        self._notes_text.insert("1.0", e.get("notes", ""))

    def _clear_form(self):
        self._edit_idx = None
        self._form_title_var.set("Item Details")
        for e in self._form_fields.values():
            e.delete(0, "end")
        self._notes_text.delete("1.0", "end")

    def _refresh(self):
        self._tree.delete(*self._tree.get_children())
        q  = self._search_var.get().lower()
        cf = self._cat_var.get()
        shown = 0
        total_val = 0.0
        for i, e in enumerate(self._data):
            if cf != "All" and e.get("category") != cf:
                continue
            if q and q not in e.get("name", "").lower() and q not in e.get("sku", "").lower():
                continue
            tags = (str(i),)
            if e.get("qty", 0) <= e.get("reorder", 0):
                tags = tags + ("low",)
            self._tree.insert("", "end", tags=tags,
                              values=(e.get("sku", ""), e["name"], e.get("category", ""),
                                      e.get("qty", 0), f"${e.get('price', 0):.2f}",
                                      e.get("reorder", 0), e.get("supplier", "")))
            shown += 1
            total_val += e.get("qty", 0) * e.get("price", 0)
        self._count_var.set(f"{shown} / {len(self._data)} items")
        self._summary_var.set(
            f"Items: {shown}\nTotal Value: ${total_val:,.2f}\nLow Stock: "
            f"{sum(1 for e in self._data if e.get('qty', 0) <= e.get('reorder', 0))}")

    def _check_alerts(self):
        low = [e["name"] for e in self._data if e.get("qty", 0) <= e.get("reorder", 0)]
        if low:
            self._status_var.set(f"⚠ Low stock: {', '.join(low[:3])}" +
                                  (" and more..." if len(low) > 3 else ""))
        else:
            self._status_var.set("All stock levels OK.")

    def _show_alerts(self):
        low = [f"{e['name']} (qty={e['qty']}, reorder@{e['reorder']})"
               for e in self._data if e.get("qty", 0) <= e.get("reorder", 0)]
        if low:
            messagebox.showwarning("Low Stock Alert",
                                   "Items at or below reorder level:\n\n" + "\n".join(low))
        else:
            messagebox.showinfo("Stock OK", "All items are above reorder level.")


if __name__ == "__main__":
    app = InventoryManager()
    app.mainloop()
