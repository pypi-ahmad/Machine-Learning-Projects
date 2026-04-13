"""Recipe Book — Tkinter desktop app.

Store, search, and manage recipes with ingredients, instructions,
servings, prep time, and category. JSON persistence.

Usage:
    python main.py
"""

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk

DATA_FILE  = os.path.join(os.path.dirname(__file__), "recipes.json")
CATEGORIES = ["Breakfast", "Lunch", "Dinner", "Dessert", "Snack",
              "Beverage", "Soup", "Salad", "Other"]


def load() -> list[dict]:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def save(data: list[dict]):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class RecipeBook(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Recipe Book")
        self.geometry("1000x640")
        self.configure(bg="#1e1e2e")

        self._data     = load()
        self._edit_idx = None
        self._build_ui()
        self._refresh()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top: search + filter
        top = tk.Frame(self, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="🔍", bg="#1e1e2e", fg="#888").pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh())
        tk.Entry(top, textvariable=self._search_var, bg="#313244", fg="#cdd6f4",
                 insertbackground="#cba6f7", font=("Consolas", 11),
                 relief="flat", width=24).pack(side="left", padx=4)
        tk.Label(top, text="Category:", bg="#1e1e2e", fg="#888").pack(side="left", padx=(8, 2))
        self._cat_filter = tk.StringVar(value="All")
        tk.OptionMenu(top, self._cat_filter, "All", *CATEGORIES,
                      command=lambda _: self._refresh()).pack(side="left")
        tk.Button(top, text="+ New Recipe", command=self._new,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="right")

        # Paned: list left, detail right
        paned = tk.PanedWindow(self, orient="horizontal", bg="#1e1e2e",
                                sashwidth=6, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        # Left: recipe list
        lf = tk.Frame(paned, bg="#1e1e2e")
        paned.add(lf, minsize=220)
        cols = ("name", "category", "time")
        self._tree = ttk.Treeview(lf, columns=cols, show="headings", height=28)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        for col, w in zip(cols, [180, 100, 70]):
            self._tree.heading(col, text=col.title())
            self._tree.column(col, width=w, anchor="w")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        sb = ttk.Scrollbar(lf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # Right: detail / edit form
        rf = tk.Frame(paned, bg="#1e1e2e")
        paned.add(rf, minsize=460)

        # Form
        self._form_title = tk.StringVar(value="Select or create a recipe")
        tk.Label(rf, textvariable=self._form_title, bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 13, "bold")).pack(anchor="w", padx=8, pady=4)

        grid = tk.Frame(rf, bg="#1e1e2e")
        grid.pack(fill="x", padx=8)
        self._fields = {}
        for r, (lbl, default) in enumerate([("Recipe Name", ""), ("Servings", "4"),
                                             ("Prep Time (min)", "30"), ("Cook Time (min)", "30")]):
            tk.Label(grid, text=lbl, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9)).grid(row=r, column=0, sticky="w", pady=2)
            e = tk.Entry(grid, bg="#313244", fg="#cdd6f4", insertbackground="#cba6f7",
                         font=("Consolas", 11), relief="flat", width=28)
            e.insert(0, default)
            e.grid(row=r, column=1, padx=8, pady=2, sticky="w")
            self._fields[lbl] = e

        tk.Label(grid, text="Category", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).grid(row=4, column=0, sticky="w", pady=2)
        self._cat_var = tk.StringVar(value=CATEGORIES[0])
        cm = tk.OptionMenu(grid, self._cat_var, *CATEGORIES)
        cm.config(bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                  relief="flat", font=("Consolas", 10))
        cm.grid(row=4, column=1, padx=8, sticky="w")

        # Ingredients & Instructions
        for attr, lbl, height in [("_ingr_text", "Ingredients (one per line)", 6),
                                    ("_inst_text", "Instructions", 9)]:
            tk.Label(rf, text=lbl, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9)).pack(anchor="w", padx=8)
            t = tk.Text(rf, bg="#313244", fg="#cdd6f4", height=height,
                        font=("Consolas", 10), insertbackground="#cba6f7",
                        relief="flat", padx=4, pady=4)
            t.pack(fill="x", padx=8, pady=(0, 4))
            setattr(self, attr, t)

        # Buttons
        btn = tk.Frame(rf, bg="#1e1e2e")
        btn.pack(fill="x", padx=8, pady=4)
        tk.Button(btn, text="💾 Save", command=self._save,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left", padx=4)
        tk.Button(btn, text="🗑 Delete", command=self._delete,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(side="left", padx=4)
        tk.Button(btn, text="✕ Clear", command=self._clear_form,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=4)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def _new(self):
        self._edit_idx = None
        self._clear_form()
        self._form_title.set("New Recipe")

    def _save(self):
        name  = self._fields["Recipe Name"].get().strip()
        serv  = self._fields["Servings"].get().strip()
        prep  = self._fields["Prep Time (min)"].get().strip()
        cook  = self._fields["Cook Time (min)"].get().strip()
        cat   = self._cat_var.get()
        ingr  = self._ingr_text.get("1.0", "end-1c").strip()
        inst  = self._inst_text.get("1.0", "end-1c").strip()

        if not name:
            messagebox.showerror("Missing", "Recipe name is required.")
            return

        entry = {"name": name, "servings": serv, "prep": prep,
                 "cook": cook, "category": cat, "ingredients": ingr, "instructions": inst}

        if self._edit_idx is not None:
            self._data[self._edit_idx] = entry
        else:
            self._data.append(entry)

        save(self._data)
        self._refresh()
        self._form_title.set(f"Saved: {name}")

    def _delete(self):
        if self._edit_idx is None:
            return
        name = self._data[self._edit_idx]["name"]
        if messagebox.askyesno("Delete", f"Delete '{name}'?"):
            self._data.pop(self._edit_idx)
            save(self._data)
            self._edit_idx = None
            self._clear_form()
            self._refresh()

    def _on_select(self, _):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        self._edit_idx = idx
        r = self._data[idx]
        self._form_title.set(r["name"])
        for lbl, key in [("Recipe Name", "name"), ("Servings", "servings"),
                          ("Prep Time (min)", "prep"), ("Cook Time (min)", "cook")]:
            self._fields[lbl].delete(0, "end")
            self._fields[lbl].insert(0, r.get(key, ""))
        self._cat_var.set(r.get("category", CATEGORIES[0]))
        self._ingr_text.delete("1.0", "end")
        self._ingr_text.insert("1.0", r.get("ingredients", ""))
        self._inst_text.delete("1.0", "end")
        self._inst_text.insert("1.0", r.get("instructions", ""))

    def _clear_form(self):
        self._edit_idx = None
        for e in self._fields.values():
            e.delete(0, "end")
        self._ingr_text.delete("1.0", "end")
        self._inst_text.delete("1.0", "end")
        self._form_title.set("New Recipe")

    def _refresh(self):
        self._tree.delete(*self._tree.get_children())
        q  = self._search_var.get().lower()
        cf = self._cat_filter.get()
        for i, r in enumerate(self._data):
            if cf != "All" and r.get("category") != cf:
                continue
            if q and q not in r.get("name", "").lower():
                continue
            time_str = f"{r.get('prep', 0)}+{r.get('cook', 0)}m"
            self._tree.insert("", "end", tags=(str(i),),
                              values=(r["name"], r.get("category", ""), time_str))


if __name__ == "__main__":
    app = RecipeBook()
    app.mainloop()
