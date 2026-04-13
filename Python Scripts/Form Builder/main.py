"""Form Builder — Tkinter desktop app.

Build custom forms visually, collect responses, and export
results to CSV. Supports text, number, dropdown, checkbox,
and textarea field types.

Usage:
    python main.py
"""

import csv
import json
import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, simpledialog, ttk

DATA_FILE = os.path.join(os.path.dirname(__file__), "forms.json")

FIELD_TYPES = ["Text", "Number", "Email", "Dropdown", "Checkbox", "Textarea", "Date"]


def load() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"forms": {}, "responses": {}}


def save(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class FormBuilder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Form Builder")
        self.geometry("960x620")
        self.configure(bg="#1e1e2e")

        self._data      = load()
        self._forms     = self._data.get("forms", {})
        self._responses = self._data.get("responses", {})
        self._cur_form  = None

        self._build_ui()
        self._refresh_forms()

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

        self._designer_tab  = tk.Frame(nb, bg="#1e1e2e")
        self._fill_tab      = tk.Frame(nb, bg="#1e1e2e")
        self._responses_tab = tk.Frame(nb, bg="#1e1e2e")
        nb.add(self._designer_tab,  text=" Designer ")
        nb.add(self._fill_tab,      text=" Fill Form ")
        nb.add(self._responses_tab, text=" Responses ")

        self._build_designer()
        self._build_filler()
        self._build_responses()

    # ── Designer tab ─────────────────────────────────────────────────────────

    def _build_designer(self):
        # Left: form list
        left = tk.Frame(self._designer_tab, bg="#1e1e2e", width=180)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.pack_propagate(False)
        tk.Label(left, text="Forms", bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 11, "bold")).pack(anchor="w")
        self._form_lb = tk.Listbox(left, bg="#313244", fg="#cdd6f4", font=("Consolas", 10),
                                    selectbackground="#45475a", activestyle="none")
        self._form_lb.pack(fill="both", expand=True)
        self._form_lb.bind("<<ListboxSelect>>", self._on_form_select)
        for lbl, cmd in [("+ New Form", self._new_form), ("✕ Delete Form", self._del_form)]:
            tk.Button(left, text=lbl, command=cmd, bg="#313244", fg="#cdd6f4",
                      relief="flat").pack(fill="x", pady=2)

        # Right: field designer
        right = tk.Frame(self._designer_tab, bg="#1e1e2e")
        right.pack(fill="both", expand=True, padx=(0, 6), pady=6)

        self._form_name_var = tk.StringVar(value="Select a form")
        tk.Label(right, textvariable=self._form_name_var, bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 12, "bold")).pack(anchor="w")

        # Add field row
        add_row = tk.Frame(right, bg="#1e1e2e")
        add_row.pack(fill="x", pady=4)
        tk.Label(add_row, text="Label:", bg="#1e1e2e", fg="#888").pack(side="left")
        self._fl_entry = tk.Entry(add_row, bg="#313244", fg="#cdd6f4", width=20,
                                   insertbackground="#cba6f7", relief="flat", font=("Consolas", 10))
        self._fl_entry.pack(side="left", padx=4)
        tk.Label(add_row, text="Type:", bg="#1e1e2e", fg="#888").pack(side="left")
        self._ft_var = tk.StringVar(value=FIELD_TYPES[0])
        tk.OptionMenu(add_row, self._ft_var, *FIELD_TYPES).pack(side="left", padx=4)
        tk.Label(add_row, text="Options (comma sep):", bg="#1e1e2e", fg="#888").pack(side="left")
        self._fo_entry = tk.Entry(add_row, bg="#313244", fg="#cdd6f4", width=18,
                                   insertbackground="#cba6f7", relief="flat", font=("Consolas", 10))
        self._fo_entry.pack(side="left", padx=4)
        tk.Button(add_row, text="+ Add Field", command=self._add_field,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat").pack(side="left", padx=4)

        # Fields table
        cols = ("label", "type", "options", "required")
        self._fields_tree = ttk.Treeview(right, columns=cols, show="headings", height=16)
        style = ttk.Style()
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        for col, w in zip(cols, [220, 100, 220, 80]):
            self._fields_tree.heading(col, text=col.title())
            self._fields_tree.column(col, width=w, anchor="w")
        self._fields_tree.pack(fill="both", expand=True)

        btn = tk.Frame(right, bg="#1e1e2e")
        btn.pack(fill="x", pady=4)
        tk.Button(btn, text="↑ Move Up",   command=lambda: self._move_field(-1),
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)
        tk.Button(btn, text="↓ Move Down", command=lambda: self._move_field(1),
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)
        tk.Button(btn, text="✕ Remove",    command=self._remove_field,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(side="left", padx=2)

    # ── Fill tab ──────────────────────────────────────────────────────────────

    def _build_filler(self):
        top = tk.Frame(self._fill_tab, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=8)
        tk.Label(top, text="Select form to fill:", bg="#1e1e2e", fg="#888").pack(side="left")
        self._fill_form_var = tk.StringVar(value="")
        self._fill_menu = tk.OptionMenu(top, self._fill_form_var, "")
        self._fill_menu.config(bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                                relief="flat", font=("Consolas", 10))
        self._fill_menu.pack(side="left", padx=6)
        tk.Button(top, text="Load Form", command=self._load_fill_form,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left")

        self._fill_canvas = tk.Canvas(self._fill_tab, bg="#1e1e2e", highlightthickness=0)
        self._fill_canvas.pack(fill="both", expand=True, padx=8)
        self._fill_scroll = ttk.Scrollbar(self._fill_tab, orient="vertical",
                                          command=self._fill_canvas.yview)
        self._fill_canvas.configure(yscrollcommand=self._fill_scroll.set)
        self._fill_scroll.pack(side="right", fill="y")

        self._fill_inner = tk.Frame(self._fill_canvas, bg="#1e1e2e")
        self._fill_canvas.create_window((0, 0), window=self._fill_inner, anchor="nw")
        self._fill_inner.bind("<Configure>",
                               lambda e: self._fill_canvas.configure(
                                   scrollregion=self._fill_canvas.bbox("all")))
        self._fill_widgets = {}

        tk.Button(self._fill_tab, text="✔ Submit Response", command=self._submit,
                  bg="#a6e3a1", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 11, "bold")).pack(pady=6)

    # ── Responses tab ─────────────────────────────────────────────────────────

    def _build_responses(self):
        top = tk.Frame(self._responses_tab, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="Form:", bg="#1e1e2e", fg="#888").pack(side="left")
        self._resp_form_var = tk.StringVar(value="")
        self._resp_menu = tk.OptionMenu(top, self._resp_form_var, "")
        self._resp_menu.config(bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                                relief="flat", font=("Consolas", 10))
        self._resp_menu.pack(side="left", padx=4)
        tk.Button(top, text="Load", command=self._load_responses,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=2)
        tk.Button(top, text="⬇ Export CSV", command=self._export_csv,
                  bg="#89b4fa", fg="#1e1e2e", relief="flat").pack(side="right")

        self._resp_text = tk.Text(self._responses_tab, bg="#313244", fg="#cdd6f4",
                                   font=("Consolas", 10), relief="flat",
                                   state="disabled")
        self._resp_text.pack(fill="both", expand=True, padx=8, pady=4)

    # ── Form management ───────────────────────────────────────────────────────

    def _new_form(self):
        name = simpledialog.askstring("New Form", "Form name:", parent=self)
        if name and name not in self._forms:
            self._forms[name] = []
            self._responses[name] = []
            self._persist()
            self._refresh_forms()

    def _del_form(self):
        sel = self._form_lb.curselection()
        if not sel:
            return
        name = self._form_lb.get(sel[0])
        if messagebox.askyesno("Delete", f"Delete form '{name}' and all responses?"):
            del self._forms[name]
            self._responses.pop(name, None)
            self._persist()
            self._cur_form = None
            self._refresh_forms()

    def _on_form_select(self, _):
        sel = self._form_lb.curselection()
        if sel:
            self._cur_form = self._form_lb.get(sel[0])
            self._form_name_var.set(self._cur_form)
            self._refresh_fields_tree()

    def _add_field(self):
        if not self._cur_form:
            messagebox.showinfo("Select", "Select or create a form first.")
            return
        label = self._fl_entry.get().strip()
        if not label:
            messagebox.showerror("Missing", "Enter a field label.")
            return
        ftype   = self._ft_var.get()
        options = self._fo_entry.get().strip()
        self._forms[self._cur_form].append({
            "label": label, "type": ftype, "options": options, "required": True
        })
        self._persist()
        self._refresh_fields_tree()
        self._fl_entry.delete(0, "end")
        self._fo_entry.delete(0, "end")

    def _remove_field(self):
        if not self._cur_form:
            return
        sel = self._fields_tree.selection()
        if sel:
            idx = int(self._fields_tree.item(sel[0], "tags")[0])
            self._forms[self._cur_form].pop(idx)
            self._persist()
            self._refresh_fields_tree()

    def _move_field(self, direction: int):
        if not self._cur_form:
            return
        sel = self._fields_tree.selection()
        if not sel:
            return
        idx  = int(self._fields_tree.item(sel[0], "tags")[0])
        new  = idx + direction
        lst  = self._forms[self._cur_form]
        if 0 <= new < len(lst):
            lst[idx], lst[new] = lst[new], lst[idx]
            self._persist()
            self._refresh_fields_tree()

    def _refresh_fields_tree(self):
        self._fields_tree.delete(*self._fields_tree.get_children())
        if not self._cur_form:
            return
        for i, f in enumerate(self._forms.get(self._cur_form, [])):
            self._fields_tree.insert("", "end", tags=(str(i),),
                                     values=(f["label"], f["type"],
                                             f.get("options", ""),
                                             "Yes" if f.get("required") else "No"))

    def _refresh_forms(self):
        self._form_lb.delete(0, "end")
        names = list(self._forms.keys())
        for n in names:
            self._form_lb.insert("end", n)
        # Update fill and responses menus
        for menu_w, var in [(self._fill_menu, self._fill_form_var),
                            (self._resp_menu, self._resp_form_var)]:
            m = menu_w["menu"]
            m.delete(0, "end")
            for n in names:
                m.add_command(label=n, command=lambda v=n, sv=var: sv.set(v))
        if names:
            self._fill_form_var.set(names[0])
            self._resp_form_var.set(names[0])

    # ── Fill form ─────────────────────────────────────────────────────────────

    def _load_fill_form(self):
        form_name = self._fill_form_var.get()
        if form_name not in self._forms:
            return
        for w in self._fill_inner.winfo_children():
            w.destroy()
        self._fill_widgets = {}

        tk.Label(self._fill_inner, text=form_name, bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 13, "bold")).pack(anchor="w", padx=8, pady=(8, 4))

        for field in self._forms[form_name]:
            lbl  = field["label"]
            ftype = field["type"]
            tk.Label(self._fill_inner, text=lbl + ("  *" if field.get("required") else ""),
                     bg="#1e1e2e", fg="#cdd6f4", font=("Consolas", 10)).pack(anchor="w", padx=8)

            if ftype in ("Text", "Number", "Email", "Date"):
                e = tk.Entry(self._fill_inner, bg="#313244", fg="#cdd6f4", width=40,
                             insertbackground="#cba6f7", font=("Consolas", 10), relief="flat")
                e.pack(anchor="w", padx=8, pady=(0, 6))
                self._fill_widgets[lbl] = e
            elif ftype == "Textarea":
                t = tk.Text(self._fill_inner, bg="#313244", fg="#cdd6f4", height=4, width=50,
                            font=("Consolas", 10), insertbackground="#cba6f7", relief="flat")
                t.pack(anchor="w", padx=8, pady=(0, 6))
                self._fill_widgets[lbl] = t
            elif ftype == "Dropdown":
                opts = [o.strip() for o in field.get("options", "").split(",") if o.strip()]
                var = tk.StringVar(value=opts[0] if opts else "")
                dd = tk.OptionMenu(self._fill_inner, var, *opts if opts else ["(no options)"])
                dd.config(bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                          relief="flat", font=("Consolas", 10))
                dd.pack(anchor="w", padx=8, pady=(0, 6))
                self._fill_widgets[lbl] = var
            elif ftype == "Checkbox":
                var = tk.BooleanVar()
                tk.Checkbutton(self._fill_inner, variable=var, bg="#1e1e2e",
                               fg="#cdd6f4", selectcolor="#313244",
                               activebackground="#1e1e2e").pack(anchor="w", padx=8, pady=(0, 6))
                self._fill_widgets[lbl] = var

    def _submit(self):
        form_name = self._fill_form_var.get()
        if form_name not in self._forms:
            messagebox.showinfo("Load", "Load a form first.")
            return
        response = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for field in self._forms[form_name]:
            lbl = field["label"]
            w   = self._fill_widgets.get(lbl)
            if w is None:
                response[lbl] = ""
            elif isinstance(w, (tk.Entry, tk.StringVar)):
                val = w.get()
                response[lbl] = val
            elif isinstance(w, tk.Text):
                response[lbl] = w.get("1.0", "end-1c").strip()
            elif isinstance(w, tk.BooleanVar):
                response[lbl] = str(w.get())
        self._responses.setdefault(form_name, []).append(response)
        self._persist()
        messagebox.showinfo("Submitted", "Response recorded!")
        self._load_fill_form()   # reset

    # ── Responses ─────────────────────────────────────────────────────────────

    def _load_responses(self):
        form_name = self._resp_form_var.get()
        resps = self._responses.get(form_name, [])
        lines = [f"Form: {form_name}  ({len(resps)} responses)\n"]
        for i, r in enumerate(resps, 1):
            lines.append(f"--- Response #{i} ({r.get('timestamp', '')}) ---")
            for k, v in r.items():
                if k != "timestamp":
                    lines.append(f"  {k}: {v}")
            lines.append("")
        self._resp_text.config(state="normal")
        self._resp_text.delete("1.0", "end")
        self._resp_text.insert("1.0", "\n".join(lines))
        self._resp_text.config(state="disabled")

    def _export_csv(self):
        form_name = self._resp_form_var.get()
        resps = self._responses.get(form_name, [])
        if not resps:
            messagebox.showinfo("No Data", "No responses to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=resps[0].keys())
                writer.writeheader()
                writer.writerows(resps)
            messagebox.showinfo("Exported", f"Saved to {path}")

    def _persist(self):
        self._data["forms"]     = self._forms
        self._data["responses"] = self._responses
        save(self._data)


if __name__ == "__main__":
    app = FormBuilder()
    app.mainloop()
