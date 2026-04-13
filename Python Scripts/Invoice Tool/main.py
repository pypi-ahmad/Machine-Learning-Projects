"""Invoice Tool — Tkinter desktop app.

Create, manage, and print invoices with line items,
tax calculation, and PDF/text export.

Usage:
    python main.py
"""

import json
import os
import tkinter as tk
from datetime import date, datetime, timedelta
from tkinter import filedialog, messagebox, ttk

DATA_FILE = os.path.join(os.path.dirname(__file__), "invoices.json")


def load() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"invoices": [], "next_id": 1001, "company": {}}


def save(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def fmt_currency(val: float) -> str:
    return f"${val:,.2f}"


class InvoiceTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Invoice Tool")
        self.geometry("1000x660")
        self.configure(bg="#1e1e2e")

        self._data     = load()
        self._invoices = self._data.get("invoices", [])
        self._next_id  = self._data.get("next_id", 1001)
        self._company  = self._data.get("company", {})
        self._edit_idx = None
        self._items    = []    # list of {desc, qty, rate}

        self._build_ui()
        self._refresh_invoice_list()

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

        self._list_tab   = tk.Frame(nb, bg="#1e1e2e")
        self._create_tab = tk.Frame(nb, bg="#1e1e2e")
        self._settings_tab = tk.Frame(nb, bg="#1e1e2e")
        nb.add(self._list_tab,     text=" Invoices ")
        nb.add(self._create_tab,   text=" Create / Edit ")
        nb.add(self._settings_tab, text=" Company Settings ")

        self._build_list_tab()
        self._build_create_tab()
        self._build_settings_tab()

    # ── Invoice list tab ─────────────────────────────────────────────────────

    def _build_list_tab(self):
        top = tk.Frame(self._list_tab, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        tk.Button(top, text="+ New Invoice", command=self._new_invoice,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left")
        tk.Button(top, text="✏ Edit", command=self._edit_selected,
                  bg="#89b4fa", fg="#1e1e2e", relief="flat").pack(side="left", padx=4)
        tk.Button(top, text="🗑 Delete", command=self._delete_selected,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(side="left")
        tk.Button(top, text="⬇ Export TXT", command=self._export_txt,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="right")

        cols = ("inv_no", "client", "date", "due", "total", "status")
        self._tree = ttk.Treeview(self._list_tab, columns=cols, show="headings")
        style = ttk.Style()
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        for col, w in zip(cols, [90, 200, 100, 100, 100, 90]):
            self._tree.heading(col, text=col.upper().replace("_", " "))
            self._tree.column(col, width=w, anchor="w")
        self._tree.tag_configure("overdue", foreground="#f38ba8")
        self._tree.tag_configure("paid",    foreground="#a6e3a1")
        self._tree.pack(fill="both", expand=True, padx=8, pady=4)
        self._tree.bind("<Double-Button-1>", lambda _: self._edit_selected())
        sb = ttk.Scrollbar(self._list_tab, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self._summary_var = tk.StringVar()
        tk.Label(self._list_tab, textvariable=self._summary_var,
                 bg="#1e1e2e", fg="#888", font=("Consolas", 9)).pack(anchor="w", padx=8)

    # ── Create/Edit tab ──────────────────────────────────────────────────────

    def _build_create_tab(self):
        # Client & dates
        top = tk.Frame(self._create_tab, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)

        self._create_fields = {}
        for r, (lbl, default) in enumerate([("Client Name", ""), ("Client Email", ""),
                                              ("Client Address", ""), ("Invoice Date", str(date.today())),
                                              ("Due Date", str(date.today() + timedelta(days=30))),
                                              ("Notes", "")]):
            tk.Label(top, text=lbl, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9)).grid(row=r, column=0, sticky="w", pady=2)
            e = tk.Entry(top, bg="#313244", fg="#cdd6f4", insertbackground="#cba6f7",
                         font=("Consolas", 10), relief="flat", width=36)
            e.insert(0, default)
            e.grid(row=r, column=1, padx=8, pady=2)
            self._create_fields[lbl] = e

        tk.Label(top, text="Status", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).grid(row=6, column=0, sticky="w", pady=2)
        self._status_var = tk.StringVar(value="Draft")
        tk.OptionMenu(top, self._status_var, "Draft", "Sent", "Paid", "Overdue", "Cancelled").grid(
            row=6, column=1, padx=8, sticky="w")

        tk.Label(top, text="Tax %", bg="#1e1e2e", fg="#888",
                 font=("Consolas", 9)).grid(row=7, column=0, sticky="w")
        self._tax_var = tk.StringVar(value="10")
        tk.Entry(top, textvariable=self._tax_var, bg="#313244", fg="#cdd6f4", width=6,
                 font=("Consolas", 10), relief="flat").grid(row=7, column=1, padx=8, sticky="w")

        # Line items
        tk.Label(self._create_tab, text="Line Items", bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 11, "bold")).pack(anchor="w", padx=8, pady=(8, 4))

        item_top = tk.Frame(self._create_tab, bg="#1e1e2e")
        item_top.pack(fill="x", padx=8)
        self._item_desc = tk.Entry(item_top, bg="#313244", fg="#cdd6f4", width=28,
                                    insertbackground="#cba6f7", font=("Consolas", 10),
                                    relief="flat")
        self._item_desc.pack(side="left", padx=(0, 4))
        self._item_qty  = tk.Entry(item_top, bg="#313244", fg="#cdd6f4", width=6,
                                    insertbackground="#cba6f7", font=("Consolas", 10),
                                    relief="flat")
        self._item_qty.insert(0, "1")
        self._item_qty.pack(side="left", padx=4)
        self._item_rate = tk.Entry(item_top, bg="#313244", fg="#cdd6f4", width=10,
                                    insertbackground="#cba6f7", font=("Consolas", 10),
                                    relief="flat")
        self._item_rate.insert(0, "0.00")
        self._item_rate.pack(side="left", padx=4)
        for lbl in ("Description", "Qty", "Rate"):
            tk.Label(item_top, text=lbl, bg="#1e1e2e", fg="#555",
                     font=("Consolas", 8)).pack(side="left", padx=2)
        tk.Button(item_top, text="+ Add", command=self._add_line_item,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=4)
        tk.Button(item_top, text="✕ Remove", command=self._remove_line_item,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(side="left")

        cols = ("desc", "qty", "rate", "amount")
        self._items_tree = ttk.Treeview(self._create_tab, columns=cols, show="headings", height=8)
        for col, w in zip(cols, [280, 60, 100, 100]):
            self._items_tree.heading(col, text=col.upper())
            self._items_tree.column(col, width=w, anchor="w")
        self._items_tree.pack(fill="x", padx=8, pady=4)

        # Totals
        tot = tk.Frame(self._create_tab, bg="#1e1e2e")
        tot.pack(fill="x", padx=8, pady=4)
        self._subtotal_var = tk.StringVar(value="Subtotal: $0.00")
        self._tax_amt_var  = tk.StringVar(value="Tax: $0.00")
        self._total_var    = tk.StringVar(value="TOTAL: $0.00")
        for var in (self._subtotal_var, self._tax_amt_var, self._total_var):
            tk.Label(tot, textvariable=var, bg="#1e1e2e", fg="#cdd6f4",
                     font=("Consolas", 11)).pack(anchor="e")

        btn = tk.Frame(self._create_tab, bg="#1e1e2e")
        btn.pack(fill="x", padx=8, pady=6)
        tk.Button(btn, text="💾 Save Invoice", command=self._save_invoice,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 11, "bold")).pack(side="left", padx=4)
        tk.Button(btn, text="✕ Clear", command=self._clear_create,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left")

    # ── Settings tab ──────────────────────────────────────────────────────────

    def _build_settings_tab(self):
        tk.Label(self._settings_tab, text="Your Company Details", bg="#1e1e2e",
                 fg="#cba6f7", font=("Consolas", 13, "bold")).pack(anchor="w", padx=12, pady=8)
        self._company_fields = {}
        for lbl in ["Company Name", "Email", "Phone", "Address", "Website", "Tax ID"]:
            tk.Label(self._settings_tab, text=lbl, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9)).pack(anchor="w", padx=12)
            e = tk.Entry(self._settings_tab, bg="#313244", fg="#cdd6f4", width=40,
                         insertbackground="#cba6f7", font=("Consolas", 11), relief="flat")
            e.insert(0, self._company.get(lbl, ""))
            e.pack(anchor="w", padx=12, pady=(0, 4))
            self._company_fields[lbl] = e
        tk.Button(self._settings_tab, text="💾 Save Settings", command=self._save_settings,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(anchor="w", padx=12, pady=8)

    # ── Invoice logic ─────────────────────────────────────────────────────────

    def _new_invoice(self):
        self._edit_idx = None
        self._items = []
        self._clear_create()

    def _add_line_item(self):
        desc = self._item_desc.get().strip()
        if not desc:
            return
        try:
            qty  = float(self._item_qty.get())
            rate = float(self._item_rate.get())
        except ValueError:
            messagebox.showerror("Invalid", "Qty and Rate must be numbers.")
            return
        self._items.append({"desc": desc, "qty": qty, "rate": rate})
        self._item_desc.delete(0, "end")
        self._item_qty.delete(0, "end"); self._item_qty.insert(0, "1")
        self._item_rate.delete(0, "end"); self._item_rate.insert(0, "0.00")
        self._refresh_items_tree()

    def _remove_line_item(self):
        sel = self._items_tree.selection()
        if sel:
            idx = int(self._items_tree.item(sel[0], "tags")[0])
            self._items.pop(idx)
            self._refresh_items_tree()

    def _refresh_items_tree(self):
        self._items_tree.delete(*self._items_tree.get_children())
        subtotal = 0.0
        for i, it in enumerate(self._items):
            amt = it["qty"] * it["rate"]
            subtotal += amt
            self._items_tree.insert("", "end", tags=(str(i),),
                                    values=(it["desc"], it["qty"],
                                            fmt_currency(it["rate"]), fmt_currency(amt)))
        try:
            tax_pct = float(self._tax_var.get()) / 100
        except ValueError:
            tax_pct = 0
        tax_amt = subtotal * tax_pct
        total   = subtotal + tax_amt
        self._subtotal_var.set(f"Subtotal: {fmt_currency(subtotal)}")
        self._tax_amt_var.set(f"Tax ({self._tax_var.get()}%): {fmt_currency(tax_amt)}")
        self._total_var.set(f"TOTAL: {fmt_currency(total)}")

    def _save_invoice(self):
        f = {k: e.get().strip() for k, e in self._create_fields.items()}
        if not f["Client Name"]:
            messagebox.showerror("Missing", "Client name is required.")
            return
        try:
            tax_pct = float(self._tax_var.get()) / 100
        except ValueError:
            tax_pct = 0
        subtotal = sum(it["qty"] * it["rate"] for it in self._items)
        tax_amt  = subtotal * tax_pct
        total    = subtotal + tax_amt
        inv = {
            "inv_no":  f"INV-{self._next_id}",
            "client":  f["Client Name"], "client_email": f["Client Email"],
            "client_addr": f["Client Address"],
            "date":    f["Invoice Date"], "due": f["Due Date"],
            "notes":   f["Notes"], "status": self._status_var.get(),
            "tax_pct": float(self._tax_var.get()),
            "items":   self._items, "subtotal": subtotal,
            "tax_amt": tax_amt, "total": total,
        }
        if self._edit_idx is not None:
            inv["inv_no"] = self._invoices[self._edit_idx]["inv_no"]
            self._invoices[self._edit_idx] = inv
        else:
            self._invoices.append(inv)
            self._next_id += 1
        self._data["invoices"] = self._invoices
        self._data["next_id"]  = self._next_id
        save(self._data)
        self._refresh_invoice_list()
        messagebox.showinfo("Saved", f"Invoice {inv['inv_no']} saved.")

    def _edit_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        inv = self._invoices[idx]
        self._edit_idx = idx
        self._items    = list(inv.get("items", []))
        for lbl, key in [("Client Name", "client"), ("Client Email", "client_email"),
                          ("Client Address", "client_addr"), ("Invoice Date", "date"),
                          ("Due Date", "due"), ("Notes", "notes")]:
            self._create_fields[lbl].delete(0, "end")
            self._create_fields[lbl].insert(0, inv.get(key, ""))
        self._status_var.set(inv.get("status", "Draft"))
        self._tax_var.set(str(inv.get("tax_pct", 10)))
        self._refresh_items_tree()

    def _delete_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        no  = self._invoices[idx]["inv_no"]
        if messagebox.askyesno("Delete", f"Delete {no}?"):
            self._invoices.pop(idx)
            self._data["invoices"] = self._invoices
            save(self._data)
            self._refresh_invoice_list()

    def _clear_create(self):
        self._items = []
        for e in self._create_fields.values():
            e.delete(0, "end")
        self._create_fields["Invoice Date"].insert(0, str(date.today()))
        self._create_fields["Due Date"].insert(0, str(date.today() + timedelta(days=30)))
        self._tax_var.set("10")
        self._status_var.set("Draft")
        self._refresh_items_tree()
        self._edit_idx = None

    def _refresh_invoice_list(self):
        self._tree.delete(*self._tree.get_children())
        total_unpaid = 0.0
        today = str(date.today())
        for i, inv in enumerate(self._invoices):
            tags = (str(i),)
            if inv.get("status") == "Paid":
                tags = tags + ("paid",)
            elif inv.get("due", "") < today and inv.get("status") != "Paid":
                tags = tags + ("overdue",)
            if inv.get("status") not in ("Paid", "Cancelled"):
                total_unpaid += inv.get("total", 0)
            self._tree.insert("", "end", tags=tags,
                              values=(inv["inv_no"], inv["client"],
                                      inv.get("date", ""), inv.get("due", ""),
                                      fmt_currency(inv.get("total", 0)),
                                      inv.get("status", "")))
        self._summary_var.set(
            f"Total invoices: {len(self._invoices)}  |  "
            f"Outstanding: {fmt_currency(total_unpaid)}")

    def _export_txt(self):
        sel = self._tree.selection()
        if not sel:
            messagebox.showinfo("Select", "Select an invoice to export.")
            return
        idx = int(self._tree.item(sel[0], "tags")[0])
        inv = self._invoices[idx]
        co  = self._company

        lines = [
            co.get("Company Name", "Your Company"), co.get("Address", ""),
            f"Email: {co.get('Email', '')}  Tel: {co.get('Phone', '')}",
            "", "=" * 56,
            f"INVOICE {inv['inv_no']}",
            f"Date: {inv.get('date','')}   Due: {inv.get('due','')}",
            f"Status: {inv.get('status','')}", "",
            f"Bill To: {inv['client']}", inv.get("client_addr", ""), "",
            f"{'Description':<30}{'Qty':>6}{'Rate':>10}{'Amount':>10}",
            "-" * 56,
        ]
        for it in inv.get("items", []):
            amt = it["qty"] * it["rate"]
            lines.append(f"{it['desc'][:29]:<30}{it['qty']:>6.1f}"
                         f"{fmt_currency(it['rate']):>10}{fmt_currency(amt):>10}")
        lines += ["=" * 56,
                  f"{'Subtotal':>46}: {fmt_currency(inv.get('subtotal',0))}",
                  f"{'Tax (' + str(inv.get('tax_pct','')) + '%)':>46}: {fmt_currency(inv.get('tax_amt',0))}",
                  f"{'TOTAL':>46}: {fmt_currency(inv.get('total',0))}",
                  "", f"Notes: {inv.get('notes','')}"]

        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            messagebox.showinfo("Exported", f"Saved to {path}")

    def _save_settings(self):
        self._company = {k: e.get().strip() for k, e in self._company_fields.items()}
        self._data["company"] = self._company
        save(self._data)
        messagebox.showinfo("Saved", "Company settings saved.")


if __name__ == "__main__":
    app = InvoiceTool()
    app.mainloop()
