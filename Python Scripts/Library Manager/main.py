"""Library Manager — Tkinter desktop app.

Manage a book collection with checkout, return, member management,
due date tracking, and overdue alerts. JSON persistence.

Usage:
    python main.py
"""

import json
import os
import tkinter as tk
from datetime import date, datetime, timedelta
from tkinter import messagebox, ttk

DATA_FILE = os.path.join(os.path.dirname(__file__), "library.json")
GENRES    = ["Fiction", "Non-Fiction", "Science", "History", "Biography",
             "Technology", "Philosophy", "Children", "Reference", "Other"]


def load() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return {"books": [], "members": [], "loans": []}


def save(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class LibraryManager(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Library Manager")
        self.geometry("1020x640")
        self.configure(bg="#1e1e2e")

        self._data    = load()
        self._books   = self._data.get("books", [])
        self._members = self._data.get("members", [])
        self._loans   = self._data.get("loans", [])
        self._build_ui()
        self._refresh_all()

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

        tabs = {}
        for name in ("Books", "Members", "Loans", "Dashboard"):
            f = tk.Frame(nb, bg="#1e1e2e")
            nb.add(f, text=f" {name} ")
            tabs[name] = f

        self._build_books_tab(tabs["Books"])
        self._build_members_tab(tabs["Members"])
        self._build_loans_tab(tabs["Loans"])
        self._build_dashboard_tab(tabs["Dashboard"])

    # ── Books tab ─────────────────────────────────────────────────────────────

    def _build_books_tab(self, parent):
        top = tk.Frame(parent, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        self._book_search = tk.StringVar()
        self._book_search.trace_add("write", lambda *_: self._refresh_books())
        tk.Entry(top, textvariable=self._book_search, bg="#313244", fg="#cdd6f4",
                 width=24, font=("Consolas", 10), relief="flat",
                 insertbackground="#cba6f7").pack(side="left", padx=(0, 4))
        tk.Label(top, text="🔍", bg="#1e1e2e", fg="#888").pack(side="left")
        tk.Button(top, text="+ Add Book", command=self._add_book_dialog,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="right")
        tk.Button(top, text="🗑 Delete", command=self._delete_book,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(side="right", padx=4)

        cols = ("isbn", "title", "author", "genre", "year", "copies", "available")
        self._books_tree = ttk.Treeview(parent, columns=cols, show="headings")
        style = ttk.Style()
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 10))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        for col, w in zip(cols, [90, 220, 160, 90, 60, 70, 80]):
            self._books_tree.heading(col, text=col.upper())
            self._books_tree.column(col, width=w, anchor="w")
        self._books_tree.tag_configure("unavail", foreground="#f38ba8")
        self._books_tree.pack(fill="both", expand=True, padx=8, pady=4)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self._books_tree.yview)
        self._books_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

    # ── Members tab ───────────────────────────────────────────────────────────

    def _build_members_tab(self, parent):
        top = tk.Frame(parent, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        tk.Button(top, text="+ Add Member", command=self._add_member_dialog,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left")
        tk.Button(top, text="🗑 Delete", command=self._delete_member,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(side="left", padx=4)

        cols = ("mid", "name", "email", "phone", "active_loans")
        self._members_tree = ttk.Treeview(parent, columns=cols, show="headings")
        for col, w in zip(cols, [80, 200, 200, 120, 100]):
            self._members_tree.heading(col, text=col.upper().replace("_", " "))
            self._members_tree.column(col, width=w, anchor="w")
        self._members_tree.pack(fill="both", expand=True, padx=8, pady=4)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self._members_tree.yview)
        self._members_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

    # ── Loans tab ─────────────────────────────────────────────────────────────

    def _build_loans_tab(self, parent):
        top = tk.Frame(parent, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        tk.Button(top, text="📤 Checkout", command=self._checkout_dialog,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left")
        tk.Button(top, text="📥 Return", command=self._return_selected,
                  bg="#a6e3a1", fg="#1e1e2e", relief="flat").pack(side="left", padx=4)

        cols = ("loan_id", "book_title", "member", "checkout", "due", "returned")
        self._loans_tree = ttk.Treeview(parent, columns=cols, show="headings")
        for col, w in zip(cols, [70, 220, 160, 100, 100, 100]):
            self._loans_tree.heading(col, text=col.upper().replace("_", " "))
            self._loans_tree.column(col, width=w, anchor="w")
        self._loans_tree.tag_configure("overdue",  foreground="#f38ba8")
        self._loans_tree.tag_configure("returned", foreground="#a6e3a1")
        self._loans_tree.pack(fill="both", expand=True, padx=8, pady=4)
        sb = ttk.Scrollbar(parent, orient="vertical", command=self._loans_tree.yview)
        self._loans_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

    # ── Dashboard tab ─────────────────────────────────────────────────────────

    def _build_dashboard_tab(self, parent):
        self._dash_text = tk.Text(parent, bg="#313244", fg="#cdd6f4",
                                   font=("Consolas", 11), relief="flat", state="disabled")
        self._dash_text.pack(fill="both", expand=True, padx=8, pady=8)
        tk.Button(parent, text="🔄 Refresh", command=self._update_dashboard,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(pady=4)

    # ── Dialogs ────────────────────────────────────────────────────────────────

    def _add_book_dialog(self):
        win = tk.Toplevel(self)
        win.title("Add Book")
        win.configure(bg="#1e1e2e")
        win.resizable(False, False)
        fields = {}
        for lbl, default in [("ISBN", ""), ("Title", ""), ("Author", ""),
                               ("Year", "2024"), ("Copies", "1")]:
            tk.Label(win, text=lbl, bg="#1e1e2e", fg="#888").grid(row=len(fields), column=0, padx=8, pady=4, sticky="w")
            e = tk.Entry(win, bg="#313244", fg="#cdd6f4", width=30, font=("Consolas", 10), relief="flat")
            e.insert(0, default)
            e.grid(row=len(fields), column=1, padx=8, pady=4)
            fields[lbl] = e
        tk.Label(win, text="Genre", bg="#1e1e2e", fg="#888").grid(row=len(fields), column=0, padx=8, pady=4, sticky="w")
        genre_var = tk.StringVar(value=GENRES[0])
        tk.OptionMenu(win, genre_var, *GENRES).grid(row=len(fields), column=1, padx=8, sticky="w")

        def save_book():
            isbn  = fields["ISBN"].get().strip()
            title = fields["Title"].get().strip()
            if not title:
                messagebox.showerror("Missing", "Title is required.", parent=win)
                return
            try:
                copies = int(fields["Copies"].get())
            except ValueError:
                copies = 1
            self._books.append({"isbn": isbn, "title": title,
                                 "author": fields["Author"].get().strip(),
                                 "genre": genre_var.get(),
                                 "year": fields["Year"].get().strip(),
                                 "copies": copies, "available": copies})
            self._persist()
            self._refresh_books()
            win.destroy()

        tk.Button(win, text="Add Book", command=save_book,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat").grid(row=10, column=0, columnspan=2, pady=8)

    def _add_member_dialog(self):
        win = tk.Toplevel(self)
        win.title("Add Member")
        win.configure(bg="#1e1e2e")
        win.resizable(False, False)
        fields = {}
        for lbl in ["Name", "Email", "Phone"]:
            tk.Label(win, text=lbl, bg="#1e1e2e", fg="#888").grid(row=len(fields), column=0, padx=8, pady=4, sticky="w")
            e = tk.Entry(win, bg="#313244", fg="#cdd6f4", width=28, font=("Consolas", 10), relief="flat")
            e.grid(row=len(fields), column=1, padx=8, pady=4)
            fields[lbl] = e

        def save_member():
            name = fields["Name"].get().strip()
            if not name:
                messagebox.showerror("Missing", "Name is required.", parent=win)
                return
            mid = f"M{len(self._members)+1:04d}"
            self._members.append({"mid": mid, "name": name,
                                   "email": fields["Email"].get().strip(),
                                   "phone": fields["Phone"].get().strip()})
            self._persist()
            self._refresh_members()
            win.destroy()

        tk.Button(win, text="Add Member", command=save_member,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat").grid(row=5, column=0, columnspan=2, pady=8)

    def _checkout_dialog(self):
        if not self._books or not self._members:
            messagebox.showinfo("Setup", "Add books and members first.")
            return
        win = tk.Toplevel(self)
        win.title("Checkout Book")
        win.configure(bg="#1e1e2e")

        book_names   = [f"{b['title']} (avail={b.get('available',0)})" for b in self._books]
        member_names = [f"{m['mid']} — {m['name']}" for m in self._members]
        book_var   = tk.StringVar(value=book_names[0])
        member_var = tk.StringVar(value=member_names[0])
        days_var   = tk.StringVar(value="14")

        for lbl, var, opts in [("Book", book_var, book_names),
                                 ("Member", member_var, member_names)]:
            tk.Label(win, text=lbl, bg="#1e1e2e", fg="#888").pack(padx=8, anchor="w")
            tk.OptionMenu(win, var, *opts).pack(padx=8, fill="x")
        tk.Label(win, text="Loan Period (days)", bg="#1e1e2e", fg="#888").pack(padx=8, anchor="w")
        tk.Entry(win, textvariable=days_var, bg="#313244", fg="#cdd6f4", width=6,
                 font=("Consolas", 10), relief="flat").pack(padx=8, anchor="w")

        def do_checkout():
            bi = book_names.index(book_var.get())
            mi = member_names.index(member_var.get())
            book = self._books[bi]
            if book.get("available", 0) <= 0:
                messagebox.showerror("Unavailable", "No copies available.", parent=win)
                return
            book["available"] = book.get("available", 1) - 1
            lid  = f"L{len(self._loans)+1:05d}"
            due  = str(date.today() + timedelta(days=int(days_var.get() or 14)))
            self._loans.append({
                "loan_id": lid, "book_idx": bi, "member_idx": mi,
                "book_title": book["title"],
                "member":     self._members[mi]["name"],
                "checkout":   str(date.today()),
                "due":        due, "returned": "",
            })
            self._persist()
            self._refresh_all()
            win.destroy()

        tk.Button(win, text="✔ Checkout", command=do_checkout,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat").pack(pady=8)

    def _return_selected(self):
        sel = self._loans_tree.selection()
        if not sel:
            return
        tags = self._loans_tree.item(sel[0], "tags")
        idx  = int(tags[0])
        loan = self._loans[idx]
        if loan.get("returned"):
            messagebox.showinfo("Already returned", "This item is already returned.")
            return
        loan["returned"] = str(date.today())
        bi = loan.get("book_idx")
        if bi is not None and bi < len(self._books):
            self._books[bi]["available"] = self._books[bi].get("available", 0) + 1
        self._persist()
        self._refresh_all()

    def _delete_book(self):
        sel = self._books_tree.selection()
        if sel:
            idx = int(self._books_tree.item(sel[0], "tags")[0])
            if messagebox.askyesno("Delete", f"Delete '{self._books[idx]['title']}'?"):
                self._books.pop(idx)
                self._persist()
                self._refresh_books()

    def _delete_member(self):
        sel = self._members_tree.selection()
        if sel:
            idx = int(self._members_tree.item(sel[0], "tags")[0])
            if messagebox.askyesno("Delete", f"Delete member '{self._members[idx]['name']}'?"):
                self._members.pop(idx)
                self._persist()
                self._refresh_members()

    # ── Refresh methods ───────────────────────────────────────────────────────

    def _refresh_all(self):
        self._refresh_books()
        self._refresh_members()
        self._refresh_loans()
        self._update_dashboard()

    def _refresh_books(self):
        self._books_tree.delete(*self._books_tree.get_children())
        q = self._book_search.get().lower()
        for i, b in enumerate(self._books):
            if q and q not in b.get("title", "").lower():
                continue
            tags = (str(i),)
            if b.get("available", 0) == 0:
                tags = tags + ("unavail",)
            self._books_tree.insert("", "end", tags=tags,
                                     values=(b.get("isbn", ""), b.get("title", ""),
                                             b.get("author", ""), b.get("genre", ""),
                                             b.get("year", ""), b.get("copies", 1),
                                             b.get("available", 1)))

    def _refresh_members(self):
        self._members_tree.delete(*self._members_tree.get_children())
        today = str(date.today())
        for i, m in enumerate(self._members):
            active = sum(1 for l in self._loans
                         if l.get("member") == m["name"] and not l.get("returned"))
            self._members_tree.insert("", "end", tags=(str(i),),
                                       values=(m.get("mid", ""), m.get("name", ""),
                                               m.get("email", ""), m.get("phone", ""),
                                               active))

    def _refresh_loans(self):
        self._loans_tree.delete(*self._loans_tree.get_children())
        today = str(date.today())
        for i, l in enumerate(self._loans):
            tags = [str(i)]
            if l.get("returned"):
                tags.append("returned")
            elif l.get("due", "") < today:
                tags.append("overdue")
            self._loans_tree.insert("", "end", tags=tuple(tags),
                                     values=(l.get("loan_id", ""), l.get("book_title", ""),
                                             l.get("member", ""), l.get("checkout", ""),
                                             l.get("due", ""),
                                             l.get("returned", "—") or "—"))

    def _update_dashboard(self):
        today  = str(date.today())
        total_b = len(self._books)
        avail   = sum(b.get("available", 0) for b in self._books)
        total_m = len(self._members)
        active  = sum(1 for l in self._loans if not l.get("returned"))
        overdue = sum(1 for l in self._loans
                      if not l.get("returned") and l.get("due", "") < today)

        lines = [
            "═" * 40, "  LIBRARY DASHBOARD", "═" * 40,
            f"  Books in catalog    : {total_b}",
            f"  Available copies   : {avail}",
            f"  Members            : {total_m}",
            f"  Active loans       : {active}",
            f"  Overdue books      : {overdue}", "",
        ]
        if overdue:
            lines.append("⚠  OVERDUE LOANS:")
            for l in self._loans:
                if not l.get("returned") and l.get("due", "") < today:
                    lines.append(f"   {l['book_title']} — {l['member']} (due {l['due']})")

        self._dash_text.config(state="normal")
        self._dash_text.delete("1.0", "end")
        self._dash_text.insert("1.0", "\n".join(lines))
        self._dash_text.config(state="disabled")

    def _persist(self):
        self._data["books"]   = self._books
        self._data["members"] = self._members
        self._data["loans"]   = self._loans
        save(self._data)


if __name__ == "__main__":
    app = LibraryManager()
    app.mainloop()
