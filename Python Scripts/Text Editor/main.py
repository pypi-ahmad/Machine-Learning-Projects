"""Text Editor — Tkinter desktop app.

A lightweight plain-text editor with file open/save,
find & replace, line numbers, word count, and dark theme.

Usage:
    python main.py
"""

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


class TextEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Text Editor — Untitled")
        self.geometry("900x640")
        self.configure(bg="#1e1e2e")

        self._filepath = None
        self._modified = False

        self._build_ui()
        self._build_menu()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Status bar (bottom)
        self._status = tk.StringVar(value="Ln 1, Col 1  |  0 words  |  UTF-8")
        tk.Label(self, textvariable=self._status, bg="#181825", fg="#888",
                 anchor="w", padx=8, font=("Consolas", 10)).pack(side="bottom", fill="x")

        # Main frame: line numbers + text area
        main = tk.Frame(self, bg="#1e1e2e")
        main.pack(fill="both", expand=True)

        self._ln = tk.Text(main, width=4, bg="#181825", fg="#555", bd=0,
                           state="disabled", font=("Consolas", 12),
                           padx=4, pady=4, selectbackground="#181825")
        self._ln.pack(side="left", fill="y")

        scroll = tk.Scrollbar(main)
        scroll.pack(side="right", fill="y")

        self._text = tk.Text(
            main, bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cba6f7",
            font=("Consolas", 12), wrap="word", bd=0,
            undo=True, padx=8, pady=4,
            yscrollcommand=self._on_scroll,
            selectbackground="#313244",
        )
        self._text.pack(fill="both", expand=True)
        scroll.config(command=self._text.yview)

        self._text.bind("<KeyRelease>", self._on_key)
        self._text.bind("<ButtonRelease>", self._on_key)

    def _build_menu(self):
        mb = tk.Menu(self, bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                     activeforeground="#cdd6f4", tearoff=False)
        self.config(menu=mb)

        fm = tk.Menu(mb, tearoff=False, bg="#313244", fg="#cdd6f4",
                     activebackground="#45475a", activeforeground="#cdd6f4")
        mb.add_cascade(label="File", menu=fm)
        fm.add_command(label="New         Ctrl+N", command=self._new)
        fm.add_command(label="Open...     Ctrl+O", command=self._open)
        fm.add_command(label="Save        Ctrl+S", command=self._save)
        fm.add_command(label="Save As...  Ctrl+Shift+S", command=self._save_as)
        fm.add_separator()
        fm.add_command(label="Exit", command=self._on_close)

        em = tk.Menu(mb, tearoff=False, bg="#313244", fg="#cdd6f4",
                     activebackground="#45475a", activeforeground="#cdd6f4")
        mb.add_cascade(label="Edit", menu=em)
        em.add_command(label="Undo  Ctrl+Z", command=self._text.edit_undo)
        em.add_command(label="Redo  Ctrl+Y", command=self._text.edit_redo)
        em.add_separator()
        em.add_command(label="Cut   Ctrl+X", command=lambda: self._text.event_generate("<<Cut>>"))
        em.add_command(label="Copy  Ctrl+C", command=lambda: self._text.event_generate("<<Copy>>"))
        em.add_command(label="Paste Ctrl+V", command=lambda: self._text.event_generate("<<Paste>>"))
        em.add_separator()
        em.add_command(label="Select All  Ctrl+A", command=self._select_all)
        em.add_command(label="Find & Replace  Ctrl+H", command=self._find_replace)

    def _bind_shortcuts(self):
        self.bind("<Control-n>", lambda e: self._new())
        self.bind("<Control-o>", lambda e: self._open())
        self.bind("<Control-s>", lambda e: self._save())
        self.bind("<Control-S>", lambda e: self._save_as())
        self.bind("<Control-a>", lambda e: self._select_all())
        self.bind("<Control-h>", lambda e: self._find_replace())

    # ── File operations ─────────────────────────────────────────────────────

    def _new(self):
        if self._check_unsaved():
            self._text.delete("1.0", "end")
            self._filepath = None
            self._modified = False
            self.title("Text Editor — Untitled")

    def _open(self):
        if not self._check_unsaved():
            return
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            self._text.delete("1.0", "end")
            self._text.insert("1.0", content)
            self._filepath = path
            self._modified = False
            self.title(f"Text Editor — {os.path.basename(path)}")
            self._update_status()

    def _save(self):
        if self._filepath:
            self._write(self._filepath)
        else:
            self._save_as()

    def _save_as(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self._write(path)
            self._filepath = path
            self.title(f"Text Editor — {os.path.basename(path)}")

    def _write(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._text.get("1.0", "end-1c"))
        self._modified = False
        self.title(self.title().rstrip(" *"))

    def _check_unsaved(self) -> bool:
        if self._modified:
            return messagebox.askyesno("Unsaved Changes",
                                       "You have unsaved changes. Continue anyway?")
        return True

    def _on_close(self):
        if self._check_unsaved():
            self.destroy()

    # ── Edit helpers ─────────────────────────────────────────────────────────

    def _select_all(self):
        self._text.tag_add("sel", "1.0", "end")
        return "break"

    def _find_replace(self):
        win = tk.Toplevel(self)
        win.title("Find & Replace")
        win.configure(bg="#1e1e2e")
        win.resizable(False, False)

        tk.Label(win, text="Find:",    bg="#1e1e2e", fg="#cdd6f4").grid(row=0, column=0, padx=8, pady=4, sticky="e")
        tk.Label(win, text="Replace:", bg="#1e1e2e", fg="#cdd6f4").grid(row=1, column=0, padx=8, pady=4, sticky="e")
        find_e    = tk.Entry(win, width=28, bg="#313244", fg="#cdd6f4", insertbackground="#cba6f7")
        replace_e = tk.Entry(win, width=28, bg="#313244", fg="#cdd6f4", insertbackground="#cba6f7")
        find_e.grid(row=0, column=1, padx=8, pady=4)
        replace_e.grid(row=1, column=1, padx=8, pady=4)
        count_lbl = tk.Label(win, text="", bg="#1e1e2e", fg="#a6e3a1")
        count_lbl.grid(row=2, column=0, columnspan=2)

        def do_replace():
            pattern = find_e.get()
            repl    = replace_e.get()
            if not pattern:
                return
            content = self._text.get("1.0", "end-1c")
            new, n  = re.subn(re.escape(pattern), repl, content)
            self._text.delete("1.0", "end")
            self._text.insert("1.0", new)
            count_lbl.config(text=f"Replaced {n} occurrence(s).")

        tk.Button(win, text="Replace All", command=do_replace,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat").grid(row=3, column=0, columnspan=2, pady=8)

    # ── Status & line numbers ─────────────────────────────────────────────────

    def _on_key(self, _event=None):
        self._modified = True
        self.title(self.title().rstrip(" *") + " *")
        self._update_status()
        self._update_line_numbers()

    def _on_scroll(self, *args):
        self._text.yview(*args)   # handled by scrollbar
        self._update_line_numbers()

    def _update_status(self):
        content  = self._text.get("1.0", "end-1c")
        words    = len(content.split())
        pos      = self._text.index("insert")
        ln, col  = pos.split(".")
        self._status.set(f"Ln {ln}, Col {int(col)+1}  |  {words} words  |  UTF-8")

    def _update_line_numbers(self):
        self._ln.config(state="normal")
        self._ln.delete("1.0", "end")
        lines = int(self._text.index("end-1c").split(".")[0])
        self._ln.insert("1.0", "\n".join(str(i) for i in range(1, lines + 1)))
        self._ln.config(state="disabled")


if __name__ == "__main__":
    app = TextEditor()
    app.mainloop()
