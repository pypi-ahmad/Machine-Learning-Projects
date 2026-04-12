"""File Browser GUI — Tkinter app.

Browse the filesystem with a tree view.  Preview text files,
view file properties, and perform basic operations
(open, copy path, delete, rename).

Usage:
    python main.py [start_path]
"""

import os
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk


class FileBrowser(tk.Tk):
    def __init__(self, start_path: str | None = None):
        super().__init__()
        self.title("File Browser")
        self.geometry("1000x650")
        self.configure(bg="#f0f0f0")

        self._cwd = Path(start_path) if start_path else Path.home()
        self._history: list[Path] = [self._cwd]
        self._hist_idx = 0

        self._build_ui()
        self._populate(self._cwd)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        lf = tkfont.Font(family="Segoe UI", size=10)

        # Toolbar
        bar = tk.Frame(self, bg="#ddd")
        bar.pack(fill="x")
        for text, cmd in [("◀ Back", self._back), ("▶ Forward", self._forward),
                           ("⬆ Up", self._up), ("🏠 Home", self._home),
                           ("📂 Go to…", self._go_to)]:
            tk.Button(bar, text=text, command=cmd, font=lf, relief="flat",
                      bg="#ddd", padx=8, pady=4, activebackground="#bbb").pack(side="left")

        self.path_var = tk.StringVar(value=str(self._cwd))
        path_entry = tk.Entry(bar, textvariable=self.path_var, font=lf)
        path_entry.pack(side="left", fill="x", expand=True, padx=4)
        path_entry.bind("<Return>", lambda _: self._go_to_path())

        # Paned window
        pane = tk.PanedWindow(self, orient="horizontal", bg="#bbb", sashwidth=4)
        pane.pack(fill="both", expand=True)

        # Left: file list
        left = tk.Frame(pane, bg="white")
        pane.add(left, minsize=300)

        self.tree = ttk.Treeview(left, columns=("size", "type"), show="headings",
                                  selectmode="browse")
        self.tree.heading("#0",    text="Name")
        self.tree.heading("size",  text="Size")
        self.tree.heading("type",  text="Type")
        self.tree.column("size", width=80,  anchor="e")
        self.tree.column("type", width=80,  anchor="w")

        # Re-configure to show tree column
        self.tree = ttk.Treeview(left, columns=("size", "modified"), show="tree headings",
                                  selectmode="browse")
        self.tree.heading("#0",       text="Name")
        self.tree.heading("size",     text="Size")
        self.tree.heading("modified", text="Modified")
        self.tree.column("#0",       width=260)
        self.tree.column("size",     width=80,  anchor="e")
        self.tree.column("modified", width=140, anchor="w")

        vsb = ttk.Scrollbar(left, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(left, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.tree.bind("<Double-1>",   self._on_double_click)
        self.tree.bind("<Button-3>",   self._context_menu)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Right: preview + info
        right = tk.Frame(pane, bg="white")
        pane.add(right, minsize=250)

        self.info_label = tk.Label(right, text="Select a file to preview",
                                    font=lf, bg="white", anchor="nw", justify="left",
                                    wraplength=260)
        self.info_label.pack(fill="x", padx=8, pady=8)

        self.preview = tk.Text(right, font=tkfont.Font(family="Consolas", size=9),
                                wrap="none", bg="#f8f8f8", relief="flat", state="disabled")
        vsb2 = ttk.Scrollbar(right, command=self.preview.yview)
        self.preview.configure(yscrollcommand=vsb2.set)
        self.preview.pack(side="left", fill="both", expand=True, padx=(8,0), pady=(0,8))
        vsb2.pack(side="right", fill="y", pady=(0,8))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, bg="#ddd", anchor="w",
                 font=tkfont.Font(family="Segoe UI", size=9)).pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # Directory population
    # ------------------------------------------------------------------

    def _populate(self, path: Path):
        self.tree.delete(*self.tree.get_children())
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            messagebox.showwarning("Access denied", f"Cannot access: {path}")
            return

        self._cwd = path
        self.path_var.set(str(path))
        self.title(f"File Browser — {path}")

        for entry in entries:
            try:
                stat = entry.stat()
                size = self._human_size(stat.st_size) if entry.is_file() else ""
                import datetime
                mod  = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                size, mod = "", ""
            icon = "📁" if entry.is_dir() else "📄"
            self.tree.insert("", "end", iid=str(entry),
                              text=f"{icon} {entry.name}", values=(size, mod))

        count = len(list(path.iterdir()))
        self.status_var.set(f"{count} items  |  {path}")

    def _human_size(self, n: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024:
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate(self, path: Path):
        self._history = self._history[:self._hist_idx+1]
        self._history.append(path)
        self._hist_idx = len(self._history) - 1
        self._populate(path)

    def _back(self):
        if self._hist_idx > 0:
            self._hist_idx -= 1
            self._populate(self._history[self._hist_idx])

    def _forward(self):
        if self._hist_idx < len(self._history) - 1:
            self._hist_idx += 1
            self._populate(self._history[self._hist_idx])

    def _up(self):
        parent = self._cwd.parent
        if parent != self._cwd:
            self._navigate(parent)

    def _home(self):
        self._navigate(Path.home())

    def _go_to(self):
        folder = filedialog.askdirectory(initialdir=str(self._cwd))
        if folder:
            self._navigate(Path(folder))

    def _go_to_path(self):
        p = Path(self.path_var.get())
        if p.is_dir():
            self._navigate(p)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_double_click(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        if path.is_dir():
            self._navigate(path)
        else:
            self._open_file(path)

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        self._show_info(path)
        if path.is_file():
            self._show_preview(path)

    def _show_info(self, path: Path):
        try:
            stat = path.stat()
            import datetime
            mod = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            info = (f"Name: {path.name}\n"
                    f"Type: {'Directory' if path.is_dir() else path.suffix or 'File'}\n"
                    f"Size: {self._human_size(stat.st_size) if path.is_file() else '—'}\n"
                    f"Modified: {mod}\n"
                    f"Path: {path}")
        except Exception as e:
            info = str(e)
        self.info_label.config(text=info)

    def _show_preview(self, path: Path):
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        try:
            if path.stat().st_size > 200_000:
                self.preview.insert("end", "[File too large to preview]")
            else:
                text = path.read_text(errors="replace")
                self.preview.insert("end", text[:5000] + ("…" if len(text) > 5000 else ""))
        except Exception as e:
            self.preview.insert("end", f"[Cannot preview: {e}]")
        self.preview.configure(state="disabled")

    def _open_file(self, path: Path):
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _context_menu(self, event):
        sel = self.tree.identify_row(event.y)
        if not sel:
            return
        self.tree.selection_set(sel)
        path = Path(sel)
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Open",          command=lambda: self._on_double_click(None))
        menu.add_command(label="Copy path",      command=lambda: self._copy_path(path))
        menu.add_separator()
        menu.add_command(label="Rename…",        command=lambda: self._rename(path))
        menu.add_command(label="Delete",         command=lambda: self._delete(path))
        menu.tk_popup(event.x_root, event.y_root)

    def _copy_path(self, path: Path):
        self.clipboard_clear()
        self.clipboard_append(str(path))
        self.status_var.set(f"Copied: {path}")

    def _rename(self, path: Path):
        new_name = simpledialog.askstring("Rename", "New name:", initialvalue=path.name, parent=self)
        if new_name and new_name.strip():
            new_path = path.parent / new_name.strip()
            try:
                path.rename(new_path)
                self._populate(self._cwd)
            except Exception as e:
                messagebox.showerror("Rename failed", str(e))

    def _delete(self, path: Path):
        if messagebox.askyesno("Delete", f"Delete '{path.name}'?"):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                self._populate(self._cwd)
            except Exception as e:
                messagebox.showerror("Delete failed", str(e))


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else None
    FileBrowser(start).mainloop()


if __name__ == "__main__":
    main()
