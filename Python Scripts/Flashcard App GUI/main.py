"""Flashcard App GUI — Tkinter desktop app.

Create, study, and manage flashcard decks with spaced-repetition
scoring and JSON persistence.

Usage:
    python main.py
"""

import json
import os
import random
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

DATA_FILE = os.path.join(os.path.dirname(__file__), "flashcards.json")


def load() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return {}


def save(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


class FlashcardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flashcard App")
        self.geometry("700x520")
        self.configure(bg="#1e1e2e")

        self._decks     = load()          # {deck_name: [{q, a, correct, wrong}]}
        self._deck_var  = tk.StringVar()
        self._study_deck = []
        self._study_idx  = 0
        self._flipped    = False
        self._session    = {"correct": 0, "wrong": 0}

        self._build_ui()
        self._refresh_deck_list()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#1e1e2e")
        style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4",
                         padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#45475a")])

        self._study_tab = tk.Frame(nb, bg="#1e1e2e")
        self._manage_tab = tk.Frame(nb, bg="#1e1e2e")
        nb.add(self._study_tab, text=" Study ")
        nb.add(self._manage_tab, text=" Manage Decks ")

        self._build_study_tab()
        self._build_manage_tab()

    # ── Study tab ─────────────────────────────────────────────────────────────

    def _build_study_tab(self):
        top = tk.Frame(self._study_tab, bg="#1e1e2e")
        top.pack(fill="x", padx=12, pady=8)
        tk.Label(top, text="Deck:", bg="#1e1e2e", fg="#888").pack(side="left")
        self._deck_menu = tk.OptionMenu(top, self._deck_var, "")
        self._deck_menu.config(bg="#313244", fg="#cdd6f4", activebackground="#45475a",
                                relief="flat", font=("Consolas", 10))
        self._deck_menu.pack(side="left", padx=6)
        tk.Button(top, text="▶ Start", command=self._start_session,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(side="left")
        tk.Button(top, text="🔀 Shuffle", command=self._shuffle,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=4)

        # Card area
        self._card_frame = tk.Frame(self._study_tab, bg="#313244",
                                     width=540, height=220)
        self._card_frame.pack(pady=12)
        self._card_frame.pack_propagate(False)
        self._card_frame.bind("<Button-1>", self._flip)

        self._card_label = tk.Label(self._card_frame, text="Select a deck to start",
                                     bg="#313244", fg="#cdd6f4",
                                     font=("Consolas", 14), wraplength=500,
                                     justify="center")
        self._card_label.pack(expand=True)
        self._card_label.bind("<Button-1>", self._flip)

        self._side_var = tk.StringVar(value="")
        tk.Label(self._study_tab, textvariable=self._side_var,
                 bg="#1e1e2e", fg="#888", font=("Consolas", 9)).pack()

        # Progress
        self._prog_var = tk.StringVar(value="Card 0 / 0")
        tk.Label(self._study_tab, textvariable=self._prog_var,
                 bg="#1e1e2e", fg="#888", font=("Consolas", 10)).pack()

        # Buttons
        btn = tk.Frame(self._study_tab, bg="#1e1e2e")
        btn.pack(pady=8)
        tk.Button(btn, text="✓ Know it", command=lambda: self._mark(True),
                  bg="#a6e3a1", fg="#1e1e2e", width=10, relief="flat",
                  font=("Consolas", 11)).pack(side="left", padx=8)
        tk.Button(btn, text="✗ Review again", command=lambda: self._mark(False),
                  bg="#f38ba8", fg="#1e1e2e", width=12, relief="flat",
                  font=("Consolas", 11)).pack(side="left", padx=8)
        tk.Button(btn, text="← Prev", command=self._prev_card,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=4)
        tk.Button(btn, text="Next →", command=self._next_card,
                  bg="#313244", fg="#cdd6f4", relief="flat").pack(side="left", padx=4)

        # Session stats
        self._sess_var = tk.StringVar(value="✓ 0  ✗ 0")
        tk.Label(self._study_tab, textvariable=self._sess_var,
                 bg="#1e1e2e", fg="#cdd6f4", font=("Consolas", 11)).pack()

    # ── Manage tab ────────────────────────────────────────────────────────────

    def _build_manage_tab(self):
        left = tk.Frame(self._manage_tab, bg="#1e1e2e", width=200)
        left.pack(side="left", fill="y", padx=8, pady=8)
        left.pack_propagate(False)
        tk.Label(left, text="Decks", bg="#1e1e2e", fg="#cba6f7",
                 font=("Consolas", 12, "bold")).pack(anchor="w")
        self._deck_lb = tk.Listbox(left, bg="#313244", fg="#cdd6f4",
                                    font=("Consolas", 10), selectbackground="#45475a",
                                    activestyle="none")
        self._deck_lb.pack(fill="both", expand=True)
        self._deck_lb.bind("<<ListboxSelect>>", self._on_deck_select)
        for lbl, cmd in [("+ New Deck", self._new_deck), ("✕ Delete Deck", self._del_deck)]:
            tk.Button(left, text=lbl, command=cmd, bg="#313244", fg="#cdd6f4",
                      relief="flat").pack(fill="x", pady=2)

        right = tk.Frame(self._manage_tab, bg="#1e1e2e")
        right.pack(fill="both", expand=True, padx=(0, 8), pady=8)

        # Form
        form = tk.Frame(right, bg="#1e1e2e")
        form.pack(fill="x", pady=(0, 8))
        for label in ["Question", "Answer"]:
            tk.Label(form, text=label, bg="#1e1e2e", fg="#888",
                     font=("Consolas", 9)).pack(anchor="w")
            e = tk.Text(form, bg="#313244", fg="#cdd6f4", height=3,
                        font=("Consolas", 11), insertbackground="#cba6f7", relief="flat")
            e.pack(fill="x", pady=(0, 4))
            setattr(self, f"_{label.lower()}_text", e)
        tk.Button(form, text="+ Add Card", command=self._add_card,
                  bg="#cba6f7", fg="#1e1e2e", relief="flat",
                  font=("Consolas", 10, "bold")).pack(anchor="w")

        # Card list
        cols = ("question", "answer", "correct", "wrong")
        self._card_tree = ttk.Treeview(right, columns=cols, show="headings", height=12)
        style = ttk.Style()
        style.configure("Treeview", background="#313244", foreground="#cdd6f4",
                         fieldbackground="#313244", font=("Consolas", 9))
        style.configure("Treeview.Heading", background="#45475a", foreground="#cdd6f4")
        for col, w in zip(cols, [220, 220, 60, 60]):
            self._card_tree.heading(col, text=col.title())
            self._card_tree.column(col, width=w, anchor="w")
        self._card_tree.pack(fill="both", expand=True)
        tk.Button(right, text="✕ Delete Card", command=self._del_card,
                  bg="#f38ba8", fg="#1e1e2e", relief="flat").pack(anchor="w", pady=4)

    # ── Study logic ────────────────────────────────────────────────────────────

    def _start_session(self):
        deck = self._deck_var.get()
        if deck not in self._decks or not self._decks[deck]:
            messagebox.showinfo("Empty", "No cards in this deck.")
            return
        self._study_deck = list(self._decks[deck])
        self._study_idx  = 0
        self._session    = {"correct": 0, "wrong": 0}
        self._show_card()

    def _shuffle(self):
        if self._study_deck:
            random.shuffle(self._study_deck)
            self._study_idx = 0
            self._show_card()

    def _show_card(self):
        if not self._study_deck:
            return
        card = self._study_deck[self._study_idx]
        self._flipped = False
        self._card_label.config(text=card["q"], fg="#cdd6f4")
        self._side_var.set("(click card to reveal answer)")
        total = len(self._study_deck)
        self._prog_var.set(f"Card {self._study_idx + 1} / {total}")
        self._sess_var.set(f"✓ {self._session['correct']}  ✗ {self._session['wrong']}")

    def _flip(self, _=None):
        if not self._study_deck:
            return
        card = self._study_deck[self._study_idx]
        if not self._flipped:
            self._card_label.config(text=card["a"], fg="#a6e3a1")
            self._side_var.set("(answer shown)")
            self._flipped = True
        else:
            self._card_label.config(text=card["q"], fg="#cdd6f4")
            self._side_var.set("(click card to reveal answer)")
            self._flipped = False

    def _mark(self, correct: bool):
        if not self._study_deck:
            return
        deck = self._deck_var.get()
        card = self._study_deck[self._study_idx]
        if correct:
            card["correct"] = card.get("correct", 0) + 1
            self._session["correct"] += 1
        else:
            card["wrong"] = card.get("wrong", 0) + 1
            self._session["wrong"] += 1
        # Update persistent store
        for c in self._decks.get(deck, []):
            if c["q"] == card["q"]:
                c.update(card)
        save(self._decks)
        self._next_card()

    def _next_card(self):
        if self._study_idx < len(self._study_deck) - 1:
            self._study_idx += 1
            self._show_card()
        else:
            messagebox.showinfo("Done!", f"Session complete!\n"
                                f"✓ {self._session['correct']}  ✗ {self._session['wrong']}")

    def _prev_card(self):
        if self._study_idx > 0:
            self._study_idx -= 1
            self._show_card()

    # ── Deck management ───────────────────────────────────────────────────────

    def _new_deck(self):
        name = simpledialog.askstring("New Deck", "Deck name:", parent=self)
        if name and name not in self._decks:
            self._decks[name] = []
            save(self._decks)
            self._refresh_deck_list()

    def _del_deck(self):
        sel = self._deck_lb.curselection()
        if not sel:
            return
        name = self._deck_lb.get(sel[0])
        if messagebox.askyesno("Delete", f"Delete deck '{name}'?"):
            del self._decks[name]
            save(self._decks)
            self._refresh_deck_list()

    def _on_deck_select(self, _):
        sel = self._deck_lb.curselection()
        if sel:
            self._refresh_card_list(self._deck_lb.get(sel[0]))

    def _add_card(self):
        sel = self._deck_lb.curselection()
        if not sel:
            messagebox.showinfo("Select Deck", "Select a deck first.")
            return
        deck = self._deck_lb.get(sel[0])
        q = self._question_text.get("1.0", "end-1c").strip()
        a = self._answer_text.get("1.0", "end-1c").strip()
        if not q or not a:
            messagebox.showerror("Missing", "Enter both question and answer.")
            return
        self._decks[deck].append({"q": q, "a": a, "correct": 0, "wrong": 0})
        save(self._decks)
        self._question_text.delete("1.0", "end")
        self._answer_text.delete("1.0", "end")
        self._refresh_card_list(deck)

    def _del_card(self):
        sel_deck = self._deck_lb.curselection()
        sel_card = self._card_tree.selection()
        if not sel_deck or not sel_card:
            return
        deck = self._deck_lb.get(sel_deck[0])
        idx  = int(self._card_tree.item(sel_card[0], "tags")[0])
        self._decks[deck].pop(idx)
        save(self._decks)
        self._refresh_card_list(deck)

    def _refresh_deck_list(self):
        self._deck_lb.delete(0, "end")
        names = list(self._decks.keys())
        for n in names:
            self._deck_lb.insert("end", n)
        menu = self._deck_menu["menu"]
        menu.delete(0, "end")
        for n in names:
            menu.add_command(label=n, command=lambda v=n: self._deck_var.set(v))
        if names:
            self._deck_var.set(names[0])

    def _refresh_card_list(self, deck: str):
        self._card_tree.delete(*self._card_tree.get_children())
        for i, card in enumerate(self._decks.get(deck, [])):
            self._card_tree.insert("", "end", tags=(str(i),),
                                   values=(card["q"][:30], card["a"][:30],
                                           card.get("correct", 0), card.get("wrong", 0)))


if __name__ == "__main__":
    app = FlashcardApp()
    app.mainloop()
