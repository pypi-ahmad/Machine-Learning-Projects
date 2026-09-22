"""Fetch and display introductory summaries from English Wikipedia."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox

import requests


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WikipediaSummaryGUI/1.0 (local desktop utility)"


def article_title(value: str) -> str:
    """Normalize a required article title."""
    title = " ".join(value.split())
    if not title:
        raise ValueError("Enter a Wikipedia article title.")
    return title


def fetch_summary(title: str, timeout: float = 15) -> tuple[str, str]:
    """Fetch the introductory plaintext extract for an English Wikipedia page."""
    response = requests.get(
        API_URL,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
            "format": "json",
            "titles": article_title(title),
        },
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    if not pages:
        raise ValueError("Wikipedia did not return a page.")
    page = list(pages.values())[0]
    if "missing" in page:
        raise ValueError("Wikipedia could not find that article.")
    extract = page.get("extract", "").strip()
    if not extract:
        raise ValueError("Wikipedia did not return an introductory summary.")
    return page.get("title", title), extract


class SummaryApp:
    """Tkinter interface that fetches summaries outside the event loop."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Wikipedia Summary")
        self.root.minsize(700, 500)
        frame = tk.Frame(root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Wikipedia article title").pack(anchor="w")
        self.title_var = tk.StringVar()
        self.entry = tk.Entry(frame, textvariable=self.title_var, width=60)
        self.entry.pack(fill="x", pady=(0, 8))
        self.entry.focus_set()
        self.button = tk.Button(frame, text="Get summary", command=self.start_fetch)
        self.button.pack(anchor="w")
        self.status = tk.StringVar(value="Enter an article title.")
        tk.Label(frame, textvariable=self.status).pack(anchor="w", pady=(8, 4))
        self.output = tk.Text(frame, wrap="word", state="disabled")
        self.output.pack(fill="both", expand=True)

    def start_fetch(self) -> None:
        """Validate input and start one summary request thread."""
        try:
            title = article_title(self.title_var.get())
        except ValueError as error:
            messagebox.showerror("Wikipedia Summary", str(error))
            return
        self.button.config(state="disabled")
        self.status.set("Fetching Wikipedia summary...")
        threading.Thread(target=self.fetch_worker, args=(title,), daemon=True).start()

    def fetch_worker(self, title: str) -> None:
        """Fetch remotely, then return the result to Tkinter's event thread."""
        try:
            resolved_title, extract = fetch_summary(title)
        except (requests.RequestException, ValueError) as error:
            self.root.after(0, self.show_error, str(error))
            return
        self.root.after(0, self.show_summary, resolved_title, extract)

    def show_summary(self, title: str, extract: str) -> None:
        """Render a successful title and summary."""
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"{title}\n\n{extract}")
        self.output.config(state="disabled")
        self.status.set("Summary displayed.")
        self.button.config(state="normal")

    def show_error(self, message: str) -> None:
        """Report a request error and restore the search control."""
        self.status.set("No summary displayed.")
        self.button.config(state="normal")
        messagebox.showerror("Wikipedia Summary", message)


def main() -> None:
    """Launch the local Tkinter summary interface."""
    root = tk.Tk()
    SummaryApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
