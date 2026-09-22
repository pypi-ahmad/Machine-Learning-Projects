"""Look up and display infobox fields from English Wikipedia."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/{}"
USER_AGENT = "WikipediaInfoboxScraper/1.0 (local desktop utility)"


def page_url(title: str) -> str:
    """Build an English Wikipedia page URL from a non-empty title."""
    cleaned = " ".join(title.split())
    if not cleaned:
        raise ValueError("Enter a Wikipedia page title.")
    return WIKIPEDIA_URL.format(quote(cleaned.replace(" ", "_")))


def parse_infobox(html: str) -> dict[str, str]:
    """Extract direct infobox header and value cells from page HTML."""
    soup = BeautifulSoup(html, "html.parser")
    infobox = soup.select_one("table.infobox")
    if infobox is None:
        raise ValueError("This page does not contain an infobox.")
    fields: dict[str, str] = {}
    for row in infobox.select("tr"):
        header = row.find("th", recursive=False)
        value = row.find("td", recursive=False)
        if header is None or value is None:
            continue
        label = header.get_text(" ", strip=True)
        text = value.get_text(" ", strip=True)
        if label and text:
            fields[label] = text
    if not fields:
        raise ValueError("The infobox did not contain readable fields.")
    return fields


def fetch_infobox(title: str, timeout: float = 15) -> tuple[str, dict[str, str]]:
    """Fetch and parse one English Wikipedia page with a bounded request."""
    url = page_url(title)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return url, parse_infobox(response.text)


class InfoboxApp:
    """Small Tkinter interface that keeps requests off the UI thread."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Wikipedia Infobox")
        self.title_var = tk.StringVar()
        frame = tk.Frame(root, padx=12, pady=12)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Wikipedia page title").pack(anchor="w")
        self.entry = tk.Entry(frame, textvariable=self.title_var, width=50)
        self.entry.pack(fill="x", pady=(0, 8))
        self.entry.focus_set()
        self.button = tk.Button(frame, text="Fetch infobox", command=self.start_fetch)
        self.button.pack(anchor="w")
        self.status = tk.StringVar(value="Enter a page title.")
        tk.Label(frame, textvariable=self.status).pack(anchor="w", pady=(8, 4))
        self.output = tk.Text(frame, width=80, height=20, wrap="word", state="disabled")
        self.output.pack(fill="both", expand=True)

    def start_fetch(self) -> None:
        """Start one request thread after validating the entered title."""
        title = self.title_var.get()
        try:
            page_url(title)
        except ValueError as error:
            messagebox.showerror("Wikipedia Infobox", str(error))
            return
        self.button.config(state="disabled")
        self.status.set("Fetching Wikipedia page...")
        threading.Thread(target=self.fetch_worker, args=(title,), daemon=True).start()

    def fetch_worker(self, title: str) -> None:
        """Fetch data outside the UI thread and send the result back to Tkinter."""
        try:
            url, fields = fetch_infobox(title)
        except (requests.RequestException, ValueError) as error:
            self.root.after(0, self.show_error, str(error))
            return
        self.root.after(0, self.show_fields, url, fields)

    def show_fields(self, url: str, fields: dict[str, str]) -> None:
        """Display parsed fields after a successful request."""
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"{url}\n\n")
        for label, value in fields.items():
            self.output.insert(tk.END, f"{label}: {value}\n")
        self.output.config(state="disabled")
        self.status.set(f"Displayed {len(fields)} field(s).")
        self.button.config(state="normal")

    def show_error(self, message: str) -> None:
        """Show a fetch or parsing error and re-enable the UI."""
        self.status.set("No infobox displayed.")
        self.button.config(state="normal")
        messagebox.showerror("Wikipedia Infobox", message)


def main() -> None:
    """Launch the local desktop interface."""
    root = tk.Tk()
    InfoboxApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
