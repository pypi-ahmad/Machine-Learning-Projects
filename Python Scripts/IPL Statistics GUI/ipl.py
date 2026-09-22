"""Browse IPL statistics from the official IPL statistics pages."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import requests
from bs4 import BeautifulSoup


CATEGORIES = {
    "Most Runs": "most-runs",
    "Most Fours": "most-fours",
    "Most Sixes": "most-sixes",
    "Most Fifties": "most-fifties",
    "Most Centuries": "most-centuries",
    "Highest Scores": "highest-scores",
    "Most Wickets": "most-wickets",
    "Most Maidens": "most-maidens",
    "Most Dot Balls": "most-dot-balls",
    "Best Bowling Average": "best-bowling-average",
    "Best Bowling Economy": "best-bowling-economy",
    "Best Bowling Strike Rate": "best-bowling-strike-rate",
}
SEASONS = ("All time", "2021", "2020", "2019", "2018", "2017", "2016", "2015", "2014", "2013", "2012", "2011", "2010", "2009", "2008")
REQUEST_TIMEOUT_SECONDS = 20


def build_stats_url(category: str, season: str) -> str:
    """Return the official IPL URL for a selected category and season."""
    season_slug = "all-time" if season == "All time" else season
    return f"https://www.iplt20.com/stats/{season_slug}/{CATEGORIES[category]}"


def parse_stats_table(html: str) -> str:
    """Extract readable text from the IPL statistics table."""
    table = BeautifulSoup(html, "html.parser").select_one("table.top-players")
    if table is None:
        raise ValueError("The IPL statistics table was not found in the response.")

    records = table.get_text("\n", strip=True)
    if not records:
        raise ValueError("The IPL statistics table did not contain any rows.")
    return records


def fetch_stats(category: str, season: str) -> str:
    """Download and parse statistics for a category and season."""
    response = requests.get(
        build_stats_url(category, season),
        headers={"User-Agent": "IPL-Statistics-GUI/0.1"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return parse_stats_table(response.text)


class StatisticsApp:
    """Tkinter interface for requesting and displaying IPL statistics."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("IPL Statistics")
        self.root.geometry("1000x700")

        controls = ttk.Frame(root, padding=12)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Category").grid(row=0, column=0, padx=(0, 6))
        self.category = ttk.Combobox(controls, values=list(CATEGORIES), state="readonly", width=27)
        self.category.grid(row=0, column=1, padx=(0, 12))
        self.category.current(0)

        ttk.Label(controls, text="Season").grid(row=0, column=2, padx=(0, 6))
        self.season = ttk.Combobox(controls, values=SEASONS, state="readonly", width=12)
        self.season.grid(row=0, column=3, padx=(0, 12))
        self.season.current(0)

        self.search_button = ttk.Button(controls, text="Search", command=self.search)
        self.search_button.grid(row=0, column=4)

        self.results = tk.Text(root, wrap=tk.NONE, state=tk.DISABLED)
        self.results.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def search(self) -> None:
        """Fetch a selection without freezing the Tkinter event loop."""
        self.search_button.configure(state=tk.DISABLED)
        self.root.config(cursor="watch")
        thread = threading.Thread(
            target=self._fetch_in_background,
            args=(self.category.get(), self.season.get()),
            daemon=True,
        )
        thread.start()

    def _fetch_in_background(self, category: str, season: str) -> None:
        try:
            records = fetch_stats(category, season)
        except (requests.RequestException, ValueError) as error:
            self.root.after(0, self._show_error, str(error))
            return
        self.root.after(0, self._show_records, records)

    def _finish_request(self) -> None:
        self.search_button.configure(state=tk.NORMAL)
        self.root.config(cursor="")

    def _show_error(self, error: str) -> None:
        self._finish_request()
        messagebox.showerror("Unable to load IPL statistics", error, parent=self.root)

    def _show_records(self, records: str) -> None:
        self._finish_request()
        self.results.configure(state=tk.NORMAL)
        self.results.delete("1.0", tk.END)
        self.results.insert(tk.END, records)
        self.results.configure(state=tk.DISABLED)


def main() -> None:
    """Launch the desktop application."""
    root = tk.Tk()
    StatisticsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
