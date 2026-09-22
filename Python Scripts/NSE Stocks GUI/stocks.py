"""Browse selected NSE stock tables in a Tkinter desktop window."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as expected
from selenium.webdriver.support.ui import WebDriverWait


CATEGORY_TABLES = {
    "Most Active equities - Main Board": "mae_mainboard_tableC",
    "Most Active equities - SME": "mae_sme_tableC",
    "Most Active equities - ETFs": "mae_etf_tableC",
    "Most Active equities - Price Spurts": "mae_pricespurts_tableC",
    "Most Active equities - Volume Spurts": "mae_volumespurts_tableC",
    "NIFTY 50 Top 20 Gainers": "topgainer-Table",
    "NIFTY 50 Top 20 Losers": "toplosers-Table",
}
MOST_ACTIVE_CATEGORIES = frozenset(tuple(CATEGORY_TABLES)[:5])
NSE_URL = "https://www.nseindia.com/market-data/{}"
WAIT_SECONDS = 20


def category_url(category: str) -> str:
    """Return the appropriate NSE page for a configured category."""
    page = "most-active-equities" if category in MOST_ACTIVE_CATEGORIES else "top-gainers-loosers"
    return NSE_URL.format(page)


def extract_table_text(html: str, table_id: str) -> str:
    """Return readable text from the selected NSE table."""
    table = BeautifulSoup(html, "html.parser").find("table", id=table_id)
    if table is None:
        raise ValueError("The selected NSE table was not present in the loaded page.")
    text = table.get_text("\n", strip=True)
    if not text:
        raise ValueError("The selected NSE table did not contain any data.")
    return text


def fetch_table_html(category: str) -> str:
    """Load the category page and wait for its table using Selenium Manager."""
    driver = webdriver.Chrome()
    try:
        driver.get(category_url(category))
        WebDriverWait(driver, WAIT_SECONDS).until(
            expected.presence_of_element_located((By.ID, CATEGORY_TABLES[category]))
        )
        return driver.page_source
    except TimeoutException as error:
        raise RuntimeError("NSE did not load the selected table in time.") from error
    finally:
        driver.quit()


class StocksApp:
    """Tkinter interface for one NSE table at a time."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NSE Stock data")
        self.root.geometry("1000x700")

        controls = ttk.Frame(root, padding=12)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Market data").grid(row=0, column=0, padx=(0, 8))
        self.category = ttk.Combobox(controls, values=tuple(CATEGORY_TABLES), state="readonly", width=42)
        self.category.grid(row=0, column=1, padx=(0, 12))
        self.category.current(0)
        self.fetch_button = ttk.Button(controls, text="Get stock data", command=self.fetch)
        self.fetch_button.grid(row=0, column=2)

        self.results = tk.Text(root, wrap=tk.NONE, state=tk.DISABLED)
        self.results.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def fetch(self) -> None:
        """Load a category table without blocking the GUI."""
        self.fetch_button.configure(state=tk.DISABLED)
        self.root.config(cursor="watch")
        thread = threading.Thread(target=self._fetch_in_background, args=(self.category.get(),), daemon=True)
        thread.start()

    def _fetch_in_background(self, category: str) -> None:
        try:
            table_text = extract_table_text(fetch_table_html(category), CATEGORY_TABLES[category])
        except (RuntimeError, ValueError, WebDriverException) as error:
            self.root.after(0, self._show_error, str(error))
            return
        self.root.after(0, self._show_results, table_text)

    def _finish_request(self) -> None:
        self.fetch_button.configure(state=tk.NORMAL)
        self.root.config(cursor="")

    def _show_error(self, error: str) -> None:
        self._finish_request()
        messagebox.showerror("Unable to load NSE data", error, parent=self.root)

    def _show_results(self, table_text: str) -> None:
        self._finish_request()
        self.results.configure(state=tk.NORMAL)
        self.results.delete("1.0", tk.END)
        self.results.insert(tk.END, table_text)
        self.results.configure(state=tk.DISABLED)


def main() -> None:
    """Launch the NSE table viewer."""
    root = tk.Tk()
    StocksApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
