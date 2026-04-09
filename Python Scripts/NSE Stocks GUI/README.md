# NSE Stocks GUI

> A Tkinter-based GUI application that scrapes live NSE (National Stock Exchange of India) stock data using Selenium and BeautifulSoup.

## Overview

This application provides a graphical interface for browsing NSE stock market data. Users select a category from a dropdown menu (most active equities, top gainers, top losers, etc.), and the app uses Selenium with ChromeDriver to load the NSE website, scrape the dynamic stock table data, and display it in a formatted text widget.

## Features

- **7 data categories** available via dropdown:
  - Most Active Equities — Main Board, SME, ETFs, Price Spurts, Volume Spurts
  - NIFTY 50 Top 20 Gainers
  - NIFTY 50 Top 20 Losers
- Selenium-powered scraping to handle JavaScript-rendered NSE pages
- Formatted tabular display of stock data in the GUI
- Category-based URL routing (most-active-equities vs. top-gainers-loosers pages)
- Read-only results display area

## Project Structure

```
NSE Stocks GUI/
├── stocks.py          # Main application script
└── requirements.txt   # Python dependencies
```

## Requirements

- Python 3.x
- requests
- beautifulsoup4
- selenium
- tkinter (included with standard Python installations)
- ChromeDriver (must match your Chrome browser version)

## Installation

```bash
cd "NSE Stocks GUI"
pip install -r requirements.txt
```

Additionally, download ChromeDriver from [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads) matching your Chrome version, and note the path to the executable.

## Usage

```bash
python stocks.py
```

1. When prompted in the terminal, enter the path to your ChromeDriver executable
2. The GUI window opens (1200×1000)
3. Select a category from the dropdown
4. Click **"Get Stock Data!"**
5. A Chrome window opens briefly to load the NSE page, then stock data appears in the app

## How It Works

1. **ChromeDriver path**: Prompts user via `input()` for the ChromeDriver path at startup
2. **GUI setup**: Creates a Tkinter window with:
   - A title label ("NSE Stock market data")
   - A `ttk.Combobox` with 7 category options
   - A "Get Stock Data!" button
   - A `tk.Text` widget for displaying results
3. **`generate_url()`**: Maps the selected category to either the `most-active-equities` or `top-gainers-loosers` NSE page
4. **`scraper()`**:
   - Opens Chrome via Selenium, navigates to the generated URL
   - Waits 5 seconds for JavaScript to render the page
   - Parses the page source with BeautifulSoup
   - Locates the table by its HTML `id` (mapped from the category dictionaries)
   - Extracts all rows and cells, formats them into fixed-width columns (20 chars each)
   - Displays the formatted data in the text widget
   - Closes the Chrome window

## Configuration

| Setting | Location | Description |
|---------|----------|-------------|
| ChromeDriver path | Runtime `input()` prompt | Path to the ChromeDriver executable |
| `time.sleep(5)` | `scraper()` function | Wait time for page JavaScript to load |
| Column width | `scraper()` format string | Fixed at 20 characters per column |

### Category-to-Table ID Mapping

| Category | Table HTML ID |
|----------|---------------|
| Most Active — Main Board | `mae_mainboard_tableC` |
| Most Active — SME | `mae_sme_tableC` |
| Most Active — ETFs | `mae_etf_tableC` |
| Most Active — Price Spurts | `mae_pricespurts_tableC` |
| Most Active — Volume Spurts | `mae_volumespurts_tableC` |
| NIFTY 50 Top 20 Gainers | `topgainer-Table` |
| NIFTY 50 Top 20 Losers | `toplosers-Table` |

## Limitations

- Requires ChromeDriver to be manually downloaded and path entered at runtime
- Opens a visible Chrome window for each scrape operation (not headless)
- The 5-second sleep is a fixed delay — may be too short on slow connections or too long on fast ones
- Column width is truncated to 20 characters, which may cut off long stock names
- NSE website structure changes will break the table ID mappings
- The `requests` import is unused — only Selenium is used for fetching
- No caching — every button click triggers a full browser session
- The ChromeDriver path prompt is in the terminal, not in the GUI
- Uses deprecated Selenium APIs (modern Selenium uses `Service` objects for drivers)

## License

Not specified.
