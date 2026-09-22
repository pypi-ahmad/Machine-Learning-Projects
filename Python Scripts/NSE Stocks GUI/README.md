# NSE Stocks GUI

Tkinter desktop viewer for selected public NSE market-data tables.

## Setup

Requirements: Python 3.13+ and a locally installed Chrome browser.

```powershell
cd "NSE Stocks GUI"
uv sync
```

Selenium Manager automatically provides a compatible ChromeDriver. Do not download or enter a driver path manually.

## Usage

```powershell
uv run python stocks.py
```

Choose one category, then click **Get stock data**. The interface stays usable while Chrome loads the NSE table. The application displays table text after it loads.

Available categories:

- Most Active equities: Main Board, SME, ETFs, Price Spurts, Volume Spurts.
- NIFTY 50 Top 20 Gainers.
- NIFTY 50 Top 20 Losers.

## Behavior

1. Opens the matching NSE page in Chrome.
2. Waits up to 20 seconds for the configured table ID.
3. Extracts readable table text from the rendered page.
4. Shows a clear error if Chrome, NSE, or the table is unavailable.

NSE can change page markup, block or limit automated requests, or display data that differs by session. This application does not bypass access controls, rate limits, or CAPTCHA checks. It is a read-only viewer, not trading software or financial advice.

## Project files

```text
NSE Stocks GUI/
├── stocks.py
├── pyproject.toml
└── uv.lock
```
