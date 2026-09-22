# Stock Watchlist

A Streamlit watchlist for locally saved ticker symbols, optional price lookups,
simple price-threshold alerts, history charts, and portfolio-value estimates.

## Run

```powershell
uv sync
uv run streamlit run main.py
```

Add or remove ticker symbols from the sidebar. Enable **Fetch live market data**
only when you want the app to request quotes and price history through yfinance.
The app starts with live data disabled.

## Local storage

The watchlist is stored in `watchlist.json` beside `main.py` after the first
change. It records tickers plus shares, buy prices, and optional alert levels.
It is not encrypted or synchronized.

## Financial-data limitations

Prices and historical data are supplied by yfinance and may be delayed,
incomplete, unavailable, or unsuitable for trading decisions. Alert text does
not place orders or guarantee that a threshold was reached at a particular time.
This app is an educational tracker, not financial advice or a brokerage tool.

## Dependencies

uv manages pandas, Streamlit, and yfinance in `pyproject.toml`; exact resolved
versions are recorded in `uv.lock`.
