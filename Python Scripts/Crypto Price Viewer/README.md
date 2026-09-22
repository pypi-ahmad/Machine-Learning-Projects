# Crypto Price Viewer

Local Streamlit viewer for CoinGecko price data, price charts, and a private local portfolio.

```powershell
uv sync
uv run streamlit run main.py
```

Portfolio data is stored in `crypto_portfolio.json` beside the app and ignored by Git. Prices are live data from CoinGecko and can be delayed, unavailable, or incomplete. This learning tool is not financial advice and must not be used as the sole basis for trading or investment decisions.

Verify syntax with `uv run python -m py_compile main.py`.
