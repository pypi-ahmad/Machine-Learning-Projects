"""Stock Watchlist — Streamlit app.

Track a list of stock tickers with live prices (via yfinance),
price change alerts, and a simple portfolio value tracker.

Usage:
    uv run streamlit run main.py
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stock Watchlist", layout="wide")
st.title("📈 Stock Watchlist")

DATA_FILE = Path(__file__).with_name("watchlist.json")

DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]


def load_watchlist() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {t: {"shares": 0, "buy_price": 0, "alert_low": 0, "alert_high": 0}
            for t in DEMO_TICKERS}


def save_watchlist(wl: dict) -> None:
    DATA_FILE.write_text(json.dumps(wl, indent=2), encoding="utf-8")


@st.cache_data(ttl=300)
def fetch_quote(ticker: str) -> dict | None:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        return {
            "price":  round(info.last_price, 2),
            "prev":   round(info.previous_close, 2),
            "high52": round(info.year_high, 2),
            "low52":  round(info.year_low, 2),
        }
    except Exception:
        return None


@st.cache_data(ttl=600)
def fetch_history(ticker: str, days: int = 90) -> pd.DataFrame:
    try:
        import yfinance as yf
        df = yf.download(ticker, start=date.today() - timedelta(days=days),
                         progress=False, auto_adjust=True)
        return df[["Close"]].rename(columns={"Close": ticker})
    except Exception:
        return pd.DataFrame()


if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

wl = st.session_state.watchlist

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Manage Watchlist")
new_ticker = st.sidebar.text_input("Add ticker (e.g. AAPL)").upper().strip()
if st.sidebar.button("Add") and new_ticker:
    if new_ticker not in wl:
        wl[new_ticker] = {"shares": 0, "buy_price": 0, "alert_low": 0, "alert_high": 0}
        save_watchlist(wl)
        st.rerun()

del_ticker = st.sidebar.selectbox("Remove ticker", [""] + list(wl.keys()))
if st.sidebar.button("Remove") and del_ticker:
    wl.pop(del_ticker, None)
    save_watchlist(wl)
    st.rerun()

fetch_market_data = st.sidebar.checkbox("Fetch live market data", value=False)

# ── Main ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Watchlist", "Charts", "Portfolio"])

with tab1:
    rows = []
    for ticker, meta in wl.items():
        q = fetch_quote(ticker) if fetch_market_data else None
        if q:
            change    = q["price"] - q["prev"]
            change_pct = change / q["prev"] * 100 if q["prev"] else 0
            alert = ""
            if meta["alert_low"] and q["price"] <= meta["alert_low"]:
                alert = "🔴 Below low alert"
            elif meta["alert_high"] and q["price"] >= meta["alert_high"]:
                alert = "🟢 Above high alert"
            rows.append({
                "Ticker":  ticker,
                "Price":   q["price"],
                "Change":  round(change, 2),
                "Change%": round(change_pct, 2),
                "52W High": q["high52"],
                "52W Low":  q["low52"],
                "Alert":   alert,
            })
        else:
            status = "Live data disabled" if not fetch_market_data else "No data"
            rows.append({"Ticker": ticker, "Price": "—", "Change": "—",
                         "Change%": "—", "52W High": "—", "52W Low": "—", "Alert": status})

    if rows:
        df_wl = pd.DataFrame(rows)
        st.dataframe(df_wl, hide_index=True)

with tab2:
    if not wl:
        st.info("Add a ticker to view chart options.")
    elif not fetch_market_data:
        st.info("Enable live market data in the sidebar to load price history.")
    else:
        sel = st.selectbox("Select ticker", list(wl.keys()))
        days = st.slider("Days of history", 30, 365, 90)
        hist = fetch_history(sel, days)
        if not hist.empty:
            st.line_chart(hist)
        else:
            st.info("Could not fetch price history.")

with tab3:
    st.subheader("Portfolio")
    total_value = 0
    if not fetch_market_data:
        st.info("Enable live market data in the sidebar to value the portfolio.")
    else:
        for ticker, meta in wl.items():
            q = fetch_quote(ticker)
            if q and meta["shares"] > 0:
                value = q["price"] * meta["shares"]
                cost  = meta["buy_price"] * meta["shares"]
                pl    = value - cost
                st.metric(ticker,
                           f"${value:,.2f}",
                           delta=f"${pl:+,.2f} ({pl/cost*100:+.1f}%)" if cost else f"${value:,.2f}")
                total_value += value
        if total_value:
            st.metric("**Total Portfolio Value**", f"${total_value:,.2f}")
        else:
            st.info("Add shares and buy prices in the JSON file to track portfolio value.")
