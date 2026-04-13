"""Crypto Price Viewer — Streamlit app.

Live crypto prices via CoinGecko public API (no API key needed).
Track portfolio, view price charts, and compare coins.

Usage:
    streamlit run main.py
"""

import json
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crypto Price Viewer", layout="wide")
st.title("₿ Crypto Price Viewer")

DATA_FILE = Path("crypto_portfolio.json")

DEFAULT_COINS = ["bitcoin", "ethereum", "binancecoin", "solana", "cardano",
                 "ripple", "polkadot", "dogecoin", "avalanche-2", "chainlink"]

COINGECKO_BASE = "https://api.coingecko.com/api/v3"


@st.cache_data(ttl=60)
def fetch_prices(coin_ids: list[str], vs: str = "usd") -> dict:
    ids = ",".join(coin_ids)
    url = f"{COINGECKO_BASE}/simple/price?ids={ids}&vs_currencies={vs}&include_24hr_change=true&include_market_cap=true"
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_history(coin_id: str, days: int = 30, vs: str = "usd") -> pd.DataFrame:
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart?vs_currency={vs}&days={days}"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
            prices = data.get("prices", [])
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df.set_index("date")[["price"]]
    except Exception:
        return pd.DataFrame()


def load_portfolio() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {}


def save_portfolio(p: dict):
    DATA_FILE.write_text(json.dumps(p, indent=2))


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
vs_currency = st.sidebar.selectbox("Currency", ["usd", "eur", "gbp", "jpy", "btc"], index=0)
extra = st.sidebar.text_input("Add coin ID (from CoinGecko)").lower().strip()
if st.sidebar.button("Add") and extra:
    if extra not in DEFAULT_COINS:
        DEFAULT_COINS.append(extra)
        st.rerun()

# ── Fetch prices ──────────────────────────────────────────────────────────────
prices = fetch_prices(DEFAULT_COINS, vs_currency)
last_update = datetime.now().strftime("%H:%M:%S")

tab1, tab2, tab3 = st.tabs(["Live Prices", "Chart", "Portfolio"])

with tab1:
    st.caption(f"Last updated: {last_update}  (auto-refresh every 60s)")
    rows = []
    for coin in DEFAULT_COINS:
        if coin in prices:
            p = prices[coin]
            rows.append({
                "Coin":       coin.replace("-", " ").title(),
                "Price":      p.get(vs_currency, 0),
                "24h Change%": p.get(f"{vs_currency}_24h_change", 0),
                "Market Cap": p.get(f"{vs_currency}_market_cap", 0),
            })
    if rows:
        df = pd.DataFrame(rows)
        df["Price"]       = df["Price"].map(lambda x: f"{x:,.6f}" if x < 1 else f"{x:,.2f}")
        df["24h Change%"] = df["24h Change%"].map(lambda x: f"{x:+.2f}%")
        df["Market Cap"]  = df["Market Cap"].map(lambda x: f"${x/1e9:.2f}B" if x > 1e9 else f"${x/1e6:.2f}M")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("Could not fetch prices. Check internet connection.")

with tab2:
    sel_coin = st.selectbox("Select coin", DEFAULT_COINS)
    days     = st.slider("Days", 7, 365, 30)
    hist     = fetch_history(sel_coin, days, vs_currency)
    if not hist.empty:
        st.subheader(f"{sel_coin.title()} — {days}d price chart ({vs_currency.upper()})")
        st.line_chart(hist)
    else:
        st.info("Could not fetch chart data.")

with tab3:
    portfolio = load_portfolio()
    st.subheader("My Portfolio")
    with st.form("add_holding"):
        col1, col2, col3 = st.columns(3)
        coin_sel = col1.selectbox("Coin", DEFAULT_COINS)
        amount   = col2.number_input("Amount held", min_value=0.0, step=0.01, format="%.6f")
        buy_price = col3.number_input(f"Buy price ({vs_currency.upper()})", min_value=0.0, step=0.01)
        if st.form_submit_button("Add/Update"):
            portfolio[coin_sel] = {"amount": amount, "buy_price": buy_price}
            save_portfolio(portfolio)
            st.rerun()

    total = 0.0
    for coin, meta in portfolio.items():
        if coin in prices:
            current = prices[coin].get(vs_currency, 0)
            value   = current * meta["amount"]
            cost    = meta["buy_price"] * meta["amount"]
            pl      = value - cost
            st.metric(coin.title(), f"{vs_currency.upper()} {value:,.2f}",
                       delta=f"{pl:+,.2f} ({pl/cost*100:+.1f}%)" if cost else None)
            total += value
    if total:
        st.metric("Total Portfolio", f"{vs_currency.upper()} {total:,.2f}")
