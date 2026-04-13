"""Sales Forecasting — Streamlit ML demo.

Forecast future sales using moving average, exponential smoothing,
and linear trend extrapolation. Upload CSV or use sample data.

Usage:
    streamlit run main.py
"""

import math
import random
from datetime import date, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Forecasting", layout="wide")
st.title("📈 Sales Forecasting")


# ── Forecasting models ────────────────────────────────────────────────────────

def moving_average(series: list[float], window: int) -> list[float]:
    result = [None] * (window - 1)
    for i in range(window - 1, len(series)):
        result.append(sum(series[i - window + 1:i + 1]) / window)
    return result


def exp_smoothing(series: list[float], alpha: float = 0.3) -> list[float]:
    result = [series[0]]
    for v in series[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return result


def linear_trend(series: list[float]) -> tuple[float, float]:
    """Return (slope, intercept) via OLS."""
    n   = len(series)
    xs  = list(range(n))
    mx  = sum(xs) / n
    my  = sum(series) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, series))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0
    return slope, my - slope * mx


def forecast_linear(series: list[float], horizon: int) -> list[float]:
    slope, intercept = linear_trend(series)
    n = len(series)
    return [slope * (n + i) + intercept for i in range(horizon)]


def forecast_exp(series: list[float], horizon: int, alpha: float = 0.3) -> list[float]:
    smoothed = exp_smoothing(series, alpha)
    last     = smoothed[-1]
    slope    = smoothed[-1] - smoothed[-2] if len(smoothed) > 1 else 0
    return [last + slope * (i + 1) for i in range(horizon)]


def mae(actual: list[float], predicted: list[float]) -> float:
    pairs = [(a, p) for a, p in zip(actual, predicted) if p is not None]
    return sum(abs(a - p) for a, p in pairs) / len(pairs) if pairs else 0


def mape(actual: list[float], predicted: list[float]) -> float:
    pairs = [(a, p) for a, p in zip(actual, predicted) if p is not None and a != 0]
    return sum(abs(a - p) / abs(a) * 100 for a, p in pairs) / len(pairs) if pairs else 0


# ── Sample data ───────────────────────────────────────────────────────────────

def make_sample_sales(months: int = 24, seed: int = 42) -> pd.DataFrame:
    rng   = random.Random(seed)
    start = date(2023, 1, 1)
    rows  = []
    base  = 50000
    for m in range(months):
        dt     = start + timedelta(days=30 * m)
        trend  = base + m * 1200
        season = math.sin(math.pi * m / 6) * 8000   # ~12-month cycle
        noise  = rng.gauss(0, 3000)
        sales  = max(10000, trend + season + noise)
        rows.append({"date": str(dt.replace(day=1))[:7], "sales": round(sales, 2)})
    return pd.DataFrame(rows)


# ── UI ────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Forecast", "Model Comparison", "Sample Data"])

with tab1:
    st.subheader("Sales Time Series Forecast")
    uploaded = st.file_uploader("Upload CSV (date, sales columns)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        date_col  = st.selectbox("Date column",  df.columns.tolist())
        sales_col = st.selectbox("Sales column", df.columns.tolist())
        df = df.rename(columns={date_col: "date", sales_col: "sales"})
    else:
        st.info("Using sample monthly sales data.")
        df = make_sample_sales()

    series = df["sales"].tolist()
    dates  = df["date"].tolist()
    n      = len(series)

    c1, c2, c3 = st.columns(3)
    horizon = c1.slider("Forecast horizon (months)", 1, 24, 6)
    window  = c2.slider("Moving average window", 2, min(12, n-1), 3)
    alpha   = c3.slider("EXP smoothing α", 0.1, 0.9, 0.3, step=0.05)

    # Compute
    ma_vals   = moving_average(series, window)
    exp_vals  = exp_smoothing(series, alpha)
    fc_linear = forecast_linear(series, horizon)
    fc_exp    = forecast_exp(series, horizon, alpha)

    # Chart data
    future_dates = [f"F+{i+1}" for i in range(horizon)]
    all_dates    = dates + future_dates

    chart_data = {
        "Actual":       series + [None]*horizon,
        "MA":           ma_vals + [None]*horizon,
        "EXP":          exp_vals + [None]*horizon,
        "Linear Trend": [None]*n + fc_linear,
        "EXP Forecast": [None]*n + fc_exp,
    }
    chart_df = pd.DataFrame(chart_data, index=all_dates)
    st.line_chart(chart_df)

    # Metrics
    valid_ma  = [(a, p) for a, p in zip(series, ma_vals)  if p is not None]
    valid_exp = list(zip(series, exp_vals))
    c1, c2 = st.columns(2)
    c1.metric("MA MAE",  f"${mae(series, ma_vals):,.0f}")
    c2.metric("EXP MAPE", f"{mape(series, exp_vals):.1f}%")

    st.subheader(f"Next {horizon} Month Forecast")
    fc_df = pd.DataFrame({
        "Period":       future_dates,
        "Linear Trend": [f"${v:,.0f}" for v in fc_linear],
        "EXP Smooth":   [f"${v:,.0f}" for v in fc_exp],
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

with tab2:
    train_n = int(n * 0.8)
    train   = series[:train_n]
    test    = series[train_n:]

    ma_test  = moving_average(train + test, window)[train_n:]
    exp_test = exp_smoothing(train + test, alpha)[train_n:]
    lin_test = forecast_linear(train, len(test))

    if test:
        rows = []
        for model_name, preds in [("Moving Average", ma_test),
                                   ("EXP Smoothing",  exp_test),
                                   ("Linear Trend",   lin_test)]:
            rows.append({
                "Model": model_name,
                "MAE":   f"${mae(test, preds):,.0f}",
                "MAPE":  f"{mape(test, preds):.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Train: {train_n} months, Test: {len(test)} months")

with tab3:
    df2 = make_sample_sales(36)
    st.dataframe(df2, use_container_width=True, hide_index=True)
    st.metric("Total Sales", f"${df2['sales'].sum():,.0f}")
    st.metric("Average Monthly", f"${df2['sales'].mean():,.0f}")
