"""
Modern Time Series Forecasting Pipeline (April 2026)

Primary models (foundation-model forecasting):
  - AutoGluon TimeSeries  (AutoML ensemble, ~3 min fit with time_limit=180)
  - Chronos-Bolt          (Amazon zero-shot foundation model, ~30s on GPU)
  - Chronos-2             (Amazon universal foundation model, ~60s on GPU)
  - TimesFM               (Google foundation model, ~20s on GPU)

Classical baselines (kept for comparison only):
  - ARIMA(5,1,0)          (statsmodels, fast, CPU-only, <5s)
  - Prophet               (Meta, fast, CPU-only, <10s)

Tabular lag-feature baselines:
  - LightGBM / CatBoost / XGBoost (GBDT with lag features, ~10s each on GPU)
  - FLAML AutoML           (automated lag-feature model selection, 60s budget)

Compute requirements:
  - GPU recommended for foundation models (RTX 3060+ / 8 GB VRAM minimum)
  - AutoGluon-TS: 4+ GB RAM, ~3 min on CPU, ~1 min on GPU
  - Chronos / TimesFM: GPU strongly recommended (CPU fallback 5-10x slower)
  - Classical baselines (ARIMA / Prophet / GBDT): CPU-only, <30s each
  - FLAML: CPU-only, budget-capped at 60s

Metrics: RMSE, MAE, MAPE (where denominator is non-zero)
Export : metrics.json + metrics.csv + forecast.png
Data: Auto-downloaded at runtime
"""
import os, json, warnings, time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TARGET = "Close"
HORIZON = 30
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def mape_score(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def score(name, y_true, y_pred, table):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mp = mape_score(y_true, y_pred)
    table.append({"Model": name, "RMSE": rmse, "MAE": mae, "MAPE(%)": mp})
    mp_s = f"{mp:.2f}%" if not np.isnan(mp) else "N/A"
    print(f"  {name}: RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mp_s}")
    return rmse


def load_data():
    import yfinance as yf
    df = yf.download("AAPL MSFT GOOGL AMZN NVDA", period="5y", auto_adjust=True).reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Auto-detect date and target
    target = TARGET
    if target not in df.columns:
        for c in df.select_dtypes("number").columns:
            if any(kw in str(c).lower() for kw in ["close","price","value","sales","demand","total"]):
                target = c; break
        else:
            target = df.select_dtypes("number").columns[-1]
    for c in df.columns:
        if "date" in str(c).lower() or "time" in str(c).lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).sort_values(c).set_index(c); break
    print(f"Dataset: {df.shape}, target: {target}")
    return df, target


def forecast(df, target):
    results = {}
    metrics = []
    series = df[target].dropna().values.astype(float)
    n = len(series); split = n - HORIZON
    train, test = series[:split], series[split:]

    # === PRIMARY: AutoGluon TimeSeries ===
    try:
        t0 = time.perf_counter()
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        ts_df = pd.DataFrame({"item_id": ["s"] * split, "timestamp": pd.date_range("2020-01-01", periods=split, freq="D"), "target": train})
        ts_data = TimeSeriesDataFrame.from_data_frame(ts_df)
        predictor = TimeSeriesPredictor(prediction_length=HORIZON, eval_metric="RMSE",
                                         path=os.path.join(SAVE_DIR, "ag_ts"))
        predictor.fit(ts_data, time_limit=120, presets="medium_quality")
        ag_preds = predictor.predict(ts_data)
        y_pred = ag_preds["mean"].values[:len(test)]
        results["AutoGluon-TS"] = y_pred
        print(f"  AutoGluon-TS ({time.perf_counter()-t0:.1f}s)")
        score("AutoGluon-TS", test, y_pred, metrics)
        lb = predictor.leaderboard(ts_data)
        print("  Leaderboard (top 5):")
        for line in lb.head().to_string().splitlines():
            print(f"    {line}")
    except Exception as e: print(f"  AutoGluon-TS failed: {e}")

    # === FOUNDATION MODELS ===

    # Chronos-Bolt (fast zero-shot)
    try:
        t0 = time.perf_counter()
        import torch
        from chronos import ChronosPipeline
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                  device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        context = torch.tensor(train, dtype=torch.float32)
        y_pred = np.median(pipe.predict(context, HORIZON)[0].numpy(), axis=0)[:len(test)]
        results["Chronos-Bolt"] = y_pred
        print(f"  Chronos-Bolt ({time.perf_counter()-t0:.1f}s)")
        score("Chronos-Bolt", test, y_pred, metrics)
    except Exception as e: print(f"  Chronos-Bolt failed: {e}")

    # Chronos-2 (universal forecasting)
    try:
        t0 = time.perf_counter()
        import torch
        from chronos import ChronosPipeline
        pipe2 = ChronosPipeline.from_pretrained("amazon/chronos-2-base",
                   device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        context = torch.tensor(train, dtype=torch.float32)
        y_pred = np.median(pipe2.predict(context, HORIZON)[0].numpy(), axis=0)[:len(test)]
        results["Chronos-2"] = y_pred
        print(f"  Chronos-2 ({time.perf_counter()-t0:.1f}s)")
        score("Chronos-2", test, y_pred, metrics)
    except Exception as e: print(f"  Chronos-2 failed: {e}")

    # TimesFM (Google foundation model)
    try:
        t0 = time.perf_counter()
        import timesfm
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32,
                                            horizon_len=HORIZON),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
        freq = [0] * 1  # freq=0 -> daily
        y_pred, _ = tfm.forecast([train], freq)
        y_pred = y_pred[0][:len(test)]
        results["TimesFM"] = y_pred
        print(f"  TimesFM ({time.perf_counter()-t0:.1f}s)")
        score("TimesFM", test, y_pred, metrics)
    except Exception as e: print(f"  TimesFM failed: {e}")

    # === CLASSICAL BASELINES (comparison only) ===

    # ARIMA (statsmodels)
    try:
        t0 = time.perf_counter()
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train, order=(5, 1, 0))
        fitted = model.fit()
        y_pred = fitted.forecast(steps=HORIZON)[:len(test)]
        results["ARIMA(5,1,0)"] = y_pred
        print(f"  ARIMA(5,1,0) baseline ({time.perf_counter()-t0:.1f}s)")
        score("ARIMA(5,1,0)", test, y_pred, metrics)
    except Exception as e: print(f"  ARIMA failed: {e}")

    # Prophet (Meta)
    try:
        t0 = time.perf_counter()
        from prophet import Prophet
        p_df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=split, freq="D"), "y": train})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(p_df)
        future = m.make_future_dataframe(periods=HORIZON)
        fc = m.predict(future)
        y_pred = fc["yhat"].values[-HORIZON:][:len(test)]
        results["Prophet"] = y_pred
        print(f"  Prophet baseline ({time.perf_counter()-t0:.1f}s)")
        score("Prophet", test, y_pred, metrics)
    except Exception as e: print(f"  Prophet failed: {e}")

    # === TABULAR LAG-FEATURE BASELINES (GBDT + FLAML) ===
    lags = [1, 2, 3, 5, 7, 14, 21]
    lag_df = pd.DataFrame({"y": series})
    for lg in lags:
        lag_df[f"lag_{lg}"] = lag_df["y"].shift(lg)
    lag_df["rolling_7"] = lag_df["y"].rolling(7).mean()
    lag_df["rolling_14"] = lag_df["y"].rolling(14).mean()
    lag_df["rolling_28"] = lag_df["y"].rolling(28).mean()
    lag_df["diff_1"] = lag_df["y"].diff(1)
    lag_df["diff_7"] = lag_df["y"].diff(7)
    lag_df = lag_df.dropna()
    offset = max(lags) + 28  # account for rolling window
    lag_train = lag_df.iloc[:split - offset]
    lag_test = lag_df.iloc[split - offset:split - offset + HORIZON]

    if len(lag_test) >= HORIZON:
        X_lag_tr = lag_train.drop(columns=["y"]); y_lag_tr = lag_train["y"]
        X_lag_te = lag_test.drop(columns=["y"]); y_lag_te = lag_test["y"]

        for name, builder in [
            ("LightGBM-Lag", lambda: __import__("lightgbm").LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                device="gpu", verbose=-1, n_jobs=-1)),
            ("CatBoost-Lag", lambda: __import__("catboost").CatBoostRegressor(
                iterations=500, learning_rate=0.05, depth=6, task_type="GPU",
                devices="0", verbose=0)),
            ("XGBoost-Lag", lambda: __import__("xgboost").XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                device="cuda", tree_method="hist", verbosity=0, n_jobs=-1)),
        ]:
            try:
                t0 = time.perf_counter()
                m = builder()
                m.fit(X_lag_tr, y_lag_tr)
                y_pred = m.predict(X_lag_te)[:len(test)]
                results[name] = y_pred
                print(f"  {name} ({time.perf_counter()-t0:.1f}s)")
                score(name, y_lag_te.values[:len(y_pred)], y_pred, metrics)
            except Exception as e:
                print(f"  {name} failed: {e}")

        # FLAML AutoML on lag features (tabularized forecasting only)
        try:
            t0 = time.perf_counter()
            from flaml import AutoML
            flaml_model = AutoML()
            flaml_model.fit(X_lag_tr, y_lag_tr, task="regression", time_budget=60,
                           metric="rmse", verbose=0)
            y_pred = flaml_model.predict(X_lag_te)[:len(test)]
            results["FLAML-Lag"] = y_pred
            best = flaml_model.best_estimator
            print(f"  FLAML-Lag [best: {best}] ({time.perf_counter()-t0:.1f}s)")
            score("FLAML-Lag", y_lag_te.values[:len(y_pred)], y_pred, metrics)
        except Exception as e:
            print(f"  FLAML-Lag failed: {e}")

    # === METRICS SUMMARY ===
    if metrics:
        print()
        print("=" * 65)
        print("METRICS SUMMARY")
        print("=" * 65)
        summary = pd.DataFrame(metrics).sort_values("RMSE")
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)
        best_model = summary.iloc[0]["Model"]
        print(f"  Best model by RMSE: {best_model}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(train)), train, alpha=0.5, label="Train")
    ax.plot(range(len(train), len(train)+len(test)), test, linewidth=2, label="Actual")
    for name, y_pred in results.items():
        ax.plot(range(len(train), len(train)+len(y_pred)), y_pred, "--", label=name)
    ax.legend(); ax.set_title("Forecast Comparison")
    fig.savefig(os.path.join(SAVE_DIR, "forecast.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return metrics


def run_eda(df, target, save_dir):
    """Time Series Exploratory Data Analysis."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}" if hasattr(df.index, 'min') else "")
    print(f"Target column: {target}")
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\nMissing values ({len(n_miss)} columns):")
        print(n_miss.sort_values(ascending=False).head(10).to_string())
    else:
        print("\nNo missing values")
    desc = df.describe().T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    print("Summary statistics saved to eda_summary.csv")
    # Target time series plot
    fig, ax = plt.subplots(figsize=(14, 5))
    if target in df.columns:
        df[target].plot(ax=ax, color="steelblue")
    ax.set_title(f"Time Series: {target}")
    ax.set_xlabel("Time")
    fig.savefig(os.path.join(save_dir, "eda_timeseries.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    # Stationarity test (ADF)
    if target in df.columns:
        try:
            from statsmodels.tsa.stattools import adfuller
            series = df[target].dropna()
            if len(series) > 20:
                result = adfuller(series, maxlag=min(30, len(series)//3))
                print(f"\nADF Stationarity Test:")
                print(f"  Test Statistic: {result[0]:.4f}")
                print(f"  p-value: {result[1]:.4f}")
                print(f"  Stationary: {'Yes' if result[1] < 0.05 else 'No (p >= 0.05)'}")
        except Exception:
            pass
        # Seasonal decomposition
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            series = df[target].dropna()
            freq = min(max(7, len(series) // 10), 365)
            if len(series) > 2 * freq:
                decomp = seasonal_decompose(series, period=freq, extrapolate_trend="freq")
                fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
                decomp.observed.plot(ax=axes[0]); axes[0].set_title("Observed")
                decomp.trend.plot(ax=axes[1]); axes[1].set_title("Trend")
                decomp.seasonal.plot(ax=axes[2]); axes[2].set_title("Seasonal")
                decomp.resid.plot(ax=axes[3]); axes[3].set_title("Residual")
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir, "eda_decomposition.png"), dpi=100, bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass
    print("EDA plots saved.")


def main():
    print("=" * 60)
    print("TIME SERIES FORECASTING | April 2026")
    print("Primary: AutoGluon-TS, Chronos-Bolt, Chronos-2, TimesFM")
    print("Baselines: ARIMA, Prophet, LightGBM/CatBoost/XGBoost Lag, FLAML")
    print("=" * 60)
    df, target = load_data()
    run_eda(df, target, SAVE_DIR)
    metrics = forecast(df, target)

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
