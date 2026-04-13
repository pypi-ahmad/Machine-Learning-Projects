"""
Modern Time Series Forecasting Pipeline (April 2026)

Metrics: RMSE, MAE, MAPE (where denominator is non-zero)
Export : metrics.json + metrics.csv + forecast.png
Data: Auto-downloaded at runtime via Kaggle (usdot/flight-delays)
       Flight-level data is aggregated to daily delayed-flight counts.
"""
import os, json, warnings, time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TARGET = "delay_count"
HORIZON = 28
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def mape_score(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0: return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def score(name, y_true, y_pred, table):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mp = mape_score(y_true, y_pred)
    table.append({"Model": name, "RMSE": rmse, "MAE": mae, "MAPE(%)": mp})
    print(f"  {name}: RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={f'{mp:.2f}%' if not np.isnan(mp) else 'N/A'}")
    return rmse


def load_data():
    import os, glob as _glob
    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(_data_dir, exist_ok=True)
    _csvs = _glob.glob(os.path.join(_data_dir, "**", "flights.csv"), recursive=True)
    if not _csvs:
        _csvs = _glob.glob(os.path.join(_data_dir, "**", "*.csv"), recursive=True)
    if not _csvs:
        from kaggle.api.kaggle_api_extended import KaggleApi
        _api = KaggleApi(); _api.authenticate()
        _api.dataset_download_files("usdot/flight-delays", path=_data_dir, unzip=True)
        _csvs = _glob.glob(os.path.join(_data_dir, "**", "flights.csv"), recursive=True)
        if not _csvs: _csvs = _glob.glob(os.path.join(_data_dir, "**", "*.csv"), recursive=True)
        print("Downloaded usdot/flight-delays from Kaggle")
    # Pick flights.csv (largest file with flight records)
    _fp = None
    for f in _csvs:
        if "flight" in os.path.basename(f).lower(): _fp = f; break
    if _fp is None: _fp = sorted(_csvs, key=os.path.getsize, reverse=True)[0]
    df = pd.read_csv(_fp, nrows=500000)
    # Find date and delay columns
    date_col = None
    for c in df.columns:
        if "fl_date" in str(c).lower() or "date" in str(c).lower(): date_col = c; break
    delay_col = None
    for c in df.columns:
        if "dep_delay" in str(c).lower() or "delay" in str(c).lower(): delay_col = c; break
    if date_col is None: date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    # Aggregate: count flights delayed by 15+ minutes per day
    if delay_col and delay_col in df.columns:
        df["delayed"] = (pd.to_numeric(df[delay_col], errors="coerce") >= 15).astype(int)
    else:
        df["delayed"] = 1
    df["day"] = df[date_col].dt.date
    daily = df.groupby("day")["delayed"].sum().reset_index()
    daily.columns = ["day", "delay_count"]
    daily["day"] = pd.to_datetime(daily["day"])
    daily = daily.sort_values("day").set_index("day")
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0); daily.index.name = "day"
    target = TARGET
    print(f"Dataset: {daily.shape}, target: {target}")
    return daily, target


def forecast(df, target):
    results = {}; metrics = []
    series = df[target].dropna().values.astype(float)
    n = len(series); split = n - HORIZON
    train, test = series[:split], series[split:]

    try:
        t0 = time.perf_counter()
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        ts_df = pd.DataFrame({"item_id": ["s"]*split, "timestamp": pd.date_range("2020-01-01", periods=split, freq="D"), "target": train})
        ts_data = TimeSeriesDataFrame.from_data_frame(ts_df)
        predictor = TimeSeriesPredictor(prediction_length=HORIZON, eval_metric="RMSE", path=os.path.join(SAVE_DIR, "ag_ts"))
        predictor.fit(ts_data, time_limit=120, presets="medium_quality")
        y_pred = predictor.predict(ts_data)["mean"].values[:len(test)]
        results["AutoGluon-TS"] = y_pred
        print(f"  AutoGluon-TS ({time.perf_counter()-t0:.1f}s)"); score("AutoGluon-TS", test, y_pred, metrics)
        lb = predictor.leaderboard(ts_data); print("  Leaderboard (top 5):")
        for line in lb.head().to_string().splitlines(): print(f"    {line}")
    except Exception as e: print(f"  AutoGluon-TS failed: {e}")

    try:
        t0 = time.perf_counter(); import torch; from chronos import ChronosPipeline
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        y_pred = np.median(pipe.predict(torch.tensor(train, dtype=torch.float32), HORIZON)[0].numpy(), axis=0)[:len(test)]
        results["Chronos-Bolt"] = y_pred; print(f"  Chronos-Bolt ({time.perf_counter()-t0:.1f}s)"); score("Chronos-Bolt", test, y_pred, metrics)
    except Exception as e: print(f"  Chronos-Bolt failed: {e}")

    try:
        t0 = time.perf_counter(); import torch; from chronos import ChronosPipeline
        pipe2 = ChronosPipeline.from_pretrained("amazon/chronos-2-base", device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        y_pred = np.median(pipe2.predict(torch.tensor(train, dtype=torch.float32), HORIZON)[0].numpy(), axis=0)[:len(test)]
        results["Chronos-2"] = y_pred; print(f"  Chronos-2 ({time.perf_counter()-t0:.1f}s)"); score("Chronos-2", test, y_pred, metrics)
    except Exception as e: print(f"  Chronos-2 failed: {e}")

    try:
        t0 = time.perf_counter(); import timesfm
        tfm = timesfm.TimesFm(hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32, horizon_len=HORIZON), checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
        y_pred, _ = tfm.forecast([train], [0]); y_pred = y_pred[0][:len(test)]
        results["TimesFM"] = y_pred; print(f"  TimesFM ({time.perf_counter()-t0:.1f}s)"); score("TimesFM", test, y_pred, metrics)
    except Exception as e: print(f"  TimesFM failed: {e}")

    try:
        t0 = time.perf_counter(); from statsmodels.tsa.arima.model import ARIMA
        y_pred = ARIMA(train, order=(5,1,0)).fit().forecast(steps=HORIZON)[:len(test)]
        results["ARIMA(5,1,0)"] = y_pred; print(f"  ARIMA(5,1,0) baseline ({time.perf_counter()-t0:.1f}s)"); score("ARIMA(5,1,0)", test, y_pred, metrics)
    except Exception as e: print(f"  ARIMA failed: {e}")

    try:
        t0 = time.perf_counter(); from prophet import Prophet
        p_df = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=split, freq="D"), "y": train})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False); m.fit(p_df)
        y_pred = m.predict(m.make_future_dataframe(periods=HORIZON))["yhat"].values[-HORIZON:][:len(test)]
        results["Prophet"] = y_pred; print(f"  Prophet baseline ({time.perf_counter()-t0:.1f}s)"); score("Prophet", test, y_pred, metrics)
    except Exception as e: print(f"  Prophet failed: {e}")

    lags = [1,2,3,5,7,14,21]
    lag_df = pd.DataFrame({"y": series})
    for lg in lags: lag_df[f"lag_{lg}"] = lag_df["y"].shift(lg)
    lag_df["rolling_7"] = lag_df["y"].rolling(7).mean(); lag_df["rolling_14"] = lag_df["y"].rolling(14).mean()
    lag_df["rolling_28"] = lag_df["y"].rolling(28).mean()
    lag_df["diff_1"] = lag_df["y"].diff(1); lag_df["diff_7"] = lag_df["y"].diff(7)
    lag_df = lag_df.dropna(); offset = max(lags) + 28
    lag_train = lag_df.iloc[:split-offset]; lag_test_df = lag_df.iloc[split-offset:split-offset+HORIZON]
    if len(lag_test_df) >= HORIZON:
        X_tr = lag_train.drop(columns=["y"]); y_tr = lag_train["y"]
        X_te = lag_test_df.drop(columns=["y"]); y_te = lag_test_df["y"]
        for name, builder in [("LightGBM-Lag", lambda: __import__("lightgbm").LGBMRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,device="gpu",verbose=-1,n_jobs=-1)),("CatBoost-Lag", lambda: __import__("catboost").CatBoostRegressor(iterations=500,learning_rate=0.05,depth=6,task_type="GPU",devices="0",verbose=0)),("XGBoost-Lag", lambda: __import__("xgboost").XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,device="cuda",tree_method="hist",verbosity=0,n_jobs=-1))]:
            try:
                t0 = time.perf_counter(); m = builder(); m.fit(X_tr, y_tr)
                y_pred = m.predict(X_te)[:len(test)]; results[name] = y_pred
                print(f"  {name} ({time.perf_counter()-t0:.1f}s)"); score(name, y_te.values[:len(y_pred)], y_pred, metrics)
            except Exception as e: print(f"  {name} failed: {e}")
        try:
            t0 = time.perf_counter(); from flaml import AutoML; fl = AutoML()
            fl.fit(X_tr, y_tr, task="regression", time_budget=60, metric="rmse", verbose=0)
            y_pred = fl.predict(X_te)[:len(test)]; results["FLAML-Lag"] = y_pred
            print(f"  FLAML-Lag [best: {fl.best_estimator}] ({time.perf_counter()-t0:.1f}s)"); score("FLAML-Lag", y_te.values[:len(y_pred)], y_pred, metrics)
        except Exception as e: print(f"  FLAML-Lag failed: {e}")

    if metrics:
        print(); print("="*65); print("METRICS SUMMARY"); print("="*65)
        summary = pd.DataFrame(metrics).sort_values("RMSE"); print(summary.to_string(index=False))
        summary.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False); print(f"  Best model by RMSE: {summary.iloc[0]['Model']}")
    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(range(len(train)), train, alpha=0.5, label="Train")
    ax.plot(range(len(train), len(train)+len(test)), test, linewidth=2, label="Actual")
    for name, yp in results.items(): ax.plot(range(len(train), len(train)+len(yp)), yp, "--", label=name)
    ax.legend(); ax.set_title("Forecast Comparison"); fig.savefig(os.path.join(SAVE_DIR, "forecast.png"), dpi=100, bbox_inches="tight"); plt.close(fig)
    return metrics


def run_eda(df, target, save_dir):
    print("\n"+"="*60); print("EXPLORATORY DATA ANALYSIS"); print("="*60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}" if hasattr(df.index, 'min') else ""); print(f"Target column: {target}")
    missing = df.isnull().sum(); n_miss = missing[missing>0]
    if len(n_miss): print(f"\nMissing values ({len(n_miss)} columns):"); print(n_miss.sort_values(ascending=False).head(10).to_string())
    else: print("\nNo missing values")
    df.describe().T.to_csv(os.path.join(save_dir, "eda_summary.csv")); print("Summary statistics saved to eda_summary.csv")
    fig, ax = plt.subplots(figsize=(14,5))
    if target in df.columns: df[target].plot(ax=ax, color="steelblue")
    ax.set_title(f"Time Series: {target}"); ax.set_xlabel("Time")
    fig.savefig(os.path.join(save_dir, "eda_timeseries.png"), dpi=100, bbox_inches="tight"); plt.close(fig)
    if target in df.columns:
        try:
            from statsmodels.tsa.stattools import adfuller; s = df[target].dropna()
            if len(s)>20: r = adfuller(s, maxlag=min(30, len(s)//3)); print(f"\nADF Stationarity Test:\n  Test Statistic: {r[0]:.4f}\n  p-value: {r[1]:.4f}\n  Stationary: {'Yes' if r[1]<0.05 else 'No (p >= 0.05)'}")
        except Exception: pass
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose; s = df[target].dropna(); freq = min(max(7, len(s)//10), 365)
            if len(s) > 2*freq:
                decomp = seasonal_decompose(s, period=freq, extrapolate_trend="freq")
                fig, axes = plt.subplots(4,1,figsize=(14,10),sharex=True)
                decomp.observed.plot(ax=axes[0]); axes[0].set_title("Observed"); decomp.trend.plot(ax=axes[1]); axes[1].set_title("Trend")
                decomp.seasonal.plot(ax=axes[2]); axes[2].set_title("Seasonal"); decomp.resid.plot(ax=axes[3]); axes[3].set_title("Residual")
                fig.tight_layout(); fig.savefig(os.path.join(save_dir, "eda_decomposition.png"), dpi=100, bbox_inches="tight"); plt.close(fig)
        except Exception: pass
    print("EDA plots saved.")


def main():
    print("="*60); print("TIME SERIES FORECASTING | April 2026"); print("Primary: AutoGluon-TS, Chronos-Bolt, Chronos-2, TimesFM")
    print("Baselines: ARIMA, Prophet, LightGBM/CatBoost/XGBoost Lag, FLAML"); print("="*60)
    df, target = load_data(); run_eda(df, target, SAVE_DIR); metrics = forecast(df, target)
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {os.path.join(SAVE_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()

