"""
Modern Time Series Forecasting Pipeline (April 2026)
Models: AutoGluon TimeSeries + Chronos-Bolt + Chronos-2 + TimesFM + StatsForecast
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TARGET = "Close"
HORIZON = 30


def load_data():
    import yfinance as yf
    df = yf.download("AMZN", period="10y", auto_adjust=True).reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Auto-detect date and target
    target = TARGET
    if target not in df.columns:
        for c in df.select_dtypes("number").columns:
            if any(kw in c.lower() for kw in ["close","price","value","sales","demand","total"]):
                target = c; break
        else:
            target = df.select_dtypes("number").columns[-1]
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).sort_values(c).set_index(c); break
    print(f"Dataset: {df.shape}, target: {target}")
    return df, target


def forecast(df, target):
    results = {}
    series = df[target].dropna().values.astype(float)
    n = len(series); split = n - HORIZON
    train, test = series[:split], series[split:]

    # Chronos-Bolt
    try:
        import torch
        from chronos import ChronosPipeline
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                  device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        context = torch.tensor(train, dtype=torch.float32)
        y_pred = np.median(pipe.predict(context, HORIZON)[0].numpy(), axis=0)[:len(test)]
        rmse = mean_squared_error(test, y_pred, squared=False)
        results["Chronos-Bolt"] = y_pred
        print(f"✓ Chronos-Bolt RMSE: {rmse:.4f}")
    except Exception as e: print(f"✗ Chronos-Bolt: {e}")

    # StatsForecast
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS, AutoTheta, AutoARIMA
        sf_df = pd.DataFrame({"unique_id": ["s"]*split, "ds": pd.date_range("2020-01-01", periods=split, freq="D"), "y": train})
        sf = StatsForecast(models=[AutoETS(season_length=7), AutoTheta(season_length=7)], freq="D", n_jobs=-1)
        preds = sf.forecast(h=HORIZON, df=sf_df)
        for col in preds.columns:
            if col not in ("unique_id","ds"):
                y_pred = preds[col].values[:len(test)]
                rmse = mean_squared_error(test, y_pred, squared=False)
                results[col] = y_pred
                print(f"✓ {col} RMSE: {rmse:.4f}")
    except Exception as e: print(f"✗ StatsForecast: {e}")

    # Chronos-2
    try:
        import torch
        from chronos import ChronosPipeline
        pipe2 = ChronosPipeline.from_pretrained("amazon/chronos-2-base",
                   device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        context = torch.tensor(train, dtype=torch.float32)
        y_pred = np.median(pipe2.predict(context, HORIZON)[0].numpy(), axis=0)[:len(test)]
        rmse = mean_squared_error(test, y_pred, squared=False)
        results["Chronos-2"] = y_pred
        print(f"✓ Chronos-2 RMSE: {rmse:.4f}")
    except Exception as e: print(f"✗ Chronos-2: {e}")

    # TimesFM
    try:
        import timesfm
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32,
                                            horizon_len=HORIZON),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
        freq = [0] * 1  # freq=0 → daily
        y_pred, _ = tfm.forecast([train], freq)
        y_pred = y_pred[0][:len(test)]
        rmse = mean_squared_error(test, y_pred, squared=False)
        results["TimesFM"] = y_pred
        print(f"✓ TimesFM RMSE: {rmse:.4f}")
    except Exception as e: print(f"✗ TimesFM: {e}")

    # AutoGluon TimeSeries
    try:
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        ts_df = pd.DataFrame({"item_id": ["s"] * split, "timestamp": pd.date_range("2020-01-01", periods=split, freq="D"), "target": train})
        ts_data = TimeSeriesDataFrame.from_data_frame(ts_df)
        predictor = TimeSeriesPredictor(prediction_length=HORIZON, eval_metric="RMSE",
                                         path=os.path.join(os.path.dirname(__file__), "ag_ts"))
        predictor.fit(ts_data, time_limit=120, presets="best_quality")
        ag_preds = predictor.predict(ts_data)
        y_pred = ag_preds["mean"].values[:len(test)]
        rmse = mean_squared_error(test, y_pred, squared=False)
        results["AutoGluon-TS"] = y_pred
        print(f"✓ AutoGluon-TS RMSE: {rmse:.4f}")
    except Exception as e: print(f"✗ AutoGluon-TS: {e}")

    # ── Baseline: lag-feature tabular reframing with FLAML ──
    try:
        from flaml import AutoML
        lags = [1, 2, 3, 5, 7, 14, 21]
        lag_df = pd.DataFrame({"y": series})
        for lg in lags:
            lag_df[f"lag_{lg}"] = lag_df["y"].shift(lg)
        lag_df["rolling_7"] = lag_df["y"].rolling(7).mean()
        lag_df["rolling_14"] = lag_df["y"].rolling(14).mean()
        lag_df = lag_df.dropna()
        lag_train = lag_df.iloc[:split - max(lags)]
        lag_test = lag_df.iloc[split - max(lags):split - max(lags) + HORIZON]
        if len(lag_test) >= HORIZON:
            X_lag_tr = lag_train.drop(columns=["y"]); y_lag_tr = lag_train["y"]
            X_lag_te = lag_test.drop(columns=["y"]); y_lag_te = lag_test["y"]
            automl = AutoML()
            automl.fit(X_lag_tr, y_lag_tr, task="regression", time_budget=60, verbose=0)
            y_pred = automl.predict(X_lag_te)[:len(test)]
            rmse = mean_squared_error(y_lag_te.values[:len(y_pred)], y_pred, squared=False)
            results["FLAML-Lag"] = y_pred
            print(f"✓ FLAML-Lag ({automl.best_estimator}) RMSE: {rmse:.4f}")
    except Exception as e: print(f"✗ FLAML-Lag: {e}")

    # ── Baseline: LazyPredict on lag features ──
    try:
        from lazypredict.Supervised import LazyRegressor
        if "X_lag_tr" in dir() and len(X_lag_tr) > 0:
            lazy = LazyRegressor(verbose=0, ignore_warnings=True)
            lazy_models, _ = lazy.fit(X_lag_tr, X_lag_te, y_lag_tr, y_lag_te)
            print(f"\n✓ LazyPredict (lag-tabular) — Top 5 regressors:")
            print(lazy_models.head().to_string())
    except Exception as e: print(f"✗ LazyPredict-Lag: {e}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(train)), train, alpha=0.5, label="Train")
    ax.plot(range(len(train), len(train)+len(test)), test, linewidth=2, label="Actual")
    for name, y_pred in results.items():
        ax.plot(range(len(train), len(train)+len(y_pred)), y_pred, "--", label=name)
    ax.legend(); ax.set_title("Forecast Comparison")
    fig.savefig(os.path.join(os.path.dirname(__file__), "forecast.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return results


def main():
    print("=" * 60)
    print("TIME SERIES: AutoGluon + Chronos-Bolt + Chronos-2 + TimesFM + StatsForecast + FLAML/LazyPredict")
    print("=" * 60)
    df, target = load_data()
    forecast(df, target)


if __name__ == "__main__":
    main()
