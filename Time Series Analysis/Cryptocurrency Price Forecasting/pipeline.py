"""
Modern Time Series Forecasting Pipeline (April 2026)
Models: AutoGluon TimeSeries + Chronos-Bolt + Chronos-2 + TimesFM + GBDT lag-feature baselines
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
    df = yf.download("BTC-USD", period="5y", auto_adjust=True).reset_index()
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

    # ═══ PRIMARY: AutoGluon TimeSeries ═══
    try:
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        ts_df = pd.DataFrame({"item_id": ["s"] * split, "timestamp": pd.date_range("2020-01-01", periods=split, freq="D"), "target": train})
        ts_data = TimeSeriesDataFrame.from_data_frame(ts_df)
        predictor = TimeSeriesPredictor(prediction_length=HORIZON, eval_metric="RMSE",
                                         path=os.path.join(os.path.dirname(__file__), "ag_ts"))
        predictor.fit(ts_data, time_limit=180, presets="best_quality")
        ag_preds = predictor.predict(ts_data)
        y_pred = ag_preds["mean"].values[:len(test)]
        rmse = mean_squared_error(test, y_pred, squared=False)
        results["AutoGluon-TS"] = y_pred
        print(f"✓ AutoGluon-TS RMSE: {rmse:.4f}")
        lb = predictor.leaderboard(ts_data)
        print(f"  AutoGluon leaderboard:\n{lb.head().to_string()}")
    except Exception as e: print(f"✗ AutoGluon-TS: {e}")

    # ═══ FOUNDATION MODELS ═══

    # Chronos-Bolt (fast zero-shot)
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

    # Chronos-2 (universal forecasting)
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

    # TimesFM (Google foundation model)
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

    # ═══ TABULAR LAG-FEATURE BASELINES (GBDT) ═══
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
                iterations=500, lr=0.05, depth=6, task_type="GPU",
                devices="0", verbose=0)),
            ("XGBoost-Lag", lambda: __import__("xgboost").XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                device="cuda", tree_method="hist", verbosity=0, n_jobs=-1)),
        ]:
            try:
                m = builder()
                m.fit(X_lag_tr, y_lag_tr)
                y_pred = m.predict(X_lag_te)[:len(test)]
                rmse = mean_squared_error(y_lag_te.values[:len(y_pred)], y_pred, squared=False)
                results[name] = y_pred
                print(f"✓ {name} RMSE: {rmse:.4f}")
            except Exception as e:
                print(f"✗ {name}: {e}")

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
    print("TIME SERIES: AutoGluon-TS + Chronos-Bolt + Chronos-2 + TimesFM + LightGBM/CatBoost/XGBoost Lag")
    print("=" * 60)
    df, target = load_data()
    forecast(df, target)


if __name__ == "__main__":
    main()
