"""
Modern Time Series Forecasting Pipeline (April 2026)
Models: Chronos-Bolt, StatsForecast (ETS/Theta/ARIMA)
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TARGET = "sales"
HORIZON = 30


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("thedevastator/store-item-demand-forecasting", split="train").to_pandas()
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
    print("TIME SERIES: Chronos-Bolt + StatsForecast")
    print("=" * 60)
    df, target = load_data()
    forecast(df, target)


if __name__ == "__main__":
    main()
