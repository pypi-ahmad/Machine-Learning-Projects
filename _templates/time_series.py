"""Time Series Forecasting template: AutoGluon TimeSeries + Chronos-Bolt — April 2026"""
import textwrap


def generate(project_path, config):
    target = config.get("target", "value")
    date_col = config.get("date_col", "date")

    return textwrap.dedent(f'''\
        """
        Modern Time Series Forecasting Pipeline (April 2026)
        Models: AutoGluon-TimeSeries, Chronos-Bolt, StatsForecast (ETS/Theta)
        """
        import os, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        warnings.filterwarnings("ignore")

        TARGET = "{target}"
        DATE_COL = "{date_col}"
        FORECAST_HORIZON = 30


        def load_data():
            data_dir = Path(os.path.dirname(__file__))
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                raise FileNotFoundError("No CSV data found.")

            # Auto-detect date column
            date_col = DATE_COL
            if date_col not in df.columns:
                for c in df.columns:
                    if "date" in c.lower() or "time" in c.lower() or "period" in c.lower():
                        date_col = c
                        break
                else:
                    # Try parsing first column as date
                    try:
                        pd.to_datetime(df.iloc[:, 0])
                        date_col = df.columns[0]
                    except Exception:
                        date_col = None

            # Auto-detect target column
            target = TARGET
            if target not in df.columns:
                num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                for c in num_cols:
                    if any(kw in c.lower() for kw in ["close", "price", "value", "sales",
                                                       "count", "amount", "demand", "total"]):
                        target = c
                        break
                else:
                    target = num_cols[-1] if num_cols else df.columns[-1]

            if date_col and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).sort_values(date_col)
                df = df.set_index(date_col)

            print(f"Dataset shape: {{df.shape}}")
            print(f"Target: {{target}}, Date: {{date_col}}")
            print(f"Date range: {{df.index.min()}} → {{df.index.max()}}")
            return df, target


        def forecast(df, target):
            results = {{}}
            series = df[target].dropna().values.astype(float)
            n = len(series)
            split = n - FORECAST_HORIZON
            train, test = series[:split], series[split:]

            # ── 1. AutoGluon TimeSeries ──
            try:
                from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

                ag_df = pd.DataFrame({{
                    "item_id": ["series"] * n,
                    "timestamp": df.index[:n] if hasattr(df.index, 'freq') else pd.date_range("2020-01-01", periods=n, freq="D"),
                    "target": series,
                }})
                ts_df = TimeSeriesDataFrame.from_data_frame(ag_df)

                predictor = TimeSeriesPredictor(
                    prediction_length=FORECAST_HORIZON,
                    eval_metric="MASE",
                    path=os.path.join(os.path.dirname(__file__), "ag_ts_model"),
                )
                predictor.fit(ts_df, time_limit=120, presets="medium_quality")
                preds = predictor.predict(ts_df)
                y_pred = preds["mean"].values[-FORECAST_HORIZON:]

                rmse = mean_squared_error(test, y_pred, squared=False)
                results["AutoGluon"] = {{"preds": y_pred, "rmse": rmse}}
                print(f"✓ AutoGluon RMSE: {{rmse:.4f}}")
            except Exception as e:
                print(f"✗ AutoGluon: {{e}}")

            # ── 2. Chronos-Bolt ──
            try:
                import torch
                from chronos import ChronosPipeline

                pipeline = ChronosPipeline.from_pretrained(
                    "amazon/chronos-bolt-base",
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.float32,
                )
                context = torch.tensor(train, dtype=torch.float32)
                forecast_out = pipeline.predict(context, FORECAST_HORIZON)
                y_pred = np.median(forecast_out[0].numpy(), axis=0)

                rmse = mean_squared_error(test, y_pred[:len(test)], squared=False)
                results["Chronos-Bolt"] = {{"preds": y_pred[:len(test)], "rmse": rmse}}
                print(f"✓ Chronos-Bolt RMSE: {{rmse:.4f}}")
            except Exception as e:
                print(f"✗ Chronos-Bolt: {{e}}")

            # ── 3. StatsForecast (ETS + Theta) ──
            try:
                from statsforecast import StatsForecast
                from statsforecast.models import AutoETS, AutoTheta, AutoARIMA

                sf_df = pd.DataFrame({{
                    "unique_id": ["series"] * split,
                    "ds": pd.date_range("2020-01-01", periods=split, freq="D"),
                    "y": train,
                }})
                sf = StatsForecast(
                    models=[AutoETS(season_length=7), AutoTheta(season_length=7), AutoARIMA(season_length=7)],
                    freq="D", n_jobs=-1,
                )
                sf_preds = sf.forecast(h=FORECAST_HORIZON, df=sf_df)
                for col in sf_preds.columns:
                    if col not in ("unique_id", "ds"):
                        y_pred = sf_preds[col].values
                        rmse = mean_squared_error(test, y_pred[:len(test)], squared=False)
                        results[col] = {{"preds": y_pred[:len(test)], "rmse": rmse}}
                        print(f"✓ {{col}} RMSE: {{rmse:.4f}}")
            except Exception as e:
                print(f"✗ StatsForecast: {{e}}")

            return results, train, test


        def report(results, train, test, save_dir="."):
            print("\\n" + "=" * 60)
            print("TIME SERIES FORECAST COMPARISON")
            print("=" * 60)

            best_name, best_rmse = None, float("inf")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(range(len(train)), train, label="Train", alpha=0.5)
            ax.plot(range(len(train), len(train) + len(test)), test, label="Actual", linewidth=2)

            for name, res in results.items():
                y_pred = res["preds"]
                rmse = res["rmse"]
                mae = mean_absolute_error(test, y_pred)
                print(f"\\n── {{name}} ──")
                print(f"  RMSE: {{rmse:.4f}}  |  MAE: {{mae:.4f}}")

                ax.plot(range(len(train), len(train) + len(y_pred)), y_pred,
                        label=f"{{name}} (RMSE={{rmse:.2f}})", linestyle="--")

                if rmse < best_rmse:
                    best_rmse, best_name = rmse, name

            ax.legend()
            ax.set_title("Forecast Comparison")
            fig.savefig(os.path.join(save_dir, "forecast_comparison.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

            print(f"\\n🏆 Best: {{best_name}} (RMSE: {{best_rmse:.4f}})")


        def main():
            print("=" * 60)
            print("MODERN TIME SERIES PIPELINE")
            print("AutoGluon | Chronos-Bolt | StatsForecast (ETS/Theta/ARIMA)")
            print("=" * 60)
            df, target = load_data()
            results, train, test = forecast(df, target)
            if results:
                report(results, train, test, os.path.dirname(os.path.abspath(__file__)))


        if __name__ == "__main__":
            main()
    ''')
