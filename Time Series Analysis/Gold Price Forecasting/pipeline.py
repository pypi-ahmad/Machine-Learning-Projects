#!/usr/bin/env python3
"""
Full pipeline for Gold Price Forecasting

Auto-generated from: code.ipynb
Project: Gold Price Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    """Run the complete pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    import pandas as pd
    import numpy as np

    df = load_dataset('gold_price_forecasting')
    df.head(10)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.columns



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df['Date'] = pd.to_datetime(df['Date'])

    all_corr = df.corr().abs()['Adj Close'].sort_values(ascending = False)
    all_corr

    corr_drop = all_corr[all_corr < 0.35]
    corr_drop



    # --- FEATURE ENGINEERING ─────────────────────────────────

    to_drop = list(corr_drop.index)
    df2 = df.drop(to_drop, axis = 1)
    df2.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df2 = df2.set_index("Date")
    df2

    import matplotlib.pyplot as plt

    titles = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SP_open', 'SP_high', 'SP_low', 'SP_close',
              'SP_Ajclose','SP_volume','DJ_open', 'DJ_high' ]
    feature_keys = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'SP_open', 'SP_high', 'SP_low', 'SP_close',
                    'SP_Ajclose', 'SP_volume','DJ_open', 'DJ_high']

    colors = [ "blue","orange","green","red","purple","brown","pink","gray","olive", "cyan"]

    date_time_key = "Date"

    def show_raw_visualization(data):
        time_data = data[date_time_key]
        fig, axes = plt.subplots(
            nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
        )
        for i in range(len(feature_keys)):
            key = feature_keys[i]
            c = colors[i % (len(colors))]
            t_data = data[key]
            t_data.index = time_data
            t_data.head()
            ax = t_data.plot(
                ax=axes[i // 2, i % 2],
                color=c,
                title="{} - {}".format(titles[i], key),
                rot=25,
            )
            ax.legend([titles[i]])
        plt.tight_layout()


    show_raw_visualization(df)

    titles = ['DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume',
           'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume',
           'EU_Price', 'EU_open', 'EU_high', 'EU_low']
    feature_keys = ['DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume',
           'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume',
           'EU_Price', 'EU_open', 'EU_high', 'EU_low']
    show_raw_visualization(df)

    titles = ['EU_Trend', 'OF_Price',
           'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend', 'OS_Price',
           'OS_Open', 'OS_High', 'OS_Low', 'OS_Trend', 'SF_Price', 'SF_Open']
    feature_keys = ['EU_Trend', 'OF_Price',
           'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend', 'OS_Price',
           'OS_Open', 'OS_High', 'OS_Low', 'OS_Trend', 'SF_Price', 'SF_Open']
    show_raw_visualization(df)

    titles = ['SF_High', 'SF_Low', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Open',
           'USB_High', 'USB_Low', 'USB_Trend', 'PLT_Price', 'PLT_Open', 'PLT_High',
           'PLT_Low', 'PLT_Trend']
    feature_keys = ['SF_High', 'SF_Low', 'SF_Volume', 'SF_Trend', 'USB_Price', 'USB_Open',
           'USB_High', 'USB_Low', 'USB_Trend', 'PLT_Price', 'PLT_Open', 'PLT_High',
           'PLT_Low', 'PLT_Trend']
    show_raw_visualization(df)

    titles = ['RHO_PRICE', 'USDI_Price', 'USDI_Open', 'USDI_High',
           'USDI_Low', 'USDI_Volume', 'USDI_Trend', 'GDX_Open', 'GDX_High',
           'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume', 'USO_Open',
           ]
    feature_keys = ['RHO_PRICE', 'USDI_Price', 'USDI_Open', 'USDI_High',
           'USDI_Low', 'USDI_Volume', 'USDI_Trend', 'GDX_Open', 'GDX_High',
           'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume', 'USO_Open',
           ]
    show_raw_visualization(df)

    titles = ['USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume']
    feature_keys = ['USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume']

    def show_raw_visualization_small(data):
        time_data = data[date_time_key]
        fig, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
        )
        for i in range(len(feature_keys)):
            key = feature_keys[i]
            c = colors[i % (len(colors))]
            t_data = data[key]
            t_data.index = time_data
            t_data.head()
            ax = t_data.plot(
                ax=axes[i // 2, i % 2],
                color=c,
                title="{} - {}".format(titles[i], key),
                rot=25,
            )
            ax.legend([titles[i]])
        plt.tight_layout()


    show_raw_visualization_small(df)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    from sklearn.model_selection import TimeSeriesSplit

    tss = TimeSeriesSplit(n_splits = 6)
    X = df2.drop(['Adj Close'], axis = 1)
    y = df2['Adj Close']



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for train_index, test_index in tss.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Gold Price Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
