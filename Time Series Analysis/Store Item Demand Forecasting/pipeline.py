#!/usr/bin/env python3
"""
Full pipeline for Store Item Demand Forecasting

Auto-generated from: code.ipynb
Project: Store Item Demand Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import itertools

import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')
# Additional imports extracted from mixed cells
from pycaret.time_series import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546, 730])
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
df.tail()
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

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

    train = load_dataset('store_item_demand_forecasting')
    test = pd.read_csv('data/test.csv', parse_dates=['date'])
    df = pd.concat([train, test], sort=False)
    df.head()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    print("Train setinin boyutu:",train.shape)
    print("Test setinin boyutu:",test.shape)

    df.shape



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T

    df["date"].min()

    df["date"].max()

    df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df["store"].nunique()

    df["item"].nunique()

    df.groupby(["store"])["item"].nunique()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})

    df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)
    df.head()



    # --- PREPROCESSING ───────────────────────────────────────

    df = pd.get_dummies(df, columns=['day_of_week', 'month'])



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df['sales'] = np.log1p(df["sales"].values)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train = df.loc[(df["date"] < "2017-01-01"), :]

    val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

    cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

    Y_train = train['sales']

    X_train = train[cols]

    Y_val = val['sales']

    X_val = val[cols]

    Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

    # LightGBM parameters
    lgb_params = {'metric': {'mae'},
                  'num_leaves': 10,
                  'learning_rate': 0.02,
                  'feature_fraction': 0.8,
                  'max_depth': 5,
                  'verbose': 0,
                  'num_boost_round': 2000,
                  'early_stopping_rounds': 200,
                  'nthread': -1}



    # --- PYCARET AUTOML ──────────────────────────────────────

    from pycaret.time_series import *

    ts_setup = setup(data=train, target='sales', fh=12, session_id=42, verbose=False)

    # Compare models and select best
    best_model = compare_models()

    # Display comparison results
    print(best_model)

    # Plot forecast
    plot_model(best_model, plot='forecast')

    # Finalize the model
    final_model = finalize_model(best_model)

    # Make predictions
    predictions = predict_model(final_model)
    print(predictions)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Store Item Demand Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
