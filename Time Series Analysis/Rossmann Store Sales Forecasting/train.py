#!/usr/bin/env python3
"""
Model training for Rossmann Store Sales Forecasting

Auto-generated from: code.ipynb
Project: Rossmann Store Sales Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from prophet import Prophet
# Additional imports extracted from mixed cells
from pycaret.time_series import *

# ======================================================================
# TRAINING PIPELINE
# ======================================================================

def main():
    """Run the training pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    sales_train_df = load_dataset('rossmann_store_sales_forecasting')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    sales_train_df['DayOfWeek'].unique()

    sales_train_df['Open'].unique()

    sales_train_df['Promo'].unique()

    sales_train_df['StateHoliday'].unique()

    sales_train_df['SchoolHoliday'].unique()



    # --- DATA LOADING ────────────────────────────────────────

    store_info_df = pd.read_csv('store.csv')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    sales_train_df['Customers'].max()

    closed_train_df = sales_train_df[sales_train_df['Open'] == 0]
    open_train_df = sales_train_df[sales_train_df['Open'] == 1]

    172817 / len(store_info_df)

    sales_train_df = sales_train_df[sales_train_df['Open'] == 1]

    sales_train_df



    # --- FEATURE ENGINEERING ─────────────────────────────────

    sales_train_df.drop(['Open'], axis = 1, inplace=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    store_info_df[store_info_df['Promo2'] == 0]



    # --- PREPROCESSING ───────────────────────────────────────

    str_cols = ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval',
                'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']
    for str in str_cols:
      store_info_df[str].fillna(0, inplace=True)

    store_info_df['CompetitionDistance'].fillna(store_info_df['CompetitionDistance'].mean(), inplace = True)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    sales_train_all_df = pd.merge(sales_train_df, store_info_df, how = 'inner', on = 'Store')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    correlations = sales_train_all_df.corr()['Sales'].sort_values()
    correlations

    sales_train_all_df['Year'] = pd.DatetimeIndex(sales_train_all_df['Date']).year

    sales_train_all_df['Month'] = pd.DatetimeIndex(sales_train_all_df['Date']).month
    sales_train_all_df['Day'] = pd.DatetimeIndex(sales_train_all_df['Date']).day

    axis = sales_train_all_df.groupby('Month')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
    axis.set_title('Média de vendas por mês')

    axis = sales_train_all_df.groupby('Month')[['Customers']].mean().plot(figsize = (10,5), marker = '^', color = 'b')
    axis.set_title('Média de clientes por mês')

    axis = sales_train_all_df.groupby('Day')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
    axis.set_title('Média de vendas por dia')

    axis = sales_train_all_df.groupby('Day')[['Customers']].mean().plot(figsize = (10,5), marker = '^', color = 'b')
    axis.set_title('Média de clientes por dia')

    axis = sales_train_all_df.groupby('DayOfWeek')[['Sales']].mean().plot(figsize = (10,5), marker = 'o', color = 'r')
    axis.set_title('Média de vendas por dia da semana')

    axis = sales_train_all_df.groupby('DayOfWeek')[['Customers']].mean().plot(figsize = (10,5), marker = '^', color = 'b')
    axis.set_title('Média de clientes por dia da semana')



    # --- PYCARET AUTOML ──────────────────────────────────────

    from pycaret.time_series import *

    ts_setup = setup(data=sales_train_df, target='Sales', fh=12, session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Rossmann Store Sales Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
