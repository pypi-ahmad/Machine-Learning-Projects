#!/usr/bin/env python3
"""
Model evaluation for Crop yield prediction

Auto-generated from: crop_yield_prediction.ipynb
Project: Crop yield prediction
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np 
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# Additional imports extracted from mixed cells
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from pycaret.regression import *

# ======================================================================
# EVALUATION PIPELINE
# ======================================================================

def main():
    """Run the evaluation pipeline."""
    USE_AUTOML = True  # Set to False to skip AutoML comparison

    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    df_yield = load_dataset('crop_yield_prediction')
    df_yield.shape



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # rename columns.
    df_yield = df_yield.rename(index=str, columns={"Value": "hg/ha_yield"})
    df_yield.head()

    # drop unwanted columns.
    df_yield = df_yield.drop(['Year Code','Element Code','Element','Year Code','Area Code','Domain Code','Domain','Unit','Item Code'], axis=1)
    df_yield.head()



    # --- DATA LOADING ────────────────────────────────────────

    df_rain = pd.read_csv('../../data/crop_yield_prediction/rainfall.csv')
    df_rain.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_rain = df_rain.rename(index=str, columns={" Area": 'Area'})



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(df_rain['average_rain_fall_mm_per_year'],errors = 'coerce')
    df_rain.info()



    # --- PREPROCESSING ───────────────────────────────────────

    df_rain = df_rain.dropna()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    yield_df = pd.merge(df_yield, df_rain, on=['Year','Area'])



    # --- DATA LOADING ────────────────────────────────────────

    df_pes = pd.read_csv('../../data/crop_yield_prediction/pesticides.csv')
    df_pes.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_pes = df_pes.rename(index=str, columns={"Value": "pesticides_tonnes"})
    df_pes = df_pes.drop(['Element','Domain','Unit','Item'], axis=1)
    df_pes.head()

    yield_df = pd.merge(yield_df, df_pes, on=['Year','Area'])
    yield_df.shape



    # --- DATA LOADING ────────────────────────────────────────

    avg_temp=  pd.read_csv('../../data/crop_yield_prediction/temp.csv')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    avg_temp = avg_temp.rename(index=str, columns={"year": "Year", "country":'Area'})
    avg_temp.head()

    yield_df = pd.merge(yield_df,avg_temp, on=['Area','Year'])
    yield_df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    yield_df.groupby('Item').count()

    yield_df.groupby(['Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)

    yield_df.groupby(['Item','Area'],sort=True)['hg/ha_yield'].sum().nlargest(10)

    correlation_data=yield_df.select_dtypes(include=[np.number]).corr()

    mask = np.zeros_like(correlation_data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.palette="vlag"

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_data, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5});



    # --- PREPROCESSING ───────────────────────────────────────

    yield_df_onehot = pd.get_dummies(yield_df, columns=['Area',"Item"], prefix = ['Country',"Item"])
    features=yield_df_onehot.loc[:, yield_df_onehot.columns != 'hg/ha_yield']
    label=yield_df['hg/ha_yield']
    features.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    features = features.drop(['Year'], axis=1)



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    features=scaler.fit_transform(features)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    features



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split
    train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.2, random_state=42)

    from sklearn.model_selection import train_test_split
    train_data, test_data, train_labels, test_labels = train_test_split(features, label, test_size=0.2, random_state=42)



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- LAZYPREDICT BASELINE ────────────────────────

            from lazypredict.Supervised import LazyRegressor

            lazy_reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = lazy_reg.fit(X_train, X_test, y_train, y_test)

            print(models)



    # --- PYCARET AUTOML ──────────────────────────────────────

            from pycaret.regression import *

            reg_setup = setup(data=df_yield, target='Value', session_id=42, verbose=False)

            # Compare models and select best
            best_model = compare_models()

            # Display comparison results
            print(best_model)

            # Evaluate the best model
            evaluate_model(best_model)

            # Finalize the model (train on full dataset)
            final_model = finalize_model(best_model)

            print('Final model:', final_model)



        except ImportError:

            print('[AutoML] LazyPredict/PyCaret not installed — skipping AutoML block')

        except Exception as _automl_err:

            print(f'[AutoML] AutoML block failed: {_automl_err}')


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model evaluation for Crop yield prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
