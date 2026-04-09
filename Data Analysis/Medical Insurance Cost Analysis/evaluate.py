#!/usr/bin/env python3
"""
Model evaluation for Medical Insurance Cost Analysis

Auto-generated from: code.ipynb
Project: Medical Insurance Cost Analysis
Category: Data Analysis | Task: data_analysis
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Additional imports extracted from mixed cells
from sklearn.preprocessing import LabelEncoder
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

    df = load_dataset('medical_insurance_cost_analysis')
    df.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax = sns.distplot(np.log10(df['charges']), kde = True, color = 'r' )



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    charges = df['charges'].groupby(df.region).sum().sort_values(ascending = True)
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax = sns.barplot(x=charges.head(), y=charges.head().index, palette='Blues')

    f, ax = plt.subplots(1,1, figsize=(12,8))
    ax = sns.barplot(x = 'region', y = 'charges',
                     hue='smoker', data=df, palette='Reds_r')

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = sns.violinplot(x = 'children', y = 'charges', data=df,
                     orient='v', hue='smoker', palette='inferno')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    ##Converting objects labels into categorical
    df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype('category')
    df.dtypes



    # --- MODEL TRAINING ──────────────────────────────────────

    ##Converting category labels into numerical using LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    label = LabelEncoder()
    label.fit(df.sex.drop_duplicates())
    df.sex = label.transform(df.sex)
    label.fit(df.smoker.drop_duplicates())
    df.smoker = label.transform(df.smoker)
    label.fit(df.region.drop_duplicates())
    df.region = label.transform(df.region)
    df.dtypes



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    # Define features and target
    X = df.drop(columns=['charges'])
    y = df['charges']

    # Handle non-numeric columns for modeling
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )



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

            reg_setup = setup(data=df, target='charges', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model evaluation for Medical Insurance Cost Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
