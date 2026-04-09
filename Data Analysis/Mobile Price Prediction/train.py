#!/usr/bin/env python3
"""
Model training for Mobile Price Prediction

Auto-generated from: code.ipynb
Project: Mobile Price Prediction
Category: Data Analysis | Task: classification
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
from sklearn.model_selection import train_test_split
# Additional imports extracted from mixed cells
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# TRAINING PIPELINE
# ======================================================================

def main():
    """Run the training pipeline."""
    USE_AUTOML = True  # Set to False to skip AutoML comparison

    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    dataset=load_dataset('mobile_price_prediction')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    labels = ["3G-supported",'Not supported']
    values=dataset['three_g'].value_counts().values

    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%',shadow=True,startangle=90)
    plt.show()

    labels4g = ["4G-supported",'Not supported']
    values4g = dataset['four_g'].value_counts().values
    fig1, ax1 = plt.subplots()
    ax1.pie(values4g, labels=labels4g, autopct='%1.1f%%',shadow=True,startangle=90)
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X=dataset.drop('price_range',axis=1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    y=dataset['price_range']



    # --- PREPROCESSING ───────────────────────────────────────

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- LAZYPREDICT BASELINE ────────────────────────

            from lazypredict.Supervised import LazyClassifier

            lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)

            print(models)



    # --- PYCARET AUTOML ──────────────────────────────────────

            from pycaret.classification import *

            clf_setup = setup(data=dataset, target='price_range', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Mobile Price Prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
