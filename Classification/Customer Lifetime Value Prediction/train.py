#!/usr/bin/env python3
"""
Model training for Predicting customer lifetime value

Auto-generated from: customer-lifetime-value-prediction.ipynb
Project: Predicting customer lifetime value
Category: Classification | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
#import libraries
from __future__ import division

from datetime import datetime, timedelta,date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans


import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import xgboost as xgb
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

    #Read data
    tx_data = load_dataset('predicting_customer_lifetime_value')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #initate plotly
    pyoff.init_notebook_mode()

    #read data from csv and redo the data work we done before
    tx_data.head()

    #converting the type of Invoice Date Field from string to datetime.
    tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #creating YearMonth field for the ease of reporting and visualization
    tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(lambda date: 100*date.year + date.month)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #we will be using only UK data
    tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)

    #create a generic user dataframe to keep CustomerID and new segmentation scores
    tx_user = pd.DataFrame(tx_data['CustomerID'].unique())
    tx_user.columns = ['CustomerID']
    tx_user.head()

    #get the max purchase date for each customer and create a dataframe with it
    tx_max_purchase = tx_uk.groupby('CustomerID').InvoiceDate.max().reset_index()
    tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
    tx_max_purchase.head()

    # Compare the last transaction of the dataset with last transaction dates of the individual customer IDs.
    tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
    tx_max_purchase.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #merge this dataframe to our new user dataframe
    tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')
    tx_user.head()



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

            clf_setup = setup(data=tx_data, target='LTVCluster', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Predicting customer lifetime value")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
