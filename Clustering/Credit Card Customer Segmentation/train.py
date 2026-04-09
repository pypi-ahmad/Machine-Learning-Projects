#!/usr/bin/env python3
"""
Model training for 2 Credit Card customer segmentation

Auto-generated from: 2 Credit Card customer segmentation.ipynb
Project: 2 Credit Card customer segmentation
Category: Clustering | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings(action="ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Additional imports extracted from mixed cells
from pycaret.clustering import *

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

    data= load_dataset('credit_card_customer_segmentation')
    print(data.shape)
    data.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    columns=['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
            'PAYMENTS', 'MINIMUM_PAYMENTS']

    for c in columns:
    
        Range=c+'_RANGE'
        data[Range]=0        
        data.loc[((data[c]>0)&(data[c]<=500)),Range]=1
        data.loc[((data[c]>500)&(data[c]<=1000)),Range]=2
        data.loc[((data[c]>1000)&(data[c]<=3000)),Range]=3
        data.loc[((data[c]>3000)&(data[c]<=5000)),Range]=4
        data.loc[((data[c]>5000)&(data[c]<=10000)),Range]=5
        data.loc[((data[c]>10000)),Range]=6

    columns=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 
             'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']

    for c in columns:
    
        Range=c+'_RANGE'
        data[Range]=0
        data.loc[((data[c]>0)&(data[c]<=0.1)),Range]=1
        data.loc[((data[c]>0.1)&(data[c]<=0.2)),Range]=2
        data.loc[((data[c]>0.2)&(data[c]<=0.3)),Range]=3
        data.loc[((data[c]>0.3)&(data[c]<=0.4)),Range]=4
        data.loc[((data[c]>0.4)&(data[c]<=0.5)),Range]=5
        data.loc[((data[c]>0.5)&(data[c]<=0.6)),Range]=6
        data.loc[((data[c]>0.6)&(data[c]<=0.7)),Range]=7
        data.loc[((data[c]>0.7)&(data[c]<=0.8)),Range]=8
        data.loc[((data[c]>0.8)&(data[c]<=0.9)),Range]=9
        data.loc[((data[c]>0.9)&(data[c]<=1.0)),Range]=10

    columns=['PURCHASES_TRX', 'CASH_ADVANCE_TRX']  

    for c in columns:
    
        Range=c+'_RANGE'
        data[Range]=0
        data.loc[((data[c]>0)&(data[c]<=5)),Range]=1
        data.loc[((data[c]>5)&(data[c]<=10)),Range]=2
        data.loc[((data[c]>10)&(data[c]<=15)),Range]=3
        data.loc[((data[c]>15)&(data[c]<=20)),Range]=4
        data.loc[((data[c]>20)&(data[c]<=30)),Range]=5
        data.loc[((data[c]>30)&(data[c]<=50)),Range]=6
        data.loc[((data[c]>50)&(data[c]<=100)),Range]=7
        data.loc[((data[c]>100)),Range]=8



    # --- FEATURE ENGINEERING ─────────────────────────────────

    data.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
           'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
           'PURCHASES_FREQUENCY',  'ONEOFF_PURCHASES_FREQUENCY',
           'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
           'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
           'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT' ], axis=1, inplace=True)

    X= np.asarray(data)



    # --- PREPROCESSING ───────────────────────────────────────

    scale = StandardScaler()
    X = scale.fit_transform(X)
    X.shape



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- PYCARET AUTOML ──────────────────────────────

            from pycaret.clustering import *

            clust_setup = setup(data=data, normalize=True, session_id=42, verbose=False)

            # Create K-Means model
            kmeans_model = create_model('kmeans')
            print(kmeans_model)

            # Assign cluster labels to data
            clustered_df = assign_model(kmeans_model)
            clustered_df.head()

            # Evaluate clustering
            plot_model(kmeans_model, plot='elbow')

            # Silhouette plot
            plot_model(kmeans_model, plot='silhouette')

            # Distribution plot
            plot_model(kmeans_model, plot='distribution')



        except ImportError:

            print('[AutoML] LazyPredict/PyCaret not installed — skipping AutoML block')

        except Exception as _automl_err:

            print(f'[AutoML] AutoML block failed: {_automl_err}')


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model training for 2 Credit Card customer segmentation")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
