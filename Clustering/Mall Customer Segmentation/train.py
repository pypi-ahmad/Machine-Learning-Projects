#!/usr/bin/env python3
"""
Model training for 4 Mall customer segmentation

Auto-generated from: 4 Mall customer segmentation.ipynb
Project: 4 Mall customer segmentation
Category: Clustering | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
plt.style.use('fivethirtyeight')
# Additional imports extracted from mixed cells
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
import os
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

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt 
    import seaborn as sns 
    import plotly as py
    import plotly.graph_objs as go
    from sklearn.cluster import KMeans
    import warnings
    import os
    warnings.filterwarnings("ignore")
    py.offline.init_notebook_mode(connected = True)
    #print(os.listdir("../input"))

    df = load_dataset('mall_customer_segmentation')
    df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    plt.figure(1 , figsize = (15 , 6))
    n = 0 
    for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(1 , 3 , n)
        plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
        sns.distplot(df[x] , bins = 20)
        plt.title('Distplot of {}'.format(x))
    plt.show()

    plt.figure(1 , figsize = (15 , 7))
    n = 0 
    for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
            n += 1
            plt.subplot(3 , 3 , n)
            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
            sns.regplot(x = x , y = y , data = df)
            plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
    plt.show()

    plt.figure(1 , figsize = (15 , 6))
    for gender in ['Male' , 'Female']:
        plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = df[df['Gender'] == gender] ,
                    s = 200 , alpha = 0.5 , label = gender)
    plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 
    plt.title('Age vs Annual Income w.r.t Gender')
    plt.legend()
    plt.show()

    plt.figure(1 , figsize = (15 , 6))
    for gender in ['Male' , 'Female']:
        plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
                    data = df[df['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)
    plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 
    plt.title('Annual Income vs Spending Score w.r.t Gender')
    plt.legend()
    plt.show()

    plt.figure(1 , figsize = (15 , 7))
    n = 0 
    for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1 
        plt.subplot(1 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.violinplot(x = cols , y = 'Gender' , data = df , palette = 'vlag')
        sns.swarmplot(x = cols , y = 'Gender' , data = df)
        plt.ylabel('Gender' if n == 1 else '')
        plt.title('Boxplots & Swarmplots' if n == 2 else '')
    plt.show()



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- PYCARET AUTOML ──────────────────────────────

            from pycaret.clustering import *

            clust_setup = setup(data=df, normalize=True, session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for 4 Mall customer segmentation")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
