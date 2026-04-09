#!/usr/bin/env python3
"""
Model training for 1 Customer segmentation for a bank

Auto-generated from: 1 Customer segmentation for a bank.ipynb
Project: 1 Customer segmentation for a bank
Category: Clustering | Task: clustering
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation
import warnings
warnings.filterwarnings("ignore")
# Additional imports extracted from mixed cells
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycaret.clustering import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def scatters(data, h=None, pal=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
    sns.scatterplot(x="Credit amount",y="Duration", hue=h, palette=pal, data=data, ax=ax1)
    sns.scatterplot(x="Age",y="Credit amount", hue=h, palette=pal, data=data, ax=ax2)
    sns.scatterplot(x="Age",y="Duration", hue=h, palette=pal, data=data, ax=ax3)
    plt.tight_layout()
def boxes(x,y,h,r=45):
    fig, ax = plt.subplots(figsize=(10,6))
    box = sns.boxplot(x=x,y=y, hue=h, data=data)
    box.set_xticklabels(box.get_xticklabels(), rotation=r)
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
def distributions(df):
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
    sns.distplot(df["Age"], ax=ax1)
    sns.distplot(df["Credit amount"], ax=ax2)
    sns.distplot(df["Duration"], ax=ax3)
    plt.tight_layout()

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

    data = load_dataset('customer_segmentation_for_a_bank')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    data.drop(data.columns[0], inplace=True, axis=1)
    print("Database has {} obserwations (customers) and {} columns (attributes).".format(data.shape[0],data.shape[1]))
    print("Missing values in each column:\n{}".format(data.isnull().sum()))
    print("Columns data types:\n{}".format(data.dtypes))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    scatters(data, h="Sex")

    import seaborn as sns
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    # Assuming you have 'data' as your dataset

    r1 = sns.jointplot(x="Credit amount", y="Duration", data=data, kind="reg", height=8)

    # Calculate Pearson correlation coefficient
    pearson_corr, _ = stats.pearsonr(data["Credit amount"], data["Duration"])

    # Annotate the plot with the Pearson correlation coefficient
    r1.ax_joint.annotate(f"Pearson Corr: {pearson_corr:.2f}", xy=(0.6, 0.9), xycoords="axes fraction")

    plt.show()

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming you have 'data' as your dataset

    sns.jointplot(x="Credit amount", y="Duration", data=data, kind="kde", space=0, color="g", height=8)
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    n_credits = data.groupby("Purpose")["Age"].count().rename("Count").reset_index()
    n_credits.sort_values(by=["Count"], ascending=False, inplace=True)

    plt.figure(figsize=(10,6))
    bar = sns.barplot(x="Purpose",y="Count",data=n_credits)
    bar.set_xticklabels(bar.get_xticklabels(), rotation=60)
    plt.ylabel("Number of granted credits")
    plt.tight_layout()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    boxes("Purpose","Credit amount","Sex")

    boxes("Purpose","Duration","Sex")

    boxes("Housing","Credit amount","Sex",r=0)

    boxes("Job","Credit amount","Sex",r=0)

    boxes("Job","Duration","Sex",r=0)

    from mpl_toolkits.mplot3d import Axes3D 
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data["Credit amount"], data["Duration"], data["Age"])
    ax.set_xlabel("Credit amount")
    ax.set_ylabel("Duration")
    ax.set_zlabel("Age")

    #Selecting columns for clusterisation with k-means
    selected_cols = ["Age","Credit amount", "Duration"]
    cluster_data = data.loc[:,selected_cols]

    distributions(cluster_data)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    cluster_log = np.log(cluster_data)
    distributions(cluster_log)



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
    _parser = _ap.ArgumentParser(description="Model training for 1 Customer segmentation for a bank")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
