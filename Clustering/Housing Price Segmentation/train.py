#!/usr/bin/env python3
"""
Model training for 6 Housing price segmentation

Auto-generated from: 6 Housing price segmentation.ipynb
Project: 6 Housing price segmentation
Category: Clustering | Task: clustering
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
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Correlation heatmap
    plt.figure(figsize=(18, 12))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.show()

    # Pairplot for selected features
    selected_features = ["Price", "number of bedrooms", "number of bathrooms", "living area", "lot area", "house_age"]
    sns.pairplot(data[selected_features])
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # 1. House prices across postal codes:

    plt.figure(figsize=(15, 8))
    sns.boxplot(x="Postal Code", y="Price", data=data)
    plt.xticks(rotation=90)
    plt.show()

    #2. Relationship between house size and other features:

    # Scatterplot matrix
    size_features = ["living area", "number of views", "number of bedrooms", "number of bathrooms", "living_area_renov", "Area of the house(excluding basement)", "grade of the house"]
    sns.pairplot(data[size_features])
    plt.show()

    #3. Age and renovation year's effect on house prices:

    # Create a new column for renovation status
    data["renovated"] = data["Renovation Year"].apply(lambda x: 1 if x > 0 else 0)

    # Scatterplot for age and price
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="house_age", y="Price", hue="renovated", data=data, alpha=0.5)
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
    _parser = _ap.ArgumentParser(description="Model training for 6 Housing price segmentation")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
