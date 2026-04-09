#!/usr/bin/env python3
"""
Model evaluation for Spotify Health Clustering

Auto-generated from: code.ipynb
Project: Spotify Health Clustering
Category: Data Analysis | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Additional imports extracted from mixed cells
import pandas as pd
import os
import matplotlib.pyplot as plt
from pycaret.clustering import *

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

    import pandas as pd
    import os

    # --- Schema Reconciliation: load from centralized data directory ---
    _data_dir = os.path.join(os.path.dirname(os.path.abspath("__file__")), "..", "..", "data", "spotify_health_clustering")
    _csv_path = os.path.join(_data_dir, "train.csv") if os.path.exists(os.path.join(_data_dir, "train.csv")) else "data.csv"
    df = load_dataset('spotify_health_clustering')

    # Column mapping: substitute dataset uses different column names
    _col_map = {"artists": "artist_name", "track_genre": "genre"}
    df = df.rename(columns={k: v for k, v in _col_map.items() if k in df.columns})
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Validation
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    average_popularity = df['popularity'].mean()
    print("Average popularity:", average_popularity)

    max_popularity = df['popularity'].max()
    max_popularity_track = df[df['popularity'] == max_popularity][['track_name', 'artist_name']]
    print("Max popularity track:")
    print(max_popularity_track)

    import matplotlib.pyplot as plt

    plt.hist(df['popularity'], bins=20)
    plt.xlabel('Popularity')
    plt.ylabel('Count')
    plt.title('Distribution of Popularity')
    plt.show()

    average_danceability = df['danceability'].mean()
    average_energy = df['energy'].mean()
    print("Average danceability:", average_danceability)
    print("Average energy:", average_energy)

    genre_danceability = df.groupby('genre')['danceability'].mean().sort_values(ascending=False)

    plt.figure(figsize=(15, 6))
    plt.bar(genre_danceability.index, genre_danceability.values)
    plt.xlabel('Genre')
    plt.ylabel('Average Danceability')
    plt.title('Top Danceability by Genre')
    plt.xticks(rotation=90)
    plt.show()

    genre_danceability = df.groupby('genre')['energy'].mean().sort_values(ascending=False)

    plt.figure(figsize=(15, 6))
    plt.bar(genre_danceability.index, genre_danceability.values)
    plt.xlabel('Genre')
    plt.ylabel('Average energy')
    plt.title('Top energy by Genre')
    plt.xticks(rotation=90)
    plt.show()



    # --- DATA LOADING ────────────────────────────────────────

    # --- Schema Reconciliation: load from centralized data directory ---
    _data_dir = os.path.join(os.path.dirname(os.path.abspath("__file__")), "..", "..", "data", "spotify_health_clustering")
    _csv_path = os.path.join(_data_dir, "train.csv") if os.path.exists(os.path.join(_data_dir, "train.csv")) else "data.csv"
    data = pd.read_csv(_csv_path)

    # Column mapping: substitute dataset uses different column names
    _col_map = {"artists": "artist_name", "track_genre": "genre"}
    data = data.rename(columns={k: v for k, v in _col_map.items() if k in data.columns})
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    print("Columns:", data.columns.tolist())
    print("Shape:", data.shape)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    features = data[['danceability', 'energy', 'key']]



    # --- PREPROCESSING ───────────────────────────────────────

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)



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
    _parser = _ap.ArgumentParser(description="Model evaluation for Spotify Health Clustering")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
