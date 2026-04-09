#!/usr/bin/env python3
"""
Full pipeline for Smart Home_s Temperature Forecasting

Auto-generated from: code.ipynb
Project: Smart Home_s Temperature Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    """Run the complete pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    train=load_dataset('smart_home_s_temperature_forecasting')
    test=pd.read_csv('data/test.csv')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    total_data=pd.concat([train.drop(['Indoor_temperature_room'],axis=1),test],ignore_index=True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    total_data.head()

    total_data.info()

    total_data.describe()

    #No nan values in the dataset.
    total_data.isnull().sum()

    plt.figure(figsize = (18,18))
    sns.heatmap(total_data.corr(), annot = True, cmap = "RdYlGn")
    plt.show()

    train.columns

    train.hist(bins=10, figsize=(15, 10))
    plt.tight_layout()



    # --- PREPROCESSING ───────────────────────────────────────

    X=train.drop(['Indoor_temperature_room','Id','Date','Time'],axis=1)
    Y=train['Indoor_temperature_room']
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=40)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Smart Home_s Temperature Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
