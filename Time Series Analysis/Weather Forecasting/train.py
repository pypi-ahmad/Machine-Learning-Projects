#!/usr/bin/env python3
"""
Model training for Weather Forecasting

Auto-generated from: code.ipynb
Project: Weather Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_validate
# Additional imports extracted from mixed cells
from sklearn.model_selection import train_test_split

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def feature_engineering(df):
    df = df.drop(["Date"],axis=1)
    print(df.dtypes.value_counts()) # Compte les nombre de types de variables
    return(df)
def imputation(df):
    #df = df.fillna(-999)
    df = df.dropna(axis=0)
    return df
def encodage(df):
    return df
def preprocessing(df):
    df = imputation(df)
    df = encodage(df)
    df = feature_engineering(df)

    X = df.drop(['Next_Tmax','Next_Tmin'],axis=1)
    y_max = df["Next_Tmax"]
    y_min = df["Next_Tmin"]

    print(X.shape)
    print(y_max.shape)

    return X,y_max,y_min

# ======================================================================
# TRAINING PIPELINE
# ======================================================================

def main():
    """Run the training pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    # Importing the data set
    df = load_dataset('weather_forecasting')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    pd.set_option('display.max_row',25) #Affiche au plus 25 éléments dans les résultats de pandas
    pd.set_option('display.max_column',25) #Affiche au plus 25 éléments dans les résultats de pandas
    df.head()

    for col in ["Next_Tmax","Next_Tmin"]:
        plt.figure()
        sns.displot(df[col],kind='kde')
        plt.show()
    print(df["Next_Tmax"].mean())
    print(df["Next_Tmax"].std())
    print(df["Next_Tmin"].mean())
    print(df["Next_Tmin"].std())



    # --- DATA LOADING ────────────────────────────────────────

    # Importing the data set
    df = pd.read_csv('data.csv')
    Save = df.copy()



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split
    trainset, testset = train_test_split(df, test_size=0.2, random_state=0)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train, y_min_train, y_max_train = preprocessing(trainset)
    X_test, y_min_test, y_max_test = preprocessing(testset)



    # --- MODEL TRAINING ──────────────────────────────────────

    reg_max = make_pipeline(StandardScaler(),
                        SGDRegressor(loss='squared_error', penalty='l2', max_iter=1000, tol=1e-3))
    reg_max.fit(X_train, y_max_train)

    reg_min = make_pipeline(StandardScaler(),
                        SGDRegressor(loss='squared_error', penalty='l2', max_iter=1000, tol=1e-3))
    reg_min.fit(X_train, y_min_train)

    cv_results_min = cross_validate(reg_min, X_train, y_min_train, cv=5, scoring=('r2', "neg_root_mean_squared_error"), return_train_score=True)
    cv_results_max = cross_validate(reg_max, X_train, y_max_train, cv=5, scoring=('r2', "neg_root_mean_squared_error"), return_train_score=True)

    print('Pour le Next_Tmin :')
    print('Test RMSE :' , -cv_results_min['test_neg_root_mean_squared_error'].mean())
    print('Test r2 :' , cv_results_min['test_r2'].mean())
    print("Train RMSE :" , -cv_results_min['train_neg_root_mean_squared_error'].mean())
    print("Train r2 :" , cv_results_min['train_r2'].mean())
    print("*------------------------------------------*")
    print('Pour le Next_Tmax :')
    print('Test RMSE :' , -cv_results_max['test_neg_root_mean_squared_error'].mean())
    print('Test r2 :' , cv_results_max['test_r2'].mean())
    print("Train RMSE :" , -cv_results_max['train_neg_root_mean_squared_error'].mean())
    print("Train r2 :" , cv_results_max['train_r2'].mean())


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model training for Weather Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
