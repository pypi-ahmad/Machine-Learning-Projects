#!/usr/bin/env python3
"""
Full pipeline for Heart Failure Prediction

Auto-generated from: code.ipynb
Project: Heart Failure Prediction
Category: Data Analysis | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import svm
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from keras import callbacks
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
# Additional imports extracted from mixed cells
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    """Run the complete pipeline."""
    USE_AUTOML = True  # Set to False to skip AutoML comparison

    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    #loading data
    data_df = load_dataset('heart_failure_prediction')
    data_df.head()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data_df.info()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Evaluating the target and finding out the potential skewness in the data
    cols= ["#CD5C5C","#FF0000"]
    ax = sns.countplot(x= data_df["DEATH_EVENT"], palette= cols)
    ax.bar_label(ax.containers[0])



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data_df.describe().T

    #Doing Bivariate Analysis by examaning a corelation matrix of all the features using heatmap
    cmap = sns.diverging_palette(2, 165, s=80, l=55, n=9)
    corrmat = data_df.corr()
    plt.subplots(figsize=(20,20))
    sns.heatmap(corrmat,cmap= cmap,annot=True, square=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Evauating age distribution as per the deaths happened
    plt.figure(figsize=(15,10))
    Days_of_week=sns.countplot(x=data_df['age'],data=data_df, hue ="DEATH_EVENT",palette = cols)
    Days_of_week.set_title("Distribution Of Age", color="#774571")

    # Checking for potential outliers using the "Boxen and Swarm plots" of non binary features.
    feature = ["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium", "time"]
    for i in feature:
        plt.figure(figsize=(10,7))
        sns.swarmplot(x=data_df["DEATH_EVENT"], y=data_df[i], color="black", alpha=0.7)
        sns.boxenplot(x=data_df["DEATH_EVENT"], y=data_df[i], palette=cols)
        plt.show()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # Plotting "Kernel Density Estimation (kde plot)" of time and age features -  both of which are significant ones.
    sns.kdeplot(x=data_df["time"], y=data_df["age"], hue =data_df["DEATH_EVENT"], palette=cols)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Defining independent and dependent attributes in training and test sets
    X=data_df.drop(["DEATH_EVENT"],axis=1)
    y=data_df["DEATH_EVENT"]



    # --- PREPROCESSING ───────────────────────────────────────

    # Setting up a standard scaler for the features and analyzing it thereafter
    col_names = list(X.columns)
    s_scaler = preprocessing.StandardScaler()
    X_scaled= s_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=col_names)
    X_scaled.describe().T



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Plotting the scaled features using boxen plots
    colors =["#CD5C5C","#F08080","#FA8072","#E9967A","#FFA07A"]
    plt.figure(figsize=(20,10))
    sns.boxenplot(data = X_scaled,palette = colors)
    plt.xticks(rotation=60)
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    #spliting variables into training and test sets
    X_train, X_test, y_train,y_test = train_test_split(X_scaled,y,test_size=0.30,random_state=25)



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

            clf_setup = setup(data=data_df, target='DEATH_EVENT', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Heart Failure Prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
