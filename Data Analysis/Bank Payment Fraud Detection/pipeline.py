#!/usr/bin/env python3
"""
Full pipeline for Bank Payment Fraud Detection

Auto-generated from: code.ipynb
Project: Bank Payment Fraud Detection
Category: Data Analysis | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
## Data loading, processing and for more
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

## Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# set seaborn style because it prettier
sns.set()

## Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

## Models
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
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

    data = load_dataset('bank_payment_fraud_detection')
    data.head(5)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.info()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Create two dataframes with fraud and non-fraud data
    df_fraud = data.loc[data.fraud == 1]
    df_non_fraud = data.loc[data.fraud == 0]

    sns.countplot(x="fraud",data=data)
    plt.title("Count of Fraudulent Payments")
    plt.show()
    print("Number of normal examples: ",df_non_fraud.fraud.count())
    print("Number of fradulent examples: ",df_fraud.fraud.count())
    #print(data.fraud.value_counts()) # does the same thing above



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    print("Mean feature values per category",data.groupby('category')['amount','fraud'].mean())



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Create two dataframes with fraud and non-fraud data
    pd.concat([df_fraud.groupby('category')['amount'].mean(),df_non_fraud.groupby('category')['amount'].mean(),\
               data.groupby('category')['fraud'].mean()*100],keys=["Fraudulent","Non-Fraudulent","Percent(%)"],axis=1,\
              sort=False).sort_values(by=['Non-Fraudulent'])



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # Plot histograms of the amounts in fraud and non-fraud data
    plt.figure(figsize=(30,10))
    sns.boxplot(x=data.category,y=data.amount)
    plt.title("Boxplot for the Amount spend in category")
    plt.ylim(0,4000)
    plt.legend()
    plt.show()

    # Plot histograms of the amounts in fraud and non-fraud data
    plt.hist(df_fraud.amount, alpha=0.5, label='fraud',bins=100)
    plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud',bins=100)
    plt.title("Histogram for fraudulent and nonfraudulent payments")
    plt.ylim(0,10000)
    plt.xlim(0,1000)
    plt.legend()
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    print((data.groupby('age')['fraud'].mean()*100).reset_index().rename(columns={'age':'Age','fraud' : 'Fraud Percent'}).sort_values(by='Fraud Percent'))

    print("Unique zipCodeOri values: ",data.zipcodeOri.nunique())
    print("Unique zipMerchant values: ",data.zipMerchant.nunique())
    # dropping zipcodeori and zipMerchant since they have only one unique value
    data_reduced = data.drop(['zipcodeOri','zipMerchant'],axis=1)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data_reduced.columns



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # turning object columns type to categorical for easing the transformation process
    col_categorical = data_reduced.select_dtypes(include= ['object']).columns
    for col in col_categorical:
        data_reduced[col] = data_reduced[col].astype('category')
    # categorical values ==> numeric values
    data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)
    data_reduced.head(5)

    X = data_reduced.drop(['fraud'],axis=1)
    y = data['fraud']
    print(X.head(),"\n")
    print(y.head())



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    y[y==1].count()



    # --- PREPROCESSING ───────────────────────────────────────

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    y_res = pd.DataFrame(y_res)
    print(y_res.value_counts())

    # I won't do cross validation since we have a lot of instances
    X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.3,random_state=42,shuffle=True,stratify=y_res)



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

            clf_setup = setup(data=data, target='fraud', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Bank Payment Fraud Detection")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
