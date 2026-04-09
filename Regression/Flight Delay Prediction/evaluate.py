#!/usr/bin/env python3
"""
Model evaluation for Flight Delay Prediction

Auto-generated from: predict_flight_cancelled.ipynb
Project: Flight Delay Prediction
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import missingno as msno
from sklearn.impute import SimpleImputer
import seaborn as sns
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def bar_plot(variable):
    var = df_2020[variable] # get feature
    varValue = var.value_counts() # count number of categorical variable(value/sample)
    
    plt.figure(figsize = (9,6))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{} \n {}".format(variable,varValue))

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

    # --- ADDITIONAL PROCESSING ───────────────────────────────

    import numpy as np 
    import pandas as pd 
    import matplotlib.pyplot as plt

    import os
    for dirname, _, filenames in os.walk('./archive/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))



    # --- DATA LOADING ────────────────────────────────────────

    df_2019 = load_dataset('flight_delay_prediction')
    df_2020 = pd.read_csv('../../data/flight_delay_prediction/Jan_2020_ontime.csv')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    column_names = df_2020.columns
    j=0
    for i in df_2020.columns:
        print("  {} has got {} Null Sample " .format(df_2020.columns[j],df_2020[i].isnull().sum()))
        j=j+1

    import missingno as msno
    plt.figure(figsize=(4,4))
    msno.bar(df_2020)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #Data Preprocessing
    df_2020 = df_2020.drop(['Unnamed: 21'],axis=1)
    df_2020.shape



    # --- PREPROCESSING ───────────────────────────────────────

    #Drop NaN TAIL_NUM rows
    df_2020 = df_2020.dropna(subset=['TAIL_NUM'])
    print(df_2020['TAIL_NUM'].isna().sum())
    print(df_2020.shape)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_2020['DEP_DEL15'] = df_2020['DEP_DEL15'].replace(np.NaN,0)
    df_2020['DEP_DEL15'].isnull().sum()

    df_2020['ARR_DEL15'] = df_2020['ARR_DEL15'].replace(np.NaN,0)
    df_2020['ARR_DEL15'].isnull().sum()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    from sklearn.impute import SimpleImputer
    imp_mean = SimpleImputer(missing_values=np.nan,strategy='mean')
    #DEP_TIME

    df_2020['DEP_TIME'] = imp_mean.fit_transform(df_2020[['DEP_TIME']])
    #ARR_TIME

    df_2020['ARR_TIME'] = imp_mean.fit_transform(df_2020[['ARR_TIME']])

    column_names = df_2020.columns
    j=0
    for i in df_2020.columns:
        print("  {} has got {} NaN Sample " .format(df_2020.columns[j],df_2020[i].isnull().sum()))
        j=j+1

    import seaborn as sns
    f,ax= plt.subplots(figsize=(15,15))
    sns.heatmap(df_2020.corr(),linewidths=.5,annot=True,fmt='.4f',ax=ax)
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_2020 = df_2020.drop(['DEST_AIRPORT_SEQ_ID'],axis=1)
    df_2020 = df_2020.drop(['ORIGIN_AIRPORT_SEQ_ID'],axis=1)
    print(df_2020.shape)

    y = df_2020.CANCELLED
    df_2020 = df_2020.drop('CANCELLED',axis=1)
    X = df_2020



    # --- PREPROCESSING ───────────────────────────────────────

    categorical_columns = ['OP_CARRIER','OP_UNIQUE_CARRIER','TAIL_NUM','ORIGIN','DEST','DEP_TIME_BLK']
    for col in categorical_columns:
        X_encoded = pd.get_dummies(X[col],prefix_sep = '_')
        df_2020 = df_2020.drop([col],axis=1)

    df_2020 = pd.concat([df_2020, X_encoded], axis=1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X = df_2020



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=42)



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

            clf_setup = setup(data=df_2019, target='CANCELLED', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model evaluation for Flight Delay Prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
