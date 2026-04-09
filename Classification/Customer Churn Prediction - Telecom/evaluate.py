#!/usr/bin/env python3
"""
Model evaluation for Predicting customer churn for a telecom company

Auto-generated from: customer-churn-prediction-on-telecom-dataset.ipynb
Project: Predicting customer churn for a telecom company
Category: Classification | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_profiling.profile_report as report
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
from plotly import offline
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score 
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer, recall_score, log_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_roc_curve
import xgboost
import shap
from lime.lime_tabular import LimeTabularExplainer
import eli5
from eli5.sklearn import PermutationImportance
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
# Additional imports extracted from mixed cells
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def encode_data(dataframe):
    if dataframe.dtype == "object":
        dataframe = LabelEncoder().fit_transform(dataframe)
    return dataframe

data = df.apply(lambda x: encode_data(x))
data.head()

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

    df = load_dataset('predicting_customer_churn_for_a_telecom_company')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df.columns=df.columns.str.replace(" ","").str.lower()



    # --- PREPROCESSING ───────────────────────────────────────

    df.avgmonthlylongdistancecharges=df.avgmonthlylongdistancecharges.fillna(0.0)

    df.multiplelines=df.multiplelines.fillna('no phone service')

    no_internet=['internettype','onlinesecurity','onlinebackup','deviceprotectionplan','premiumtechsupport','streamingtv',
                 'streamingmovies','streamingmusic','unlimiteddata']
    df[no_internet]=df[no_internet].fillna('no internet service')

    df.avgmonthlygbdownload=df.avgmonthlygbdownload.fillna(0)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df=df.drop(columns=['customerid','churncategory','churnreason','totalrefunds','zipcode','longitude','latitude','city'])



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df=df.loc[~df.customerstatus.str.contains('Join')]
    df.reset_index(drop=True,inplace=True)

    type_ = ["No", "yes"]
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Pie(labels=type_, values=df['customerstatus'].value_counts(), name="customerstatus"))

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)

    fig.update_layout(
        title_text="Churn Distributions",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Churn', x=0.5, y=0.5, font_size=20, showarrow=False)])
    fig.show()

    df.customerstatus[df.customerstatus == 'Stayed'].groupby(by = df.gender).count()

    df.customerstatus[df.customerstatus == 'Churned'].groupby(by = df.gender).count()

    fig = px.histogram(df, x="customerstatus", color = "contract", barmode = "group", title = "<b>Customer contract distribution<b>")
    fig.update_layout(width=700, height=500, bargap=0.2)
    fig.show()



    # --- MODEL TRAINING ──────────────────────────────────────

    #Create a label encoder object
    le = LabelEncoder()
    # Label Encoding will be used for columns with 2 or less unique 

    le_count = 0
    for col in df.columns[1:]:
        if df[col].dtype == 'object':
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                le_count += 1
    print('{} columns were label encoded.'.format(le_count))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df['gender'] = [1 if each == 'Female' else 0 for each in df['gender']]

    data.to_csv('mycsvfile.csv',index=False)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X = data.drop(columns = "customerstatus")
    y = data["customerstatus"].values



    # --- PREPROCESSING ───────────────────────────────────────

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4, stratify =y)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    col=['totalcharges','avgmonthlylongdistancecharges','monthlycharge','totalrevenue','totallongdistancecharges',
         'tenureinmonths','totallongdistancecharges','totalextradatacharges']



    # --- PREPROCESSING ───────────────────────────────────────

    scaler = StandardScaler()
    X_train[col] = StandardScaler().fit_transform(X_train[col])
    X_test[col] = StandardScaler().fit_transform(X_test[col])



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

            clf_setup = setup(data=df, target='customerstatus', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model evaluation for Predicting customer churn for a telecom company")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
