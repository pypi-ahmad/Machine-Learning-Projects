#!/usr/bin/env python3
"""
Model training for Airbnb Data Analysis

Auto-generated from: code.ipynb
Project: Airbnb Data Analysis
Category: Data Analysis | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
# Additional imports extracted from mixed cells
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
#Encode the input Variables
def Encode(airbnb):
    for column in airbnb.columns[airbnb.columns.isin(['neighbourhood_group', 'room_type'])]:
        airbnb[column] = airbnb[column].factorize()[0]
    return airbnb

airbnb_en = Encode(airbnb.copy())

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

    airbnb=load_dataset('airbnb_data_analysis')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    airbnb.duplicated().sum()
    airbnb.drop_duplicates(inplace=True)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    airbnb.drop(['name','id','host_name','last_review'], axis=1, inplace=True)



    # --- PREPROCESSING ───────────────────────────────────────

    airbnb.fillna({'reviews_per_month':0}, inplace=True)
    #examing changes
    airbnb.reviews_per_month.isnull().sum()

    airbnb.isnull().sum()
    airbnb.dropna(how='any',inplace=True)
    airbnb.info() #.info() function is used to get a concise summary of the dataframe



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    corr = airbnb.corr(method='kendall')
    plt.figure(figsize=(15,8))
    sns.heatmap(corr, annot=True)
    airbnb.columns

    airbnb['neighbourhood_group'].unique()

    sns.countplot(x = airbnb['neighbourhood_group'], palette="plasma")
    fig = plt.gcf()
    fig.set_size_inches(10,10)
    plt.title('Neighbourhood Group')

    sns.countplot(x = airbnb['neighbourhood'], palette="plasma")
    fig = plt.gcf()
    fig.set_size_inches(25,6)
    plt.title('Neighbourhood')

    #Restaurants delivering Online or not
    sns.countplot(x = airbnb['room_type'], palette="plasma")
    fig = plt.gcf()
    fig.set_size_inches(10,10)
    plt.title('Restaurants delivering online or Not')

    plt.subplots(figsize=(25,15))
    wordcloud = WordCloud(
                              background_color='white',
                              width=1920,
                              height=1080
                             ).generate(" ".join(airbnb.neighbourhood))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig('neighbourhood.png')
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    airbnb.drop(['host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis=1, inplace=True)
    #examing the changes
    airbnb.head(5)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Get Correlation between different variables
    corr = airbnb_en.corr(method='kendall')
    plt.figure(figsize=(18,12))
    sns.heatmap(corr, annot=True)
    airbnb_en.columns
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    #Defining the independent variables and dependent variables
    x = airbnb_en.iloc[:,[0,1,3,4,5]]
    y = airbnb_en['price']
    #Getting Test and Training Set
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
    x_train.head()
    y_train.head()



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

            clf_setup = setup(data=airbnb, target='price', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Airbnb Data Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
