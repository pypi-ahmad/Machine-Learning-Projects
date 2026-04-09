#!/usr/bin/env python3
"""
Full pipeline for Indians Diabetes Prediction

Auto-generated from: code.ipynb
Project: Indians Diabetes Prediction
Category: Data Analysis | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
#plt.style.use('ggplot')
#ggplot is R based visualisation package that provides better graphics with higher level of abstraction
# Additional imports extracted from mixed cells
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

    #Loading the dataset
    diabetes_data = load_dataset('indians_diabetes_prediction')

    #Print the first 5 rows of the dataframe.
    diabetes_data.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    ## gives information about the data types,columns, null value counts, memory usage etc
    ## function reference : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
    diabetes_data.info(verbose=True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    ## basic statistic details about the data (note only numerical columns would be displayed here unless parameter include="all")
    ## for reference: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html#pandas.DataFrame.describe
    diabetes_data.describe()

    ## Also see :
    ##to return columns of a specific dtype: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.select_dtypes.html#pandas.DataFrame.select_dtypes

    diabetes_data.describe().T



    # --- FEATURE ENGINEERING ─────────────────────────────────

    diabetes_data_copy = diabetes_data.copy(deep = True)
    diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

    ## showing the count of Nans
    print(diabetes_data_copy.isnull().sum())



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    p = diabetes_data.hist(figsize = (20,20))



    # --- PREPROCESSING ───────────────────────────────────────

    diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
    diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
    diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
    diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
    diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    p = diabetes_data_copy.hist(figsize = (20,20))

    ## observing the shape of the data
    diabetes_data.shape

    ## data type analysis
    #plt.figure(figsize=(5,5))
    #sns.set(font_scale=2)
    sns.countplot(y=diabetes_data.dtypes ,data=diabetes_data)
    plt.xlabel("count of each data type")
    plt.ylabel("data types")
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    ## null count analysis
    import missingno as msno
    p=msno.bar(diabetes_data)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    ## checking the balance of the data by plotting the count of outcomes by their value
    color_wheel = {1: "#0392cf",
                   2: "#7bc043"}
    colors = diabetes_data["Outcome"].map(lambda x: color_wheel.get(x + 1))
    print(diabetes_data.Outcome.value_counts())
    p=diabetes_data.Outcome.value_counts().plot(kind="bar")



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    from pandas.plotting import scatter_matrix
    p=scatter_matrix(diabetes_data,figsize=(25, 25))



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    p=sns.pairplot(diabetes_data_copy, hue = 'Outcome')

    plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
    p=sns.heatmap(diabetes_data.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap

    plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
    p=sns.heatmap(diabetes_data_copy.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),
            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age'])



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    X.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #X = diabetes_data.drop("Outcome",axis = 1)
    y = diabetes_data_copy.Outcome



    # --- PREPROCESSING ───────────────────────────────────────

    #importing train_test_split
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)



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

            clf_setup = setup(data=diabetes_data, target='Outcome', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Indians Diabetes Prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
