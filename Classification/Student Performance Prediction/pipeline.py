#!/usr/bin/env python3
"""
Full pipeline for Student performance prediction

Auto-generated from: student-performance-explained.ipynb
Project: Student performance prediction
Category: Classification | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.model_selection import GridSearchCV #for hypertuning
from sklearn.linear_model import LinearRegression,LogisticRegression, Lasso, Ridge
from lightgbm import LGBMRegressor
# Additional imports extracted from mixed cells
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

    df = load_dataset('student_performance_prediction')
    df



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.info()

    df.isna().any()

    df.describe()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Identify the categorical features
    cat_cols = [col for col in df.columns if df[col].dtype=='O']
    cat_cols

    for col in cat_cols:
        print(df[col].unique())

    # Get list of categorical columns
    cat_cols = [col for col in df.columns if df[col].dtype == 'O']

    # Loop over categorical columns
    for col in cat_cols:
        unique_vals = df[col].nunique()
        total_vals = len(df[col])
        unique_pct = unique_vals / total_vals * 100
        print(f"{col}: {unique_vals} unique values ({unique_pct:.2f} of total)")



    # --- FEATURE ENGINEERING ─────────────────────────────────

    for col in cat_cols:
        df[col] = df[col].astype('category')
    df.memory_usage(deep=True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # Creating Bar chart as the Target variable is Continuous
    df['writing score'].hist();

    plt.scatter(df['math score'],df['writing score'],marker = '*', color = 'g')
    plt.scatter(df['reading score'],df['writing score'],marker = '+', color = 'b')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    CorrelationData=df[['math score','reading score','writing score']].corr()
    CorrelationData

    final_cols = ['gender', 'race/ethnicity','parental level of education','lunch', 'test preparation course', 'math score','reading score']

    df_final = df[final_cols]
    X = df_final[final_cols]
    y = df['writing score']
    X
    y

    num_cols = ['math score', 'reading score']



    # --- PREPROCESSING ───────────────────────────────────────

    # Create a pipeline for categorical data
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    categorical_cols = ['gender',
     'race/ethnicity',
     'parental level of education',
     'lunch',
     'test preparation course']

     # Apply the pipeline to the categorical columns
    categorical_df = categorical_pipeline.fit_transform(df[categorical_cols])

    # Convert the sparse matrix to a Pandas DataFrame
    categorical_df = pd.DataFrame(categorical_df.toarray())

    # Concatenate the categorical data with the original DataFrame
    df = pd.concat([df.drop(categorical_cols, axis=1), categorical_df], axis=1)

    # define the preprocessing pipelines for numerical and categorical features
    num_cols = ['math score', 'reading score']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_cols = ['gender',
     'race/ethnicity',
     'parental level of education',
     'lunch',
     'test preparation course']

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())])



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # convert all column names to strings
    df.columns = df.columns.astype(str)
    df



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    num_pipeline = Pipeline([
        ('num_smoothening',PowerTransformer())
    ])

    # define the column transformer to preprocess both numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, categorical_cols)])



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split
    X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

    # check the shapes of the training and test data
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')
    X_train



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # define the final pipeline that includes the column transformer and a logistic regression model
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LinearRegression())])



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

            clf_setup = setup(data=df, target='G3', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Student performance prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
