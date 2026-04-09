#!/usr/bin/env python3
"""
Full pipeline for Job Salary prediction

Auto-generated from: job_salary_prediction.ipynb
Project: Job Salary prediction
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Importing some tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Additional imports extracted from mixed cells
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor
from pycaret.regression import *

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

    import os

    # --- Schema Reconciliation: load substitute dataset from centralized data directory ---
    _data_dir = os.path.join(os.path.dirname(os.path.abspath("__file__")), "..", "..", "data", "job_salary_prediction")
    _csv_path = os.path.join(_data_dir, "adzuna_global_job_listings_2025.csv")
    if not os.path.exists(_csv_path):
        _csv_path = './job-salary-prediction/Train_rev1.zip'

    df_train = load_dataset('job_salary_prediction')

    # --- Schema mapping ---
    # Original: SalaryNormalized (target), SalaryRaw, Id, Title, FullDescription, etc.
    # Substitute: salary_min, salary_max, job_id, title, description, company, etc.

    # Create unified salary target from min/max
    if 'SalaryNormalized' not in df_train.columns:
        if 'salary_min' in df_train.columns and 'salary_max' in df_train.columns:
            df_train['SalaryNormalized'] = (df_train['salary_min'] + df_train['salary_max']) / 2
            df_train['SalaryRaw'] = df_train['salary_min'].astype(str) + " - " + df_train['salary_max'].astype(str)
        else:
            raise KeyError("Cannot derive SalaryNormalized: salary_min/salary_max columns not found")

    # Map column names to match original schema expectations
    _col_map = {
        'job_id': 'Id',
        'title': 'Title',
        'company': 'Company',
        'location_display': 'LocationNormalized',
        'description': 'FullDescription',
        'contract_time': 'ContractTime',
        'contract_type': 'ContractType',
        'category_label': 'Category',
    }
    df_train = df_train.rename(columns={k: v for k, v in _col_map.items() if k in df_train.columns})

    # Drop rows where salary is NaN or zero (invalid records)
    df_train = df_train.dropna(subset=['SalaryNormalized'])
    df_train = df_train[df_train['SalaryNormalized'] > 0].reset_index(drop=True)

    # Validation
    print("Columns:", df_train.columns.tolist())
    print("Shape:", df_train.shape)
    df_train.head()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df_train.describe()

    df_train.info()

    # Check missing values
    df_train.isna().sum()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Check for string label 
    for label,content in df_train.items():
        if pd.api.types.is_string_dtype(content):
            print(label)

    # Check for numerical label
    for label,content in df_train.items():
        if pd.api.types.is_numeric_dtype(content):
            print(label)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # This will turn all of the string value into category values
    for label, content in df_train.items():
        if pd.api.types.is_string_dtype(content):
            df_train[label] = content.astype("category").cat.as_ordered()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Filling missing values
    for label,content in df_train.items():
        if not pd.api.types.is_numeric_dtype(content):
            # Add binary column to indicate whether sample had missing value
            df_train[label+"is_missing"]=pd.isnull(content)
            # Turn categories into numbers and add+1
            df_train[label] = pd.Categorical(content).codes+1



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df_train.isna().sum()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    ms = df_train["SalaryNormalized"][:10].plot.barh(figsize=(15,10))



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df_train["SalaryNormalized"].hist()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # For more security,copy the train set
    df_tmp = df_train.copy()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df_tmp.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Split the data into X & y
    X = df_tmp.drop("SalaryNormalized",axis=1)
    y = df_tmp["SalaryNormalized"]



    # --- MODEL TRAINING ──────────────────────────────────────

    # # Let's build a machine learning model 
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_jobs=-1)
    model.fit(X_train,y_train)



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- LAZYPREDICT BASELINE ────────────────────────

            from lazypredict.Supervised import LazyRegressor

            lazy_reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = lazy_reg.fit(X_train, X_test, y_train, y_test)

            print(models)



    # --- PYCARET AUTOML ──────────────────────────────────────

            from pycaret.regression import *

            reg_setup = setup(data=df_train, target='SalaryNormalized', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Job Salary prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
