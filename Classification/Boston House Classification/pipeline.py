#!/usr/bin/env python3
"""
Full pipeline for Boston House Classification

Auto-generated from: boston_house_classification.ipynb
Project: Boston House Classification
Category: Classification | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import pandas as pd
# Additional imports extracted from mixed cells
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

    housing = load_dataset('boston_house_classification')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    housing.head()

    housing.info()

    housing['CHAS'].value_counts()

    housing.describe()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # # For plotting histogram
    import matplotlib.pyplot as plt
    housing.hist(bins=50, figsize=(20, 15))

    # For learning purpose
    import numpy as np
    def split_train_test(data, test_ratio):
        np.random.seed(42)
        shuffled = np.random.permutation(len(data))
        print(shuffled)
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled[:test_set_size]
        train_indices = shuffled[test_set_size:] 
        return data.iloc[train_indices], data.iloc[test_indices]



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split
    train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
    print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['CHAS']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    strat_test_set['CHAS'].value_counts()

    strat_train_set['CHAS'].value_counts()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    95/7

    376/28

    housing = strat_train_set.copy()

    corr_matrix = housing.corr()
    corr_matrix['MEDV'].sort_values(ascending=False)

    from pandas.plotting import scatter_matrix
    attributes = ["MEDV", "RM", "ZN", "LSTAT"]
    scatter_matrix(housing[attributes], figsize = (12,8))



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    housing["TAXRM"] = housing['TAX']/housing['RM']



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    housing.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    corr_matrix = housing.corr()
    corr_matrix['MEDV'].sort_values(ascending=False)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    housing = strat_train_set.drop("MEDV", axis=1)
    housing_labels = strat_train_set["MEDV"].copy()



    # --- PREPROCESSING ───────────────────────────────────────

    a = housing.dropna(subset=["RM"]) #Option 1
    a.shape
    # Note that the original housing dataframe will remain unchanged



    # --- FEATURE ENGINEERING ─────────────────────────────────

    housing.drop("RM", axis=1).shape # Option 2
    # Note that there is no RM column and also note that the original housing dataframe will remain unchanged



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    median = housing["RM"].median() # Compute median for Option 3



    # --- PREPROCESSING ───────────────────────────────────────

    housing["RM"].fillna(median) # Option 3
    # Note that the original housing dataframe will remain unchanged



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    housing.shape

    housing.describe() # before we started filling missing attributes



    # --- MODEL TRAINING ──────────────────────────────────────

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    imputer.statistics_

    X = imputer.transform(housing)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    housing_tr = pd.DataFrame(X, columns=housing.columns)

    housing_tr.describe()



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    my_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        #     ..... add as many as you want in your pipeline
        ('std_scaler', StandardScaler()),
    ])



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    housing_num_tr = my_pipeline.fit_transform(housing)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    housing_num_tr.shape



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

            reg_setup = setup(data=housing, target='MEDV', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Boston House Classification")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
