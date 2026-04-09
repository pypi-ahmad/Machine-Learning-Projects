#!/usr/bin/env python3
"""
Model training for Student Alcohol Consumption Analysis

Auto-generated from: code.ipynb
Project: Student Alcohol Consumption Analysis
Category: Data Analysis | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

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

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Read the data into a DataFrame
    data = load_dataset('student_alcohol_consumption_analysis')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Gender-Based Analysis
    gender_grades = data.groupby('sex')['failures'].mean()
    print("\nAverage Grades by Gender:")
    print(gender_grades)

    cross_tab = pd.crosstab(data['sex'], data['Pstatus'])
    print(cross_tab)



    # --- MODEL TRAINING ──────────────────────────────────────

    import pandas as pd
    import statsmodels.api as sm

    # Load the dataset
    data = pd.read_csv('data.csv')

    # Convert 'failures' to binary format (0 or 1)
    data['failures'] = data['failures'].apply(lambda x: 0 if x == 0 else 1)

    # Convert categorical variables to dummy variables
    categorical_vars = ['school', 'sex', 'famsize', 'Pstatus']
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

    # Define the predictors and target variable
    X = data[['age', 'Medu', 'Fedu', 'traveltime']]
    X = sm.add_constant(X)
    y = data['failures']

    # Perform logistic regression
    model = sm.Logit(y, X)
    result = model.fit()
    print(result.summary())



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    # Define features and target
    X = data.drop(columns=['failures'])
    y = data['failures']

    # Handle non-numeric columns for modeling
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )



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

            clf_setup = setup(data=data, target='failures', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Student Alcohol Consumption Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
