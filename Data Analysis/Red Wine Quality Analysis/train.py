#!/usr/bin/env python3
"""
Model training for Red Wine Quality Analysis

Auto-generated from: code.ipynb
Project: Red Wine Quality Analysis
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
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
# Additional imports extracted from mixed cells
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

    data = load_dataset('red_wine_quality_analysis')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data.corr

    data['quality'].unique()

    #count of each target variable
    from collections import Counter
    Counter(data['quality'])

    #next we shall create a new column called Review. This column will contain the values of 1,2, and 3.
    #1 - Bad
    #2 - Average
    #3 - Excellent
    #This will be split in the following way.
    #1,2,3 --> Bad
    #4,5,6,7 --> Average
    #8,9,10 --> Excellent
    #Create an empty list called Reviews
    reviews = []
    for i in data['quality']:
        if i >= 1 and i <= 3:
            reviews.append('1')
        elif i >= 4 and i <= 7:
            reviews.append('2')
        elif i >= 8 and i <= 10:
            reviews.append('3')
    data['Reviews'] = reviews

    data['Reviews'].unique()

    Counter(data['Reviews'])

    x = data.iloc[:,:11]
    y = data['Reviews']



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x = sc.fit_transform(x)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    from sklearn.decomposition import PCA
    pca = PCA()
    x_pca = pca.fit_transform(x)

    #AS per the graph, we can see that 8 principal components attribute for 90% of variation in the data.
    #we shall pick the first 8 components for our prediction.
    pca_new = PCA(n_components=8)
    x_new = pca_new.fit_transform(x)



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25, random_state=42)



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

            clf_setup = setup(data=data, target='quality', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Red Wine Quality Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
