#!/usr/bin/env python3
"""
Full pipeline for Red Wine Quality Analysis

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

    data = load_dataset('red_wine_quality_analysis')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data.corr



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.columns

    data.info()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data['quality'].unique()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #Check correleation between the variables using Seaborn's pairplot.
    sns.pairplot(data)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #count of each target variable
    from collections import Counter
    Counter(data['quality'])



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #count of the target variable
    sns.countplot(x='quality', data=data)

    #Plot a boxplot to check for Outliers
    #Target variable is Quality. So will plot a boxplot each column against target variable
    sns.boxplot(data = data, x = 'quality', y = 'fixed acidity')

    sns.boxplot(x = 'quality', y = 'volatile acidity', data = data)

    sns.boxplot(x = 'quality', y = 'citric acid', data = data)

    sns.boxplot(x = 'quality', y = 'residual sugar', data = data)

    sns.boxplot(x = 'quality', y = 'chlorides', data = data)

    sns.boxplot(x = 'quality', y = 'free sulfur dioxide', data = data)

    sns.boxplot(x = 'quality', y = 'total sulfur dioxide', data = data)

    sns.boxplot(x = 'quality', y = 'density', data = data)

    sns.boxplot(x = 'quality', y = 'pH', data = data)

    sns.boxplot(x = 'quality', y = 'sulphates', data = data)

    sns.boxplot(x = 'quality', y = 'alcohol', data = data)

    #boxplots show many outliers for quite a few columns. Describe the dataset to get a better idea on what's happening
    data.describe()
    #fixed acidity - 25% - 7.1 and 50% - 7.9. Not much of a variance. Could explain the huge number of outliers
    #volatile acididty - similar reasoning
    #citric acid - seems to be somewhat uniformly distributed
    #residual sugar - min - 0.9, max - 15!! Waaaaay too much difference. Could explain the outliers.
    #chlorides - same as residual sugar. Min - 0.012, max - 0.611
    #free sulfur dioxide, total suflur dioxide - same explanation as above



    # --- ADDITIONAL PROCESSING ───────────────────────────────

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



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #view final data
    data.columns



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data['Reviews'].unique()

    Counter(data['Reviews'])

    x = data.iloc[:,:11]
    y = data['Reviews']



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    x.head(10)

    y.head(10)



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x = sc.fit_transform(x)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #view the scaled features
    print(x)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    from sklearn.decomposition import PCA
    pca = PCA()
    x_pca = pca.fit_transform(x)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #plot the graph to find the principal components
    plt.figure(figsize=(10,10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
    plt.grid()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #AS per the graph, we can see that 8 principal components attribute for 90% of variation in the data.
    #we shall pick the first 8 components for our prediction.
    pca_new = PCA(n_components=8)
    x_new = pca_new.fit_transform(x)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    print(x_new)



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25, random_state=42)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)



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
    _parser = _ap.ArgumentParser(description="Full pipeline for Red Wine Quality Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
