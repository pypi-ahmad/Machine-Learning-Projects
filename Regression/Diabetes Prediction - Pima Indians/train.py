#!/usr/bin/env python3
"""
Model training for Predicting diabetes using the prima indians diabetes dataset

Auto-generated from: diabetes_prediction.ipynb
Project: Predicting diabetes using the prima indians diabetes dataset
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load in 

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

    import os
    for dirname, _, filenames in os.walk('./archive/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # Any results you write to the current directory are saved as output.

    dataset = load_dataset('predicting_diabetes_using_the_prima_indians_diabetes_dataset')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    print(sns.distplot(dataset['Pregnancies']))

    _, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
    sns.boxplot(data=dataset['Pregnancies'], ax=axes[0]);
    sns.violinplot(data=dataset['Pregnancies'], ax=axes[1])

    sns.FacetGrid(data=dataset, hue='Outcome', height=5) \
     .map(sns.distplot, 'Pregnancies') \
     .add_legend()
    plt.title('PDF with Pregnancies')
    plt.show()

    sns.FacetGrid(data=dataset, hue='Outcome', height=5) \
     .map(plt.scatter, 'Outcome', 'Pregnancies')\
     .add_legend()
    plt.title('Sebaran Pasien Berdasarkan Pregnancies')
    plt.show()

    print(sns.distplot(dataset['Glucose']))

    _, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
    sns.boxplot(data=dataset['Glucose'], ax=axes[0]);
    sns.violinplot(data=dataset['Glucose'], ax=axes[1]);

    sns.FacetGrid(dataset, hue="Outcome", height=5) \
     .map(sns.distplot, "Glucose") \
     .add_legend()
    plt.title('PDF with Glucose')
    plt.show()

    sns.FacetGrid(dataset, hue = 'Outcome', height = 5)\
    .map(plt.scatter, 'Outcome', 'Glucose')\
    .add_legend()
    plt.title('Distribusi Pasien Berdasarkan Glucose')
    plt.show()

    sns.distplot(dataset['BloodPressure'])

    _, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
    sns.boxplot(data=dataset['BloodPressure'], ax=axes[0]);
    sns.violinplot(data=dataset['BloodPressure'], ax=axes[1]);

    sns.FacetGrid(dataset, hue="Outcome", height=5) \
     .map(sns.distplot, "BloodPressure") \
     .add_legend()
    plt.title('PDF with BloodPressure')
    plt.show()

    sns.FacetGrid(dataset, hue = 'Outcome', height = 5)\
    .map(plt.scatter, 'Outcome', 'BloodPressure')\
    .add_legend()
    plt.title('Distribusi Pasien Berdasarkan BloodPressure')
    plt.show()

    sns.distplot(dataset['SkinThickness'])

    _, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
    sns.boxplot(data=dataset['SkinThickness'], ax=axes[0]);
    sns.violinplot(data=dataset['SkinThickness'], ax=axes[1]);

    sns.FacetGrid(dataset, hue="Outcome", height = 5) \
     .map(sns.distplot, "SkinThickness") \
     .add_legend()
    plt.title('PDF with SkinThickness')
    plt.show()

    sns.FacetGrid(dataset, hue = 'Outcome', height = 5)\
    .map(plt.scatter, 'Outcome', 'SkinThickness')\
    .add_legend()
    plt.title('Distribusi Pasien Berdasarkan SkinThickness')
    plt.show()

    sns.distplot(dataset['Insulin'])

    _, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
    sns.boxplot(data=dataset['Insulin'], ax=axes[0]);
    sns.violinplot(data=dataset['Insulin'], ax=axes[1]);

    sns.FacetGrid(dataset, hue="Outcome", height=5) \
     .map(sns.distplot, "Insulin") \
     .add_legend()
    plt.title('PDF with Insulin')
    plt.show()

    sns.FacetGrid(dataset, hue = 'Outcome', height = 5)\
    .map(plt.scatter, 'Outcome', 'Insulin')\
    .add_legend()
    plt.title('Distribusi Pasien Berdasarkan Insulin')
    plt.show()

    sns.distplot(dataset['BMI'])

    _, axes = plt.subplots(1,2, sharey=True, figsize=(10,5))
    sns.barplot(data=dataset['BMI'], ax=axes[0]);
    sns.violinplot(data=dataset['BMI'], ax=axes[1]);

    sns.FacetGrid(dataset, hue='Outcome', height=5) \
     .map(sns.distplot, 'BMI') \
     .add_legend()
    plt.title('PDF with BMI')
    plt.show()

    sns.FacetGrid(dataset, hue='Outcome', height=5) \
     .map(plt.scatter, 'Outcome', 'BMI') \
     .add_legend()
    plt.title('Sebaran pasien berdasarkan BMI')
    plt.show

    sns.distplot(dataset['DiabetesPedigreeFunction'])

    _, axes = plt.subplots(1,2, sharey=True, figsize=(10,6))
    sns.boxplot(data=dataset['DiabetesPedigreeFunction'], ax=axes[0]);
    sns.violinplot(data=dataset['DiabetesPedigreeFunction'], ax=axes[1])

    sns.FacetGrid(data=dataset, hue='Outcome', height=5) \
     .map(sns.distplot, 'DiabetesPedigreeFunction') \
     .add_legend()
    plt.title('PDF with DiabetesPedigreeFunction')
    plt.show

    sns.FacetGrid(data=dataset, hue='Outcome', height=5) \
     .map(plt.scatter, 'Outcome','DiabetesPedigreeFunction') \
     .add_legend()
    plt.title('Sebaran pasien berdasarkan DiabetesPedigreeFunction')
    plt.show()

    sns.distplot(dataset['Age'])

    _, axes = plt.subplots(1,2, sharey=True, figsize=(10,6))
    sns.boxplot(data=dataset['Age'], ax=axes[0]);
    sns.violinplot(data=dataset['Age'], ax=axes[1])

    sns.FacetGrid(data=dataset, hue='Outcome', height=5) \
     .map(sns.distplot, 'Age') \
     .add_legend()
    plt.title('PDF with Age')
    plt.show

    sns.FacetGrid(data=dataset, hue='Outcome', height=5) \
     .map(plt.scatter, 'Outcome','Age') \
     .add_legend()
    plt.title('Sebaran pasien berdasarkan Age')
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    # Define features and target
    X = dataset.drop(columns=['Outcome'])
    y = dataset['Outcome']

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

            clf_setup = setup(data=dataset, target='Outcome', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Predicting diabetes using the prima indians diabetes dataset")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
