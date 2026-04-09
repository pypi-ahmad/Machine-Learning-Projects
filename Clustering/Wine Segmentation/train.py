#!/usr/bin/env python3
"""
Model training for 5 Wine segmentation

Auto-generated from: 5 Wine segmentation.ipynb
Project: 5 Wine segmentation
Category: Clustering | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from sklearn.metrics import confusion_matrix
from sklearn import metrics
# Additional imports extracted from mixed cells
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    df = load_dataset('wine_segmentation')
    df.head(10)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    plt.figure(figsize=(12,8))
    sns.heatmap(df.describe()[1:].transpose(),
                annot=True,linecolor="w",
                linewidth=2,cmap=sns.color_palette("Set1"))
    plt.title("Data summary")
    plt.show()

    cor_mat= df[:].corr()
    mask = np.array(cor_mat)
    mask[np.tril_indices_from(mask)] = False
    fig=plt.gcf()
    fig.set_size_inches(30,12)
    sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

    corr=df.corr()
    corr.sort_values(by=["Customer_Segment"],ascending=False).iloc[0].sort_values(ascending=False)

    plt.rcParams['figure.figsize'] = (20, 10)
    size = [59, 71, 48]
    colors = ['mediumseagreen', 'c', 'gold']
    labels = "Group A", "Group B", "Group C"
    explode = [0, 0, 0.1]
    plt.subplot(1, 2, 1)
    plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
    #plt.title('Different Visitors', fontsize = 20)
    plt.axis('off')
    plt.legend()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X = df.drop('Customer_Segment',axis=1).values
    y = df['Customer_Segment'].values



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)



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

            clf_setup = setup(data=df, target='Customer_Segment', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for 5 Wine segmentation")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
