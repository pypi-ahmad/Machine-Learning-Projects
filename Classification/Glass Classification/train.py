#!/usr/bin/env python3
"""
Model training for Glass Classification

Auto-generated from: Glass_classification.ipynb
Project: Glass Classification
Category: Classification | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np  # linear algebra
import pandas as pd  # read and wrangle dataframes
import matplotlib.pyplot as plt # visualization
import seaborn as sns # statistical visualizations and aesthetics
from sklearn.base import TransformerMixin # To create new classes for transformations
from sklearn.preprocessing import (FunctionTransformer, StandardScaler) # preprocessing 
from sklearn.decomposition import PCA # dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import boxcox # data transform
from sklearn.model_selection import (train_test_split, KFold , StratifiedKFold, 
                                     cross_val_score, GridSearchCV, 
                                     learning_curve, validation_curve) # model selection modules
from sklearn.pipeline import Pipeline # streaming pipelines
from sklearn.base import BaseEstimator, TransformerMixin # To create a box-cox transformation class
from collections import Counter
import warnings
# load models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import (XGBClassifier, plot_importance)
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from time import time

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
# Additional imports extracted from mixed cells
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
# Detect observations with more than one outlier

def outlier_hunt(df):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than 2 outliers. 
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in df.columns.tolist():
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        
        # Interquartile rrange (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2 )
    
    return multiple_outliers   

print('The dataset contains %d observations with more than 2 outliers' %(len(outlier_hunt(df[features]))))

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

    df = load_dataset('glass_classification')
    features = df.columns[:-1].tolist()
    print(df.shape)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    features

    for feat in features:
        skew = df[feat].skew()
        sns.distplot(df[feat], kde= False, label='Skew = %.3f' %(skew), bins=30)
        plt.legend(loc='best')
        plt.show()

    corr = df[features].corr()
    plt.figure(figsize=(16,16))
    sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
               xticklabels= features, yticklabels= features, alpha = 0.7,   cmap= 'coolwarm')
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    outlier_indices = outlier_hunt(df[features])
    df = df.drop(outlier_indices).reset_index(drop=True)
    print(df.shape)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for feat in features:
        skew = df[feat].skew()
        sns.distplot(df[feat], kde=False, label='Skew = %.3f' %(skew), bins=30)
        plt.legend(loc='best')
        plt.show()

    # Define X as features and y as lablels
    X = df[features] 
    y = df['Type']



    # --- PREPROCESSING ───────────────────────────────────────

    # Standardize the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    seed = 7
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=seed)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    features_boxcox = []

    for feature in features:
        bc_transformed, _ = boxcox(df[feature]+1)  # shift by 1 to avoid computing log of negative values
        features_boxcox.append(bc_transformed)

    features_boxcox = np.column_stack(features_boxcox)
    df_bc = pd.DataFrame(data=features_boxcox, columns=features)
    df_bc['Type'] = df['Type']

    for feature in features:
        fig, ax = plt.subplots(1,2,figsize=(7,3.5))    
        ax[0].hist(df[feature], color='blue', bins=30, alpha=0.3, label='Skew = %s' %(str(round(df[feature].skew(),3))) )
        ax[0].set_title(str(feature))   
        ax[0].legend(loc=0)
        ax[1].hist(df_bc[feature], color='red', bins=30, alpha=0.3, label='Skew = %s' %(str(round(df_bc[feature].skew(),3))) )
        ax[1].set_title(str(feature)+' after a Box-Cox transformation')
        ax[1].legend(loc=0)
        plt.show()

    # check if skew is closer to zero after a box-cox transform
    for feature in features:
        delta = np.abs( df_bc[feature].skew() / df[feature].skew() )
        if delta < 1.0 :
            print('Feature %s is less skewed after a Box-Cox transform' %(feature))
        else:
            print('Feature %s is more skewed after a Box-Cox transform'  %(feature))



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

            clf_setup = setup(data=df, target='Type', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Glass Classification")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
