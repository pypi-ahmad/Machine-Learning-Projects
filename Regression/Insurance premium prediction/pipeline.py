#!/usr/bin/env python3
"""
Full pipeline for Insurance premium prediction

Auto-generated from: predicting_insurance_premium.ipynb
Project: Insurance premium prediction
Category: Regression | Task: regression
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
# Additional imports extracted from mixed cells
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from scipy import stats
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


    import os
    for dirname, _, filenames in os.walk('./archive/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    filename = './archive/insurance.csv'
    df = load_dataset('insurance_premium_prediction')
    df.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df.rename(columns = {'expenses':'charges'}, inplace = True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.head()

    df.shape

    df.info()

    df.describe()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    corr = df.corr()
    corr



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.isnull().sum()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot = True)

    fig, axes = plt.subplots(ncols = 3, figsize = (15,6), squeeze=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
    df.plot(kind='scatter', x='age', y='charges', ax=axes[0])
    df.plot(kind='scatter', x='children', y='charges', ax=axes[1])
    df.plot(kind='scatter', x='bmi', y='charges', ax=axes[2])

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15,10))

    df.plot(kind='hist', y='age', ax=axes[0][0], color = 'blue')
    df.plot(kind='hist', y='bmi', ax=axes[0][1], color = 'orange', bins = 54)
    df.plot(kind='hist', y='children', ax=axes[1][0], color = 'red', bins = 6)
    df.plot(kind='hist', y='charges', ax=axes[1][1], color = 'green', bins = 80)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    palette=['#EB5050','#3EA2FF']
    fig, axes = plt.subplots(ncols = 3, figsize = (15,6), squeeze=True)
    sns.scatterplot(x='bmi', y='charges', ax=axes[0], data=df,hue='sex', palette=palette)
    sns.scatterplot(x='bmi', y='charges', ax=axes[1], data=df,hue='smoker', palette=palette)
    sns.scatterplot(x='bmi', y='charges', ax=axes[2], data=df,hue='region')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    fig, axes = plt.subplots(ncols=3, figsize = (15,6))
    df['sex'].value_counts().plot(kind='bar', color = 'orange', ax=axes[0],title="Sex", legend = 'sex') 
    df['region'].value_counts().plot(kind='bar', color = 'green', ax=axes[1],title="Region", legend = 'region')
    df['smoker'].value_counts().plot(kind='bar', ax=axes[2],title="Smoker", legend = 'smoker')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    palette=['#EB5050','#3EA2FF']
    sns.catplot(x='sex', y='charges', kind='violin', palette=palette, data=df)

    palette=['#EB5050','#2DFFAB'] 
    sns.catplot(x='sex', y='charges', kind='violin', hue='smoker', palette=palette, data=df)



    # --- MODEL TRAINING ──────────────────────────────────────

    from scipy import stats
    from scipy.stats import norm
    fig =plt.figure(figsize=(18,6))
    plt.subplot(1,2,1)
    sns.distplot(df['charges'], fit=norm)
    (mu,sigma)= norm.fit(df['charges'])
    plt.legend(['For Normal dist. mean: {:.2f} | std: {:.2f}'.format(mu,sigma)])
    plt.ylabel('Frequency')
    plt.title('Distribution of Charges')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    palette=['#EB5050','#2DFFAB'] 
    sns.set(style="ticks")
    sns.pairplot(data=df, hue='smoker', palette=palette)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df.drop(["region"], axis=1, inplace=True) 
    df.head()

    # Changing binary categories to 1s and 0s
    df['sex'] = df['sex'].map(lambda s :1  if s == 'female' else 0)
    df['smoker'] = df['smoker'].map(lambda s :1  if s == 'yes' else 0)

    df.head()

    X = df.drop(['charges'], axis = 1)
    y = df.charges
    print('Shape of X: ', X.shape)
    print('Shape of y: ', y.shape)



    # --- MODEL TRAINING ──────────────────────────────────────

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    lr = LinearRegression().fit(X_train, y_train)

    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    print(lr.score(X_test, y_test))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
    results



    # --- PREPROCESSING ───────────────────────────────────────

    # Normalize the data
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    pd.DataFrame(X_train).head()

    pd.DataFrame(y_train).head()



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

            reg_setup = setup(data=df, target='charges', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Insurance premium prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
