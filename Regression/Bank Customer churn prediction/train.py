#!/usr/bin/env python3
"""
Model training for Bank Customer churn prediction

Auto-generated from: bank_customer_churn_prediction.ipynb
Project: Bank Customer churn prediction
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
## REQUIRED LIBRARIES
# For data wrangling 
import numpy as np
import pandas as pd

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_rows = None
pd.options.display.max_columns = None
# Additional imports extracted from mixed cells
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
# data prep pipeline for test data
def DfPrepPipeline(df_predict,df_train_Cols,minVec,maxVec):
    # Add new features
    df_predict['BalanceSalaryRatio'] = df_predict.Balance/df_predict.EstimatedSalary
    df_predict['TenureByAge'] = df_predict.Tenure/(df_predict.Age - 18)
    df_predict['CreditScoreGivenAge'] = df_predict.CreditScore/(df_predict.Age - 18)
    # Reorder the columns
    continuous_vars = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','BalanceSalaryRatio',
                   'TenureByAge','CreditScoreGivenAge']
    cat_vars = ['HasCrCard','IsActiveMember',"Geography", "Gender"] 
    df_predict = df_predict[['Exited'] + continuous_vars + cat_vars]
    # Change the 0 in categorical variables to -1
    df_predict.loc[df_predict.HasCrCard == 0, 'HasCrCard'] = -1
    df_predict.loc[df_predict.IsActiveMember == 0, 'IsActiveMember'] = -1
    # One hot encode the categorical variables
    lst = ["Geography", "Gender"]
    remove = list()
    for i in lst:
        for j in df_predict[i].unique():
            df_predict[i+'_'+j] = np.where(df_predict[i] == j,1,-1)
        remove.append(i)
    df_predict = df_predict.drop(remove, axis=1)
    # Ensure that all one hot encoded variables that appear in the train data appear in the subsequent data
    L = list(set(df_train_Cols) - set(df_predict.columns))
    for l in L:
        df_predict[str(l)] = -1        
    # MinMax scaling coontinuous variables based on min and max from the train data
    df_predict[continuous_vars] = (df_predict[continuous_vars]-minVec)/(maxVec-minVec)
    # Ensure that The variables are ordered in the same way as was ordered in the train set
    df_predict = df_predict[df_train_Cols]
    return df_predict

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

    # Read the data frame
    df = load_dataset('bank_customer_churn_prediction')
    df.shape



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Drop the columns as explained above
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    labels = 'Exited', 'Retained'
    sizes = [df.Exited[df['Exited']==1].count(), df.Exited[df['Exited']==0].count()]
    explode = (0, 0.1)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title("Proportion of customer churned and retained", size = 20)
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Split Train, test data
    df_train = df.sample(frac=0.8,random_state=200)
    df_test = df.drop(df_train.index)
    print(len(df_train))
    print(len(df_test))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df_train['BalanceSalaryRatio'] = df_train.Balance/df_train.EstimatedSalary
    sns.boxplot(y='BalanceSalaryRatio',x = 'Exited', hue = 'Exited',data = df_train)
    plt.ylim(-1, 5)

    # Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
    df_train['TenureByAge'] = df_train.Tenure/(df_train.Age)
    sns.boxplot(y='TenureByAge',x = 'Exited', hue = 'Exited',data = df_train)
    plt.ylim(-1, 1)
    plt.show()

    '''Lastly we introduce a variable to capture credit score given age to take into account credit behaviour visavis adult life
    :-)'''
    df_train['CreditScoreGivenAge'] = df_train.CreditScore/(df_train.Age)

    # Arrange columns by data type for easier manipulation
    continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                       'TenureByAge','CreditScoreGivenAge']
    cat_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
    df_train = df_train[['Exited'] + continuous_vars + cat_vars]
    df_train.head()

    '''For the one hot variables, we change 0 to -1 so that the models can capture a negative relation 
    where the attribute in inapplicable instead of 0'''
    df_train.loc[df_train.HasCrCard == 0, 'HasCrCard'] = -1
    df_train.loc[df_train.IsActiveMember == 0, 'IsActiveMember'] = -1
    df_train.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # One hot encode the categorical variables
    lst = ['Geography', 'Gender']
    remove = list()
    for i in lst:
        if (df_train[i].dtype == np.str or df_train[i].dtype == np.object):
            for j in df_train[i].unique():
                df_train[i+'_'+j] = np.where(df_train[i] == j,1,-1)
            remove.append(i)
    df_train = df_train.drop(remove, axis=1)
    df_train.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # minMax scaling the continuous variables
    minVec = df_train[continuous_vars].min().copy()
    maxVec = df_train[continuous_vars].max().copy()
    df_train[continuous_vars] = (df_train[continuous_vars]-minVec)/(maxVec-minVec)
    df_train.head()



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    # Define features and target
    X = df.drop(columns=['Exited'])
    y = df['Exited']

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

            clf_setup = setup(data=df, target='Exited', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Bank Customer churn prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
