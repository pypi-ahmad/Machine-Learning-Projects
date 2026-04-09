#!/usr/bin/env python3
"""
Full pipeline for Predicting loan default

Auto-generated from: loan-default-prediction.ipynb
Project: Predicting loan default
Category: Classification | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
# # Missing Values

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
# # # Feature Engineering and Selection

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between loss
    y = x['loss']
    x = x.drop(columns = ['loss'])
    
    # Calculate the correlation matrix
    corr_matrix = x.corr(numeric_only=True)
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns = drops)
    
    # Add the score back in to the data
    x['loss'] = y
               
    return x

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

    # --- PREPROCESSING ───────────────────────────────────────

    # # Imports

    # Pandas and numpy for data manipulation
    import pandas as pd
    import numpy as np

    # No warnings about setting value on copy of slice
    pd.options.mode.chained_assignment = None

    # Display up to 60 columns of a dataframe
    pd.set_option('display.max_columns', 60)

    # Matplotlib visualization
    import matplotlib.pyplot as plt

    # Set default font size
    plt.rcParams['font.size'] = 24

    # Internal ipython tool for setting figure size
    from IPython.core.pylabtools import figsize

    # Seaborn for visualization
    import seaborn as sns
    sns.set(font_scale = 2)

    # Splitting data into training and testing
    from sklearn.model_selection import train_test_split



    # --- DATA LOADING ────────────────────────────────────────

    # # # Data Cleaning and Formatting

    # # Load in the Data and Examine

    import os

    # --- Schema Reconciliation: load substitute dataset from centralized data directory ---
    _data_dir = os.path.join(os.path.dirname(os.path.abspath("__file__")), "..", "..", "data", "loan_default_prediction")
    _csv_path = os.path.join(_data_dir, "Loan_Default.csv")
    if not os.path.exists(_csv_path):
        _csv_path = './loan-default-prediction/train_v2.csv'

    data = load_dataset('predicting_loan_default')

    # --- Schema mapping ---
    # Original dataset had 771 anonymous features + 'loss' (continuous).
    # Substitute dataset has 34 named columns + 'Status' (binary 0/1).
    # Map 'Status' -> 'loss' so downstream code referencing 'loss' still works.
    if 'Status' in data.columns and 'loss' not in data.columns:
        data = data.rename(columns={'Status': 'loss'})
    # Drop ID column (not a feature)
    if 'ID' in data.columns:
        data = data.drop(columns=['ID'])

    # Validation
    print("Columns:", data.columns.tolist())
    print("Shape:", data.shape)
    data.head()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.shape

    # # Data Types and Missing Values

    # See the column data types and non-missing values
    data.info()

    data.select_dtypes(include=['object']).head()

    # Statistics for each column
    data.describe()

    missing_values_table(data).head(50)



    # --- PREPROCESSING ───────────────────────────────────────

    # Fill numeric columns with mean, categorical with mode
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else "Unknown")



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    missing_values_table(data).head(50)



    # --- PREPROCESSING ───────────────────────────────────────

    data.dropna(inplace=True)
    missing_values_table(data)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.shape



    # --- PREPROCESSING ───────────────────────────────────────

    # # # Exploratory Data Analysis

    # Encode categorical features using label encoding (substitute dataset has named categorical columns)
    from sklearn.preprocessing import LabelEncoder
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # # Single Variable Plots

    figsize=(8, 8)

    # Histogram of the loss
    plt.style.use('fivethirtyeight')
    plt.hist(data['loss'], bins = 100, edgecolor = 'k')
    plt.xlabel('Loss') 
    plt.ylabel('Number of Clients');
    plt.title('Loss Distribution')

    # # Correlations between Features and Target

    # Find all correlations and sort 
    correlations_data = data.corr(numeric_only=True)['loss'].sort_values()

    # Print the most negative correlations
    print(correlations_data.head(15), '\n')

    # Print the most positive correlations
    print(correlations_data.tail(15))



    # --- FEATURE ENGINEERING ─────────────────────────────────

    for i in data.columns:
        if len(set(data[i]))==1:
            data.drop(labels=[i], axis=1, inplace=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Find all correlations and sort 
    correlations_data = data.corr(numeric_only=True)['loss'].sort_values()

    # Print the most negative correlations
    print(correlations_data.head(15), '\n')

    # Print the most positive correlations
    print(correlations_data.tail(15))



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.shape



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Remove the collinear features above a specified correlation coefficient
    data = remove_collinear_features(data, 0.6);



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.shape



    # --- PREPROCESSING ───────────────────────────────────────

    # # # Split Into Training and Testing Sets

    # Separate out the features and targets
    features = data.drop(columns='loss')
    targets = pd.DataFrame(data['loss'])

    # Split into 80% training and 20% testing set
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # Convert y to one-dimensional array (vector)
    y_train = np.array(y_train).reshape((-1, ))
    y_test = np.array(y_test).reshape((-1, ))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train

    X_test



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

            clf_setup = setup(data=data, target='Status', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Predicting loan default")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
