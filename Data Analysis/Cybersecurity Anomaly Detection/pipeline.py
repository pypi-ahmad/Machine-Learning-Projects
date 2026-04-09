#!/usr/bin/env python3
"""
Full pipeline for Cybersecurity Anomaly Detection

Auto-generated from: code.ipynb
Project: Cybersecurity Anomaly Detection
Category: Data Analysis | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # viz
import matplotlib.pyplot as plt # viz
from scipy import stats
import json
from typing import List, Tuple

from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn import metrics, linear_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings('ignore')
# Additional imports extracted from mixed cells
import os
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def dataset_to_corr_heatmap(dataframe, title, ax):
    corr = dataframe.corr()
    sns.heatmap(corr, ax = ax, annot=True, cmap="YlGnBu")
    ax.set_title(f'Correlation Plot for {title}')
def column_uniques(df, col):
    print(f'{col} - Uniques:\n\n{df[col].unique()} \n\nNo. Uniques: {df[col].nunique()}')
def strip_string(input_str):
    """
    Takes an input string and replaces specific
    puncutation marks with nothing

    Args:
        input_str: The string to be processed

    Returns:
        The processed string
    """
    assert isinstance(input_str, str)
    return input_str.replace("[", "").replace("]", "").replace("'", '"')
def process_args_row(row):
    """
    Takes an single value from the 'args' column
    and returns a processed dataframe row

    Args:
        row: A single 'args' value/row

    Returns:
        final_df: The processed dataframe row
    """

    row = row.split('},')
    row = [string.replace("[", "").replace("]", "").replace("{", "").replace("'", "").replace("}", "").lstrip(" ") for string in row]
    row = [item.split(',') for item in row]

    processed_row = []
    for lst in row:
        for key_value in lst:
            key, value = key_value.split(': ', 1)
            if not processed_row or key in processed_row[-1]:
                processed_row.append({})
            processed_row[-1][key] = value

    json_row = json.dumps(processed_row)
    row_df = pd.json_normalize(json.loads(json_row))

    final_df = row_df.unstack().to_frame().T.sort_index(1,1)
    final_df.columns = final_df.columns.map('{0[0]}_{0[1]}'.format)

    return final_df
def process_args_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Processes the `args` column within the dataset
    """

    processed_dataframes = []
    data = df[column_name].tolist()

    # Debug counter
    counter = 0

    for row in data:
        if row == '[]': # If there are no args
            pass
        else:
            try:
                ret = process_args_row(row)
                processed_dataframes.append(ret)
            except:
                print(f'Error Encounter: Row {counter} - {row}')

            counter+=1

    processed = pd.concat(processed_dataframes).reset_index(drop=True)
    processed.columns = processed.columns.str.lstrip()

    df = pd.concat([df, processed], axis=1)

    return df

def prepare_dataset(df: pd.DataFrame, process_args=False) -> pd.DataFrame:
    """
    Prepare the dataset by completing the standard feature engineering tasks
    """

    df["processId"] = train_df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    df["parentProcessId"] = train_df["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    df["userId"] = train_df["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
    df["mountNamespace"] = train_df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
    df["eventId"] = train_df["eventId"]  # Keep eventId values (requires knowing max value)
    df["returnValue"] = train_df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error

    if process_args is True:
        df = process_args_dataframe(df, 'args')

    features = df[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]]
    labels = df['sus']

    return features, labels

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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- DATA LOADING ────────────────────────────────────────

    import os

    # Training data file is missing from dataset; use testing data as training substitute
    _train_path = 'data/labelled_training_data.csv'
    if not os.path.exists(_train_path):
        print("WARNING: labelled_training_data.csv not found. Using labelled_testing_data.csv as training data.")
        _train_path = 'data/labelled_testing_data.csv'

    train_df = load_dataset('cybersecurity_anomaly_detection')
    test_df = pd.read_csv('data/labelled_testing_data.csv')
    validation_df = pd.read_csv('data/labelled_validation_data.csv')

    # Validation
    print("train_df columns:", train_df.columns.tolist())
    print("train_df shape:", train_df.shape)
    print("test_df shape:", test_df.shape)
    print("validation_df shape:", validation_df.shape)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    assert train_df.columns.all() == test_df.columns.all() == validation_df.columns.all()

    train_df.dtypes

    train_df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train_df.describe(include=['object', 'float', 'int'])



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    train_df.evil.value_counts().plot(kind='bar', title='Label Frequency for evil label in Train Dataset')

    train_df.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Train Dataset')

    test_df.evil.value_counts().plot(kind='bar', title='Label Frequency for evil label in Test Dataset')

    test_df.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Test Dataset')

    validation_df.evil.value_counts().plot(kind='bar', title='Label Frequency for evil label in Validation Dataset')

    validation_df.sus.value_counts().plot(kind='bar', title='Label Frequency for sus label in Validation Dataset')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train_df.groupby(['sus', 'evil'])[['timestamp']].count()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    train_df.groupby(['sus', 'evil'])[['timestamp']].count().plot(kind='bar')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    test_df.groupby(['sus', 'evil'])[['timestamp']].count()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    test_df.loc[(test_df['sus'] == 1) & (test_df['evil'] == 1)].shape[0]

    test_df.groupby(['sus', 'evil'])[['timestamp']].count().plot(kind='bar')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    validation_df.groupby(['sus', 'evil'])[['timestamp']].count()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    validation_df.groupby(['sus', 'evil'])[['timestamp']].count().plot(kind='bar')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (15,20))
    fig.tight_layout(pad=10.0)
    datasets = [train_df, test_df, validation_df]
    dataset_names = ['train', 'test', 'validation']
    axs = [ax1, ax2, ax3]

    for dataset, name, ax in zip(datasets, dataset_names, axs):
        dataset_to_corr_heatmap(dataset, name, ax)

    datasets = [train_df, test_df, validation_df]

    entropy_values = []
    for dataset in datasets:
        dataset_entropy_values = []
        for col in dataset.columns:
            if col == 'timestamp':
                pass
            else:
                counts = dataset[col].value_counts()
                col_entropy = stats.entropy(counts)
                dataset_entropy_values.append(col_entropy)

        entropy_values.append(dataset_entropy_values)

    plt.boxplot(entropy_values)
    plt.title('Boxplot of Entropy Values')
    plt.ylabel("entropy values")
    plt.xticks([0,1,2,3],labels=['','train', 'test', 'validate'])
    plt.show()

    datasets = [train_df, test_df, validation_df]

    variation_values = []
    for dataset in datasets:
        dataset_variation_values = []
        for col in dataset.columns:
            if col == 'timestamp':
                pass
            else:
                counts = dataset[col].value_counts()
                col_variation = stats.variation(counts)
                dataset_variation_values.append(col_variation)

        variation_values.append(dataset_variation_values)

    plt.boxplot(variation_values)
    plt.title('Boxplot of Variation Values')
    plt.ylabel("Variation values")
    plt.xticks([0,1,2,3],labels=['','train', 'test', 'validate'])
    plt.show()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    train_df.loc[:, ['eventId', 'eventName']].head(10)

    train_df.loc[:, ['processName', 'hostName', 'args']].head(10)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    column_uniques(train_df, 'processName')

    column_uniques(train_df, 'hostName')

    column_uniques(train_df, 'args')

    sample = train_df['args'].sample(n=15, random_state=1)
    sample

    sample_df = pd.DataFrame(sample)
    sample_df

    sample_df.iloc[0]



    # --- FEATURE ENGINEERING ─────────────────────────────────

    sample1 = sample_df.iloc[0]
    sample1 = sample1.replace("[", "").replace("]", "").replace("'", '"')
    sample1 = sample1[0]
    sample1



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    sample1 = json.dumps(sample1)
    test1 = json.loads(sample1)

    test1



    # --- FEATURE ENGINEERING ─────────────────────────────────

    sample_df['stripped_args'] = sample_df['args'].apply(strip_string)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for i in sample_df['stripped_args']:
        print(i)
        print('\n')

    sample_df['args'].iloc[2]

    test2 = sample_df['args'].iloc[2]

    split_test2 = test2.split('},')
    split_test2



    # --- FEATURE ENGINEERING ─────────────────────────────────

    strings = [string.replace("[", "").replace("]", "").replace("{", "").replace("'", "").replace("}", "").lstrip(" ") for string in split_test2]
    strings



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    list_of_lists = [item.split(',') for item in strings]
    list_of_lists

    output = []
    for lst in list_of_lists:
        for key_value in lst:
            key, value = key_value.split(': ', 1)
            if not output or key in output[-1]:
                output.append({})
            output[-1][key] = value

    output

    json_output = json.dumps(output)

    interim_df = pd.json_normalize(json.loads(json_output))
    interim_df

    interim_df.unstack()

    interim_df.unstack().to_frame()

    interim_df.unstack().to_frame().T

    interim_df.unstack().to_frame().T.sort_index(1,1)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    final_df = interim_df.unstack().to_frame().T.sort_index(1,1)
    final_df.columns = final_df.columns.map('{0[0]}_{0[1]}'.format)
    final_df



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data = sample_df['args'].tolist()

    processed_dataframes = []

    for row in data:
        ret = process_args_row(row)
        processed_dataframes.append(ret)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    processed = pd.concat(processed_dataframes).reset_index(drop=True)
    processed.columns = processed.columns.str.lstrip()
    processed

    sample_df = sample_df.reset_index(drop=True)
    merged_sample = pd.concat([sample_df, processed], axis=1)
    merged_sample

    # Taken from here - https://github.com/jinxmirror13/BETH_Dataset_Analysis
    train_df["processId"] = train_df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    train_df["parentProcessId"] = train_df["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)  # Map to OS/not OS
    train_df["userId"] = train_df["userId"].map(lambda x: 0 if x < 1000 else 1)  # Map to OS/not OS
    train_df["mountNamespace"] = train_df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)  # Map to mount access to mnt/ (all non-OS users) /elsewhere
    train_df["eventId"] = train_df["eventId"]  # Keep eventId values (requires knowing max value)
    train_df["returnValue"] = train_df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))  # Map to success/success with value/error



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    train_df.head(5)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train = train_df[["processId", "parentProcessId", "userId", "mountNamespace", "eventId", "argsNum", "returnValue"]]
    train_labels = train_df['sus']



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    train.head(5)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train_labels



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    assert len(train_labels) == train.shape[0]



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train_no_args_feats, train_no_args_labels = prepare_dataset(train_df)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    train_no_args_feats.head()

    train_no_args_labels.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train_df_feats, train_df_labels = prepare_dataset(train_df)
    test_df_feats, test_df_labels = prepare_dataset(test_df)
    val_df_feats, val_df_labels = prepare_dataset(validation_df)



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    # Define features and target
    X = train_df.drop(columns=['label'])
    y = train_df['label']

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

            clf_setup = setup(data=train_df, target='label', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Cybersecurity Anomaly Detection")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
