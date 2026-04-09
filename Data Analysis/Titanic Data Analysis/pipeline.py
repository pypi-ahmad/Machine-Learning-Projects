#!/usr/bin/env python3
"""
Full pipeline for Titanic Data Analysis

Auto-generated from: code.ipynb
Project: Titanic Data Analysis
Category: Data Analysis | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# Additional imports extracted from mixed cells
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

    train_df = load_dataset('titanic_data_analysis')
    test_df = pd.read_csv('../../data/titanic_data_analysis/test.csv')
    combine = [train_df, test_df]



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    print(train_df.columns.values)

    # preview the data
    train_df.head()

    train_df.tail()

    train_df.info()
    print('_'*40)
    test_df.info()

    train_df.describe()
    # Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
    # Review Parch distribution using `percentiles=[.75, .8]`
    # SibSp distribution `[.68, .69]`
    # Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train_df.describe(include=['O'])

    train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    g = sns.FacetGrid(train_df, col='Survived')
    g.map(plt.hist, 'Age', bins=20)

    # grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend();

    # grid = sns.FacetGrid(train_df, col='Embarked')
    grid = sns.FacetGrid(train_df, row='Embarked', aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()

    # grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
    grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()

    print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

    "After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)

    pd.crosstab(train_df['Title'], train_df['Sex'])



    # --- FEATURE ENGINEERING ─────────────────────────────────

    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



    # --- PREPROCESSING ───────────────────────────────────────

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    train_df.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]
    train_df.shape, test_df.shape

    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    train_df.head()

    # grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
    grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    guess_ages = np.zeros((2,3))
    guess_ages



    # --- PREPROCESSING ───────────────────────────────────────

    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & \
                                      (dataset['Pclass'] == j+1)]['Age'].dropna()

                # age_mean = guess_df.mean()
                # age_std = guess_df.std()
                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                        'Age'] = guess_ages[i,j]

        dataset['Age'] = dataset['Age'].astype(int)

    train_df.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for dataset in combine:
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']
    train_df.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    train_df = train_df.drop(['AgeBand'], axis=1)
    combine = [train_df, test_df]
    train_df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    train_df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass

    train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)



    # --- PREPROCESSING ───────────────────────────────────────

    freq_port = train_df.Embarked.dropna().mode()[0]
    freq_port

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    train_df.head()



    # --- PREPROCESSING ───────────────────────────────────────

    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
    test_df.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

    train_df.head(10)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    test_df.head(10)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    X_train.shape, Y_train.shape, X_test.shape



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    # Define features and target
    X = train_df.drop(columns=['Survived'])
    y = train_df['Survived']

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

            clf_setup = setup(data=train_df, target='Survived', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Titanic Data Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
