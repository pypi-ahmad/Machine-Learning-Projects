#!/usr/bin/env python3
"""
Full pipeline for Student Alcohol Consumption Analysis

Auto-generated from: code.ipynb
Project: Student Alcohol Consumption Analysis
Category: Data Analysis | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Read the data into a DataFrame
    data = load_dataset('student_alcohol_consumption_analysis')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    numeric_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'failures']
    categorical_columns = ['school', 'sex', 'famsize', 'Pstatus']

    # Frequency distribution for categorical columns
    for column in categorical_columns:
        freq_dist = data[column].value_counts()
        print(f"\nFrequency Distribution for {column}:")
        print(freq_dist)

    # Correlation Analysis
    corr_matrix = data[numeric_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Correlation Analysis
    correlation = data[['Medu', 'Fedu']].corr()
    print(correlation)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Gender-Based Analysis
    gender_grades = data.groupby('sex')['failures'].mean()
    print("\nAverage Grades by Gender:")
    print(gender_grades)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # Count plot of schools
    sns.countplot(data=data, x='school')
    plt.title('Count Plot of Schools')
    plt.show()

    # Count plot of sex
    sns.countplot(data=data, x='sex')
    plt.title('Count Plot of Sex')
    plt.show()

    # Box plot of age
    sns.boxplot(data=data, x='age')
    plt.title('Box Plot of Age')
    plt.show()

    # Count plot of family size
    sns.countplot(data=data, x='famsize')
    plt.title('Count Plot of Family Size')
    plt.show()

    # Count plot of parent's cohabitation status
    sns.countplot(data=data, x='Pstatus')
    plt.title("Count Plot of Parent's Cohabitation Status")
    plt.show()

    # Count plot of mother's education level
    sns.countplot(data=data, x='Medu')
    plt.title("Count Plot of Mother's Education Level")
    plt.show()

    # Count plot of father's education level
    sns.countplot(data=data, x='Fedu')
    plt.title("Count Plot of Father's Education Level")
    plt.show()

    # Count plot of travel time
    sns.countplot(data=data, x='traveltime')
    plt.title('Count Plot of Travel Time')
    plt.show()

    # Count plot of failures
    sns.countplot(data=data, x='failures')
    plt.title('Count Plot of Failures')
    plt.show()

    numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'failures']
    numerical_summary = data[numerical_columns].describe()
    print(numerical_summary)

    correlation_matrix = data[numerical_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    sns.histplot(data=data, x='age', bins=10)
    plt.title('Distribution of Age')
    plt.show()

    sns.barplot(data=data, x='famsize', y='Medu')
    plt.title("Average Mother's Education Level by Family Size")
    plt.show()

    sex_count = data['sex'].value_counts()
    sex_percentage = data['sex'].value_counts(normalize=True) * 100
    print("Sex Count:\n", sex_count)
    print("\nSex Percentage:\n", sex_percentage)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    cross_tab = pd.crosstab(data['sex'], data['Pstatus'])
    print(cross_tab)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    sns.boxplot(data=data, x='Medu')
    plt.title("Box Plot - Mother's Education Level")
    plt.show()

    sns.violinplot(data=data, x='sex', y='age')
    plt.title("Violin Plot - Age by Sex")
    plt.show()



    # --- MODEL TRAINING ──────────────────────────────────────

    import pandas as pd
    import statsmodels.api as sm

    # Load the dataset
    data = pd.read_csv('data.csv')

    # Convert 'failures' to binary format (0 or 1)
    data['failures'] = data['failures'].apply(lambda x: 0 if x == 0 else 1)

    # Convert categorical variables to dummy variables
    categorical_vars = ['school', 'sex', 'famsize', 'Pstatus']
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

    # Define the predictors and target variable
    X = data[['age', 'Medu', 'Fedu', 'traveltime']]
    X = sm.add_constant(X)
    y = data['failures']

    # Perform logistic regression
    model = sm.Logit(y, X)
    result = model.fit()
    print(result.summary())



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    # Define features and target
    X = data.drop(columns=['failures'])
    y = data['failures']

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

            clf_setup = setup(data=data, target='failures', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Student Alcohol Consumption Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
