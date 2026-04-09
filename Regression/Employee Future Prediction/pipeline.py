#!/usr/bin/env python3
"""
Full pipeline for Employee Future Prediction

Auto-generated from: employee_prediction.ipynb
Project: Employee Future Prediction
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
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

    # --- EVALUATION ──────────────────────────────────────────

    import numpy as np 
    import pandas as pd 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.preprocessing import LabelEncoder


    #Matplotlib Config
    plt.style.use('fivethirtyeight')

    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 20
    # plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20
    width, height = plt.figaspect(1.68)
    fig = plt.figure(figsize=(width,height), dpi=400)



    # --- DATA LOADING ────────────────────────────────────────

    df = load_dataset('employee_future_prediction')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.shape

    df.info()

    df.isna().sum()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()

    print('No. of duplicate records :',duplicates)
    print('Shape after dropping duplicate records :',df.shape)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.describe()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    f, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax ,annot=True)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df['JoiningYear'] = df['JoiningYear'].astype('object')
    sns.countplot(data = df ,x='JoiningYear',hue='LeaveOrNot')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    sns.countplot(data = df ,x='EverBenched',hue='LeaveOrNot')

    sns.countplot(data = df ,x='ExperienceInCurrentDomain',hue='LeaveOrNot')

    sns.countplot(data = df ,x='Gender',hue='LeaveOrNot')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df['PaymentTier'] = df['PaymentTier'].astype('category')
    sns.countplot(data = df ,x='PaymentTier',hue='LeaveOrNot')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    sns.countplot(data = df ,x='City',hue='LeaveOrNot')

    sns.countplot(data = df ,x='Education',hue='LeaveOrNot')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    groups = ['Young', 'MiddleAged', 'Adulthood']
    df['AgeGroup'] = pd.qcut(df['Age'], q=3, labels=groups)
    sns.countplot(data = df ,x='AgeGroup',hue='LeaveOrNot')



    # --- PREPROCESSING ───────────────────────────────────────

    X = df.drop('LeaveOrNot',axis=1)
    y= df.LeaveOrNot.values

    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

    #new feature engineering code which got  me lgbm's acc : 83% and precision 97%

    multi_categories = ['AgeGroup','EverBenched','City','JoiningYear']
    ordinal_cat = [['PHD','Masters','Bachelors'],[1,2,3]]
    binary_categories=['Gender','EverBenched']
    transformer = ColumnTransformer(transformers=[('ohe1', OneHotEncoder(sparse='False'), multi_categories),
                                                 ('oe',OrdinalEncoder(categories=ordinal_cat),['Education','PaymentTier']),
                                                 ('ohe2', OneHotEncoder(drop='first',sparse='False'), binary_categories)],remainder='passthrough')

    # old feat engineering code which got me lgbm's acc : 83% and precision 93% 

    # df1= df.copy()
    # df1['PaymentTier'] = df['PaymentTier'].astype(np.uint8)
    # df1['LeaveOrNot'] = df['LeaveOrNot'].astype('category')
    # dummy_col = ['JoiningYear','City','AgeGroup']
    # df1=pd.get_dummies(df, columns=dummy_col,prefix = ['year','city','age'])
    # df1["is_masters"] = df1["Education"].map(lambda is_masters: 1 if is_masters == "Masters" else 0).astype(np.uint8)
    # df1["is_male"] = df1["Gender"].map(lambda is_male: 1 if is_male == "Male" else 0).astype(np.uint8)
    # df1["EverBenched"] = df1['EverBenched'].map(lambda is_benched: 1 if is_benched =='Yes' else 0).astype(np.uint8)
    # df1['Senority'] = df1[['ExperienceInCurrentDomain','Age']].apply(lambda x: 1 if x.ExperienceInCurrentDomain >= 3 else 0, axis=1).astype(np.uint8) # 0 : junior , 1: senior
    # # df1['isRecentEmployee']= df['JoiningYear'].map(lambda x : 1 if (x == '2018'or'2017') else 0).astype(np.uint8)
    # df1=df1.drop(['Education','Gender'],axis=1)
    # df1.head()

    X_train = transformer.fit_transform(X_train)
    X_train.shape
    X_test = transformer.transform(X_test)



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

            clf_setup = setup(data=df, target='LeaveOrNot', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Employee Future Prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
