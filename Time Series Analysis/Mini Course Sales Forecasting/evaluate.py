#!/usr/bin/env python3
"""
Model evaluation for Mini-Course Sales Forecasting

Auto-generated from: code.ipynb
Project: Mini-Course Sales Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from category_encoders import OneHotEncoder, MEstimateEncoder, GLMMEncoder, OrdinalEncoder, CatBoostEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
class DateProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, x, y = None):
        return self
    def transform(self, x, y = None):
        x_copy = x.copy()
        x_copy['day'] = x_copy.date.dt.day
        x_copy['month'] = x_copy.date.dt.month
        x_copy['year'] = x_copy.date.dt.year
        x_copy['dow'] = x_copy.date.dt.dayofweek
        x_copy = x_copy.drop('date', axis = 1)
        return x_copy
def multipliers(predictors, prediction, canada = 1, japan = 1, spain = 1, estonia = 1, argentina = 1):
    prediction[predictors.country == 'Canada'] *= canada
    prediction[predictors.country == 'Japan'] *= japan
    prediction[predictors.country == 'Spain'] *= spain
    prediction[predictors.country == 'Estonia'] *= estonia
    prediction[predictors.country == 'Argentina'] *= argentina
    return prediction

# ======================================================================
# EVALUATION PIPELINE
# ======================================================================

def main():
    """Run the evaluation pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- EVALUATION ──────────────────────────────────────────

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re

    from category_encoders import OneHotEncoder, MEstimateEncoder, GLMMEncoder, OrdinalEncoder, CatBoostEncoder
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold, TimeSeriesSplit
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.linear_model import HuberRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error, roc_auc_score, roc_curve
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
    from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, PowerTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.calibration import CalibratedClassifierCV
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    sns.set_theme(style = 'white', palette = 'colorblind')
    pal = sns.color_palette('colorblind')

    pd.set_option('display.max_rows', 100)



    # --- DATA LOADING ────────────────────────────────────────

    train = load_dataset('mini_course_sales_forecasting')
    test_1 = pd.read_csv(r'data/test.csv')

    train.drop('id', axis = 1, inplace = True)
    test = test_1.drop('id', axis = 1)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    desc = pd.DataFrame(index = list(train))
    desc['count'] = train.count()
    desc['nunique'] = train.nunique()
    desc['%unique'] = desc['nunique'] / len(train) * 100
    desc['null'] = train.isnull().sum()
    desc['type'] = train.dtypes
    desc = pd.concat([desc, train.describe().T.drop('count', axis = 1)], axis = 1)
    desc

    desc = pd.DataFrame(index = list(test))
    desc['count'] = test.count()
    desc['nunique'] = test.nunique()
    desc['%unique'] = desc['nunique'] / len(test) * 100
    desc['null'] = test.isnull().sum()
    desc['type'] = test.dtypes
    desc = pd.concat([desc, test.describe().T.drop('count', axis = 1)], axis = 1)
    desc

    categorical_features = ['country', 'store', 'product']
    numerical_features = test.drop(categorical_features, axis = 1).columns



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train.date = pd.to_datetime(train.date, format = '%Y-%m-%d')
    test.date = pd.to_datetime(test.date, format = '%Y-%m-%d')

    fig, ax = plt.subplots(1, 2, figsize = (16, 5))
    ax = ax.flatten()

    ax[0].pie(
        train['country'].value_counts(),
        shadow = True,
        explode = [.1 for i in range(0, 5)],
        autopct = '%1.f%%',
        textprops = {'size' : 14, 'color' : 'white'}
    )

    sns.countplot(data = train, y = 'country', ax = ax[1], palette = 'viridis', order = train['country'].value_counts().index)
    ax[1].yaxis.label.set_size(20)
    plt.yticks(fontsize = 12)
    ax[1].set_xlabel('Count', fontsize = 20)
    ax[1].set_ylabel(None)
    plt.xticks(fontsize = 12)

    fig.suptitle('Country in Train Dataset', fontsize = 25, fontweight = 'bold')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize = (16, 5))
    ax = ax.flatten()

    ax[0].pie(
        test['country'].value_counts(),
        shadow = True,
        explode = [.1 for i in range(0, 5)],
        autopct = '%1.f%%',
        textprops = {'size' : 14, 'color' : 'white'}
    )

    sns.countplot(data = test, y = 'country', ax = ax[1], palette = 'viridis', order = test['country'].value_counts().index)
    ax[1].yaxis.label.set_size(20)
    plt.yticks(fontsize = 12)
    ax[1].set_xlabel('Count', fontsize = 20)
    ax[1].set_ylabel(None)
    plt.xticks(fontsize = 12)

    fig.suptitle('Country in Test Dataset', fontsize = 25, fontweight = 'bold')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize = (16, 5))
    ax = ax.flatten()

    ax[0].pie(
        train['store'].value_counts(),
        shadow = True,
        explode = [.1 for i in range(0, 3)],
        autopct = '%1.f%%',
        textprops = {'size' : 14, 'color' : 'white'}
    )

    sns.countplot(data = train, y = 'store', ax = ax[1], palette = 'viridis', order = train['store'].value_counts().index)
    ax[1].yaxis.label.set_size(20)
    plt.yticks(fontsize = 12)
    ax[1].set_xlabel('Count', fontsize = 20)
    ax[1].set_ylabel(None)
    plt.xticks(fontsize = 12)

    fig.suptitle('Store in Train Dataset', fontsize = 25, fontweight = 'bold')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize = (16, 5))
    ax = ax.flatten()

    ax[0].pie(
        test['store'].value_counts(),
        shadow = True,
        explode = [.1 for i in range(0, 3)],
        autopct = '%1.f%%',
        textprops = {'size' : 14, 'color' : 'white'}
    )

    sns.countplot(data = test, y = 'store', ax = ax[1], palette = 'viridis', order = test['store'].value_counts().index)
    ax[1].yaxis.label.set_size(20)
    plt.yticks(fontsize = 12)
    ax[1].set_xlabel('Count', fontsize = 20)
    ax[1].set_ylabel(None)
    plt.xticks(fontsize = 12)

    fig.suptitle('Store in Test Dataset', fontsize = 25, fontweight = 'bold')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize = (16, 5))
    ax = ax.flatten()

    ax[0].pie(
        train['product'].value_counts(),
        shadow = True,
        explode = [.1 for i in range(0, 5)],
        autopct = '%1.f%%',
        textprops = {'size' : 14, 'color' : 'white'}
    )

    sns.countplot(data = train, y = 'product', ax = ax[1], palette = 'viridis', order = train['product'].value_counts().index)
    ax[1].yaxis.label.set_size(20)
    plt.yticks(fontsize = 12)
    ax[1].set_xlabel('Count', fontsize = 20)
    ax[1].set_ylabel(None)
    plt.xticks(fontsize = 12)

    fig.suptitle('Product in Train Dataset', fontsize = 25, fontweight = 'bold')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize = (16, 5))
    ax = ax.flatten()

    ax[0].pie(
        test['product'].value_counts(),
        shadow = True,
        explode = [.1 for i in range(0, 5)],
        autopct = '%1.f%%',
        textprops = {'size' : 14, 'color' : 'white'}
    )

    sns.countplot(data = test, y = 'product', ax = ax[1], palette = 'viridis', order = test['product'].value_counts().index)
    ax[1].yaxis.label.set_size(20)
    plt.yticks(fontsize = 12)
    ax[1].set_xlabel('Count', fontsize = 20)
    ax[1].set_ylabel(None)
    plt.xticks(fontsize = 12)

    fig.suptitle('Product in Test Dataset', fontsize = 25, fontweight = 'bold')
    plt.tight_layout()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X = train.copy()
    y = X.pop('num_sold')
    y = np.log1p(y)

    seed = 42
    k = TimeSeriesSplit(n_splits = 4, test_size = 27390)

    np.random.seed(seed)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model evaluation for Mini-Course Sales Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
