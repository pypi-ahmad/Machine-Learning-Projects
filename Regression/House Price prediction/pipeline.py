#!/usr/bin/env python3
"""
Full pipeline for House Price prediction

Auto-generated from: House_Prices_Prediction.ipynb
Project: House Price prediction
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
# Additional imports extracted from mixed cells
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p
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

    #import some necessary librairies

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt  # Matlab-style plotting
    import seaborn as sns
    color = sns.color_palette()
    sns.set_style('darkgrid')
    import warnings
    def ignore_warn(*args, **kwargs):
        pass
    warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


    from scipy import stats
    from scipy.stats import norm, skew #for some statistics


    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

    #Now let's import and put the train and test datasets in  pandas dataframe

    train = load_dataset('house_price_prediction')
    test = pd.read_csv('../../data/house_price_prediction/test.csv')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    ##display the first five rows of the train dataset.
    train.head(5)

    ##display the first five rows of the test dataset.
    test.head(5)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #check the numbers of samples and features
    print("The train data size before dropping Id feature is : {} ".format(train.shape))
    print("The test data size before dropping Id feature is : {} ".format(test.shape))

    #Save the 'Id' column
    train_ID = train['Id']
    test_ID = test['Id']

    #Now drop the  'Id' colum since it's unnecessary for  the prediction process.
    train.drop("Id", axis = 1, inplace = True)
    test.drop("Id", axis = 1, inplace = True)

    #check again the data size after dropping the 'Id' variable
    print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
    print("The test data size after dropping Id feature is : {} ".format(test.shape))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig, ax = plt.subplots()
    ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('GrLivArea', fontsize=13)
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #Deleting outliers
    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

    #Check the graphic again
    fig, ax = plt.subplots()
    ax.scatter(train['GrLivArea'], train['SalePrice'])
    plt.ylabel('SalePrice', fontsize=13)
    plt.xlabel('GrLivArea', fontsize=13)
    plt.show()



    # --- MODEL TRAINING ──────────────────────────────────────

    sns.distplot(train['SalePrice'] , fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train['SalePrice'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend([r'Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(train['SalePrice'], plot=plt)
    plt.show()

    #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    train["SalePrice"] = np.log1p(train["SalePrice"])

    #Check the new distribution 
    sns.distplot(train['SalePrice'] , fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(train['SalePrice'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend([r'Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(train['SalePrice'], plot=plt)
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    print("all_data size is : {}".format(all_data.shape))

    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    missing_data.head(20)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    sns.barplot(x=all_data_na.index, y=all_data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)

    #Correlation map to see how features are correlated with SalePrice
    corrmat = train.corr()
    plt.subplots(figsize=(12,9))
    sns.heatmap(corrmat, vmax=0.9, square=True)



    # --- PREPROCESSING ───────────────────────────────────────

    all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

    all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

    all_data["Alley"] = all_data["Alley"].fillna("None")

    all_data["Fence"] = all_data["Fence"].fillna("None")

    all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

    #Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        all_data[col] = all_data[col].fillna('None')

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        all_data[col] = all_data[col].fillna(0)

    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        all_data[col] = all_data[col].fillna('None')

    all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
    all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

    all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



    # --- FEATURE ENGINEERING ─────────────────────────────────

    all_data = all_data.drop(['Utilities'], axis=1)



    # --- PREPROCESSING ───────────────────────────────────────

    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

    all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

    all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
    all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

    all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

    all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #Check remaining missing values if any 
    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
    missing_data.head()

    #MSSubClass=The building class
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


    #Changing OverallCond into a categorical variable
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)


    #Year and month sold are transformed into categorical features.
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)



    # --- MODEL TRAINING ──────────────────────────────────────

    from sklearn.preprocessing import LabelEncoder
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(all_data[c].values)) 
        all_data[c] = lbl.transform(list(all_data[c].values))

    # shape        
    print('Shape all_data: {}'.format(all_data.shape))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Adding total sqfootage feature 
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



    # --- PREPROCESSING ───────────────────────────────────────

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness.head(10)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        #all_data[feat] += 1
        all_data[feat] = boxcox1p(all_data[feat], lam)
    
    #all_data[skewed_features] = np.log1p(all_data[skewed_features])



    # --- PREPROCESSING ───────────────────────────────────────

    all_data = pd.get_dummies(all_data)
    print(all_data.shape)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train = all_data[:ntrain]
    test = all_data[ntrain:]



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

            reg_setup = setup(data=train, target='SalePrice', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for House Price prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
