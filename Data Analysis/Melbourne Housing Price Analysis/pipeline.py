#!/usr/bin/env python3
"""
Full pipeline for Melbourne Housing Price Analysis

Auto-generated from: code.ipynb
Project: Melbourne Housing Price Analysis
Category: Data Analysis | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# data operation libraries
import numpy as np
import pandas as pd

# for datetime operation
import datetime as dt

# visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

# stats for Q-Q plot
import scipy.stats as stats

# general pandas setting
pd.options.display.max_columns = None
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from feature_engine.selection import SmartCorrelatedSelection
# for car and CouncilArea missing values are less so I will use random
# sample imputation

from feature_engine.imputation import RandomSampleImputer
from feature_engine.encoding import RareLabelEncoder
from feature_engine.encoding import MeanEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from feature_engine.selection import RecursiveFeatureElimination
# Additional imports extracted from mixed cells
from lazypredict.Supervised import LazyRegressor
from pycaret.regression import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
# I will create a function which will help in identifying
# distribution and outliers at the sametime

def diagnostic_plot(data, var):
    fig = plt.figure(figsize=(15,4))

    plt.subplot(1,3,1)
    data[var].hist(bins=50)
    plt.title('Distribution of {}'.format(var))

    plt.subplot(1,3,2)
    stats.probplot(data[var], dist='norm', plot=plt)
    plt.ylabel('RM Quantiles')

    plt.subplot(1,3,3)
    sns.boxplot(y=data[var])
    plt.title('Boxplot')

    plt.show()

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

    data = load_dataset('melbourne_housing_price_analysis')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.head()

    data.tail()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # correcting the Date column
    # we could have directly pass the parse date variable while importing the file
    # but we have some impurities in our original date column which pandas is not
    # to understand so it interprets day as month and month as date

    data[['Day','Month','Year']] = data['Date'].str.split('/', expand=True)

    data['Date'] = data['Year'] + '/' + data['Month'] + '/' + data['Day']

    # overwriting the existing date
    data['Date'] = pd.to_datetime(data['Date'])

    # dropping the helping columns
    data.drop(labels=['Day','Month', 'Year'], axis=1, inplace=True)

    # creating some useful varible for our future analysis
    data['Month_name'] = data['Date'].dt.month_name()

    data['day'] = data['Date'].dt.day

    data['Year'] = data['Date'].dt.year



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.info()

    data.nunique()

    sns.histplot(data=data, x=data.Price, kde=True)
    plt.title('Distribution of Price Variable')
    plt.show()

    plt.figure(figsize=(10,5))
    data[data['Type']=='h']['Price'].hist(alpha=0.5, color='blue',bins=30, label='Type h')
    data[data['Type']=='u']['Price'].hist(alpha=0.5, color='red',bins=30, label='Type u')
    data[data['Type']=='t']['Price'].hist(alpha=0.5, color='yellow',bins=30, label='Type t')
    plt.ylabel('Price in Millions')
    plt.xlabel('Number of observations')
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(25,15))
    sns.lmplot(y='Price', x='Distance', data=data, row='Method', col='Year', palette='gist_earth')
    plt.show()

    sns.lmplot(data=data, x='Rooms',y='Price', fit_reg=True, col='Year')
    plt.show()

    sns.lmplot(data=data, x='Bedroom2',y='Price', fit_reg=True, col='Year')
    plt.show()

    sns.lmplot(data=data, x='Bathroom',y='Price', fit_reg=True, col='Year')
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    year_2016_df = data[data['Year']==2016]
    year_2017_df = data[data['Year']==2017]



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    fig = plt.figure(figsize=(25,18))

    plt.subplot(2,1,1)
    sns.boxplot(x='Regionname',y='Price', data=year_2016_df)
    plt.title('Region Name Vs Price for year 2016')

    plt.subplot(2,1,2)
    sns.boxplot(x='Regionname',y='Price', data=year_2017_df)
    plt.title('Region Name Vs Price for year 2017')

    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    grouped_df = data.groupby(['Year','Month_name', 'Suburb'])
    ym_df = grouped_df.agg({'Propertycount': 'min',
                   'Price':'mean'})
    ym_df.reset_index(inplace=True)
    ym_df.rename(columns={'Propertycount':'Total_Property', 'Price':'Mean_price'}, inplace=True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    ym_df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df_2016 = ym_df[ym_df['Year']==2016]
    df_2017 = ym_df[ym_df['Year']==2017]



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    fig = plt.figure(figsize=(25,10))

    plt.subplot(1,2,1)
    sns.barplot(x='Month_name',y='Total_Property', data=df_2016)
    plt.title('Total Property count for Year 2016')

    plt.subplot(1,2,2)
    sns.lineplot(x='Month_name',y='Mean_price', data=df_2016)
    plt.title('Mean Price for Year 2016')

    plt.show()

    fig = plt.figure(figsize=(25,10))

    plt.subplot(1,2,1)
    sns.barplot(x='Month_name',y='Total_Property', data=df_2017)
    plt.title('Total Property count for Year 2016')

    plt.subplot(1,2,2)
    sns.lineplot(x='Month_name',y='Mean_price', data=df_2017)
    plt.title('Mean Price for Year 2016')

    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    suburb_gp = data.groupby(['Year','Suburb'])
    suburb_df = suburb_gp.agg({'Price': sum})

    suburb_df.loc[(2016,),:].nlargest(5,'Price').style.background_gradient(cmap='Spectral', subset=pd.IndexSlice[:, 'Price'])

    suburb_df.loc[(2017,),:].nlargest(5,'Price').style.background_gradient(cmap='Spectral', subset=pd.IndexSlice[:, 'Price'])

    suburb_df.loc[(2016,),:].nsmallest(5,'Price').style.background_gradient(cmap='Spectral', subset=pd.IndexSlice[:, 'Price'])

    suburb_df.loc[(2017,),:].nsmallest(5,'Price').style.background_gradient(cmap='Spectral', subset=pd.IndexSlice[:, 'Price'])



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.head() #let's again remind ourselves about available variables

    print('There are {} observations in the dataset'.format(len(data)))
    print('\n')
    print('There are {} unique observations in Address variable'.format(data['Address'].nunique()))
    print('There are {} unique observations in Postcode variable'.format(data['Postcode'].nunique()))



    # --- FEATURE ENGINEERING ─────────────────────────────────

    data.drop(labels=['Address','Date','Lattitude','Longtitude'],axis=1, inplace=True)

    x = data.drop('Price', axis=1)
    y = data['Price']



    # --- PREPROCESSING ───────────────────────────────────────

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # to determine co-linearity, we evaluate the correlation of independent variables

    # we calculate the correlations using pandas corr()
    # and we round the values to 2 decimals
    correlation_matrix = np.round(x_train.corr(),2)

    # plot the correlation matric using seaborn
    # we use annot = true to print the correlation values

    fig = plt.figure(figsize=(10,10))
    sns.heatmap(data=correlation_matrix, annot=True)



    # --- MODEL TRAINING ──────────────────────────────────────

    correlated = SmartCorrelatedSelection(
        variables= None,
        method = 'pearson',
        threshold = 0.8,
        missing_values = 'ignore',
        selection_method = 'variance',
        estimator=None,
        scoring = 'roc_auc',
        cv=3)

    correlated.fit(x_train)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    correlated.features_to_drop_

    x_train = correlated.transform(x_train)
    x_test = correlated.transform(x_test)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # make a list of numeric variables

    descrete_var = [var for var in x_train.columns if x_train[var].dtypes != 'O' and x_train[var].nunique() < 30]
    continuous_var = [var for var in x_train.columns if x_train[var].dtypes != 'O' and var not in descrete_var]

    categorical_var = [var for var in x_train.columns if x_train[var].dtypes == 'O']

    print('There are {} descrete variables'.format(len(descrete_var)))
    print('There are {} continuous variables'.format(len(continuous_var)))
    print('There are {} categorical variables'.format(len(categorical_var)))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    descrete_var

    continuous_var

    categorical_var



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    x_train.isnull().mean()

    x_test.isnull().mean()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # cardinality in categorical variables

    for var in categorical_var:
        print('Number of labels in {}: {}'.format(var, x_train[var].nunique()))

    for var in continuous_var:
        diagnostic_plot(x_train, var)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # as the amount of missing values are higher in BuildingArea and YearBuilt
    #  variables I will drop them because using them for model may cause random
    #  prediction because the values we may impute will not be accurate

    x_train.drop(columns=['BuildingArea','YearBuilt'], axis=1, inplace=True)
    x_test.drop(columns=['BuildingArea','YearBuilt'], axis=1, inplace=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    continuous_var.remove('BuildingArea')
    continuous_var.remove('YearBuilt')



    # --- MODEL TRAINING ──────────────────────────────────────

    random_imputer = RandomSampleImputer(variables=['Car', 'CouncilArea'])

    random_imputer.fit(x_train)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    x_train = random_imputer.transform(x_train)
    x_test = random_imputer.transform(x_test)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    x_train.isnull().mean()

    x_test.isnull().mean()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    continuous_var.remove('Postcode')
    descrete_var.append('Postcode')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # I will treat discreate variable as categorical variable so later I can encode

    x_train[descrete_var] = x_train[descrete_var].astype('O')
    x_test[descrete_var] = x_test[descrete_var].astype('O')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for var in descrete_var:
        print(x_train.groupby([var])[var].count()/len(x_train))

    for var in categorical_var:
        print(x_train.groupby([var])[var].count()/len(x_train))



    # --- MODEL TRAINING ──────────────────────────────────────

    rare_encoder = RareLabelEncoder(
                    tol = 0.05,
                    n_categories = 1,
                    max_n_categories = None,
                    replace_with = 'Rare',
                    variables = ['Type', 'Method', 'SellerG', 'Regionname', 'Month_name', 'Bathroom', 'day', 'Bedroom2','Year', 'Car','CouncilArea'] )

    rare_encoder.fit(x_train)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    x_train = rare_encoder.transform(x_train)
    x_test = rare_encoder.transform(x_test)



    # --- MODEL TRAINING ──────────────────────────────────────

    # as Suburb and Postcode are high cardinal variables I am taking lower tolerance

    suburb_postcode_rare_encoder = RareLabelEncoder(tol = 0.02,
                    n_categories = 1,
                    max_n_categories = None,
                    replace_with = 'Rare',
                    variables = ['Suburb','Postcode'])

    suburb_postcode_rare_encoder.fit(x_train)

    x_train = suburb_postcode_rare_encoder.transform(x_train)
    x_test = suburb_postcode_rare_encoder.transform(x_test)

    mean_encoder = MeanEncoder(variables=categorical_var+descrete_var)

    mean_encoder.fit(x_train,y_train)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    x_train = mean_encoder.transform(x_train)
    x_test = mean_encoder.transform(x_test)



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

            reg_setup = setup(data=data, target='Price', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Melbourne Housing Price Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
