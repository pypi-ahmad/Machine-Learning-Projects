#!/usr/bin/env python3
"""
Model training for Prediction Future Sales

Auto-generated from: sales_forecasting.ipynb
Project: Prediction Future Sales
Category: Regression | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np      # To use np.arrays
import pandas as pd     # To use dataframes
from pandas.plotting import autocorrelation_plot as auto_corr

# To plot
import matplotlib.pyplot as plt  
import matplotlib as mpl
import seaborn as sns

#For date-time
import math
from datetime import datetime
from datetime import timedelta

# Another imports if needs
import itertools
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose as season
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose
from pmdarima import auto_arima


import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns=100 # to see columns
pd.options.display.max_columns=100 # to see columns
# Additional imports extracted from mixed cells
import os
from sklearn.preprocessing import RobustScaler

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def wmae_test(test, pred): # WMAE for test 
    weights = X_test['IsHoliday'].apply(lambda is_holiday:5 if is_holiday else 1)
    error = np.sum(weights * np.abs(test - pred), axis=0) / np.sum(weights)
    return error

# ======================================================================
# TRAINING PIPELINE
# ======================================================================

def main():
    """Run the training pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    # Load unified Walmart Sales dataset (substitute for original 3-file dataset)
    df_combined = load_dataset('prediction_future_sales')

    # Parse dates (CSV uses DD-MM-YYYY format) and convert to ISO format string for downstream compatibility
    df_combined['Date'] = pd.to_datetime(df_combined['Date'], dayfirst=True).dt.strftime('%Y-%m-%d')

    # Rename Holiday_Flag -> IsHoliday for compatibility with downstream code
    df_combined.rename(columns={'Holiday_Flag': 'IsHoliday'}, inplace=True)
    df_combined['IsHoliday'] = df_combined['IsHoliday'].astype(bool)

    # Create missing columns with sensible defaults
    df_combined['Dept'] = 1  # Single department (original had 81)
    df_combined['Type'] = 'A'  # Store type
    df_combined['Size'] = 150000  # Approximate median store size
    for md in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
        df_combined[md] = 0.0

    # Create the expected sub-DataFrames for compatibility with .head() display cells
    df_store = df_combined[['Store', 'Type', 'Size']].drop_duplicates().reset_index(drop=True)
    df_train = df_combined[['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']]
    df_features = df_combined[['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1',
                                'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                                'CPI', 'Unemployment', 'IsHoliday']]

    # Validation
    print("Combined shape:", df_combined.shape)
    print("Columns:", df_combined.columns.tolist())
    print("Date sample:", df_combined['Date'].iloc[:3].tolist())
    df_store.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Data already combined from single CSV; use it directly
    df = df_combined.copy()
    df.head(5)

    # No duplicate IsHoliday columns in unified dataset - skip
    pass

    # IsHoliday already renamed from Holiday_Flag during loading
    pass

    store_dept_table = pd.pivot_table(df, index='Store', columns='Dept',
                                      values='Weekly_Sales', aggfunc=np.mean)
    display(store_dept_table)

    df.loc[df['Weekly_Sales']<=0]

    df = df.loc[df['Weekly_Sales'] > 0]

    df_holiday = df.loc[df['IsHoliday']==True]
    df_holiday['Date'].unique()

    df_not_holiday = df.loc[df['IsHoliday']==False]
    df_not_holiday['Date'].nunique()

    # Super bowl dates in train set
    df.loc[(df['Date'] == '2010-02-12')|(df['Date'] == '2011-02-11')|(df['Date'] == '2012-02-10'),'Super_Bowl'] = True
    df.loc[(df['Date'] != '2010-02-12')&(df['Date'] != '2011-02-11')&(df['Date'] != '2012-02-10'),'Super_Bowl'] = False

    # Labor day dates in train set
    df.loc[(df['Date'] == '2010-09-10')|(df['Date'] == '2011-09-09')|(df['Date'] == '2012-09-07'),'Labor_Day'] = True
    df.loc[(df['Date'] != '2010-09-10')&(df['Date'] != '2011-09-09')&(df['Date'] != '2012-09-07'),'Labor_Day'] = False

    # Thanksgiving dates in train set
    df.loc[(df['Date'] == '2010-11-26')|(df['Date'] == '2011-11-25'),'Thanksgiving'] = True
    df.loc[(df['Date'] != '2010-11-26')&(df['Date'] != '2011-11-25'),'Thanksgiving'] = False

    #Christmas dates in train set
    df.loc[(df['Date'] == '2010-12-31')|(df['Date'] == '2011-12-30'),'Christmas'] = True
    df.loc[(df['Date'] != '2010-12-31')&(df['Date'] != '2011-12-30'),'Christmas'] = False

    df.groupby(['Christmas','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Christmas

    df.groupby(['Labor_Day','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Labor Day

    df.groupby(['Thanksgiving','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Thanksgiving

    df.groupby(['Super_Bowl','Type'])['Weekly_Sales'].mean()  # Avg weekly sales for types on Super Bowl

    my_data = [48.88, 37.77 , 13.33 ]  #percentages
    my_labels = 'Type A','Type B', 'Type C' # labels
    plt.pie(my_data,labels=my_labels,autopct='%1.1f%%', textprops={'fontsize': 15}) #plot pie type and bigger the labels
    plt.axis('equal')
    mpl.rcParams.update({'font.size': 20}) #bigger percentage labels

    plt.show()

    df.groupby('IsHoliday')['Weekly_Sales'].mean()

    # Plotting avg wekkly sales according to holidays by types
    plt.style.use('seaborn-poster')
    labels = ['Thanksgiving', 'Super_Bowl', 'Labor_Day', 'Christmas']
    A_means = [27397.77, 20612.75, 20004.26, 18310.16]
    B_means = [18733.97, 12463.41, 12080.75, 11483.97]
    C_means = [9696.56,10179.27,9893.45,8031.52]

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(16, 8))
    rects1 = ax.bar(x - width, A_means, width, label='Type_A')
    rects2 = ax.bar(x , B_means, width, label='Type_B')
    rects3 = ax.bar(x + width, C_means, width, label='Type_C')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Weekly Avg Sales')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.axhline(y=17094.30,color='r') # holidays avg
    plt.axhline(y=15952.82,color='green') # not-holiday avg

    fig.tight_layout()

    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    df = df.fillna(0) # filling null's with 0



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    x = df_combined['Dept']
    y = df_combined['Weekly_Sales']
    plt.figure(figsize=(15,5))
    plt.title('Weekly Sales by Department')
    plt.xlabel('Departments')
    plt.ylabel('Weekly Sales')
    plt.scatter(x,y)
    plt.show()

    x = df_combined['Store']
    y = df_combined['Weekly_Sales']
    plt.figure(figsize=(15,5))
    plt.title('Weekly Sales by Store')
    plt.xlabel('Stores')
    plt.ylabel('Weekly Sales')
    plt.scatter(x,y)
    plt.show()

    df["Date"] = pd.to_datetime(df["Date"]) # convert to datetime
    df['week'] =df['Date'].dt.week
    df['month'] =df['Date'].dt.month 
    df['year'] =df['Date'].dt.year

    df.groupby('month')['Weekly_Sales'].mean() # to see the best months for sales

    df.groupby('year')['Weekly_Sales'].mean() # to see the best years for sales

    fuel_price = pd.pivot_table(df, values = "Weekly_Sales", index= "Fuel_Price")
    fuel_price.plot()

    temp = pd.pivot_table(df, values = "Weekly_Sales", index= "Temperature")
    temp.plot()

    CPI = pd.pivot_table(df, values = "Weekly_Sales", index= "CPI")
    CPI.plot()

    unemployment = pd.pivot_table(df, values = "Weekly_Sales", index= "Unemployment")
    unemployment.plot()

    df.to_csv('clean_data.csv') # assign new data frame to csv for using after here



    # --- DATA LOADING ────────────────────────────────────────

    # Reload from saved clean_data.csv, or fall back to in-memory df
    import os
    if os.path.exists('./clean_data.csv'):
        df = pd.read_csv('./clean_data.csv')
    else:
        print("clean_data.csv not found; using in-memory df from above")
        # df already exists from prior cells



    # --- FEATURE ENGINEERING ─────────────────────────────────

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df['Date'] = pd.to_datetime(df['Date']) # changing datetime to divide if needs

    df_encoded = df.copy() # to keep original dataframe taking copy of it



    # --- FEATURE ENGINEERING ─────────────────────────────────

    type_group = {'A':1, 'B': 2, 'C': 3}  # changing A,B,C to 1-2-3
    df_encoded['Type'] = df_encoded['Type'].replace(type_group)

    df_encoded['Super_Bowl'] = df_encoded['Super_Bowl'].astype(bool).astype(int) # changing T,F to 0-1

    df_encoded['Thanksgiving'] = df_encoded['Thanksgiving'].astype(bool).astype(int) # changing T,F to 0-1

    df_encoded['Labor_Day'] = df_encoded['Labor_Day'].astype(bool).astype(int) # changing T,F to 0-1

    df_encoded['Christmas'] = df_encoded['Christmas'].astype(bool).astype(int) # changing T,F to 0-1

    df_encoded['IsHoliday'] = df_encoded['IsHoliday'].astype(bool).astype(int) # changing T,F to 0-1



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df_new = df_encoded.copy() # taking the copy of encoded df to keep it original



    # --- FEATURE ENGINEERING ─────────────────────────────────

    drop_col = ['Super_Bowl','Labor_Day','Thanksgiving','Christmas']
    df_new.drop(drop_col, axis=1, inplace=True) # dropping columns

    drop_col = ['Temperature','MarkDown4','MarkDown5','CPI','Unemployment']
    df_new.drop(drop_col, axis=1, inplace=True) # dropping columns



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df_new = df_new.sort_values(by='Date', ascending=True) # sorting according to date

    train_data = df_new[:int(0.7*(len(df_new)))] # taking train part
    test_data = df_new[int(0.7*(len(df_new))):] # taking test part

    target = "Weekly_Sales"
    used_cols = [c for c in df_new.columns.to_list() if c not in [target]] # all columns except weekly sales

    X_train = train_data[used_cols]
    X_test = test_data[used_cols]
    y_train = train_data[target]
    y_test = test_data[target]

    X = df_new[used_cols] # to keep train and test X values together



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X_train = X_train.drop(['Date'], axis=1) # dropping date from train
    X_test = X_test.drop(['Date'], axis=1) # dropping date from test



    # --- MODEL TRAINING ──────────────────────────────────────

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                               max_features = 'sqrt',min_samples_split = 10)

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()



    #making pipe tp use scaler and regressor together
    pipe = make_pipeline(scaler,rf)

    pipe.fit(X_train, y_train)

    # predictions on train set
    y_pred = pipe.predict(X_train)

    # predictions on test set
    y_pred_test = pipe.predict(X_test)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    wmae_test(y_test, y_pred_test)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X = X.drop(['Date'], axis=1) #dropping date column from X



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Printing the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plotting the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X1_train = X_train.drop(['month'], axis=1) # dropping month
    X1_test = X_test.drop(['month'], axis=1)



    # --- MODEL TRAINING ──────────────────────────────────────

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                               max_features = 'sqrt',min_samples_split = 10)

    scaler=RobustScaler()
    pipe = make_pipeline(scaler,rf)

    pipe.fit(X1_train, y_train)

    # predictions on train set
    y_pred = pipe.predict(X1_train)

    # predictions on test set
    y_pred_test = pipe.predict(X1_test)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    wmae_test(y_test, y_pred_test)

    # splitting train-test to whole dataset
    train_data_enc = df_encoded[:int(0.7*(len(df_encoded)))]
    test_data_enc = df_encoded[int(0.7*(len(df_encoded))):]

    target = "Weekly_Sales"
    used_cols1 = [c for c in df_encoded.columns.to_list() if c not in [target]] # all columns except price

    X_train_enc = train_data_enc[used_cols1]
    X_test_enc = test_data_enc[used_cols1]
    y_train_enc = train_data_enc[target]
    y_test_enc = test_data_enc[target]

    X_enc = df_encoded[used_cols1] # to get together train,test splits



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X_enc = X_enc.drop(['Date'], axis=1) #dropping date column for whole X

    X_train_enc = X_train_enc.drop(['Date'], axis=1) # dropping date from train and test
    X_test_enc= X_test_enc.drop(['Date'], axis=1)



    # --- MODEL TRAINING ──────────────────────────────────────

    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=35,
                               max_features = 'sqrt',min_samples_split = 10)

    scaler=RobustScaler()
    pipe = make_pipeline(scaler,rf)

    pipe.fit(X_train_enc, y_train_enc)

    # predictions on train set
    y_pred_enc = pipe.predict(X_train_enc)

    # predictions on test set
    y_pred_test_enc = pipe.predict(X_test_enc)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    wmae_test(y_test_enc, y_pred_test_enc)

    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Printing the feature ranking
    print("Feature ranking:")

    for f in range(X_enc.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plotting the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_enc.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_enc.shape[1]), indices)
    plt.xlim([-1, X_enc.shape[1]])
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_encoded_new = df_encoded.copy() # taking copy of encoded data to keep it without change.
    df_encoded_new.drop(drop_col, axis=1, inplace=True)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model training for Prediction Future Sales")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
