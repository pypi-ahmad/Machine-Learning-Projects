#!/usr/bin/env python3
"""
Full pipeline for Traffic Congestion Prediction

Auto-generated from: traffic_prediction.ipynb
Project: Traffic Congestion Prediction
Category: Classification | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import tensorflow
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras import callbacks
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")
# Additional imports extracted from mixed cells
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
# Normalize Function
def Normalize(df,col):
    average = df[col].mean()
    stdev = df[col].std()
    df_normalized = (df[col] - average) / stdev
    df_normalized = df_normalized.to_frame()
    return df_normalized, average, stdev

# Differencing Function
def Difference(df,col, interval):
    diff = []
    for i in range(interval, len(df)):
        value = df[col][i] - df[col][i - interval]
        diff.append(value)
    return diff
#Stationary Check for the time series Augmented Dickey Fuller test
def Stationary_check(df):
    check = adfuller(df.dropna())
    print(f"ADF Statistic: {check[0]}")
    print(f"p-value: {check[1]}")
    print("Critical Values:")
    for key, value in check[4].items():
        print('\t%s: %.3f' % (key, value))
    if check[0] > check[4]["1%"]:
        print("Time Series is Non-Stationary")
    else:
        print("Time Series is Stationary") 
  

#Checking if the series is stationary

List_df_ND = [ df_N1["Diff"], df_N2["Diff"], df_N3["Diff"], df_N4["Diff"]] 
print("Checking the transformed series for stationarity:")
for i in List_df_ND:
    print("\n")
    Stationary_check(i)

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    """Run the complete pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)

    # --- DATA LOADING ────────────────────────────────────────

    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    # Input data files are available in the read-only "../input/" directory
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

    import os
    for dirname, _, filenames in os.walk('./archive/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

    #Loading Data
    data = load_dataset('traffic_congestion_prediction')
    data.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    data["DateTime"]= pd.to_datetime(data["DateTime"])
    data = data.drop(["ID"], axis=1) #dropping IDs
    data.info()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #df to be used for EDA
    df=data.copy() 
    #Let's plot the Timeseries
    colors = [ "#FFD4DB","#BBE7FE","#D3B5E5","#dfe2b6"]
    plt.figure(figsize=(20,4),facecolor="#627D78")
    Time_series=sns.lineplot(x=df['DateTime'],y="Vehicles",data=df, hue="Junction", palette=colors)
    Time_series.set_title("Traffic On Junctions Over Years")
    Time_series.set_ylabel("Number of Vehicles")
    Time_series.set_xlabel("Date")

    #Exploring more features 
    df["Year"]= df['DateTime'].dt.year
    df["Month"]= df['DateTime'].dt.month
    df["Date_no"]= df['DateTime'].dt.day
    df["Hour"]= df['DateTime'].dt.hour
    df["Day"]= df.DateTime.dt.strftime("%A")
    df.head()

    #Let's plot the Timeseries
    new_features = [ "Year","Month", "Date_no", "Hour", "Day"]

    for i in new_features:
        plt.figure(figsize=(10,2),facecolor="#627D78")
        ax=sns.lineplot(x=df[i],y="Vehicles",data=df, hue="Junction", palette=colors )
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.figure(figsize=(12,5),facecolor="#627D78")
    count = sns.countplot(data=df, x =df["Year"], hue="Junction", palette=colors)
    count.set_title("Count Of Traffic On Junctions Over Years")
    count.set_ylabel("Number of Vehicles")
    count.set_xlabel("Date")



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    corrmat = df.corr()
    plt.subplots(figsize=(10,10),facecolor="#627D78")
    sns.heatmap(corrmat,cmap= "Pastel2",annot=True,square=True, )

    sns.pairplot(data=df, hue= "Junction",palette=colors)

    #Pivoting data fron junction 
    df_J = data.pivot(columns="Junction", index="DateTime")
    df_J.describe()



    # --- PREPROCESSING ───────────────────────────────────────

    #Creating new sets
    df_1 = df_J[[('Vehicles', 1)]]
    df_2 = df_J[[('Vehicles', 2)]]
    df_3 = df_J[[('Vehicles', 3)]]
    df_4 = df_J[[('Vehicles', 4)]]
    df_4 = df_4.dropna() #Junction 4 has limited data only for a few months

    #Dropping level one in dfs's index as it is a multi index data frame
    list_dfs = [df_1, df_2, df_3, df_4]
    for i in list_dfs:
        i.columns= i.columns.droplevel(level=1)   

    #Function to plot comparitive plots of dataframes
    def Sub_Plots4(df_1, df_2,df_3,df_4,title):
        fig, axes = plt.subplots(4, 1, figsize=(15, 8),facecolor="#627D78", sharey=True)
        fig.suptitle(title)
        #J1
        pl_1=sns.lineplot(ax=axes[0],data=df_1,color=colors[0])
        #pl_1=plt.ylabel()
        axes[0].set(ylabel ="Junction 1")
        #J2
        pl_2=sns.lineplot(ax=axes[1],data=df_2,color=colors[1])
        axes[1].set(ylabel ="Junction 2")
        #J3
        pl_3=sns.lineplot(ax=axes[2],data=df_3,color=colors[2])
        axes[2].set(ylabel ="Junction 3")
        #J4
        pl_4=sns.lineplot(ax=axes[3],data=df_4,color=colors[3])
        axes[3].set(ylabel ="Junction 4")
    
    
    #Plotting the dataframe to check for stationarity
    Sub_Plots4(df_1.Vehicles, df_2.Vehicles,df_3.Vehicles,df_4.Vehicles,"Dataframes Before Transformation")



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Normalizing and Differencing to make the series stationary 
    df_N1, av_J1, std_J1 = Normalize(df_1, "Vehicles")
    Diff_1 = Difference(df_N1, col="Vehicles", interval=(24*7)) #taking a week's diffrence
    df_N1 = df_N1[24*7:]
    df_N1.columns = ["Norm"]
    df_N1["Diff"]= Diff_1

    df_N2, av_J2, std_J2 = Normalize(df_2, "Vehicles")
    Diff_2 = Difference(df_N2, col="Vehicles", interval=(24)) #taking a day's diffrence
    df_N2 = df_N2[24:]
    df_N2.columns = ["Norm"]
    df_N2["Diff"]= Diff_2

    df_N3, av_J3, std_J3 = Normalize(df_3, "Vehicles")
    Diff_3 = Difference(df_N3, col="Vehicles", interval=1) #taking an hour's diffrence
    df_N3 = df_N3[1:]
    df_N3.columns = ["Norm"]
    df_N3["Diff"]= Diff_3

    df_N4, av_J4, std_J4 = Normalize(df_4, "Vehicles")
    Diff_4 = Difference(df_N4, col="Vehicles", interval=1) #taking an hour's diffrence
    df_N4 = df_N4[1:]
    df_N4.columns = ["Norm"]
    df_N4["Diff"]= Diff_4

    Sub_Plots4(df_N1.Diff, df_N2.Diff,df_N3.Diff,df_N4.Diff,"Dataframes After Transformation")



    # --- PREPROCESSING ───────────────────────────────────────

    #Differencing created some NA values as we took a weeks data into consideration while difrencing
    df_J1 = df_N1["Diff"].dropna()
    df_J1 = df_J1.to_frame()

    df_J2 = df_N2["Diff"].dropna()
    df_J2 = df_J2.to_frame()

    df_J3 = df_N3["Diff"].dropna()
    df_J3 = df_J3.to_frame()

    df_J4 = df_N4["Diff"].dropna()
    df_J4 = df_J4.to_frame()

    #Splitting the dataset
    def Split_data(df):
        training_size = int(len(df)*0.90)
        data_len = len(df)
        train, test = df[0:training_size],df[training_size:data_len] 
        train, test = train.values.reshape(-1, 1), test.values.reshape(-1, 1)
        return train, test
    #Splitting the training and test datasets 
    J1_train, J1_test = Split_data(df_J1)
    J2_train, J2_test = Split_data(df_J2)
    J3_train, J3_test = Split_data(df_J3)
    J4_train, J4_test = Split_data(df_J4)

    #Target and Feature
    def TnF(df):
        end_len = len(df)
        X = []
        y = []
        steps = 32
        for i in range(steps, end_len):
            X.append(df[i - steps:i, 0])
            y.append(df[i, 0])
        X, y = np.array(X), np.array(y)
        return X ,y

    #fixing the shape of X_test and X_train
    def FeatureFixShape(train, test):
        train = np.reshape(train, (train.shape[0], train.shape[1], 1))
        test = np.reshape(test, (test.shape[0],test.shape[1],1))
        return train, test

    #Assigning features and target 
    X_trainJ1, y_trainJ1 = TnF(J1_train)
    X_testJ1, y_testJ1 = TnF(J1_test)
    X_trainJ1, X_testJ1 = FeatureFixShape(X_trainJ1, X_testJ1)

    X_trainJ2, y_trainJ2 = TnF(J2_train)
    X_testJ2, y_testJ2 = TnF(J2_test)
    X_trainJ2, X_testJ2 = FeatureFixShape(X_trainJ2, X_testJ2)

    X_trainJ3, y_trainJ3 = TnF(J3_train)
    X_testJ3, y_testJ3 = TnF(J3_test)
    X_trainJ3, X_testJ3 = FeatureFixShape(X_trainJ3, X_testJ3)

    X_trainJ4, y_trainJ4 = TnF(J4_train)
    X_testJ4, y_testJ4 = TnF(J4_test)
    X_trainJ4, X_testJ4 = FeatureFixShape(X_trainJ4, X_testJ4)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Traffic Congestion Prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
