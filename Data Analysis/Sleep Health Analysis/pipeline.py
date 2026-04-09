#!/usr/bin/env python3
"""
Full pipeline for Sleep Health Analysis

Auto-generated from: code.ipynb
Project: Sleep Health Analysis
Category: Data Analysis | Task: data_analysis
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
# Additional imports extracted from mixed cells
import matplotlib.pyplot as plt
import seaborn as sns

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

    # --- DATA LOADING ────────────────────────────────────────

    df=load_dataset('sleep_health_analysis')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.info()

    print('Unique Values of Occupation are', df['Occupation'].unique())

    print('Unique Values of BMI Category are', df['BMI Category'].unique())

    print('Unique Values of Sleep Disorder are', df['Sleep Disorder'].unique())



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df['Blood Pressure'].unique()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df1 = pd.concat([df, df['Blood Pressure'].str.split('/', expand=True)], axis=1).drop(
        'Blood Pressure', axis=1)

    df1=df1.rename(columns={0: 'BloodPressure_high', 1: 'BloodPressure_low'})

    df1['BloodPressure_high'] = df1['BloodPressure_high'].astype(float)
    df1['BloodPressure_low'] = df1['BloodPressure_low'].astype(float)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df1.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    plt.figure(figsize=(10,6))
    sns.heatmap(df1.drop('Person ID',axis=1).corr(),annot=True,fmt="1.1f");

    sns.pairplot(df1.drop('Person ID',axis=1),hue='Sleep Disorder');



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    num_col=['Age','Sleep Duration',
           'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
           'Heart Rate', 'Daily Steps',
           'BloodPressure_high', 'BloodPressure_low']

    cat_col=['Gender','Occupation','BMI Category','Sleep Disorder']

    fig = plt.figure(figsize=(10,10))

    for i in range(len(num_col)):
        plt.subplot(3,3,i+1)
        plt.title(num_col[i])
        sns.histplot(data=df1,x=df1[num_col[i]],hue='Sleep Disorder')
        plt.legend(fontsize=7,labels=df['Sleep Disorder'].unique())
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10,10))

    for i in range(len(num_col)):
        plt.subplot(3,3,i+1)
        plt.title(num_col[i])
        sns.histplot(data=df1,x=df1[num_col[i]],hue='BMI Category')
        plt.legend(labels=df['BMI Category'].unique(),fontsize=6)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(8,8))

    for i in range(len(num_col)):
        plt.subplot(3,3,i+1)
        plt.title(num_col[i])
        sns.boxplot(data=df1,y=df1['Gender'],x=df1[num_col[i]])
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(15,8))
    for i in range(len(num_col)):
        plt.subplot(3,3,i+1)
        plt.title(num_col[i])
        sns.boxplot(data=df1,y=df1['Occupation'],x=df1[num_col[i]])
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(15,8))

    for i in range(len(num_col)):
        plt.subplot(3,3,i+1)
        plt.title(num_col[i])
        sns.boxplot(data=df1,y=df1['BMI Category'],x=df1[num_col[i]])
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(8,8))

    for i in range(len(num_col)):
        plt.subplot(3,3,i+1)
        plt.title(num_col[i])
        sns.boxplot(data=df1,y=df1['Sleep Disorder'],x=df1[num_col[i]])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.legend(fontsize=10)
    plt.tick_params(labelsize=10)
    ax=sns.scatterplot(x=df1['Age'],y=df1['Sleep Duration'],hue=df1['BMI Category'],data=df1,sizes=(50,500))
    plt.xticks(rotation=90)
    ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    x_lim = [25,60]
    y_lim = [5.5,8.5]
    plt.plot(x_lim, y_lim,color="red");



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df1=df1.replace({'BMI Category': {'Normal': 0,'Normal Weight':1,'Overweight':2,'Obese':3}})



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    plt.figure(figsize=(5, 5))
    plt.legend(fontsize=10)
    plt.tick_params(labelsize=10)
    ax=sns.scatterplot(x=df1['Age'],y=df1['BMI Category'],hue=df1['Sleep Duration'],data=df1,sizes=(50,500))
    plt.xticks(rotation=90)
    ax.legend(loc='upper left',bbox_to_anchor=(1,1))
    x_lim = [25,60]
    y_lim = [0,3]
    plt.plot(x_lim, y_lim,color="red");



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df1['Age'].describe()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df1['Age_bin'] = pd.cut(df1['Age'],[0, 30, 40, 50,60], labels=False)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df1

    df1.groupby('Age_bin')['BMI Category'].mean().plot.line();

    df1.groupby('Age_bin')['Sleep Duration'].mean().plot.line();



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    sns.boxplot(data=df1,x=df1['Age_bin'],y=df1['BMI Category']);

    sns.boxplot(data=df1,x=df1['Age_bin'],y=df1['Sleep Duration']);

    sns.boxplot(data=df1,x=df1['Occupation'],y=df1['Age_bin'])
    plt.xticks(rotation=45);

    sns.boxplot(data=df1,x=df1['Occupation'],y=df1['BMI Category'])
    plt.xticks(rotation=45);

    sns.boxplot(data=df1,x=df1['Occupation'],y=df1['Sleep Duration'])
    plt.xticks(rotation=45);



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Summary statistics
    df.describe(include='all')

    # Correlation matrix for numeric columns
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_df = df.select_dtypes(include='number')
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Sleep Health Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
