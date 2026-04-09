#!/usr/bin/env python3
"""
Full pipeline for 10 clustering vehicle crash data for safety analysi

Auto-generated from: 10 Clustering vehicle crash data.ipynb
Project: 10 clustering vehicle crash data for safety analysi
Category: Clustering | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
# Additional imports extracted from mixed cells
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from imblearn.over_sampling import SMOTE
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

    #load and read the file
    df=load_dataset('clustering_vehicle_crash_data_for_safety_analysi')
    df.head()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #shape/ size of the data
    df.shape

    #checking the numerical statistics of the data
    df.describe()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df.describe(include="all")



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #checking data types of each columns
    df.info()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #finding duplicate values
    df.duplicated().sum()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #Distribution of Accident severity
    df['Accident_severity'].value_counts()

    #plotting the final class
    sns.countplot(x = df['Accident_severity'])
    plt.title('Distribution of Accident severity')

    #checking missing values
    df.isna().sum()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #dropping columns which has more than 2500 missing values and Time column
    df.drop(['Service_year_of_vehicle','Defect_of_vehicle','Work_of_casuality', 'Fitness_of_casuality','Time'], axis = 1, inplace = True)
    df.head()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #storing categorical column names to a new variable
    categorical=[i for i in df.columns if df[i].dtype=='O']
    print('The categorical variables are',categorical)



    # --- PREPROCESSING ───────────────────────────────────────

    #for categorical values we can replace the null values with the Mode of it
    for i in categorical:
        df[i].fillna(df[i].mode()[0],inplace=True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #checking the current null values
    df.isna().sum()

    #plotting relationship between Number_of_casualties and Number_of_vehicles_involved
    sns.scatterplot(x=df['Number_of_casualties'], y=df['Number_of_vehicles_involved'], hue=df['Accident_severity'])

    #joint Plot
    sns.jointplot(x='Number_of_casualties',y='Number_of_vehicles_involved',data=df)

    #checking the correlation between numerical columns
    df.corr()

    #plotting the correlation using heatmap
    sns.heatmap(df.corr())

    #storing numerical column names to a variable
    numerical=[i for i in df.columns if df[i].dtype!='O']
    print('The numerica variables are',numerical)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #distribution for numerical columns
    plt.figure(figsize=(10,10))
    plotnumber = 1
    for i in numerical:
        if plotnumber <= df.shape[1]:
            ax1 = plt.subplot(2,2,plotnumber)
            plt.hist(df[i],color='red')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('frequency of '+i, fontsize=10)
        plotnumber +=1

    #count plot for categorical values
    plt.figure(figsize=(10,200))
    plotnumber = 1

    for col in categorical:
        if plotnumber <= df.shape[1] and col!='Pedestrian_movement':
            ax1 = plt.subplot(28,1,plotnumber)
            sns.countplot(data=df, y=col, palette='muted')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(col.title(), fontsize=14)
            plt.xlabel('')
            plt.ylabel('')
        plotnumber +=1



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.dtypes



    # --- PREPROCESSING ───────────────────────────────────────

    #importing label encoing module
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()

    #creating a new data frame from performing the chi2 analysis
    df1=pd.DataFrame()

    #adding all the categorical columns except the output to new data frame
    for i in categorical:
        if i!= 'Accident_severity':
            df1[i]=le.fit_transform(df[i])



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #confirming the data type
    df1.info()

    plt.figure(figsize=(22,17))
    sns.set(font_scale=1)
    sns.heatmap(df1.corr(), annot=True)

    #label encoded data set
    df1.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #import chi2 test
    from sklearn.feature_selection import chi2
    f_p_values=chi2(df1,df['Accident_severity'])

    #f_p_values will return Fscore and pvalues
    f_p_values

    #for better understanding and ease of access adding them to a new dataframe
    f_p_values1=pd.DataFrame({'features':df1.columns, 'Fscore': f_p_values[0], 'Pvalues':f_p_values[1]})
    f_p_values1

    #since we want lower Pvalues we are sorting the features
    f_p_values1.sort_values(by='Pvalues',ascending=True)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #after evaluating we are removing lesser important columns and storing to a new data frame
    df2=df.drop(['Owner_of_vehicle', 'Type_of_vehicle', 'Road_surface_conditions', 'Pedestrian_movement',
             'Casualty_severity','Educational_level','Day_of_week','Sex_of_driver','Road_allignment',
             'Sex_of_casualty'],axis=1)
    df2.head()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df2.shape

    df2.info()

    #to check distinct values in each categorical columns we are storing them to a new variable
    categorical_new=[i for i in df2.columns if df2[i].dtype=='O']
    print(categorical_new)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for i in categorical_new:
        print(df2[i].value_counts())



    # --- PREPROCESSING ───────────────────────────────────────

    #get_dummies
    dummy=pd.get_dummies(df2[['Age_band_of_driver', 'Vehicle_driver_relation', 'Driving_experience',
                              'Area_accident_occured', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                              'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                              'Casualty_class', 'Age_band_of_casualty', 'Cause_of_accident']],drop_first=True)
    dummy.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #concatinate dummy and old data frame
    df3=pd.concat([df2,dummy],axis=1)
    df3.head()

    #dropping dummied columns
    df3.drop(['Age_band_of_driver', 'Vehicle_driver_relation', 'Driving_experience', 'Area_accident_occured', 'Lanes_or_Medians',
              'Types_of_Junction', 'Road_surface_type', 'Light_conditions', 'Weather_conditions', 'Type_of_collision',
              'Vehicle_movement','Casualty_class', 'Age_band_of_casualty', 'Cause_of_accident'],axis=1,inplace=True)
    df3.head()

    x=df3.drop(['Accident_severity'],axis=1)
    x.shape



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    x.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    y=df3.iloc[:,2]
    y.head()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #checking the count of each item in the output column
    y.value_counts()

    #plotting count plot using seaborn
    sns.countplot(x = y, palette='muted')



    # --- PREPROCESSING ───────────────────────────────────────

    #importing SMOTE 
    from imblearn.over_sampling import SMOTE
    oversample=SMOTE()
    xo,yo=oversample.fit_resample(x,y)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #checking the oversampling output
    y1=pd.DataFrame(yo)
    y1.value_counts()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    sns.countplot(x = yo, palette='muted')



    # --- PREPROCESSING ───────────────────────────────────────

    #converting data to training data and testing data
    from sklearn.model_selection import train_test_split
    #splitting 70% of the data to training data and 30% of data to testing data
    x_train,x_test,y_train,y_test=train_test_split(xo,yo,test_size=0.30,random_state=42)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)



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

            clf_setup = setup(data=df, target='Accident_severity', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for 10 clustering vehicle crash data for safety analysi")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
