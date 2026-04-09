#!/usr/bin/env python3
"""
Model evaluation for Predicting rainfall amount

Auto-generated from: rain_prediction.ipynb
Project: Predicting rainfall amount
Category: Regression | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from keras import callbacks

np.random.seed(0)
# Additional imports extracted from mixed cells
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

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

    data = load_dataset('predicting_rainfall_amount')
    data.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #first of all let us evaluate the target and find out if our data is imbalanced or not
    cols= ["#C2C4E2","#EED4E5"]
    sns.countplot(x= data["RainTomorrow"], palette= cols)

    #Parsing datetime
    #exploring the length of date objects
    lengths = data["Date"].str.len()
    lengths.value_counts()

    #There don't seem to be any error in dates so parsing values into datetime
    data['Date']= pd.to_datetime(data["Date"])
    #Creating a collumn of year
    data['year'] = data.Date.dt.year

    # function to encode datetime into cyclic parameters. 
    #As I am planning to use this data in a neural network I prefer the months and days in a cyclic continuous feature. 

    def encode(data, col, max_val):
        data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
        data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
        return data

    data['month'] = data.Date.dt.month
    data = encode(data, 'month', 12)

    data['day'] = data.Date.dt.day
    data = encode(data, 'day', 31)

    data.head()

    # roughly a year's span section 
    section = data[:360] 
    tm = section["day"].plot(color="#C2C4E2")
    tm.set_title("Distribution Of Days Over Year")
    tm.set_ylabel("Days In month")
    tm.set_xlabel("Days In Year")

    cyclic_month = sns.scatterplot(x="month_sin",y="month_cos",data=data, color="#C2C4E2")
    cyclic_month.set_title("Cyclic Encoding of Month")
    cyclic_month.set_ylabel("Cosine Encoded Months")
    cyclic_month.set_xlabel("Sine Encoded Months")

    cyclic_day = sns.scatterplot(x='day_sin',y='day_cos',data=data, color="#C2C4E2")
    cyclic_day.set_title("Cyclic Encoding of Day")
    cyclic_day.set_ylabel("Cosine Encoded Day")
    cyclic_day.set_xlabel("Sine Encoded Day")

    # Get list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)

    print("Categorical variables:")
    print(object_cols)

    # Missing values in categorical variables

    for i in object_cols:
        print(i, data[i].isnull().sum())



    # --- PREPROCESSING ───────────────────────────────────────

    # Filling missing values with mode of the column in value

    for i in object_cols:
        data[i].fillna(data[i].mode()[0], inplace=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Get list of neumeric variables
    t = (data.dtypes == "float64")
    num_cols = list(t[t].index)

    print("Neumeric variables:")
    print(num_cols)

    # Missing values in numeric variables

    for i in num_cols:
        print(i, data[i].isnull().sum())



    # --- PREPROCESSING ───────────────────────────────────────

    # Filling missing values with median of the column in value

    for i in num_cols:
        data[i].fillna(data[i].median(), inplace=True)
    
    data.info()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #plotting a lineplot rainfall over years
    plt.figure(figsize=(12,8))
    Time_series=sns.lineplot(x=data['Date'].dt.year,y="Rainfall",data=data,color="#C2C4E2")
    Time_series.set_title("Rainfall Over Years")
    Time_series.set_ylabel("Rainfall")
    Time_series.set_xlabel("Years")

    #Evauating Wind gust speed over years
    colours = ["#D0DBEE", "#C2C4E2", "#EED4E5", "#D1E6DC", "#BDE2E2"]
    plt.figure(figsize=(12,8))
    Days_of_week=sns.barplot(x=data['Date'].dt.year,y="WindGustSpeed",data=data, ci =None,palette = colours)
    Days_of_week.set_title("Wind Gust Speed Over Years")
    Days_of_week.set_ylabel("WindGustSpeed")
    Days_of_week.set_xlabel("Year")



    # --- PREPROCESSING ───────────────────────────────────────

    # Apply label encoder to each column with categorical data
    label_encoder = LabelEncoder()
    for i in object_cols:
        data[i] = label_encoder.fit_transform(data[i])
    
    data.info()

    # Prepairing attributes of scale data

    features = data.drop(['RainTomorrow', 'Date','day', 'month'], axis=1) # dropping target and extra columns

    target = data['RainTomorrow']

    #Set up a standard scaler for the features
    col_names = list(features.columns)
    s_scaler = preprocessing.StandardScaler()
    features = s_scaler.fit_transform(features)
    features = pd.DataFrame(features, columns=col_names) 

    features.describe().T



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Detecting outliers
    #looking at the scaled features
    colours = ["#D0DBEE", "#C2C4E2", "#EED4E5", "#D1E6DC", "#BDE2E2"]
    plt.figure(figsize=(20,10))
    sns.boxenplot(data = features,palette = colours)
    plt.xticks(rotation=90)
    plt.show()

    #full data for 
    features["RainTomorrow"] = target

    #Dropping with outlier

    features = features[(features["MinTemp"]<2.3)&(features["MinTemp"]>-2.3)]
    features = features[(features["MaxTemp"]<2.3)&(features["MaxTemp"]>-2)]
    features = features[(features["Rainfall"]<4.5)]
    features = features[(features["Evaporation"]<2.8)]
    features = features[(features["Sunshine"]<2.1)]
    features = features[(features["WindGustSpeed"]<4)&(features["WindGustSpeed"]>-4)]
    features = features[(features["WindSpeed9am"]<4)]
    features = features[(features["WindSpeed3pm"]<2.5)]
    features = features[(features["Humidity9am"]>-3)]
    features = features[(features["Humidity3pm"]>-2.2)]
    features = features[(features["Pressure9am"]< 2)&(features["Pressure9am"]>-2.7)]
    features = features[(features["Pressure3pm"]< 2)&(features["Pressure3pm"]>-2.7)]
    features = features[(features["Cloud9am"]<1.8)]
    features = features[(features["Cloud3pm"]<2)]
    features = features[(features["Temp9am"]<2.3)&(features["Temp9am"]>-2)]
    features = features[(features["Temp3pm"]<2.3)&(features["Temp3pm"]>-2)]


    features.shape



    # --- PREPROCESSING ───────────────────────────────────────

    X = features.drop(["RainTomorrow"], axis=1)
    y = features["RainTomorrow"]

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    X.shape



    # --- MODEL TRAINING ──────────────────────────────────────

    #Early stopping
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=20, # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    # Initialising the NN
    model = Sequential()

    # layers

    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 26))
    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    opt = Adam(learning_rate=0.00009)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Train the ANN
    history = model.fit(X_train, y_train, batch_size = 32, epochs = 150, callbacks=[early_stopping], validation_split=0.2)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    history_df = pd.DataFrame(history.history)

    plt.plot(history_df.loc[:, ['loss']], "#BDE2E2", label='Training loss')
    plt.plot(history_df.loc[:, ['val_loss']],"#C2C4E2", label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="best")

    plt.show()

    history_df = pd.DataFrame(history.history)

    plt.plot(history_df.loc[:, ['accuracy']], "#BDE2E2", label='Training accuracy')
    plt.plot(history_df.loc[:, ['val_accuracy']], "#C2C4E2", label='Validation accuracy')

    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Predicting the test set results
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)



    # --- EVALUATION ──────────────────────────────────────────

    # confusion matrix
    cmap1 = sns.diverging_palette(260,-10,s=50, l=75, n=5, as_cmap=True)
    plt.subplots(figsize=(12,8))
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), cmap = cmap1, annot = True, annot_kws = {'size':15})

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model evaluation for Predicting rainfall amount")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
