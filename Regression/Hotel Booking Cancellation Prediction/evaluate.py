#!/usr/bin/env python3
"""
Model evaluation for Hotel Booking Cancellation Prediction

Auto-generated from: hotel_booking_prediction.ipynb
Project: Hotel Booking Cancellation Prediction
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

import folium
from folium.plugins import HeatMap
import plotly.express as px

plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 32)
# Additional imports extracted from mixed cells
import sort_dataframeby_monthorweek as sd
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# EVALUATION PIPELINE
# ======================================================================

def main():
    """Run the evaluation pipeline."""
    USE_AUTOML = True  # Set to False to skip AutoML comparison

    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    # reading data
    df = load_dataset('hotel_booking_cancellation_prediction')
    df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # checking for null values 

    null = pd.DataFrame({'Null Values' : df.isna().sum(), 'Percentage Null Values' : (df.isna().sum()) / (df.shape[0]) * (100)})
    null



    # --- PREPROCESSING ───────────────────────────────────────

    # filling null values with zero

    df.fillna(0, inplace = True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # visualizing null values
    msno.bar(df)
    plt.show()

    # adults, babies and children cant be zero at same time, so dropping the rows having all these zero at same time

    filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)
    df[filter]

    df = df[~filter]
    df

    country_wise_guests = df[df['is_canceled'] == 0]['country'].value_counts().reset_index()
    country_wise_guests.columns = ['country', 'No of guests']
    country_wise_guests

    basemap = folium.Map()
    guests_map = px.choropleth(country_wise_guests, locations = country_wise_guests['country'],
                               color = country_wise_guests['No of guests'], hover_name = country_wise_guests['country'])
    guests_map.show()

    data = df[df['is_canceled'] == 0]

    px.box(data_frame = data, x = 'reserved_room_type', y = 'adr', color = 'hotel', template = 'plotly_dark')

    data_resort = df[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 0)]
    data_city = df[(df['hotel'] == 'City Hotel') & (df['is_canceled'] == 0)]

    resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
    resort_hotel

    city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
    city_hotel



    # --- FEATURE ENGINEERING ─────────────────────────────────

    final_hotel = resort_hotel.merge(city_hotel, on = 'arrival_date_month')
    final_hotel.columns = ['month', 'price_for_resort', 'price_for_city_hotel']
    final_hotel



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    import sort_dataframeby_monthorweek as sd

    def sort_month(df, column_name):
        return sd.Sort_Dataframeby_Month(df, column_name)

    final_prices = sort_month(final_hotel, 'month')
    final_prices

    plt.figure(figsize = (17, 8))

    px.line(final_prices, x = 'month', y = ['price_for_resort','price_for_city_hotel'],
            title = 'Room price per night over the Months', template = 'plotly_dark')

    resort_guests = data_resort['arrival_date_month'].value_counts().reset_index()
    resort_guests.columns=['month','no of guests']
    resort_guests

    city_guests = data_city['arrival_date_month'].value_counts().reset_index()
    city_guests.columns=['month','no of guests']
    city_guests



    # --- FEATURE ENGINEERING ─────────────────────────────────

    final_guests = resort_guests.merge(city_guests,on='month')
    final_guests.columns=['month','no of guests in resort','no of guest in city hotel']
    final_guests



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    final_guests = sort_month(final_guests,'month')
    final_guests

    px.line(final_guests, x = 'month', y = ['no of guests in resort','no of guest in city hotel'],
            title='Total no of guests per Months', template = 'plotly_dark')

    filter = df['is_canceled'] == 0
    data = df[filter]
    data.head()

    data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
    data.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    stay = data.groupby(['total_nights', 'hotel']).agg('count').reset_index()
    stay = stay.iloc[:, :3]
    stay = stay.rename(columns={'is_canceled':'Number of stays'})
    stay



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    px.bar(data_frame = stay, x = 'total_nights', y = 'Number of stays', color = 'hotel', barmode = 'group',
            template = 'plotly_dark')

    correlation = df.corr()['is_canceled'].abs().sort_values(ascending = False)
    correlation



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # dropping columns that are not useful

    useless_col = ['days_in_waiting_list', 'arrival_date_year', 'arrival_date_year', 'assigned_room_type', 'booking_changes',
                   'reservation_status', 'country', 'days_in_waiting_list']

    df.drop(useless_col, axis = 1, inplace = True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # creating numerical and categorical dataframes

    cat_cols = [col for col in df.columns if df[col].dtype == 'O']
    cat_cols

    cat_df = df[cat_cols]
    cat_df.head()

    cat_df['reservation_status_date'] = pd.to_datetime(cat_df['reservation_status_date'])

    cat_df['year'] = cat_df['reservation_status_date'].dt.year
    cat_df['month'] = cat_df['reservation_status_date'].dt.month
    cat_df['day'] = cat_df['reservation_status_date'].dt.day



    # --- FEATURE ENGINEERING ─────────────────────────────────

    cat_df.drop(['reservation_status_date','arrival_date_month'] , axis = 1, inplace = True)

    # encoding categorical variables

    cat_df['hotel'] = cat_df['hotel'].map({'Resort Hotel' : 0, 'City Hotel' : 1})

    cat_df['meal'] = cat_df['meal'].map({'BB' : 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})

    cat_df['market_segment'] = cat_df['market_segment'].map({'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                                                               'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})

    cat_df['distribution_channel'] = cat_df['distribution_channel'].map({'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3,
                                                                           'GDS': 4})

    cat_df['reserved_room_type'] = cat_df['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,
                                                                       'L': 7, 'B': 8})

    cat_df['deposit_type'] = cat_df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})

    cat_df['customer_type'] = cat_df['customer_type'].map({'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})

    cat_df['year'] = cat_df['year'].map({2015: 0, 2014: 1, 2016: 2, 2017: 3})

    num_df = df.drop(columns = cat_cols, axis = 1)
    num_df.drop('is_canceled', axis = 1, inplace = True)
    num_df



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    num_df.var()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # normalizing numerical variables

    num_df['lead_time'] = np.log(num_df['lead_time'] + 1)
    num_df['arrival_date_week_number'] = np.log(num_df['arrival_date_week_number'] + 1)
    num_df['arrival_date_day_of_month'] = np.log(num_df['arrival_date_day_of_month'] + 1)
    num_df['agent'] = np.log(num_df['agent'] + 1)
    num_df['company'] = np.log(num_df['company'] + 1)
    num_df['adr'] = np.log(num_df['adr'] + 1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    num_df.var()



    # --- PREPROCESSING ───────────────────────────────────────

    num_df['adr'] = num_df['adr'].fillna(value = num_df['adr'].mean())



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X = pd.concat([cat_df, num_df], axis = 1)
    y = df['is_canceled']



    # --- PREPROCESSING ───────────────────────────────────────

    # splitting data into training set and test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=42)



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

            clf_setup = setup(data=df, target='is_canceled', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model evaluation for Hotel Booking Cancellation Prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
