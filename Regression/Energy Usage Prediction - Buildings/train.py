#!/usr/bin/env python3
"""
Model training for Predicting energy usage in buildings

Auto-generated from: energy_prediction.ipynb
Project: Predicting energy usage in buildings
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, model_selection, metrics
import warnings
import os
from sklearn.model_selection import train_test_split
import chart_studio.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

# Function to get top correlations 

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(train_corr, 40))

# ======================================================================
# TRAINING PIPELINE
# ======================================================================

def main():
    """Run the training pipeline."""
    USE_AUTOML = True  # Set to False to skip AutoML comparison

    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load in 

    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import preprocessing, model_selection, metrics
    import warnings
    warnings.filterwarnings("ignore")

    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

    import os
    print(os.listdir("./archive/"))

    # Any results you write to the current directory are saved as output.

    data = load_dataset('predicting_energy_usage_in_buildings')



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    # 75% of the data is usedfor the training of the models and the rest is used for testing
    train, test = train_test_split(data,test_size=0.25,random_state=40)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Divide the columns based on type for clear column management 

    col_temp = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]

    col_hum = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]

    col_weather = ["T_out", "Tdewpoint","RH_out","Press_mm_hg",
                    "Windspeed","Visibility"] 
    col_light = ["lights"]

    col_randoms = ["rv1", "rv2"]

    col_target = ["Appliances"]

    # Seperate dependent and independent variables 
    feature_vars = train[col_temp + col_hum + col_weather + col_light + col_randoms ]
    target_vars = train[col_target]



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Due to lot of zero enteries this column is of not much use and will be ignored in rest of the model
    _ = feature_vars.drop(['lights'], axis=1 , inplace= True) ;



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # plotly
    import chart_studio.plotly as py
    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode(connected=True)
    import plotly.graph_objs as go

    # To understand the timeseries variation of the applaince energy consumption
    visData = go.Scatter( x= data.date  ,  mode = "lines", y = data.Appliances )
    layout = go.Layout(title = 'Appliance energy consumption measurement' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))
    fig = go.Figure(data=[visData],layout=layout)

    iplot(fig)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Adding column to mark weekdays (0) and weekends(1) for time series evaluation , 
    # decided not to use it for model evaluation as it has least impact

    data['WEEKDAY'] = ((pd.to_datetime(data['date']).dt.dayofweek)// 5 == 1).astype(float)
    # There are 5472 weekend recordings 
    data['WEEKDAY'].value_counts()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Find rows with weekday 
    temp_weekday =  data[data['WEEKDAY'] == 0]
    # To understand the timeseries variation of the applaince energy consumption
    visData = go.Scatter( x= temp_weekday.date  ,  mode = "lines", y = temp_weekday.Appliances )
    layout = go.Layout(title = 'Appliance energy consumption measurement on weekdays' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))
    fig = go.Figure(data=[visData],layout=layout)

    iplot(fig)

    # Find rows with weekday 

    temp_weekend =  data[data['WEEKDAY'] == 1]

    # To understand the timeseries variation of the applaince energy consumption
    visData = go.Scatter( x= temp_weekend.date  ,  mode = "lines", y = temp_weekend.Appliances )
    layout = go.Layout(title = 'Appliance energy consumption measurement on weekend' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))
    fig = go.Figure(data=[visData],layout=layout)

    iplot(fig)

    # Use the weather , temperature , applainces and random column to see the correlation
    train_corr = train[col_temp + col_hum + col_weather +col_target+col_randoms]
    corr = train_corr.corr()
    # Mask the repeated values
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
  
    f, ax = plt.subplots(figsize=(16, 14))
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, annot=True, fmt=".2f" , mask=mask,)
        #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);
        #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
        #show plot
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Due to conlusion made above below columns are removed
    train_X.drop(["rv1","rv2","Visibility","T6","T9"],axis=1 , inplace=True)

    # Due to conlusion made above below columns are removed
    test_X.drop(["rv1","rv2","Visibility","T6","T9"], axis=1, inplace=True)



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()

    # Create test and training set by including Appliances column

    train = train[list(train_X.columns.values) + col_target ]

    test = test[list(test_X.columns.values) + col_target ]

    # Create dummy test and training set to hold scaled values

    sc_train = pd.DataFrame(columns=train.columns , index=train.index)

    sc_train[sc_train.columns] = sc.fit_transform(train)

    sc_test= pd.DataFrame(columns=test.columns , index=test.index)

    sc_test[sc_test.columns] = sc.fit_transform(test)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Remove Appliances column from traininig set

    train_X =  sc_train.drop(['Appliances'] , axis=1)
    train_y = sc_train['Appliances']

    test_X =  sc_test.drop(['Appliances'] , axis=1)
    test_y = sc_test['Appliances']



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

            clf_setup = setup(data=data, target='Appliances', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for Predicting energy usage in buildings")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
