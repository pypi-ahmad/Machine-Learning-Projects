#!/usr/bin/env python3
"""
Model evaluation for Predicting car prices using the car features

Auto-generated from: car_price_prediction.ipynb
Project: Predicting car prices using the car features
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import warnings
warnings.filterwarnings('ignore')

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Additional imports extracted from mixed cells
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from lazypredict.Supervised import LazyRegressor
from pycaret.regression import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def plot_count(x,fig):
    plt.subplot(4,2,fig)
    plt.title(x+' Histogram')
    sns.countplot(cars[x],palette=("magma"))
    plt.subplot(4,2,(fig+1))
    plt.title(x+' vs Price')
    sns.boxplot(x=cars[x], y=cars.price, palette=("magma"))
    
plt.figure(figsize=(15,20))

plot_count('enginelocation', 1)
plot_count('cylindernumber', 3)
plot_count('fuelsystem', 5)
plot_count('drivewheel', 7)

plt.tight_layout()
def scatter(x,fig):
    plt.subplot(5,2,fig)
    plt.scatter(cars[x],cars['price'])
    plt.title(x+' vs Price')
    plt.ylabel('Price')
    plt.xlabel(x)

plt.figure(figsize=(10,20))

scatter('carlength', 1)
scatter('carwidth', 2)
scatter('carheight', 3)
scatter('curbweight', 4)

plt.tight_layout()
def pp(x,y,z):
    sns.pairplot(cars, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter')
    plt.show()

pp('enginesize', 'boreratio', 'stroke')
pp('compressionratio', 'horsepower', 'peakrpm')
pp('wheelbase', 'citympg', 'highwaympg')
# Defining the map function
def dummies(x,df):
    temp = pd.get_dummies(df[x], drop_first = True)
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df
# Applying the function to the cars_lr

cars_lr = dummies('fueltype',cars_lr)
cars_lr = dummies('aspiration',cars_lr)
cars_lr = dummies('carbody',cars_lr)
cars_lr = dummies('drivewheel',cars_lr)
cars_lr = dummies('enginetype',cars_lr)
cars_lr = dummies('cylindernumber',cars_lr)
cars_lr = dummies('carsrange',cars_lr)
def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary()) # model summary
    return X
    
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)

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

    cars = load_dataset('predicting_car_prices_using_the_car_features')
    cars.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #Splitting company name from CarName column
    CompanyName = cars['CarName'].apply(lambda x : x.split(' ')[0])
    cars.insert(3,"CompanyName",CompanyName)
    cars.drop(['CarName'],axis=1,inplace=True)
    cars.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    cars.CompanyName.unique()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    cars.CompanyName = cars.CompanyName.str.lower()

    def replace_name(a,b):
        cars.CompanyName.replace(a,b,inplace=True)

    replace_name('maxda','mazda')
    replace_name('porcshce','porsche')
    replace_name('toyouta','toyota')
    replace_name('vokswagen','volkswagen')
    replace_name('vw','volkswagen')

    cars.CompanyName.unique()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Checking for duplicates
    cars.loc[cars.duplicated()]

    plt.figure(figsize=(25, 6))

    plt.subplot(1,3,1)
    plt1 = cars.CompanyName.value_counts().plot(kind='bar')
    plt.title('Companies Histogram')
    plt1.set(xlabel = 'Car company', ylabel='Frequency of company')

    plt.subplot(1,3,2)
    plt1 = cars.fueltype.value_counts().plot(kind='bar')
    plt.title('Fuel Type Histogram')
    plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')

    plt.subplot(1,3,3)
    plt1 = cars.carbody.value_counts().plot(kind='bar')
    plt.title('Car Type Histogram')
    plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')

    plt.show()

    plt.figure(figsize=(20,8))

    plt.subplot(1,2,1)
    plt.title('Engine Type Histogram')
    sns.countplot(cars.enginetype, palette=("Blues_d"))

    plt.subplot(1,2,2)
    plt.title('Engine Type vs Price')
    sns.boxplot(x=cars.enginetype, y=cars.price, palette=("PuBuGn"))

    plt.show()

    df = pd.DataFrame(cars.groupby(['enginetype'])['price'].mean().sort_values(ascending = False))
    df.plot.bar(figsize=(8,6))
    plt.title('Engine Type vs Average Price')
    plt.show()

    plt.figure(figsize=(25, 6))

    df = pd.DataFrame(cars.groupby(['CompanyName'])['price'].mean().sort_values(ascending = False))
    df.plot.bar()
    plt.title('Company Name vs Average Price')
    plt.show()

    df = pd.DataFrame(cars.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))
    df.plot.bar()
    plt.title('Fuel Type vs Average Price')
    plt.show()

    df = pd.DataFrame(cars.groupby(['carbody'])['price'].mean().sort_values(ascending = False))
    df.plot.bar()
    plt.title('Car Type vs Average Price')
    plt.show()

    np.corrcoef(cars['carlength'], cars['carwidth'])[0, 1]

    #Fuel economy
    cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg'])



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #Binning the Car Companies based on avg prices of each Company.
    cars['price'] = cars['price'].astype('int')
    temp = cars.copy()
    table = temp.groupby(['CompanyName'])['price'].mean()
    temp = temp.merge(table.reset_index(), how='left',on='CompanyName')
    bins = [0,10000,20000,40000]
    cars_bin=['Budget','Medium','Highend']
    cars['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)
    cars.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    plt.figure(figsize=(25, 6))

    df = pd.DataFrame(cars.groupby(['fuelsystem','drivewheel','carsrange'])['price'].mean().unstack(fill_value=0))
    df.plot.bar()
    plt.title('Car Range vs Average Price')
    plt.show()

    cars_lr = cars[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                      'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
                        'fueleconomy', 'carlength','carwidth', 'carsrange']]
    cars_lr.head()



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    np.random.seed(0)
    df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size = 0.3, random_state = 100)

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
    df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Dividing data into X and y variables
    y_train = df_train.pop('price')
    X_train = df_train



    # --- MODEL TRAINING ──────────────────────────────────────

    lm = LinearRegression()
    lm.fit(X_train,y_train)
    rfe = RFE(estimator=lm, n_features_to_select=10)
    rfe.fit(X_train, y_train)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train_new = build_model(X_train_rfe,y_train)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X_train_new = X_train_rfe.drop(["twelve"], axis = 1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train_new = build_model(X_train_new,y_train)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X_train_new = X_train_new.drop(["fueleconomy"], axis = 1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train_new = build_model(X_train_new,y_train)

    #Calculating the Variance Inflation Factor
    checkVIF(X_train_new)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X_train_new = X_train_new.drop(["curbweight"], axis = 1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train_new = build_model(X_train_new,y_train)

    checkVIF(X_train_new)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X_train_new = X_train_new.drop(["sedan"], axis = 1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train_new = build_model(X_train_new,y_train)

    checkVIF(X_train_new)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    X_train_new = X_train_new.drop(["wagon"], axis = 1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train_new = build_model(X_train_new,y_train)

    checkVIF(X_train_new)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #Dropping dohcv to see the changes in model statistics
    X_train_new = X_train_new.drop(["dohcv"], axis = 1)
    X_train_new = build_model(X_train_new,y_train)
    checkVIF(X_train_new)



    # --- MODEL TRAINING ──────────────────────────────────────

    lm = sm.OLS(y_train,X_train_new).fit()
    y_train_price = lm.predict(X_train_new)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Plot the histogram of the error terms
    fig = plt.figure()
    sns.distplot((y_train - y_train_price), bins = 20)
    fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
    plt.xlabel('Errors', fontsize = 18)

    #Scaling the test set
    num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
    df_test[num_vars] = scaler.fit_transform(df_test[num_vars])

    #Dividing into X and y
    y_test = df_test.pop('price')
    X_test = df_test



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Now let's use our model to make predictions.
    X_train_new = X_train_new.drop('const',axis=1)
    # Creating X_test_new dataframe by dropping variables from X_test
    X_test_new = X_test[X_train_new.columns]

    # Adding a constant variable 
    X_test_new = sm.add_constant(X_test_new)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Making predictions
    y_pred = lm.predict(X_test_new)



    # --- EVALUATION ──────────────────────────────────────────

    from sklearn.metrics import r2_score 
    r2_score(y_test, y_pred)



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

            reg_setup = setup(data=df, target='const', session_id=42, verbose=False)

            # Compare models and select best
            best_model = compare_models()

            # Display comparison results
            print(best_model)



        except ImportError:

            print('[AutoML] LazyPredict/PyCaret not installed — skipping AutoML block')

        except Exception as _automl_err:

            print(f'[AutoML] AutoML block failed: {_automl_err}')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #EVALUATION OF THE MODEL
    # Plotting y_test and y_pred to understand the spread.
    fig = plt.figure()
    plt.scatter(y_test,y_pred)
    fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
    plt.xlabel('y_test', fontsize=18)                          # X-label
    plt.ylabel('y_pred', fontsize=16)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model evaluation for Predicting car prices using the car features")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
