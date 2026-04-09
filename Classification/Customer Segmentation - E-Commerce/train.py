#!/usr/bin/env python3
"""
Model training for Customer segmentation for an E-commerce company

Auto-generated from: customer_segmentation_e-commerce.ipynb
Project: Customer segmentation for an E-commerce company
Category: Classification | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import datetime as dt
from datetime import datetime
from datetime import timedelta
# !pip install pyclustertend
from sklearn.cluster import KMeans, AgglomerativeClustering
from pyclustertend import hopkins
from sklearn.preprocessing import scale
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_samples,silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from yellowbrick.cluster import KElbowVisualizer
# Additional imports extracted from mixed cells
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
import datetime
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cufflinks as cf
import plotly.offline
import warnings
import colorama
from colorama import Fore, Style  # makes strings colored
from termcolor import colored
from termcolor import cprint
import ipywidgets
from ipywidgets import interact
import pandas_profiling
from pandas_profiling.report.presentation.flavours.html.templates import create_html_assets
from wordcloud import WordCloud
import squarify as sq
from IPython.display import display
from PIL import Image
from pycaret.clustering import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
## Some Useful User-Defined-Functions

###############################################################################

def missing_values(df):
    missing_number = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_values = pd.concat([missing_number, missing_percent], axis=1, keys=['Missing_Number', 'Missing_Percent'])
    return missing_values[missing_values['Missing_Number']>0]

###############################################################################

def first_looking(df):
    print(colored("Shape:", attrs=['bold']), df.shape,'\n', 
          colored('*'*100, 'red', attrs=['bold']),
          colored("\nInfo:\n", attrs=['bold']), sep='')
    print(df.info(), '\n', 
          colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("Number of Uniques:\n", attrs=['bold']), df.nunique(),'\n',
          colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("Missing Values:\n", attrs=['bold']), missing_values(df),'\n', 
          colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("All Columns:", attrs=['bold']), list(df.columns),'\n', 
          colored('*'*100, 'red', attrs=['bold']), sep='')

    df.columns= df.columns.str.lower().str.replace('&', '_').str.replace(' ', '_')
    print(colored("Columns after rename:", attrs=['bold']), list(df.columns),'\n',
          colored('*'*100, 'red', attrs=['bold']), sep='')
    print(colored("Descriptive Statistics \n", attrs=['bold']), df.describe().round(2),'\n',
          colored('*'*100, 'red', attrs=['bold']), sep='') # Gives a statstical breakdown of the data.
    print(colored("Descriptive Statistics (Categorical Columns) \n", attrs=['bold']), df.describe(include=object).T,'\n',
          colored('*'*100, 'red', attrs=['bold']), sep='') # Gives a statstical breakdown of the data.
    
###############################################################################

# To view summary information about the columns

def first_look(col):
    print("column name    : ", col)
    print("--------------------------------")
    print("per_of_nulls   : ", "%", round(df[col].isnull().sum()/df.shape[0]*100, 2))
    print("num_of_nulls   : ", df[col].isnull().sum())
    print("num_of_uniques : ", df[col].nunique())
    print("Value counts   : \n", df[col].value_counts(dropna = False))    

###############################################################################
    
def multicolinearity_control(df):
    feature =[]
    collinear=[]
    for col in df.corr().columns:
        for i in df.corr().index:
            if (abs(df.corr()[col][i])> .9 and abs(df.corr()[col][i]) < 1):
                    feature.append(col)
                    collinear.append(i)
                    print(colored(f"Multicolinearity alert in between:{col} - {i}", 
                                  "red", attrs=['bold']), df.shape,'\n',
                                  colored('*'*100, 'red', attrs=['bold']), sep='')

###############################################################################

def duplicate_values(df):
    print(colored("Duplicate check...", attrs=['bold']), sep='')
    duplicate_values = df.duplicated(subset=None, keep='first').sum()
    if duplicate_values > 0:
        df.drop_duplicates(keep='first', inplace=True)
        print(duplicate_values, colored(" Duplicates were dropped!"),'\n',
              colored('*'*100, 'red', attrs=['bold']), sep='')
    else:
        print(colored("There are no duplicates"),'\n',
              colored('*'*100, 'red', attrs=['bold']), sep='')     

###############################################################################
        
def drop_columns(df, drop_columns):
    if drop_columns !=[]:
        df.drop(drop_columns, axis=1, inplace=True)
        print(drop_columns, 'were dropped')
    else:
        print(colored('We will now check the missing values and if necessary, the realted columns will be dropped!', attrs=['bold']),'\n',
              colored('*'*100, 'red', attrs=['bold']), sep='')

###############################################################################

def drop_null(df, limit):
    print('Shape:', df.shape)
    for i in df.isnull().sum().index:
        if (df.isnull().sum()[i]/df.shape[0]*100)>limit:
            print(df.isnull().sum()[i], 'percent of', i ,'null and were dropped')
            df.drop(i, axis=1, inplace=True)
            print('new shape:', df.shape)       
    print(colored("New shape after missing value control:"),'\n', df.shape)

###############################################################################

def fill_most(df, group_col, col_name):
    '''Fills the missing values with the most existing value (mode) in the relevant column according to single-stage grouping'''
    for group in list(df[group_col].unique()):
        cond = df[group_col]==group
        mode = list(df[cond][col_name].mode())
        if mode != []:
            df.loc[cond, col_name] = df.loc[cond, col_name].fillna(df[cond][col_name].mode()[0])
        else:
            df.loc[cond, col_name] = df.loc[cond, col_name].fillna(df[col_name].mode()[0])
    print("Number of NaN : ",df[col_name].isnull().sum())
    print("------------------")
    print(df[col_name].value_counts(dropna=False))
    
###############################################################################
def explore(x):
    divider = "*_*"
    print("\n {} \n".format((divider*20))) #creates a divider between each method output breaking at each end.
    print("Dataframe Makeup \n") # title for output.
    x.info() # Explains what the data and values the data is madeup from.
    print("\n {} \n".format((divider*20))) # creates a dvider between each method output breaking at each end.
    print("Descriptive Statistics \n\n", x.describe().round(2)) # Gives a statstical breakdown of the data.
    print("\n {} \n".format((divider*20))) # creates a divider between each method output breaking at each end.
    print("Shape of dataframe: {}".format(x.shape)) # Gives the shape of the data.
    print("\n {} \n".format((divider*20))) # creates a dvider between each method output breaking at each end.
    return
def recency_scoring(rfm):
    if rfm.Recency <= 24.0:
        recency_score = 4
    elif rfm.Recency <= 57.0:
        recency_score = 3
    elif rfm.Recency <= 149.0:
        recency_score = 2
    else:
        recency_score = 1
    return recency_score

customer_rfm['Recency_Score'] = customer_rfm.apply(recency_scoring, axis=1)
customer_rfm.sample(10)
def frequency_scoring(rfm):
    if rfm.Frequency >= 10.0:
        frequency_score = 4
    elif rfm.Frequency >= 5.0:
        frequency_score = 3
    elif rfm.Frequency >= 2.0:
        frequency_score = 2
    else:
        frequency_score = 1
    return frequency_score

customer_rfm['Frequency_Score'] = customer_rfm.apply(frequency_scoring, axis=1)
customer_rfm.sample(10)
def monetary_scoring(rfm):
    if rfm.Monetary >= 1571.0:
        monetary_score = 4
    elif rfm.Monetary >= 645.0:
        monetary_score = 3
    elif rfm.Monetary >= 298.0:
        monetary_score = 2
    else:
        monetary_score = 1
    return monetary_score

customer_rfm['Monetary_Score'] = customer_rfm.apply(monetary_scoring, axis=1)
customer_rfm.sample(10)
def rfm_scoring(customer):
    return str(int(customer['Recency_Score'])) + str(int(customer['Frequency_Score'])) + str(int(customer['Monetary_Score']))


customer_rfm['Customer_RFM_Score'] = customer_rfm.apply(rfm_scoring, axis=1)
customer_rfm.sample(8)
def categorizer(rfm):
    
    if (rfm[0] in ['2', '3', '4']) & (rfm[1] in ['4']) & (rfm[2] in ['4']):
        rfm = 'Champion'
        
    elif (rfm[0] in ['3']) & (rfm[1] in ['1', '2', '3', '4']) & (rfm[2] in ['3', '4']):
        rfm = 'Top Loyal Customer'
        
    elif (rfm[0] in ['3']) & (rfm[1] in ['1', '2', '3', '4']) & (rfm[2] in ['1', '2']):
        rfm = 'Loyal Customer'
    
    elif (rfm[0] in ['4']) & (rfm[1] in ['1', '2', '3', '4']) & (rfm[2] in ['3', '4']):
        rfm = 'Top Recent Customer'
    
    elif (rfm[0] in ['4']) & (rfm[1] in ['1', '2', '3', '4']) & (rfm[2] in ['1', '2']):
        rfm = 'Recent Customer'
    
    elif (rfm[0] in ['2', '3']) & (rfm[1] in ['1', '2', '3', '4']) & (rfm[2] in ['3', '4']):
        rfm = 'Top Customer Needed Attention'    
   
    elif (rfm[0] in ['2', '3']) & (rfm[1] in ['1', '2', '3', '4']) & (rfm[2] in ['1', '2']):
        rfm = 'Customer Needed Attention'
    
    elif (rfm[0] in ['1']) & (rfm[1] in ['1', '2', '3', '4']) & (rfm[2] in ['3', '4']):
        rfm = 'Top Lost Customer'
                
    elif (rfm[0] in ['1']) & (rfm[1] in ['1', '2', '3', '4']) & (rfm[2] in ['1', '2']):
        rfm = 'Lost Customer'
    
    return rfm

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

    # --- PREPROCESSING ───────────────────────────────────────

    # !pip install pyforest
    # 1-Import Libraies
    import numpy as np
    import pandas as pd 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import datetime as dt
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import missingno as msno 
    import plotly.express as px
    import plotly.graph_objects as go
    import datetime

    from sklearn.compose import make_column_transformer

    # Scaling
    from sklearn.preprocessing import scale 
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import PolynomialFeatures 
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import PowerTransformer 
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import RobustScaler

    # Modelling
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # Importing plotly and cufflinks in offline mode
    import cufflinks as cf
    import plotly.offline
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)
    import plotly.graph_objects as go

    # Ignore Warnings
    import warnings
    warnings.filterwarnings("ignore")
    warnings.warn("this will not show")

    # Figure&Display options
    plt.rcParams["figure.figsize"] = (16, 9)
    pd.set_option('max_colwidth',200)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # !pip install termcolor
    import colorama
    from colorama import Fore, Style  # makes strings colored
    from termcolor import colored
    from termcolor import cprint

    import ipywidgets
    from ipywidgets import interact

    # !pip install -U pandas-profiling --user
    import pandas_profiling
    from pandas_profiling.report.presentation.flavours.html.templates import create_html_assets

    # !pip install wordcloud
    from wordcloud import WordCloud

    # !pip install squarify
    import squarify as sq



    # --- DATA LOADING ────────────────────────────────────────

    df0 = load_dataset('customer_segmentation_for_an_e_commerce_company')
    df = df0.copy()
    df.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    first_looking(df)
    duplicate_values(df)
    drop_columns(df, [])
    drop_null(df, 90)

    df[["invoiceno", "quantity", "unitprice"]].describe(include=object).T

    df['total_price'] = df['quantity'] * df['unitprice']
    df.head(3)

    df['invoiceno'].sample(10)

    cprint("Have a First Look at 'invoice' Column", 'blue')
    first_look('invoiceno')

    cprint("Total number of invoices by country :",'blue')
    df.groupby('country')['invoiceno'].count().sort_values(ascending=False)

    fig = px.histogram(df, 
                       x = 'country', 
                       title = 'The Number of Invoices by Country', 
                       color='country').update_xaxes(categoryorder="total descending")
    fig.show()

    fig = px.histogram(df, 
                       x = df.groupby('country')['invoiceno'].nunique().index,
                       y = df.groupby('country')['invoiceno'].nunique().values, 
                       title = 'The Unique Number of Invoices by Country', 
                       labels = dict(x = "Countries", y ="Invoice")).update_xaxes(categoryorder="total descending")
    fig.show();



    # --- FEATURE ENGINEERING ─────────────────────────────────

    CA_values = (df['invoiceno'].str.startswith('C') | df['invoiceno'].str.startswith('A')).value_counts()
    CA_values = pd.DataFrame(CA_values)
    CA_values.rename(index={False: 'Invoices Without C & A', True: 'Invoices With C & A'}, inplace=True)
    CA_values



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig = px.pie(CA_values, 
                 values = CA_values['invoiceno'], 
                 names = CA_values.index, 
                 title = "The Percentage of 'invoiceno' Starts With A or C")

    fig.show()

    cprint("Have a First Look at 'stockcode' Column",'blue')
    first_look('stockcode')

    cprint("Have a First Look at 'description' Column",'blue')
    first_look('description')



    # --- PREPROCESSING ───────────────────────────────────────

    cprint("Croos check", 'blue')
    df[df['stockcode']=='22139.0']['description'].value_counts(dropna=False)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    cprint("Have a First Look to 'stockcode' Column",'blue')
    first_look('quantity')

    cprint("The Number of Observations in the 'Quantity' Column Which are Above 0",'blue')
    df[df['quantity'] > 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]

    cprint("The Number of Observations in the 'Quantity' Column Which Equal to 0",'blue')
    df[df['quantity'] == 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]

    cprint("The Number of Observations in the 'Quantity' Column Which are Below 0",'blue')
    df[df['quantity'] < 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]



    # --- FEATURE ENGINEERING ─────────────────────────────────

    quantity_values = (df['quantity'] < 0).value_counts()
    quantity_values = pd.DataFrame(quantity_values)
    quantity_values.rename(index={False: 'Invoices Smaller Than 0', True: 'Invoices Bigger Than 0'}, inplace=True)
    quantity_values



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig = px.pie(quantity_values, 
                 values = quantity_values['quantity'], 
                 names = quantity_values.index, 
                 title = 'Quantity Smaller/Bigger')

    fig.show()

    cprint("Have a First Look at 'unitprice' Column",'blue')
    first_look('unitprice')

    cprint("Have a First Look at 'customerid' Column",'blue')
    first_look('customerid')

    fig = px.histogram(df, x = df.groupby('country')['customerid'].nunique().index, 
                       y = df.groupby('country')['customerid'].nunique().values, 
                       title = 'The Number of Customers By Country',
                       labels = dict(x = "Countries", y ="Customer")).update_xaxes(categoryorder="total descending")
    fig.show()

    df_wo_UK = df.groupby('country')["customerid"].nunique().sort_values(ascending=False).iloc[1:]

    fig = px.histogram(df, x = df_wo_UK.index, 
                       y = df_wo_UK.values, 
                       title = 'The Number of Customers By Country Without UK',
                       labels = dict(x = "Countries", y ="Customer")).update_xaxes(categoryorder="total descending")
    fig.show()

    cprint("Have a First Look at 'invoicedate' Column",'blue')
    first_look('invoicedate')

    df.invoicedate.max()

    df.invoicedate.min()

    df[["invoiceno", "quantity", "unitprice"]].describe(include=object).T

    df[df['quantity'] < 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]

    df[df['quantity'] == 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]

    df[df['unitprice'] < 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]

    df[df['unitprice'] > 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]

    df[df['unitprice'] == 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]

    df["invoiceno"].str.startswith('C').sum()

    df[df["invoiceno"].str.startswith('C')][["invoiceno", "quantity", "unitprice"]]

    cprint("Cancelled Orders",'blue')
    df[df['quantity'] < 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]



    # --- FEATURE ENGINEERING ─────────────────────────────────

    cancelled_orders = (df['invoiceno'].str.startswith('C').value_counts())
    cancelled_orders = pd.DataFrame(cancelled_orders)
    cancelled_orders.rename(index={False: 'Non-Cancelled Orders', True: 'Cancelled Orders'}, inplace=True)
    cancelled_orders



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig = px.pie(cancelled_orders, 
                 values = cancelled_orders['invoiceno'], 
                 names = cancelled_orders.index, 
                 title = 'The Proportion of Canceled Orders')

    fig.show()

    cprint("Non-Cancelled Orders",'blue')
    df[df['quantity'] > 0][['invoiceno', 'stockcode', 'description', 'quantity', 'invoicedate', 'unitprice', 'customerid', 'country']]

    df[df['unitprice'] < 0]

    df[df['unitprice'] == 0]

    missing_values(df)

    plt.figure(figsize = (10, 5))

    sns.displot(
        data = df.isnull().melt(value_name = "missing"),
        y = "variable",
        hue = "missing",
        multiple = "fill",
        height = 9.25)

    plt.axvline(0.3, color = "r");



    # --- PREPROCESSING ───────────────────────────────────────

    df = df.dropna(subset=["customerid"])
    df.shape



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    missing_values(df)

    plt.figure(figsize = (10, 5))

    sns.displot(
        data = df.isnull().melt(value_name = "missing"),
        y = "variable",
        hue = "missing",
        multiple = "fill",
        height = 9.25)

    plt.axvline(0.3, color = "r");

    df.sample(10)

    missing_values(df)

    df.sample(10)

    df = df[(df['unitprice'] > 0) & (df['quantity'] > 0)]

    df.sample(10)

    print("There are", df.duplicated(subset=None, keep='first').sum(), "duplicated observations in the dataset.")
    print(df.duplicated(subset=None, keep='first').sum(), "Duplicated observations are dropped!")
    df.drop_duplicates(keep='first', inplace=True)

    cprint("The Average Number of Unqiue Items By Order",'blue')
    df.groupby(["invoiceno", "stockcode", "description"])["quantity"].mean()

    cprint("The Average Number of Unqiue Items By Customer",'blue')
    df.groupby(["customerid", "stockcode", "description"])["quantity"].mean()

    df.groupby(['customerid', 'invoiceno', 'stockcode', "description"])['quantity'].mean()

    df['total_price'] = df['quantity'] * df['unitprice']
    df.head(3)

    df.groupby("country")[['total_price']].sum().sort_values(by='total_price', ascending=False)

    total_revenue = df.groupby("country")["total_price"].sum().sort_values(ascending=False)
    total_revenue

    customer_num = df.groupby("country")['customerid'].nunique().sort_values(ascending=False)
    customer_num

    customer_num.sum()

    dfg1 = df.groupby('country')["customerid"].nunique().sort_values(ascending=False)
    fig = px.bar(x=dfg1.index, 
                 y=dfg1, 
                 title="Customers By Countries", 
                 labels=dict(x="Countries", y="Total Number of Customers"))

    # also works with graph_objects:
    # fig = go.Figure(go.Bar(x=dfg.index, y=dfg))
    fig.show()

    dfg1_w_oUK = df.groupby('country')["customerid"].nunique().sort_values(ascending=False).iloc[1:]
    fig = px.bar(x=dfg1_w_oUK.index, 
                 y=dfg1_w_oUK, 
                 title="Customers By Countries Without The UK", 
                 labels=dict(x="Countries", y="Total Number of Customers Without The UK"))

    # also works with graph_objects:
    # fig = go.Figure(go.Bar(x=dfg.index, y=dfg))
    fig.show()

    fig = px.treemap(dfg1, path=[dfg1.index], values='customerid', width=950, height=600)
    fig.update_layout(title_text='Customers By Countries',
                      title_x=0.5, title_font=dict(size=20)
                      )
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.show()

    dfg1

    dfg1 = pd.DataFrame(dfg1).reset_index()
    dfg1

    dfg1["country"] = dfg1["country"].str.split(" ").str.join("_")
    dfg1

    dfg1_list = dict(zip(dfg1['country'].tolist(), dfg1['customerid'].tolist()))
    dfg1_list

    #Define a list of stop words
    stopwords = ['country', 'customerid', 'total_price']

    #A function to generate the word cloud from text
    def generate_wordcloud_frequencies(data, title):
        cloud = WordCloud(width=400,
                          height=200,
                          background_color="#32fcbc",
                          max_words=150,
                          colormap='seismic',
                          stopwords=stopwords,
                          collocations=True).generate_from_frequencies(data)
        plt.figure(figsize=(13, 13))
        plt.imshow(cloud)
        plt.axis('off')
        plt.title(title, fontsize=13)
        plt.show()
    
    #Use the function to generate the wordcloud by fequencies
    generate_wordcloud_frequencies(dfg1_list, 'Customers By Countries')

    from IPython.display import display
    from PIL import Image
    import numpy as np

    # Create an array from the image you want to use as a mask
    ## Your file path will look different

    path="./shokunin_United_Kingdom_map.png"
    display(Image.open(path))
    UK_mask = np.array(Image.open(path))
    UK_mask

    # A similar function, but using the mask

    #Define a list of stop words
    stopwords = ['country', 'customerid', 'total_price']

    def generate_wordcloud_mask(data, title, mask=None):
        cloud = WordCloud(#scale=3,
                          width=350,
                          height=400,
                          #max_words=150,
                          colormap='gist_heat',
                          mask=mask,
                          background_color='#bbfce8',
                          stopwords=stopwords,
                          collocations=True).generate_from_text(data)
        plt.figure(figsize=(10, 8))
        plt.imshow(cloud)
        plt.axis('off')
        plt.title(title)
        plt.show()
    
    # Use the function with the UK_mask and our mask to create wordcloud     
    generate_wordcloud_mask(str(dfg1), 'Customers By Countries', mask=UK_mask)

    import plotly.graph_objects as go

    dfg2 = df.groupby('country')["total_price"].sum().sort_values(ascending=False)
    fig = px.bar(x=dfg2.index, 
                 y=dfg2, 
                 title="Total Cost (£) By Countries", 
                 labels=dict(x="Countries", y="Total Cost (£)"))

    # also works with graph_objects:
    # fig = go.Figure(go.Bar(x=dfg.index, y=dfg))
    fig.show()

    dfg2_w_oUK = df.groupby('country')["customerid"].nunique().sort_values(ascending=False).iloc[1:]
    fig = px.bar(x=dfg2_w_oUK.index, 
                 y=dfg2_w_oUK, 
                 title="Customers By Countries Without The UK", 
                 labels=dict(x="Countries", y="Total Number of Customers Without The UK"))

    # also works with graph_objects:
    # fig = go.Figure(go.Bar(x=dfg.index, y=dfg))
    fig.show()

    fig = px.treemap(dfg2, path=[dfg2.index], values='total_price', width=950, height=600)
    fig.update_layout(title_text='Total Cost (£) By Countries',
                      title_x=0.5, title_font=dict(size=20)
                      )
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.show()

    dfg2 = pd.DataFrame(dfg2).reset_index()
    dfg2["country"] = dfg2["country"].str.split(" ").str.join("_")
    dfg2

    dfg2_list = dict(zip(dfg1['country'].tolist(), dfg2['total_price'].tolist()))
    dfg2_list

    #Use the function to generate the wordcloud by frequencies

    generate_wordcloud_frequencies(dfg2_list, 'Total Cost (£) By Countries')

    # Use the function with the UK_mask and our mask to create wordcloud     

    generate_wordcloud_mask(str(dfg1), 'Total Cost (£) By Countries', mask=UK_mask)

    df_uk = df[df["country"]=="United Kingdom"]
    df_uk.head(3)

    df_uk.groupby(["stockcode", "description"])[["quantity"]].sum().sort_values(by="quantity", ascending=False)

    first_looking(df_uk)
    duplicate_values(df_uk)

    min_invoice_date = min(df_uk['invoicedate'])
    min_invoice_date

    max_invoice_date = max(df_uk['invoicedate'])
    max_invoice_date

    df_uk['last_purchase_date'] = df_uk.groupby('customerid')['invoicedate'].transform(max)

    df_uk['last_purchase_date'] = pd.to_datetime(df_uk['last_purchase_date']).dt.date

    df_uk.sample(10)

    df_uk['last_purchase_date'] = pd.to_datetime(df_uk['last_purchase_date'])
    df_uk['invoicedate'] = pd.to_datetime(df_uk['invoicedate'])

    # alternative code
    # ref_date = datetime(2011, 12, 16)
    # ref_date = df_uk['invoicedate'].max()

    df_uk['ref_date'] = df_uk['invoicedate'].max() + timedelta(days=7)

    df_uk['ref_date'] = df_uk['ref_date'].dt.date

    df_uk.sample(5)

    df_uk['ref_date'] = pd.to_datetime(df_uk['ref_date'])

    df_uk['date'] = pd.to_datetime(df_uk['invoicedate'])

    df_uk['date'] = df_uk['date'].dt.date

    df_uk.sample(10)

    customer_recency = pd.DataFrame(df_uk.groupby('customerid', as_index=False).date.max())
    customer_recency.head()

    df_uk[df_uk["customerid"] == 12346.00][["customerid", 'last_purchase_date']]

    df_uk["customer_recency"] = df_uk["ref_date"] - df_uk["last_purchase_date"]
    df_uk[["customerid", 'last_purchase_date', "ref_date", "customer_recency"]]



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_uk['recency2'] = pd.to_numeric(df_uk['customer_recency'].dt.days.astype('int64'))
    df_uk[["customerid", 'last_purchase_date', "ref_date", "customer_recency", 'recency2']]

    # alternative code
    # df_uk["recency"] = df_uk.groupby('customerid')['last_purchase_date'].apply(lambda x: ref_date - x)
    # df_uk['recency'] = pd.to_numeric(df_uk['recency'].dt.days, downcast='integer')

    customer_recency = df_uk.groupby('customerid', as_index=False)['recency2'].mean()
    customer_recency.rename(columns={'recency2':'Recency'}, inplace=True)
    customer_recency.sort_values(by='Recency', ascending=False).head()

    df_uk.drop(['last_purchase_date'], axis = 1, inplace=True)
    df_uk.head(3)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    plt.figure(figsize=(10, 8))
    graph = sns.scatterplot(data=df_uk, x="customerid", y="recency2")
    graph.axhline(100, color="blue")
    plt.show();

    fig = px.scatter(df_uk, x="customerid", y="recency2")
    fig.show()

    fig = px.histogram(df_uk, x="recency2", nbins=70)
    fig.show()

    dfUK_copy = df_uk.copy()
    dfUK_copy.head(3)

    print("There are", dfUK_copy.duplicated(subset=None, keep='first').sum(), "duplicated observations in the dataset.")
    print(dfUK_copy.duplicated(subset=None, keep='first').sum(), "Duplicated observations are dropped!")
    dfUK_copy.drop_duplicates(keep='first', inplace=True)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    customer_frequency = dfUK_copy.groupby('customerid', as_index=False)['invoiceno'].nunique()
    customer_frequency.rename(columns={'invoiceno':'Frequency'}, inplace=True)
    customer_frequency.sort_values(by='Frequency', ascending=False)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    customer_frequency.nlargest(5, "Frequency")

    fig = px.scatter(customer_frequency, x="customerid", y="Frequency")
    fig.show()

    fig = px.histogram(customer_frequency, x="customerid", y="Frequency", nbins=3920)
    fig.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    customer_monetary = dfUK_copy.groupby('customerid', as_index=False)['total_price'].sum()
    customer_monetary.rename(columns={'total_price':'Monetary'}, inplace=True)
    customer_monetary.sort_values(by='Monetary', ascending=False).head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    customer_monetary.nlargest(5, "Monetary")

    fig = px.scatter(customer_monetary, x="customerid", y="Monetary")
    fig.show()

    fig = px.histogram(customer_monetary, x="Monetary", nbins=50)
    fig.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    customer_rfm = pd.merge(pd.merge(customer_recency, customer_frequency, on='customerid'), customer_monetary, on='customerid')
    customer_rfm.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    quantiles = customer_rfm.quantile(q = [0.25, 0.50, 0.75])
    quantiles

    fig = px.pie(df, values = customer_rfm['Recency_Score'].value_counts(), 
                 names = (customer_rfm["Recency_Score"].value_counts()).index, 
                 title = 'Recency Score Distribution')
    fig.show()

    fig = px.pie(df, values = customer_rfm['Frequency_Score'].value_counts(), 
                 names = (customer_rfm["Frequency_Score"].value_counts()).index, 
                 title = 'Frequency Score Distribution')
    fig.show()

    fig = px.pie(df, values = customer_rfm['Monetary_Score'].value_counts(), 
                 names = (customer_rfm["Monetary_Score"].value_counts()).index, 
                 title = 'Monetary Score Distribution')
    fig.show()

    fig = px.histogram(customer_rfm, x = customer_rfm['Customer_RFM_Score'].value_counts().index, 
                       y = customer_rfm['Customer_RFM_Score'].value_counts().values, 
                       title = 'Customer RFM Score Distribution',
                       labels = dict(x = "Customer_RFM_Score", y ="counts"))
    fig.show()

    customer_rfm['RFM_Label'] = customer_rfm['Recency_Score'] + customer_rfm['Frequency_Score'] + customer_rfm['Monetary_Score']

    customer_rfm.sample(8)

    customer_rfm.groupby(['Customer_RFM_Score']).size().sort_values(ascending=False)[:]

    fig = px.pie(df, values = customer_rfm['RFM_Label'].value_counts(), 
                 names = (customer_rfm["RFM_Label"].value_counts()).index, 
                 title = 'RFM Label Distribution')
    fig.show()

    customer_rfm['RFM_Label'].min()

    customer_rfm['RFM_Label'].max()

    np.sort(customer_rfm['RFM_Label'].unique())

    segments = {'Customer_Segment':['Champion', 
                                    'Top Loyal Customer', 
                                    'Loyal Customer', 
                                    'Top Recent Customer', 
                                    'Recent Customer', 
                                    'Top Customer Needed Attention', 
                                    'Customer Needed Attention', 
                                    'Top Lost Customer', 
                                    'Lost Customer'],
                'RFM':['(2|3|4)-(4)-(4)', 
                       '(3)-(1|2|3|4)-(3|4)', 
                       '(3)-(1|2|3|4)-(1|2)', 
                       '(4)-(1|2|3|4)-(3|4)', 
                       '(4)-(1|2|3|4)-(1|2)',
                       '(2|3)-(1|2|3|4)-(3|4)', 
                       '(2|3)-(1|2|3|4)-(1|2)',
                       '(1)-(1|2|3|4)-(3|4)', 
                       '(1)-(1|2|3|4)-(1|2)',]}

    pd.DataFrame(segments)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    customer_rfm['Customer_Category'] = customer_rfm["Customer_RFM_Score"].apply(categorizer)
    customer_rfm



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    customer_rfm.groupby('Customer_Category').RFM_Label.mean()



    # --- PREPROCESSING ───────────────────────────────────────

    customer_rfm['Customer_Category'].value_counts(dropna=False, normalize=True)*100



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    customer_rfm[customer_rfm['Customer_Category'] == "Recent Customer"].sample(8)

    customer_rfm[customer_rfm['Customer_Category'] == "Top Recent Customer"].sample(8)

    customer_rfm[customer_rfm['Customer_Category'] == "Champion"].sample(8)

    customer_rfm[customer_rfm['Customer_Category'] == "Top Loyal Customer"].sample(8)

    customer_rfm[customer_rfm['Customer_Category'] == "Lost Customer"].sample(8)

    fig = px.histogram(customer_rfm, 
                       x = customer_rfm['Customer_Category'].value_counts().index, 
                       y = customer_rfm['Customer_Category'].value_counts().values, 
                       title = 'Customer Category Distribution',
                       labels = dict(x = "Customer_Category", y ="counts"))
    fig.show()

    fig = px.pie(df, 
                 values = customer_rfm['Customer_Category'].value_counts(), 
                 names = (customer_rfm["Customer_Category"].value_counts()).index, 
                 title = 'Customer Category Distribution')
    fig.show()

    customer_rfm



    # --- FEATURE ENGINEERING ─────────────────────────────────

    Avg_RFM_Label = customer_rfm.groupby('Customer_Category').RFM_Label.mean()
    Size_RFM_Label = customer_rfm['Customer_Category'].value_counts()
    df_customer_segmentation = pd.concat([Avg_RFM_Label, Size_RFM_Label], axis=1).rename(columns={'RFM_Label':'Avg_RFM_Label',
                                                                               'Customer_Category':'Size_RFM_Label'})
    df_customer_segmentation



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig = px.histogram(customer_rfm, 
                       x = customer_rfm.groupby('Customer_Category').RFM_Label.mean().sort_values(ascending=False).index, 
                       y = customer_rfm.groupby('Customer_Category').RFM_Label.mean().sort_values(ascending=False).values, 
                       title = 'The Average of RFM Label',
                       labels = dict(x = "Customer Segments (Categories)", y ="RFM Label Mean Values"))
    fig.show()

    fig = px.treemap(df_customer_segmentation, 
                     path=[df_customer_segmentation.index], 
                     values='Avg_RFM_Label', 
                     width=950, height=600)

    fig.update_layout(title_text='The Average of Each Customer Segments',
                      title_x=0.5, title_font=dict(size=20)
                      )
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.show()

    fig = px.histogram(customer_rfm, 
                       x = customer_rfm['Customer_Category'].value_counts().index, 
                       y = customer_rfm['Customer_Category'].value_counts().values, 
                       title = 'The Size of RFM Label',
                       labels = dict(x = "Customer Segments (Categories)", y ="RFM Label Mean Values"))
    fig.show()

    fig = px.treemap(df_customer_segmentation, 
                     path=[df_customer_segmentation.index], 
                     values='Size_RFM_Label', 
                     width=950, height=600)

    fig.update_layout(title_text='Customer Segmentation',
                      title_x=0.5, title_font=dict(size=20)
                      )
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.show()



    # --- PREPROCESSING ───────────────────────────────────────

    segmentation = pd.DataFrame(customer_rfm.Customer_Category.value_counts(dropna=False).sort_values(ascending=False))
    segmentation.reset_index(inplace=True)
    segmentation.rename(columns={'index':'Customer Category', 'Customer_Category':'The Number Of Customer'}, inplace=True)
    segmentation



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig = px.bar(segmentation, x='Customer Category', y='The Number Of Customer')
    fig.show()

    segmentation

    #import squarify as sq

    fig = plt.gcf()
    ax = fig.add_subplot()
    fig.set_size_inches(13, 8)
    sq.plot(sizes=segmentation['The Number Of Customer'], 
                          label=['Lost Customer', 
                                'Customer Needed Attention', 
                                'Top Recent Customer', 
                                'Top Loyal Customer', 
                                'Top Customer Needed Attention', 
                                'Loyal Customer', 
                                'Champion', 
                                'Recent Customer', 
                                'Top Lost Customer'], 
                                alpha=0.8, 
                                color=["red", "#48BCF5", "#DD6AE1", "blue", "cyan", "magenta", '#B20CB7', "#A4E919"])
    plt.title("RFM Segments", fontsize=18, fontweight="bold")
    plt.axis('off')
    plt.show()

    # import plotly.express as px

    fig = px.treemap(segmentation,
                     path=[segmentation['Customer Category']], 
                     values='The Number Of Customer', 
                     width=900, 
                     height=600)
    fig.update_layout(title="RFM Segments",
                      title_x = 0.5, title_font = dict(size=18),
                     )
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.show()

    segment_text = segmentation["Customer Category"].str.split(" ").str.join("_")
    all_segments = " ".join(segment_text)

    wc = WordCloud(background_color="orange", 
                   #max_words=250, 
                   max_font_size=256, 
                   random_state=42,
                   width=800, height=400)
    wc.generate(all_segments)
    plt.figure(figsize = (16, 15))
    plt.imshow(wc)
    plt.title("RFM Segments", fontsize=18, fontweight="bold")
    plt.axis('off')
    plt.show()

    customer_rfm.sample(5)

    df_temp = customer_rfm.corr()

    feature =[]
    collinear=[]

    for col in df_temp.columns:
        for i in df_temp.index:
            if (df_temp[col][i]> .85 and df_temp[col][i] < 1) or (df_temp[col][i]< -.85 and df_temp[col][i] > -1) :
                    feature.append(col)
                    collinear.append(i)
                    print(Fore.RED + f"\033[1mmulticolinearity alert\033[0m between {col} - {i}")
            else:
                print(f"For {col} and {i}, there is \033[1mNO multicollinearity problem\033[0m") 

    unique_list = list(set(feature+collinear))

    print(colored('*'*80, 'cyan', attrs=['bold']))
    print("\033[1mThe total number of strong corelated features:\033[0m", len(unique_list))

    g = sns.scatterplot(data=customer_rfm, x="RFM_Label", y="Recency_Score", hue="Customer_Category")
    g.legend(loc='center left', bbox_to_anchor=(1, 0.85), ncol=1);

    g = sns.scatterplot(data=customer_rfm, x="RFM_Label", y="Monetary_Score", hue="Customer_Category")
    g.legend(loc='center left', bbox_to_anchor=(1, 0.85), ncol=1);

    g = sns.scatterplot(data=customer_rfm, x="RFM_Label", y="Frequency_Score", hue="Customer_Category")
    g.legend(loc='center left', bbox_to_anchor=(1, 0.85), ncol=1);

    fig = px.scatter_matrix(customer_rfm, 
                            dimensions=['Recency', 'Frequency', 'Monetary'], 
                            color="Customer_Category",
                            width=1000, height=800)

    fig.show()

    customer_rfm[['Recency', 'Frequency', 'Monetary']].sample(10)

    matrix = np.triu(customer_rfm[['Recency','Frequency','Monetary']].corr())
    fig, ax = plt.subplots(figsize=(11, 6)) 
    sns.heatmap (customer_rfm[['Recency','Frequency','Monetary']].corr(), 
                 annot=True, 
                 fmt= '.2f', 
                 vmin=-1, 
                 vmax=1, 
                 center=0, 
                 cmap='coolwarm',
                 mask=matrix, 
                 ax=ax);

    customer_rfm.set_index("customerid", inplace=True)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # **Handling with Skewness - np.log**

    skew_limit = 0.75 # This is our threshold-limit to evaluate skewness. Overall below abs(1) seems acceptable for the linear models. 
    skew_vals = customer_rfm[['Recency', 'Frequency', 'Monetary']].skew()
    skew_cols = skew_vals[abs(skew_vals)> skew_limit].sort_values(ascending=False)
    skew_cols

    rfm_log = customer_rfm[skew_cols.index].copy()

    for col in skew_cols.index.values:
        rfm_log[col] = rfm_log[col].apply(np.log1p)

        print(rfm_log.skew())
    print()

    rfm_log.iplot(kind='histogram', subplots=True, bins=50);



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Interpreting Skewness 

    for skew in rfm_log.skew():
        if -0.75 < skew < 0.75:
            print ("A skewness value of", '\033[1m', Fore.GREEN, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.GREEN, "symmetric", '\033[0m')
        elif  -0.75 < skew < -1.0 or 0.75 < skew < 1.0:
            print ("A skewness value of", '\033[1m', Fore.YELLOW, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.YELLOW, "moderately skewed", '\033[0m')
        else:
            print ("A skewness value of", '\033[1m', Fore.RED, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.RED, "highly skewed", '\033[0m')

    # **Handling with Skewness - Power Transformer**

    rfm_before_trans = customer_rfm[skew_cols.index].copy()
    pt = PowerTransformer(method='yeo-johnson')
    trans= pt.fit_transform(rfm_before_trans)
    rfm_trans = pd.DataFrame(trans, columns =skew_cols.index )

    print(rfm_trans.skew())
    print()

    # Interpreting Skewness 

    for skew in rfm_trans.skew():
        if -0.75 < skew < 0.75:
            print ("A skewness value of", '\033[1m', Fore.GREEN, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.GREEN, "symmetric", '\033[0m')
        elif  -0.75 < skew < -1.0 or 0.75 < skew < 1.0:
            print ("A skewness value of", '\033[1m', Fore.YELLOW, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.YELLOW, "moderately skewed", '\033[0m')
        else:
            print ("A skewness value of", '\033[1m', Fore.RED, skew, '\033[0m', "means that the distribution is approx.", '\033[1m', Fore.RED, "highly skewed", '\033[0m')

    fig = px.scatter_3d(rfm_trans, 
                        x='Recency',
                        y='Frequency',
                        z='Monetary',
                        color='Frequency')
    fig.show();



    # --- MODEL TRAINING ──────────────────────────────────────

    # Normalize the variables with StandardScaler
    # from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(rfm_trans)

    #Store it separately for clustering
    rfm_scaled = scaler.transform(rfm_trans)



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- PYCARET AUTOML ──────────────────────────────

            from pycaret.clustering import *

            clust_setup = setup(data=df0, normalize=True, session_id=42, verbose=False)

            # Create K-Means model
            kmeans_model = create_model('kmeans')
            print(kmeans_model)

            # Assign cluster labels to data
            clustered_df = assign_model(kmeans_model)
            clustered_df.head()

            # Evaluate clustering
            plot_model(kmeans_model, plot='elbow')

            # Silhouette plot
            plot_model(kmeans_model, plot='silhouette')

            # Distribution plot
            plot_model(kmeans_model, plot='distribution')



        except ImportError:

            print('[AutoML] LazyPredict/PyCaret not installed — skipping AutoML block')

        except Exception as _automl_err:

            print(f'[AutoML] AutoML block failed: {_automl_err}')


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model training for Customer segmentation for an E-commerce company")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
