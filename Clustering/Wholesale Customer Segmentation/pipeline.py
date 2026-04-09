#!/usr/bin/env python3
"""
Full pipeline for 8 wholesale customer segmentation

Auto-generated from: 8 Wholesale customer segmentation.ipynb
Project: 8 wholesale customer segmentation
Category: Clustering | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from scipy.stats import boxcox, shapiro, probplot
from scipy.cluster.hierarchy import dendrogram, linkage
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Additional imports extracted from mixed cells
from pycaret.clustering import *

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

    df=load_dataset('wholesale_customer_segmentation')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.head()

    #Checking for nulls

    df.isnull().sum()

    #No nulls -great

    #Checking for data types

    df.info()

    #Statistical summary

    df.describe()

    #Let´s convert the categorical variables to meaningful data

    df.Region.value_counts()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df.Region=df.Region.map({3:'Other',2:'Lisbon',1:'Oporto'})



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.Channel.value_counts()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df.Channel=df.Channel.map({1:'Horeca',2:'Retail'})



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #Distribution of numeric variables

    df.hist(figsize=(20,10),bins=30, color='lightblue', edgecolor='black')
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data=df.copy()

    lamb=[]
    confidence=[]
    for i in data.columns[2:]:
        data[i], coef, conf=boxcox(data[i]+0.0000001, alpha=0.05)   #We add a tiny constant as values need to be positive 
                                                                    #for Box-Cox
        lamb.append(coef)
        confidence.append(conf)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #Now the data resembles more the normal distribution

    data.hist(figsize=(20,10),bins=30, color='lightblue', edgecolor='black')
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    norm=data.iloc[:,2:] #Numerical data

    scaler=MinMaxScaler()
    columns=data.columns[2:]
    norm=scaler.fit_transform(data.iloc[:,2:]) #Only numeric variables
    norm=pd.DataFrame(norm, columns=columns)

    plt.figure(figsize=(10,7))
    sns.boxplot(data=norm)
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    for i in norm.columns:
        iqr=np.percentile(norm[i], 75)-np.percentile(norm[i], 25)
        per75=np.percentile(norm[i], 75)
        per25=np.percentile(norm[i], 25)
        norm[i]=np.where(norm[i]>per75+1.5*iqr, per75+1.5*iqr,norm[i] )
        norm[i]=np.where(norm[i]<per25-1.5*iqr, per25-1.5*iqr,norm[i] )



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #Now we don´t have outliers

    plt.figure(figsize=(15,10))
    sns.boxplot(data=norm)
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Let´s also check for normality of the numeric variables now. However it is not a required assumptions 
    #of KMeans but can produce better results. Shapiro Wilks test may be employed. 
    #The null hypothesis is that the data is normal.



    normality=pd.DataFrame(index=['p-value', 'test-statistic'])
    for i in norm.columns:
        normality[i]=shapiro(norm[i])
    
    normality.T



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #We can also plot the normality via que quantile quantile plots from scipy statistical library.

    plt.figure(figsize=(15,15))
    for i,var in enumerate(norm.columns):
        plt.subplot(3,3,i+1)
        probplot(norm[var], plot=plt)
        plt.title(f'{var}')
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    data=pd.get_dummies(data=df, columns= ['Region','Channel'], drop_first=True)

    #Uniting our categorical dummified variables with numerical normalized data.

    data.iloc[:,:6]=norm



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- PYCARET AUTOML ──────────────────────────────

            from pycaret.clustering import *

            clust_setup = setup(data=df, normalize=True, session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for 8 wholesale customer segmentation")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
