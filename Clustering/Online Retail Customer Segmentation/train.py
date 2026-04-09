#!/usr/bin/env python3
"""
Model training for 3 Online retail customer segmentation

Auto-generated from: 3 Online retail customer segmentation.ipynb
Project: 3 Online retail customer segmentation
Category: Clustering | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import nltk
from pycaret.clustering import *

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

    #__________________
    # read the datafile
    df_initial = load_dataset('online_retail_customer_segmentation')
    print('Dataframe dimensions:', df_initial.shape)
    #______
    df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])
    #____________________________________________________________
    # gives some infos on columns types and numer of null values
    tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})
    tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values (nb)'}))
    tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.
                             rename(index={0:'null values (%)'}))
    display(tab_info)
    #__________________
    # show first lines
    display(df_initial[:5])



    # --- PREPROCESSING ───────────────────────────────────────

    df_initial.dropna(axis = 0, subset = ['CustomerID'], inplace = True)
    print('Dataframe dimensions:', df_initial.shape)
    #____________________________________________________________
    # gives some infos on columns types and numer of null values
    tab_info=pd.DataFrame(df_initial.dtypes).T.rename(index={0:'column type'})
    tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0:'null values (nb)'}))
    tab_info=tab_info.append(pd.DataFrame(df_initial.isnull().sum()/df_initial.shape[0]*100).T.
                             rename(index={0:'null values (%)'}))
    display(tab_info)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    print('Entrées dupliquées: {}'.format(df_initial.duplicated().sum()))
    df_initial.drop_duplicates(inplace = True)

    temp = df_initial[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
    temp = temp.reset_index(drop = False)
    countries = temp['Country'].value_counts()
    print('Nb. de pays dans le dataframe: {}'.format(len(countries)))

    data = dict(type='choropleth',
    locations = countries.index,
    locationmode = 'country names', z = countries,
    text = countries.index, colorbar = {'title':'Order nb.'},
    colorscale=[[0, 'rgb(224,255,255)'],
                [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
                [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
                [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
                [1, 'rgb(227,26,28)']],    
    reversescale = False)
    #_______________________
    layout = dict(title='Number of orders per country',
    geo = dict(showframe = True, projection={'type':'mercator'}))
    #______________
    choromap = go.Figure(data = [data], layout = layout)
    iplot(choromap, validate=False)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    temp = df_initial.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
    nb_products_per_basket = temp.rename(columns = {'InvoiceDate':'Number of products'})
    nb_products_per_basket[:10].sort_values('CustomerID')

    nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x:int('C' in x))
    display(nb_products_per_basket[:5])
    #______________________________________________________________________________________________
    n1 = nb_products_per_basket['order_canceled'].sum()
    n2 = nb_products_per_basket.shape[0]
    print('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df_check = df_initial[df_initial['Quantity'] < 0][['CustomerID','Quantity',
                                                       'StockCode','Description','UnitPrice']]
    for index, col in  df_check.iterrows():
        if df_initial[(df_initial['CustomerID'] == col[0]) & (df_initial['Quantity'] == -col[1]) 
                    & (df_initial['Description'] == col[2])].shape[0] == 0: 
            print(df_check.loc[index])
            print(15*'-'+'>'+' HYPOTHESIS NOT FULFILLED')
            break

    df_check = df_initial[(df_initial['Quantity'] < 0) & (df_initial['Description'] != 'Discount')][
                                     ['CustomerID','Quantity','StockCode',
                                      'Description','UnitPrice']]

    for index, col in  df_check.iterrows():
        if df_initial[(df_initial['CustomerID'] == col[0]) & (df_initial['Quantity'] == -col[1]) 
                    & (df_initial['Description'] == col[2])].shape[0] == 0: 
            print(index, df_check.loc[index])
            print(15*'-'+'>'+' HYPOTHESIS NOT FULFILLED')
            break

    df_cleaned = df_initial.copy(deep = True)
    df_cleaned['QuantityCanceled'] = 0

    entry_to_remove = [] ; doubtfull_entry = []

    for index, col in  df_initial.iterrows():
        if (col['Quantity'] > 0) or col['Description'] == 'Discount': continue        
        df_test = df_initial[(df_initial['CustomerID'] == col['CustomerID']) &
                             (df_initial['StockCode']  == col['StockCode']) & 
                             (df_initial['InvoiceDate'] < col['InvoiceDate']) & 
                             (df_initial['Quantity']   > 0)].copy()
        #_________________________________
        # Cancelation WITHOUT counterpart
        if (df_test.shape[0] == 0): 
            doubtfull_entry.append(index)
        #________________________________
        # Cancelation WITH a counterpart
        elif (df_test.shape[0] == 1): 
            index_order = df_test.index[0]
            df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index)        
        #______________________________________________________________
        # Various counterparts exist in orders: we delete the last one
        elif (df_test.shape[0] > 1): 
            df_test.sort_index(axis=0 ,ascending=False, inplace = True)        
            for ind, val in df_test.iterrows():
                if val['Quantity'] < -col['Quantity']: continue
                df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
                entry_to_remove.append(index) 
                break



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_cleaned.drop(entry_to_remove, axis = 0, inplace = True)
    df_cleaned.drop(doubtfull_entry, axis = 0, inplace = True)
    remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
    print("nb of entries to delete: {}".format(remaining_entries.shape[0]))
    remaining_entries[:5]



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df_cleaned[(df_cleaned['CustomerID'] == 14048) & (df_cleaned['StockCode'] == '22464')]

    list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
    list_special_codes

    for code in list_special_codes:
        print("{:<15} -> {:<30}".format(code, df_cleaned[df_cleaned['StockCode'] == code]['Description'].unique()[0]))

    df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
    df_cleaned.sort_values('CustomerID')[:5]



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #___________________________________________
    # somme des achats / utilisateur & commande
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
    basket_price = temp.rename(columns = {'TotalPrice':'Basket Price'})
    #_____________________
    # date de la commande
    df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
    df_cleaned.drop('InvoiceDate_int', axis = 1, inplace = True)
    basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
    #______________________________________
    # selection des entrées significatives:
    basket_price = basket_price[basket_price['Basket Price'] > 0]
    basket_price.sort_values('CustomerID')[:6]



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #____________________
    # Décompte des achats
    price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
    count_price = []
    for i, price in enumerate(price_range):
        if i == 0: continue
        val = basket_price[(basket_price['Basket Price'] < price) &
                           (basket_price['Basket Price'] > price_range[i-1])]['Basket Price'].count()
        count_price.append(val)
    #____________________________________________
    # Représentation du nombre d'achats / montant        
    plt.rc('font', weight='bold')
    f, ax = plt.subplots(figsize=(11, 6))
    colors = ['yellowgreen', 'gold', 'wheat', 'c', 'violet', 'royalblue','firebrick']
    labels = [ '{}<.<{}'.format(price_range[i-1], s) for i,s in enumerate(price_range) if i != 0]
    sizes  = count_price
    explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
    ax.pie(sizes, explode = explode, labels=labels, colors = colors,
           autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
           shadow = False, startangle=0)
    ax.axis('equal')
    f.text(0.5, 1.01, "Répartition des montants des commandes", ha='center', fontsize = 18);



    # --- PREPROCESSING ───────────────────────────────────────

    is_noun = lambda pos: pos[:2] == 'NN'

    def keywords_inventory(dataframe, colonne = 'Description'):
        stemmer = nltk.stem.SnowballStemmer("english")
        keywords_roots  = dict()  # collect the words / root
        keywords_select = dict()  # association: root <-> keyword
        category_keys   = []
        count_keywords  = dict()
        icount = 0
        for s in dataframe[colonne]:
            if pd.isnull(s): continue
            lines = s.lower()
            tokenized = nltk.word_tokenize(lines)
            nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 
        
            for t in nouns:
                t = t.lower() ; racine = stemmer.stem(t)
                if racine in keywords_roots:                
                    keywords_roots[racine].add(t)
                    count_keywords[racine] += 1                
                else:
                    keywords_roots[racine] = {t}
                    count_keywords[racine] = 1
    
        for s in keywords_roots.keys():
            if len(keywords_roots[s]) > 1:  
                min_length = 1000
                for k in keywords_roots[s]:
                    if len(k) < min_length:
                        clef = k ; min_length = len(k)            
                category_keys.append(clef)
                keywords_select[s] = clef
            else:
                category_keys.append(list(keywords_roots[s])[0])
                keywords_select[s] = list(keywords_roots[s])[0]
                   
        print("Nb of keywords in variable '{}': {}".format(colonne,len(category_keys)))
        return category_keys, keywords_roots, keywords_select, count_keywords



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_produits = pd.DataFrame(df_initial['Description'].unique()).rename(columns = {0:'Description'})



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_produits)

    list_products = []
    for k,v in count_keywords.items():
        list_products.append([keywords_select[k],v])
    list_products.sort(key = lambda x:x[1], reverse = True)

    liste = sorted(list_products, key = lambda x:x[1], reverse = True)
    #_______________________________
    plt.rc('font', weight='normal')
    fig, ax = plt.subplots(figsize=(7, 25))
    y_axis = [i[1] for i in liste[:125]]
    x_axis = [k for k,i in enumerate(liste[:125])]
    x_label = [i[0] for i in liste[:125]]
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 13)
    plt.yticks(x_axis, x_label)
    plt.xlabel("Nb. of occurences", fontsize = 18, labelpad = 10)
    ax.barh(x_axis, y_axis, align = 'center')
    ax = plt.gca()
    ax.invert_yaxis()
    #_______________________________________________________________________________________
    plt.title("Words occurence",bbox={'facecolor':'k', 'pad':5}, color='w',fontsize = 25)
    plt.show()

    list_products = []
    for k,v in count_keywords.items():
        word = keywords_select[k]
        if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
        if len(word) < 3 or v < 13: continue
        if ('+' in word) or ('/' in word): continue
        list_products.append([word, v])
    #______________________________________________________    
    list_products.sort(key = lambda x:x[1], reverse = True)
    print('mots conservés:', len(list_products))

    liste_produits = df_cleaned['Description'].unique()
    X = pd.DataFrame()
    for key, occurence in list_products:
        X.loc[:, key] = list(map(lambda x:int(key.upper() in x), liste_produits))

    threshold = [0, 1, 2, 3, 5, 10]
    label_col = []
    for i in range(len(threshold)):
        if i == len(threshold)-1:
            col = '.>{}'.format(threshold[i])
        else:
            col = '{}<.<{}'.format(threshold[i],threshold[i+1])
        label_col.append(col)
        X.loc[:, col] = 0

    for i, prod in enumerate(liste_produits):
        prix = df_cleaned[ df_cleaned['Description'] == prod]['UnitPrice'].mean()
        j = 0
        while prix > threshold[j]:
            j+=1
            if j == len(threshold): break
        X.loc[i, label_col[j-1]] = 1

    print("{:<8} {:<20} \n".format('gamme', 'nb. produits') + 20*'-')
    for i in range(len(threshold)):
        if i == len(threshold)-1:
            col = '.>{}'.format(threshold[i])
        else:
            col = '{}<.<{}'.format(threshold[i],threshold[i+1])    
        print("{:<10}  {:<20}".format(col, X.loc[:, col].sum()))



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- PYCARET AUTOML ──────────────────────────────

            from pycaret.clustering import *

            clust_setup = setup(data=df_initial, normalize=True, session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model training for 3 Online retail customer segmentation")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
