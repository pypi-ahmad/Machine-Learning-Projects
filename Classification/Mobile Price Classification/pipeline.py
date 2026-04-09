#!/usr/bin/env python3
"""
Full pipeline for Mobile price classification

Auto-generated from: Mobile_price_classification.ipynb
Project: Mobile price classification
Category: Classification | Task: classification
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate

# for visuallization
import plotly.express as px
# Additional imports extracted from mixed cells
from warnings import filterwarnings
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

    # --- ADDITIONAL PROCESSING ───────────────────────────────

    from warnings import filterwarnings
    filterwarnings('ignore')



    # --- DATA LOADING ────────────────────────────────────────

    train=load_dataset('mobile_price_classification')
    test=pd.read_csv('../../data/mobile_price_classification/test.csv')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    print('trian_shape:',train.shape)
    print('test_shape',test.shape)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train.info

    test.info

    train.describe(include='all')

    test.describe(include='all')

    sns.set_style('darkgrid')
    color = 'royalblue'

    fig, ax = plt.subplots(1, 2, figsize=(16,7),dpi=100,facecolor='#459878')
    train.nunique().plot(kind='barh',color='#ff00ae',ax=ax[0],label='train')
    ax[0].grid()
    test.nunique().plot(kind='barh',color='#9845ff',ax=ax[1],label='test')
    ax[1].grid()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    test.drop('id',inplace=True,axis=1)

    train_without_target=train.drop('price_range',axis=1)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    labels = train["price_range"].value_counts().index
    sizes = train["price_range"].value_counts()
    colors = ['green','pink','magenta','cyan']
    plt.figure(figsize = (8,8))
    plt.pie(sizes, labels=labels, rotatelabels=False, autopct='%1.1f%%',colors=colors,shadow=True, startangle=90)
    plt.title('COSTS',color = 'green',fontsize = 15)
    plt.show()

    sns.set_style('darkgrid')
    color = 'royalblue'

    plt.figure(figsize = (12,55))
    i = 0
    for index, col in enumerate(list(train_without_target.columns.values)):
        i += 1 ;
        plt.subplot(21,2, index + i)
        ax = sns.histplot(x = col, data = train_without_target, color = "#326598", stat = "density", common_norm=False)
        sns.kdeplot(x = col, data = train_without_target, color = "pink", linewidth = 5)
        plt.xlabel(col, size = 15)
        plt.title('train')
        # set text on axes
        textstr_train = '\n'.join((
        r'$\mu=%.2f$' %train_without_target[col].mean(),
        r'$\sigma=%.2f$' %train_without_target[col].std(),
        r'$\mathrm{median}=%0.2f$' %np.median(train_without_target[col]),
        r'$\mathrm{min}=%.2f$' %train_without_target[col].min(),
        r'$\mathrm{max}=%.2f$' %train_without_target[col].max()
        ))
        ax.text(0.7, 0.90, textstr_train, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round',facecolor='pink', edgecolor='black', pad=0.5, alpha = 0.5))
    
        plt.subplot(21,2, index + (i+1))
        ax = sns.histplot(x = col, data = test, color = "#326598", stat = "density", common_norm=False)
        sns.kdeplot(x = col, data = test, color = "pink", linewidth = 5)
        plt.xlabel(col, size = 15)
        plt.title('test')
    
        textstr_test = '\n'.join((
        r'$\mu=%.2f$' %test[col].mean(),
        r'$\sigma=%.2f$' %test[col].std(),
        r'$\mathrm{median}=%0.2f$' %np.median(test[col]),
        r'$\mathrm{min}=%.2f$' %test[col].min(),
        r'$\mathrm{max}=%.2f$' %test[col].max()
        ))
        ax.text(0.7, 0.90, textstr_test, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round',facecolor='pink', edgecolor='black', pad=0.5, alpha = 0.5))
   
        plt.grid()

    plt.suptitle("Disturbution Of Features", y = 1, x = 0.55, size = 20,
        fontweight = "bold")
    plt.tight_layout()
    plt.show()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    num_columns=['battery_power','clock_speed', 'fc', 'int_memory','m_dep','mobile_wt','pc','px_height','px_width','ram','sc_h','sc_w','talk_time']
    cat_columns=['blue','dual_sim','four_g','n_cores', 'three_g','touch_screen','wifi']



    # --- FEATURE ENGINEERING ─────────────────────────────────

    colors_cat=['#8B008B','#DC143C','#FFA500','#ff8080','#556B2F','#D2691E','#DAA520']
    new_train=train.replace(to_replace={'blue':[0,1],'dual_sim':[0,1],'four_g':[0,1],
                          'n_cores':[1,2,3,4,5,6,7,8],'three_g':[0,1],'touch_screen':[0,1],'wifi':[0,1]},
               value={'blue':['no-blue','has blue'],
                      'dual_sim':['no-dual-sim','has dual-sim'],
                     'four_g':['no-4G','has-4G'],
                     'n_cores':['1-core','2-cores','3-cores','4-cores','5-cores','6-cores','7-cores','8-cores'],
                     'three_g':['no-3G','has-3G'],
                     'touch_screen':['no-touch','has-touch'],
                     'wifi':['no-wifi','has-wifi']})
    for i,c in enumerate(cat_columns):
        plt.figure(figsize =(5.5, 6.5))
        plt.pie(new_train[c].value_counts() ,labels=list(new_train[c].value_counts().index),shadow = True,autopct='%1.1f%%')
        plt.legend()
        plt.title(c,color=colors_cat[i],fontsize=20)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    for i,c in enumerate(num_columns):
        sns.jointplot(x=c ,y='price_range',data=train,kind='kde',palette = "dict",color='seagreen')
        plt.xlabel(c)
        plt.ylabel('price_range')
        plt.title(c,fontsize=20)

    train.isnull().sum()

    test.isnull().sum()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    train.drop_duplicates(inplace=True)
    train.shape

    test.drop_duplicates(inplace=True)
    test.shape



    # --- FEATURE ENGINEERING ─────────────────────────────────

    target = 'price_range'
    features = train.columns.drop(target)
    colors = ['#a9f943', '#ed67cd', '#23bf00','#5687da']
    CustomPalette = sns.set_palette(sns.color_palette(colors))
    fig, ax = plt.subplots(nrows=7 ,ncols=3, figsize=(15,22), dpi=200)

    for i in range(len(features)):
        x=i//3
        y=i%3
        sns.scatterplot(data=train, x=features[i], y=target,hue=target, ax=ax[x,y],style=target,palette="deep",size=target)
        ax[x,y].set_title('{} vs. {}'.format(target, features[i]), size = 15)
        ax[x,y].set_xlabel(features[i], size = 12)
        ax[x,y].set_ylabel(target, size = 12)
        ax[x,y].grid()

    ax[6, 2].axis('off')

    plt.tight_layout()
    plt.show()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    #1-
    print('train******************************')
    print('px_height: ',len(train[(train['px_height']<217)]))
    print('px_width: ',len(train[(train['px_width']<220)]))
    print('test******************************')
    print('px_height: ',len(test[(test['px_height']<217)]))
    print('px_width: ',len(test[(test['px_width']<220)]))



    # --- FEATURE ENGINEERING ─────────────────────────────────

    train['px_height'].replace(train['px_height'][(train['px_height']<217)].values,217,inplace=True)
    test['px_height'].replace(test['px_height'][(test['px_height']<217)].values,217,inplace=True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    print('train******************************')
    print('px_height: ',len(train[(train['sc_w']<2.5)]))
    print('px_width: ',len(train[(train['sc_h']<5)]))
    print('test******************************')
    print('px_height: ',len(test[(test['sc_w']<2.5)]))
    print('px_width: ',len(test[(test['sc_h']<5)]))



    # --- FEATURE ENGINEERING ─────────────────────────────────

    train['sc_w'].replace(train['sc_w'][(train['sc_w']<2.5)].values,2.5,inplace=True)
    test['sc_w'].replace(test['sc_w'][(test['sc_w']<2.5)].values,2.5,inplace=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # train_dataset
    #we dont check the test dataset
    outliers_indexes = []
    target = 'price_range'

    for col in train.select_dtypes(include='object').columns:
        for cat in train[col].unique():
            df1 = train[train[col] == cat]
            q1 = df1[target].quantile(0.25)
            q3 = df1[target].quantile(0.75)
            iqr = q3-q1
            maximum = q3 + (1.5 * iqr)
            minimum = q1 - (1.5 * iqr)
            outlier_samples = df1[(df1[target] < minimum) | (df1[target] > maximum)]
            outliers_indexes.extend(outlier_samples.index.tolist())
        
        
        
    for col in train.select_dtypes(exclude='object').columns:
        q1 = train[col].quantile(0.25)
        q3 = train[col].quantile(0.75)
        iqr = q3-q1
        maximum = q3 + (1.5 * iqr)
        minimum = q1 - (1.5 * iqr)
        outlier_samples = train[(train[col] < minimum) | (train[col] > maximum)]
        outliers_indexes.extend(outlier_samples.index.tolist())
    
    outliers_indexes = list(set(outliers_indexes))
    print('{} outliers were identified, whose indices are:\n\n{}'.format(len(outliers_indexes), outliers_indexes))

    # outlier plots for both continues and categorical features:

    sns.set_style('darkgrid')
    colors = ['#a9f943', '#ed67cd', '#23bf00','#5687da','#af28aa','#8236ba','#0ff5a6','#83912c']
    CustomPalette = sns.set_palette(sns.color_palette(colors))

    OrderedCols = np.concatenate([num_columns,cat_columns])

    fig, ax = plt.subplots(5, 4, figsize=(15,20),dpi=100)

    for i,col in enumerate(OrderedCols):
        x = i//4
        y = i%4
        if i<13:
            sns.boxplot(data=train, y=col, ax=ax[x,y])
            ax[x,y].yaxis.label.set_size(15)
        else:
            sns.boxplot(data=train, x=col, y='price_range', ax=ax[x,y])
            ax[x,y].xaxis.label.set_size(15)
            ax[x,y].yaxis.label.set_size(15)

    plt.tight_layout()    
    plt.show()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    fig,ax=plt.subplots(1,1,figsize=(18,12))
    mask=np.triu(np.ones_like(train.corr()))
    heatmap=sns.heatmap(train.corr(),vmin=-1,vmax=1,mask=mask,cmap='Pastel2',annot=True)
    heatmap.set_title('Triangle',fontdict={'fontsize':12},pad=20)



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

            clf_setup = setup(data=train, target='price_range', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Mobile price classification")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
