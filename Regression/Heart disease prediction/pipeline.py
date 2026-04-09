#!/usr/bin/env python3
"""
Full pipeline for Heart disease prediction

Auto-generated from: heart_disease_predictions.ipynb
Project: Heart disease prediction
Category: Regression | Task: regression
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from IPython.core.display import HTML
import matplotlib.pyplot as plt
from scipy.stats import uniform

import warnings
warnings.filterwarnings('ignore')

import os
# Additional imports extracted from mixed cells
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def count_plot(data, cat_feats):    
    L = len(cat_feats)
    ncol= 2
    nrow= int(np.ceil(L/ncol))
    remove_last= (nrow * ncol) - L

    fig, ax = plt.subplots(nrow, ncol,figsize=(18, 24), facecolor='#F6F5F4')    
    fig.subplots_adjust(top=0.92)
    ax.flat[-remove_last].set_visible(False)

    i = 1
    for col in cat_feats:
        plt.subplot(nrow, ncol, i, facecolor='#F6F5F4')
        ax = sns.countplot(data=data, x=col, hue="target", palette=mypal[1::4])
        ax.set_xlabel(col, fontsize=20)
        ax.set_ylabel("count", fontsize=20)
        sns.despine(right=True)
        sns.despine(offset=0, trim=False) 
        plt.legend(facecolor='#F6F5F4')
        
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.0f}'.format((height)),ha="center",
                  bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
        
        i = i +1

    plt.suptitle('Distribution of Categorical Features' ,fontsize = 24)
    return 0

count_plot(data, cat_feats[0:-1]);
# the cramers_v function is copied from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9

def cramers_v(x, y): 
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

# calculate the correlation coefficients using the above function
data_ = data[cat_feats]
rows= []
for x in data_:
    col = []
    for y in data_ :
        cramers =cramers_v(data_[x], data_[y]) 
        col.append(round(cramers,2))
    rows.append(col)
    
cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results, columns = data_.columns, index = data_.columns)

# color palette 
mypal_1= ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA', '#FC05FB', '#FEAEFE', '#FCD2FC']
# plot the heat map
mask = np.triu(np.ones_like(df, dtype=bool))
corr = df.mask(mask)
f, ax = plt.subplots(figsize=(10, 6), facecolor=None)
cmap = sns.color_palette(mypal_1, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=0, center=0, annot=True,
            square=False, linewidths=.01, cbar_kws={"shrink": 0.75})
ax.set_title("Categorical Features Correlation (Cramer's V)", fontsize=20, y= 1.05);

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

    data = load_dataset('heart_disease_prediction')
    print('Shape of the data is ', data.shape)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.head()

    data.dtypes



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data = data[data['ca'] < 4] #drop the wrong ca values
    data = data[data['thal'] > 0] # drop the wong thal value
    print(f'The length of the data now is {len(data)} instead of 303!')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    data = data.rename(
        columns = {'cp':'chest_pain_type', 
                   'trestbps':'resting_blood_pressure', 
                   'chol': 'cholesterol',
                   'fbs': 'fasting_blood_sugar',
                   'restecg' : 'resting_electrocardiogram', 
                   'thalach': 'max_heart_rate_achieved', 
                   'exang': 'exercise_induced_angina',
                   'oldpeak': 'st_depression', 
                   'slope': 'st_slope', 
                   'ca':'num_major_vessels', 
                   'thal': 'thalassemia'}, 
        errors="raise")



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data['sex'][data['sex'] == 0] = 'female'
    data['sex'][data['sex'] == 1] = 'male'

    data['chest_pain_type'][data['chest_pain_type'] == 0] = 'typical angina'
    data['chest_pain_type'][data['chest_pain_type'] == 1] = 'atypical angina'
    data['chest_pain_type'][data['chest_pain_type'] == 2] = 'non-anginal pain'
    data['chest_pain_type'][data['chest_pain_type'] == 3] = 'asymptomatic'

    data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
    data['fasting_blood_sugar'][data['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

    data['resting_electrocardiogram'][data['resting_electrocardiogram'] == 0] = 'normal'
    data['resting_electrocardiogram'][data['resting_electrocardiogram'] == 1] = 'ST-T wave abnormality'
    data['resting_electrocardiogram'][data['resting_electrocardiogram'] == 2] = 'left ventricular hypertrophy'

    data['exercise_induced_angina'][data['exercise_induced_angina'] == 0] = 'no'
    data['exercise_induced_angina'][data['exercise_induced_angina'] == 1] = 'yes'

    data['st_slope'][data['st_slope'] == 0] = 'upsloping'
    data['st_slope'][data['st_slope'] == 1] = 'flat'
    data['st_slope'][data['st_slope'] == 2] = 'downsloping'

    data['thalassemia'][data['thalassemia'] == 1] = 'fixed defect'
    data['thalassemia'][data['thalassemia'] == 2] = 'normal'
    data['thalassemia'][data['thalassemia'] == 3] = 'reversable defect'



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data.dtypes

    data.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # numerical fearures 6
    num_feats = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'num_major_vessels']
    # categorical (binary)
    bin_feats = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina', 'target']
    # caterorical (multi-)
    nom_feats= ['chest_pain_type', 'resting_electrocardiogram', 'st_slope', 'thalassemia']
    cat_feats = nom_feats + bin_feats

    mypal= ['#FC05FB', '#FEAEFE', '#FCD2FC','#F3FEFA', '#B4FFE4','#3FFEBA']

    plt.figure(figsize=(7, 5),facecolor='#F6F5F4')
    total = float(len(data))
    ax = sns.countplot(x=data['target'], palette=mypal[1::4])
    ax.set_facecolor('#F6F5F4')

    for p in ax.patches:
    
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.1f} %'.format((height/total)*100), ha="center",
               bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))

    ax.set_title('Target variable distribution', fontsize=20, y=1.05)
    sns.despine(right=True)
    sns.despine(offset=5, trim=True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    data[num_feats].describe().T



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    L = len(num_feats)
    ncol= 2
    nrow= int(np.ceil(L/ncol))
    #remove_last= (nrow * ncol) - L

    fig, ax = plt.subplots(nrow, ncol, figsize=(16, 14),facecolor='#F6F5F4')   
    fig.subplots_adjust(top=0.92)

    i = 1
    for col in num_feats:
        plt.subplot(nrow, ncol, i, facecolor='#F6F5F4')
    
        ax = sns.kdeplot(data=data, x=col, hue="target", multiple="stack", palette=mypal[1::4]) 
        ax.set_xlabel(col, fontsize=20)
        ax.set_ylabel("density", fontsize=20)
        sns.despine(right=True)
        sns.despine(offset=0, trim=False)
    
        if col == 'num_major_vessels':
            sns.countplot(data=data, x=col, hue="target", palette=mypal[1::4])
            for p in ax.patches:
                    height = p.get_height()
                    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.0f}'.format((height)),ha="center",
                          bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.5))
    
        i = i +1
    plt.suptitle('Distribution of Numerical Features' ,fontsize = 24);

    _ = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'target']
    data_ = data[_]
    g = sns.pairplot(data_, hue="target", corner=True, diag_kind='hist', palette=mypal[1::4]);
    plt.suptitle('Pairplot: Numerical Features ' ,fontsize = 24);



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    fig, ax = plt.subplots(1,4, figsize=(20, 4))
    sns.regplot(data=data[data['target'] ==1], x='age', y='cholesterol', ax = ax[0], color=mypal[0], label='1')
    sns.regplot(data=data[data['target'] ==0], x='age', y='cholesterol', ax = ax[0], color=mypal[5], label='0')
    sns.regplot(data=data[data['target'] ==1], x='age', y='max_heart_rate_achieved', ax = ax[1], color=mypal[0], label='1')
    sns.regplot(data=data[data['target'] ==0], x='age', y='max_heart_rate_achieved', ax = ax[1], color=mypal[5], label='0')
    sns.regplot(data=data[data['target'] ==1], x='age', y='resting_blood_pressure', ax = ax[2], color=mypal[0], label='1')
    sns.regplot(data=data[data['target'] ==0], x='age', y='resting_blood_pressure', ax = ax[2], color=mypal[5], label='0')
    sns.regplot(data=data[data['target'] ==1], x='age', y='st_depression', ax = ax[3], color=mypal[0], label='1')
    sns.regplot(data=data[data['target'] ==0], x='age', y='st_depression', ax = ax[3], color=mypal[5], label='0')
    plt.suptitle('Reg plots of selected features')
    plt.legend();



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df_ = data[num_feats]
    corr = df_.corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(8, 5), facecolor=None)
    cmap = sns.color_palette(mypal, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0, annot=True,
                square=False, linewidths=.5, cbar_kws={"shrink": 0.75})
    ax.set_title("Numerical features correlation (Pearson's)", fontsize=20, y= 1.05);

    feats_ = ['age', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'st_depression', 'num_major_vessels', 'target']

    def point_biserial(x, y):
        pb = stats.pointbiserialr(x, y)
        return pb[0]

    rows= []
    for x in feats_:
        col = []
        for y in feats_ :
            pbs =point_biserial(data[x], data[y]) 
            col.append(round(pbs,2))  
        rows.append(col)  
    
    pbs_results = np.array(rows)
    DF = pd.DataFrame(pbs_results, columns = data[feats_].columns, index =data[feats_].columns)

    mask = np.triu(np.ones_like(DF, dtype=bool))
    corr = DF.mask(mask)

    f, ax = plt.subplots(figsize=(8, 5), facecolor=None)
    cmap = sns.color_palette(mypal, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1, center=0, annot=True,
                square=False, linewidths=.5, cbar_kws={"shrink": 0.75})
    ax.set_title("Cont feats vs target correlation (point-biserial)", fontsize=20, y= 1.05);



    # --- PREPROCESSING ───────────────────────────────────────

    from sklearn.model_selection import train_test_split

    # Define features and target
    X = data.drop(columns=['target'])
    y = data['target']

    # Handle non-numeric columns for modeling
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )



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

            clf_setup = setup(data=data, target='target', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Heart disease prediction")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
