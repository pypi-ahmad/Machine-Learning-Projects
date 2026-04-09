#!/usr/bin/env python3
"""
Full pipeline for Pokemon Data Analysis

Auto-generated from: code.ipynb
Project: Pokemon Data Analysis
Category: Data Analysis | Task: classification
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
import missingno as msno
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.svm import SVC
from plotly.subplots import make_subplots
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

    # --- EVALUATION ──────────────────────────────────────────

    # dataframe
    import pandas as pd
    import numpy as np

    # visualization
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import missingno as msno
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go

    # styling
    plt.style.use('default')
    sns.set_theme(style="white")
    mpl.rcParams['axes.unicode_minus'] = False
    pd.set_option('display.max_columns',None)

    # modeling
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm.sklearn import LGBMClassifier
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    from sklearn.model_selection import KFold
    from sklearn.impute import SimpleImputer
    import xgboost as xgb
    from sklearn.svm import SVC

    color_scheme = px.colors.qualitative.T10



    # --- DATA LOADING ────────────────────────────────────────

    pokemon_data = load_dataset('pokemon_data_analysis')
    pokemon = pd.DataFrame(pokemon_data)
    pokemon.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    print(pokemon.shape)
    msno.matrix(pokemon)
    plt.title('Distribution of Missing values',fontsize = 20)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    pokemon.drop(['Type_2','Egg_Group_2'],axis=1,inplace=True)
    msno.matrix(pokemon)
    plt.title('After two columns removed',fontsize = 20)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    type1 = pokemon['Type_1'].value_counts()
    fig = px.bar(x=type1.index, y=type1.values, color = type1.index,
                 color_discrete_sequence=color_scheme,text = type1.values, title='Type1')

    lst = [0,1,2]
    for idx in lst:
        fig.data[idx].marker.line.width = 3
        fig.data[idx].marker.line.color='black'

    fig.update_layout(
        xaxis_title="Type 1 ",
        yaxis_title="count",
        template = 'simple_white')
    fig.show()

    egg1 = pokemon['Egg_Group_1'].value_counts()
    water = [2,-3,-4]

    fig = px.bar(x=egg1.index, y=egg1.values, color=egg1.index, text=egg1.values,
                 color_discrete_sequence=color_scheme, title='Egg Group 1')
    print(lst)
    for idx in lst:
        fig.data[idx].marker.line.width = 3
        fig.data[idx].marker.line.color='black'

    fig.update_layout(
        xaxis_title="Egg Group 1",
        yaxis_title="count",
        template = 'simple_white')

    fig.show()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    plt.figure(figsize=(20,5))
    before = pokemon['Pr_Male'].value_counts()
    sns.lineplot(data = pokemon['Pr_Male'].value_counts()).set_title('Pr_Male line plot',fontsize=20)



    # --- MODEL TRAINING ──────────────────────────────────────

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer = imputer.fit(pokemon[['Pr_Male']])
    pokemon['Pr_Male'] = imputer.transform(pokemon[['Pr_Male']])
    after = pokemon['Pr_Male'].value_counts()
    sns.set_theme(style="white")
    plt.figure(figsize = (20,5))
    sns.lineplot(data = before, linestyle='-',color='red', label='before imputation')
    sns.lineplot(data = after, color = 'black', marker="o", label='after imputation')

    plt.title('compare before & after imputation',fontsize=20)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    body_style = pokemon['Body_Style'].value_counts()
    fig = px.bar(y=body_style.values,
                 x=body_style.index,
                 color = body_style.index,
                 color_discrete_sequence=color_scheme,
                 text=body_style.values,
                 title= 'Body Style')

    for idx in lst:
        fig.data[idx].marker.line.width = 3
        fig.data[idx].marker.line.color='black'

    fig.update_layout(
        xaxis_title="Body Style",
        yaxis_title="count",
        template = 'simple_white')

    fig.show()

    color = pokemon['Color'].value_counts()

    fig = px.pie(values=color.values,
                 names=color.index,
                 color_discrete_sequence=color_scheme,
                 title= 'Color of pokemon')
    fig.update_traces(textinfo='label+percent', textfont_size=13,
                      marker=dict(line=dict(color='#100000', width=0.2)))

    fig.data[0].marker.line.width = 0.5
    fig.data[0].marker.line.color='gray'
    fig.show()

    fig, (ax1,ax2,ax3) = plt.subplots(ncols=3)
    fig.set_size_inches(20,5)
    sns.boxplot(data=pokemon.HP,ax=ax1,color=color_scheme[0]).set(title='HP')
    sns.boxplot(data=pokemon.Attack,ax=ax2,color=color_scheme[1]).set(title='Attack')
    sns.boxplot(data=pokemon.Defense,ax=ax3,color=color_scheme[2]).set(title='Defense')

    fig, ((ax1),(ax2)) = plt.subplots(ncols=2,nrows=1)
    fig.set_size_inches(20,5)

    pokemon['Height'] = pokemon['Height_m']*100
    # plt.figure(figsize=(20,5))
    sns.kdeplot(pokemon['Height'],color=color_scheme[0],shade=True,label='height',ax=ax1).set_title('Height')
    sns.kdeplot(pokemon['Weight_kg'],color = color_scheme[1], shade=True,label='weight',ax=ax2).set_title('Weight_kg')
    fig.show()

    hasmega = pokemon['hasMegaEvolution'].value_counts()
    legend = pokemon['isLegendary'].value_counts()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, specs =[[{"type": "pie"},{"type":"pie"}]])

    fig.add_trace(go.Pie(values = hasmega.values,
                         labels = hasmega.index,
                         marker = dict(colors=color_scheme),
                         title = "Has Mega Evolution", titlefont = dict(size=17)),row=1,col=1)

    fig.add_trace(go.Pie(values = legend.values,
                         labels = legend.index,
                         marker = dict(colors=color_scheme),
                         title = "Is or Not Legendary", titlefont = dict(size=17)),row=1,col=2)

    fig.show()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    print('top 3 lowest catch rate:', list(pokemon['Catch_Rate'].value_counts().index.sort_values()[:3]))



    # --- FEATURE ENGINEERING ─────────────────────────────────

    rate_3 = pokemon[pokemon['Catch_Rate']==3].reset_index().drop('index',axis=1)
    print('shape of new dataframe: ', rate_3.shape)
    rate_3.head(5)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    legend_rate = rate_3['isLegendary'].value_counts()
    fig = px.bar(x=legend_rate.index, y=legend_rate.values, color = legend_rate.index,
                 color_discrete_sequence=px.colors.sequential.dense,text = legend_rate.values, title='Legendary pokemon with lowest catch rate')

    fig.data[0].marker.line.width = 3
    fig.data[0].marker.line.color='black'
    fig.data[1].marker.line.width = 3
    fig.data[1].marker.line.color='blue'
    fig.update_traces(width=0.3)

    fig.update_layout(
        xaxis_title="Lowest Catch Rate",
        yaxis_title="count",
        template = 'simple_white')

    fig.show()

    type_rate = rate_3['Type_1'].value_counts()

    fig = go.Figure(data=[
        go.Bar(name='Category of lowest catch rate', x=type_rate.index, y=type_rate.values,text=type_rate.values,marker_color=color_scheme[0]),
        go.Bar(name='Original Category', x=type1.index, y=type1.values,text=type1.values,marker_color=color_scheme[1])
    ])
    fig.update_layout(barmode='group', xaxis_tickangle=-45,title='Category of pokemon',
                      template = 'simple_white')

    fig.show()

    color3 = rate_3['Color'].value_counts()

    fig = go.Figure(data=[
        go.Bar(name='Original Category', x=color.index, y=color.values, text=color.values, marker_color=color_scheme[0]),
        go.Bar(name='Category of lowest catch rate', x=color3.index, y=color3.values,text=color3.values, marker_color = color_scheme[1]),
    ])
    fig.update_layout(barmode='stack', xaxis_tickangle=-45,title='Color of pokemon', template = 'simple_white')

    fig.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    total = pokemon.groupby('Total').max().reset_index()
    total['isLegendary'].replace(1,'legend',inplace=True)
    total.tail(2)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig = px.treemap(total[-2:],
                     path=['isLegendary','Type_1','Name'],
                     values='Total',
                     title = 'top2 highest total stats pokemon',
                     color = 'Total',
                     color_continuous_scale=color_scheme,
                     width=1300, height=500)
    fig.update_layout(margin = dict(t=50, l=100, r=100, b=100))
    fig.show()

    fig = px.bar(x = total[-2:]['Name'], y = total[-2:]['Total'],
           color_discrete_sequence=color_scheme, color = total[-2:]['Name'], text = total[-2:]['Total'], title='Top 2 highest total stats pokemon')
    fig.data[1].marker.line.width = 4
    fig.data[1].marker.line.color = "black"
    fig.data[0].marker.line.width = 2
    fig.data[0].marker.line.color = "black"

    fig.update_traces(width=0.3)

    fig.update_layout(
        xaxis_title="Total Stats",
        yaxis_title="count",
        template = 'simple_white')

    fig.show()

    arceus = pokemon[pokemon['Name']=='Arceus']
    zekrom = pokemon[pokemon['Name']=='Zekrom']

    fig = go.Figure()
    categories = ['HP', 'Attack', 'Defense', 'Speed']
    fig.add_trace(go.Scatterpolar(
                 r = (arceus[['HP','Attack','Defense','Speed']].values/sum(arceus[['HP','Attack','Defense','Speed']].values.tolist()[0])).tolist()[0],
                 theta = categories,
                 fill = 'toself',
                 name = 'Arceus Stats'
                 ))
    fig.add_trace(go.Scatterpolar(
                 r = (zekrom[['HP','Attack','Defense','Speed']].values/sum(zekrom[['HP','Attack','Defense','Speed']].values.tolist()[0])).tolist()[0],
                 theta = categories,
                 fill = 'toself',
                 name = 'Zekrom Stats'
                 ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          range=[0, 0.4]
        )),
      showlegend=True,
      title = 'Arceus vs Zekrom (Stats)',
      template = 'simple_white',
    )

    mask = np.array(pokemon.corr())
    mask[np.tril_indices_from(mask)] = False

    fig, ax = plt.subplots()
    fig.set_size_inches(20,10)
    sns.heatmap(pokemon.corr(), mask = mask, vmax =.8, square = True, annot = True)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    pokemon.replace(False,0,inplace=True)
    pokemon.replace(True,1,inplace=True)
    pokemon.head(3)

    pokemon = pokemon.replace(['Water', 'Ice'],'Water_Ice')
    pokemon = pokemon.replace(['Grass', 'Bug'],'Grass_Bug')
    pokemon = pokemon.replace(['Ground', 'Rock'],'Ground_Rock')
    pokemon = pokemon.replace(['Psychic','Dark','Ghost',"Fairy"],'Psychic_Dark_Ghost_Fairy')
    pokemon = pokemon.replace(['Electric','Steel'],'ELectric_Steel')

    pokemon = pokemon.replace(['Water_1','Water_2','Water_3'],'Water')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    pokemon.head(2)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    body = dict(pokemon['Body_Style'].value_counts())

    pokemon['Body_Style'] = pokemon['Body_Style'].map(body)



    # --- PREPROCESSING ───────────────────────────────────────

    types_poke = pd.get_dummies(pokemon["Type_1"])
    color_poke = pd.get_dummies(pokemon['Color'])

    X = pd.concat([pokemon, types_poke],axis = 1)
    X = pd.concat([X, color_poke], axis=1)

    X.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    y = X['isLegendary']
    X = X.drop(['Number','Type_1',"Name",'Color','Egg_Group_1','isLegendary'],axis=1)



    # --- PREPROCESSING ───────────────────────────────────────

    train_X, test_X, train_y, test_y = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=116)



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

            clf_setup = setup(data=pokemon_data, target='Type_2', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for Pokemon Data Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
