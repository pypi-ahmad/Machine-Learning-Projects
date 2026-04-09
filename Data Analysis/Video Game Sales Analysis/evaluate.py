#!/usr/bin/env python3
"""
Model evaluation for Video Game Sales Analysis

Auto-generated from: code.ipynb
Project: Video Game Sales Analysis
Category: Data Analysis | Task: data_analysis
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Visualization Packages Importing
from matplotlib import pyplot as plt
import seaborn as sns
from plotly import graph_objects as go
from plotly import express as px
# import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot

# WordCloud Packages
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import r2 score for Calculation
from sklearn.metrics import r2_score
# Additional imports extracted from mixed cells
from lazypredict.Supervised import LazyRegressor
from pycaret.regression import *

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

    df = load_dataset('video_game_sales_analysis')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df.sample(5)

    for category_name in categorical_features:
        print('-' * 50)
        print("Column Name: ", category_name)
        print(' ' * 50)

        print(df[category_name].value_counts().head())

        print('-' * 50)
        print('-' * 50)

    df[['Year', 'Publisher']].describe(include='all')



    # --- PREPROCESSING ───────────────────────────────────────

    df.Year = df.Year.fillna(df.Year.mean())



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df.Year = df.Year.astype('int32')
    df.Year



    # --- PREPROCESSING ───────────────────────────────────────

    df.Publisher = df.Publisher.fillna(df.Publisher.mode()[0])



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Get Top 10 Video Games Publishers
    top_10_publishers = df.Publisher.value_counts().head(10)

    px.bar(top_10_publishers, title='Top 10 Video Game Pubishers',
           labels={
               'value': "Number of Games Publishing",
               'index': "Name of the Publisher"
           })

    # Get Top 10 Video Games Genre
    top_10_generes = df.Genre.value_counts()
    # top_10_generes

    fig =px.bar(top_10_generes, title='Top 10 Video Game Genres',
           labels={
               'value': "Number of Games Genres",
               'index': "Name of the Genre"
           })

    fig.show()


    fig = px.scatter(top_10_generes, title='Top Gernres Games',
                  labels={
                       'value': "Numbers",
                       'index': "Genre"
                   })
    fig.show()



    # px.bar(top_10_generes.index, top_10_generes.values, title='Top 10 Video Game Genres',
    #        labels={
    #            'value': "Numbers",
    #            'index': "Genre"
    #        })

    # Get Top 10 Video Games Genre
    top_10_platform = df.Platform.value_counts().sort_values()
    top_10_platform

    fig = px.line(top_10_platform, title='Top Playing Platforms',
                  labels={
                       'value': "Counts",
                       'index': "Name of the Platform"
                   })

    # fig = go.Figure(data=go.Scatter(x= top_10_platform.index, y=top_10_platform.values,
    #                                title="Top Playing Platforms"))

    fig.show()

    year_wise_sales = df.loc[:, ['Name', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].groupby(by =  'Year'  ).sum()


    fig1 = go.Scatter(x = year_wise_sales.index, y = year_wise_sales['NA_Sales'],
                      name = "North America's Sales",
                      line_shape='vh'
                     )

    fig2 = go.Scatter(x = year_wise_sales.index, y = year_wise_sales['EU_Sales'],
                      name = "Europe's Sales",
                      line_shape='vh')

    fig3 = go.Scatter(x = year_wise_sales.index, y = year_wise_sales['JP_Sales'],
                      name = "Japan's Sales",
                      line_shape='vh')

    fig4 = go.Scatter(x = year_wise_sales.index, y = year_wise_sales['Other_Sales'],
                      name = "Other Sales",
                      line_shape='vh')

    figs = [ fig1, fig2, fig3, fig4 ]

    layout = dict(title = 'Year Wise Total Game Sales of North America, Europe, Japan and Other Country',
                  xaxis= dict(title= 'Year' ),
                  yaxis= dict(title= 'Total Sales In Millions',)
                 )

    figure = dict(data = figs, layout = layout)

    iplot(figure)

    year_wise_sales = df.loc[:, ['Name', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].groupby(by =  'Year'  ).mean()


    fig1 = go.Scatter(x = year_wise_sales.index, y = year_wise_sales['NA_Sales'],
                      name = "North America's Sales",
                      line_shape='vh'
                     )

    fig2 = go.Scatter(x = year_wise_sales.index, y = year_wise_sales['EU_Sales'],
                      name = "Europe's Sales",
                      line_shape='vh')

    fig3 = go.Scatter(x = year_wise_sales.index, y = year_wise_sales['JP_Sales'],
                      name = "Japan's Sales",
                      line_shape='vh')

    fig4 = go.Scatter(x = year_wise_sales.index, y = year_wise_sales['Other_Sales'],
                      name = "Other Sales",
                      line_shape='vh')

    figs = [ fig1, fig2, fig3, fig4 ]

    layout = dict(title = 'Year Wise Average Sales for North America, Europe, Japan and Other Country',
                  xaxis= dict(title= 'Year' ),
                  yaxis= dict(title= 'Average Sales In Millions',)
                 )

    figure = dict(data = figs, layout = layout)

    iplot(figure)

    # Scatter

    fig = px.scatter(df, x="Year", y="Global_Sales", color="Genre",
                     size='Global_Sales', hover_data=['Name'],
                     title="Year Wise Global Video Game Sales by Genere",
                     labels={'x':'Years', 'y':'Global Sales In Millions'}
                    )

    fig.show()

    top_sales = df.sort_values(by=['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], ascending=False).head(10)

    # ['NA_Sales', '', '', '']
    dicts_name = {
        'NA_Sales' : "North America Sales ( In Millions)",
        'EU_Sales' : "Europe Sales ( In Millions)",
        'JP_Sales' : "Japan Sales ( In Millions)",
        'Other_Sales' : "Other Sales ( In Millions)",
    }

    for (key, title) in dicts_name.items():

        fig = px.sunburst(top_sales, path=['Genre', 'Publisher', 'Platform'], values=key, title= 'Top Selling by '+ title)

        fig.update_layout(
            grid= dict(columns=2, rows=2),
            margin = dict(t=40, l=2, r=2, b=5)
        )

        fig.show()

    global_sales = df.sort_values(by='Other_Sales', ascending=False)

    # plt.subplot(1, 2, 1)


    fig = plt.figure(figsize=(17,17))


    for index, col,  in enumerate(categorical_features):

        plt.subplot(len(categorical_features), 2, index + 1)

        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(
            stopwords=stopwords
        ).generate(" ".join(global_sales[col]))

        # Show WordCloud Image


        plt.imshow(wordcloud)
        plt.title("Video Game " + col, fontsize=20)
        plt.axis('off')
        plt.tight_layout(pad=3)

    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    data = df.copy()

    le = LabelEncoder()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    feature = ["Platform", "Genre"]

    for col in feature:
        data[col] = le.fit_transform(df[col])

    X = data[['Platform', 'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].values

    y = data['Global_Sales'].values

    X[:5], y[:5]



    # --- PREPROCESSING ───────────────────────────────────────

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=45)



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

            reg_setup = setup(data=df, target='Global_Sales', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model evaluation for Video Game Sales Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
