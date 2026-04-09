#!/usr/bin/env python3
"""
Model evaluation for 7 spotify song cluster analysis

Auto-generated from: 7 spotify song cluster analysis.ipynb
Project: 7 spotify song cluster analysis
Category: Clustering | Task: clustering
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import statsmodels.formula.api as smf
from warnings import filterwarnings
import os
import plotly.graph_objects as go
import plotly.express as px
from pandas.plotting import scatter_matrix
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import matplotlib.colors as colors
from sklearn.preprocessing import StandardScaler
from pycaret.clustering import *

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

    # --- PREPROCESSING ───────────────────────────────────────

    import numpy as np
    import pandas as pd 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy as sp
    from sklearn.preprocessing import scale 
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    import statsmodels.formula.api as smf

    from warnings import filterwarnings
    filterwarnings('ignore')


    import os
    for dirname, _, filenames in os.walk('../../data/spotify_song_cluster/top50.csv'):
        for filename in filenames:
            print(os.path.join(dirname, filename))



    # --- DATA LOADING ────────────────────────────────────────

    #Importing Dataset
    data=load_dataset('spotify_song_cluster_analysis')
    data.head()



    # --- PREPROCESSING ───────────────────────────────────────

    data=data.dropna(how='all')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    data=data.sort_values(['Unnamed: 0'])
    data=data.reindex(data['Unnamed: 0'])
    data=data.drop("Unnamed: 0",axis=1)
    data.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data=data.loc[:49,:]



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #Rename Column
    data=data.rename(columns={"Loudness..dB..": "Loudness", 
                              "Acousticness..": "Acousticness",
                              "Speechiness.":"Speechiness",
                              "Valence.":"Valence",
                              "Length.":"Length"})



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    final=data.copy()

    #Grouping Some Features According To Genre
    a=final[['Genre', 'Popularity','Energy', 'Length','Liveness','Acousticness']].groupby(
        ['Genre'], as_index=False).mean().sort_values(by='Energy', ascending=True)
    a

    sorted_energy=final.sort_values(by=['Energy'])

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=sorted_energy['Energy'],
        y=sorted_energy['Loudness'],
        name="Energy and Loudness"       # this sets its legend entry
    ))


    fig.add_trace(go.Scatter(
        x=sorted_energy['Energy'],
        y=sorted_energy['Acousticness'],
        name="Energy and Acousticness"
    ))

    fig.update_layout(
        title="Acousticness-Loudness values according to Energy",
        xaxis_title="Energy",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#7f7f7f"
        )
    )

    fig.show()

    import plotly.express as px

    fig = px.bar(a, x='Genre', y='Popularity',
                 hover_data=['Energy', 'Length'],
                 color='Energy',height=400)

    fig.update_layout(
        title="Popularity and energy comparison by Genre",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#7f7f7f"
        )
    )

    fig.show()

    fig = px.bar(a, y='Acousticness', x='Genre').update_xaxes(categoryorder='total ascending')

    fig.update_layout(
        title="Acousticness comparison by Genre",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#7f7f7f"
        )
    )
    fig.show()

    #Add to new column according to Genre

    final['GeneralGenre']=['hip hop' if each =='atl hip hop'
                          else 'hip hop' if each =='canadian hip hop'
                          else 'hip hop' if each == 'trap music'
                          else 'pop' if each == 'australian pop'
                          else 'pop' if each == 'boy band'
                          else 'pop' if each == 'canadian pop'
                          else 'pop' if each == 'dance pop'
                          else 'pop' if each == 'panamanian pop'
                          else 'pop' if each == 'pop'
                          else 'pop' if each == 'pop house'
                          else 'electronic' if each == 'big room'
                          else 'electronic' if each == 'brostep'
                          else 'electronic' if each == 'edm'
                          else 'electronic' if each == 'electropop'
                          else 'rap' if each == 'country rap'
                          else 'rap' if each == 'dfw rap'
                          else 'hip hop' if each == 'hip hop'
                          else 'latin' if each == 'latin'
                          else 'r&b' if each == 'r&n en espanol'
                          else 'raggae' for each in final['Genre']]

    # histogram
    final.hist()
    plt.gcf().set_size_inches(15, 15)    #Thanks to this graphic, we can see the feature is right or left skewed.
    plt.show()

    from pandas.plotting import scatter_matrix

    # scatter plot matrix
    scatter_matrix(final)
    plt.gcf().set_size_inches(15, 15)
    plt.show()

    color_list = ['red' if i=='electronic' 
                  else 'green' if i=='escape room' 
                  else 'blue' if i == 'hip hop' 
                  else 'purple' if i == 'latin'
                  else 'darksalmon' if i == 'pop'
                  else 'darkcyan' if i == 'raggae'
                  else 'greenyellow' for i in final.loc[:,'Genre']]
    pd.plotting.scatter_matrix(final.loc[:,['Energy','Danceability','Length','Popularity']],
                                           c=color_list,
                                           figsize= [15,15],
                                           diagonal='hist',
                                           alpha=1,
                                           s = 200,
                                           marker = '+',
                                           edgecolor= "black")
    plt.show()

    import plotly.express as px

    fig = px.scatter(final, x="Beats.Per.Minute", y="Valence",size='Acousticness'
                     ,color="GeneralGenre")
    fig.show()

    #Box Plot Each Features

    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly as py
    import plotly.graph_objs as go

    init_notebook_mode(connected=True)

    trace0 = go.Box(
        y=final['Beats.Per.Minute'],
        name = 'Beats.Per.Minute'
    )
    trace1 = go.Box(
        y=final['Energy'],
        name = 'Energy'
    )
    trace2 = go.Box(
        y=final['Danceability'],
        name = 'Danceability'
    )
    trace3 = go.Box(
        y=final['Loudness'],
        name = 'Loudness'
    )
    trace4 = go.Box(
        y=final['Liveness'],
        name = 'Liveness'
    )
    trace5 = go.Box(
        y=final['Valence'],
        name = 'Valence'
    )
    trace6 = go.Box(
        y=final['Loudness'],
        name = 'Loudness'
    )
    trace7 = go.Box(
        y=final['Length'],
        name = 'Length'
    )
    trace8 = go.Box(
        y=final['Acousticness'],
        name = 'Acousticness'
    )
    trace9 = go.Box(
        y=final['Speechiness'],
        name = 'Speechiness'
    )
    trace10 = go.Box(
        y=final['Popularity'],
        name = 'Popularity'
    )
    data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8,trace9,trace10]

    fig = go.Figure(data=data)

    py.offline.iplot(fig)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    pd.rename(columns={"Genre": "Count"},inplace=True)
    pd['Genre'] = pd.index
    pd['Ratio'] = pd['Count']/pd['Count'].sum() #Add count ratio for each genre

    pd['Ratio'] = pd['Ratio'].astype(float)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    import matplotlib.colors as colors
    labels_list = pd['Genre']
    colors_list = list(colors._colors_full_map.values())

    # Plot
    plt.figure(figsize=(20,10))
    plt.pie(pd['Ratio'], colors=colors_list[0:20],
    autopct='%1.1f%%', shadow=True, startangle=50,
           labels=labels_list)

    plt.axis('equal')
    plt.title('Distribution of the % of Genre')
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    final2=final.drop(columns=['Track.Name', 'Artist.Name', 'GeneralGenre', 'Genre'])

    #Rename Column
    final2=final2.rename(columns={"Beats.Per.Minute": "BeatsPerMinute"})



    # --- PREPROCESSING ───────────────────────────────────────

    #Standardizing to all data

    from sklearn.preprocessing import StandardScaler
    final2[['BeatsPerMinute', 'Energy',
            'Danceability','Loudness',
            'Liveness','Valence',
            'Length','Acousticness',
            'Speechiness','Popularity']] = StandardScaler().fit_transform(final2[['BeatsPerMinute','Energy',
                                                                                  'Danceability','Loudness',
                                                                                  'Liveness','Valence',
                                                                                  'Length','Acousticness',
                                                                                  'Speechiness','Popularity']])



    # --- AUTOML COMPARISON ────────────────────────────────────

    if USE_AUTOML:

        try:

            # --- PYCARET AUTOML ──────────────────────────────

            from pycaret.clustering import *

            clust_setup = setup(data=data, normalize=True, session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model evaluation for 7 spotify song cluster analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
