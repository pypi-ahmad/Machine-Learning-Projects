#!/usr/bin/env python3
"""
Full pipeline for Food Delivery Analysis

Auto-generated from: code.ipynb
Project: Food Delivery Analysis
Category: Data Analysis | Task: data_analysis
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Additional imports extracted from mixed cells
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot
import geopandas as gpd
import math
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import nltk
import re
import string
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import defaultdict
import cufflinks as cf

# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    """Run the complete pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Basic libraries
    import pandas as pd
    import numpy as np

    #Visualization libraries
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objs as go
    from plotly import tools
    from plotly.offline import iplot

    #Geospatial Analysis Libraries
    import geopandas as gpd
    import math
    import folium
    from folium import Choropleth, Circle, Marker
    from folium.plugins import HeatMap, MarkerCluster

    #NLTK libraries
    import nltk
    import re
    import string
    from wordcloud import WordCloud,STOPWORDS
    from nltk.corpus import stopwords
    from textblob import TextBlob


    #Miscellaneous libraries
    from collections import defaultdict
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)



    # --- DATA LOADING ────────────────────────────────────────

    #Reading the data
    delivery=load_dataset('food_delivery_analysis')

    #Printing the information of dataset
    print ("The shape of the  data is (row, column):"+ str(delivery.shape))
    print(delivery.info())



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    delivery.head()

    delivery.describe()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    #Pivot table
    delivery_pivot1=pd.pivot_table(delivery,index=["Gender","Marital Status"],
                                   values=['Age','Family size'],
                                   aggfunc=[np.mean,len], margins=True)

    #Adding color gradient
    cm = sns.light_palette("green", as_cmap=True)
    delivery_pivot1.style.background_gradient(cmap=cm)

    #Pivot table
    delivery_pivot2=pd.pivot_table(delivery,index=["Educational Qualifications","Occupation"],
                                   values=['Age','Family size'],
                                   aggfunc=[np.mean,len,np.std])

    #Adding bar for numbers
    delivery_pivot2.style.bar()

    #Pivot table
    delivery_pivot4=pd.pivot_table(delivery,index=["Order Time","Maximum wait time"],
                                   values=['Age','Family size'],columns=['Influence of time'],
                                   aggfunc={'Influence of time':len},
                                   fill_value=0)

    #Adding color gradient
    cm = sns.light_palette("blue", as_cmap=True)
    delivery_pivot4.style.background_gradient(cmap=cm)

    #Setting up the frame
    fig,axes = plt.subplots(nrows=2,ncols=2,dpi=120,figsize = (8,6))

    #Distribution of age with displot
    plot00=sns.distplot(delivery['Age'],ax=axes[0][0],color='green')
    axes[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0][0].set_title('Distribution of Age',fontdict={'fontsize':8})
    axes[0][0].set_xlabel('Age',fontdict={'fontsize':7})
    axes[0][0].set_ylabel('Count/Dist.',fontdict={'fontsize':7})
    plt.tight_layout()

    #Distribution of Familysize with displot
    plot01=sns.distplot(delivery['Family size'],ax=axes[0][1],color='green')
    axes[0][1].set_title('Distribution of Family Size',fontdict={'fontsize':8})
    axes[0][1].set_xlabel('Family Size',fontdict={'fontsize':7})
    axes[0][1].set_ylabel('Count/Dist.',fontdict={'fontsize':7})
    axes[0][1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.tight_layout()

    #Age-Boxplot
    plot10=sns.boxplot(delivery['Age'],ax=axes[1][0])
    axes[1][0].set_title('Age Distribution',fontdict={'fontsize':8})
    axes[1][0].set_xlabel('Distribution',fontdict={'fontsize':7})
    axes[1][0].set_ylabel(r'Age',fontdict={'fontsize':7})
    plt.tight_layout()

    #FamilySize-Boxplot
    plot11=sns.boxplot(delivery['Family size'],ax=axes[1][1])
    axes[1][1].set_title(r'Numerical Summary (Family Size)',fontdict={'fontsize':8})
    axes[1][1].set_ylabel(r'Size of Family',fontdict={'fontsize':7})
    axes[1][1].set_xlabel('Family Size',fontdict={'fontsize':7})
    plt.tight_layout()

    plt.show()

    #Setting up the frame
    plt.figure(figsize = (15, 7))
    plt.style.use('seaborn-white')

    #Gender Countplot
    plt.subplot(2,3,1)
    ax = sns.countplot(x="Gender", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Gender count',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count_Gender', fontsize = 12)
    plt.tight_layout()

    #Marital Status Countplot
    plt.subplot(2,3,2)
    ax = sns.countplot(x="Marital Status", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Marital Status count',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count_Marital', fontsize = 12)
    plt.tight_layout()

    #Occupation Countplot
    plt.subplot(2,3,3)
    ax = sns.countplot(x="Occupation", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Occupation count',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count_Occupation', fontsize = 12)
    plt.tight_layout()

    #Education Countplot
    plt.subplot(2,3,4)
    ax = sns.countplot(x="Educational Qualifications", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Educational Qualifications count',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count_Ed', fontsize = 12)
    plt.tight_layout()

    #Income Countplot
    plt.subplot(2,3,5)
    ax = sns.countplot(x="Monthly Income", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=20)
    ax.set_title('Monthly Income count',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count_Income', fontsize = 12)
    plt.tight_layout()

    #Family Size Countplot
    plt.subplot(2,3,6)
    ax = sns.countplot(x="Family size", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=20)
    ax.set_title('Family size',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count_Size', fontsize = 12)
    plt.tight_layout()

    #Setting up the frame
    plt.figure(figsize = (15, 7))
    plt.style.use('seaborn-white')

    #Meal Countplot
    plt.subplot(1,3,1)
    ax = sns.countplot(x="Meal(P1)", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Meal count',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count_Meal', fontsize = 12)
    plt.tight_layout()

    #Medium Countplot
    plt.subplot(1,3,2)
    ax = sns.countplot(x="Medium (P1)", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Medium Status count',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count_Medium', fontsize = 12)
    plt.tight_layout()

    #Preference Countplot
    plt.subplot(1,3,3)
    ax = sns.countplot(x="Perference(P1)", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=20)
    ax.set_title('Preference count',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count_Preference', fontsize = 12)
    plt.tight_layout()

    #Setting up the frame
    plt.figure(figsize = (15, 7))
    plt.style.use('ggplot')

    #Ease and convenient Countplot
    plt.subplot(2,4,1)
    ax = sns.countplot(x="Ease and convenient", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Ease and convenient',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Time Countplot
    plt.subplot(2,4,2)
    ax = sns.countplot(x="Time saving", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Time saving',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Restaurant Countplot
    plt.subplot(2,4,3)
    ax = sns.countplot(x="More restaurant choices", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('More restaurant choices',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Payment Countplot
    plt.subplot(2,4,4)
    ax = sns.countplot(x="Easy Payment option", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Easy Payment option',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Offers Countplot
    plt.subplot(2,4,5)
    ax = sns.countplot(x="More Offers and Discount", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('More Offers and Discount',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Preference Countplot
    plt.subplot(2,4,6)
    ax = sns.countplot(x="Good Food quality", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Good Food quality',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count_Preference', fontsize = 12)
    plt.tight_layout()

    #Tracking Countplot
    plt.subplot(2,4,7)
    ax = sns.countplot(x="Good Tracking system", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Good Tracking system',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Setting up the frame
    plt.figure(figsize = (15, 7))
    plt.style.use('seaborn-dark')

    #Self cooking Countplot
    plt.subplot(2,4,1)
    ax = sns.countplot(x="Self Cooking", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Self Cooking',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Health Countplot
    plt.subplot(2,4,2)
    ax = sns.countplot(x="Health Concern", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Health Concern',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Late delivery Countplot
    plt.subplot(2,4,3)
    ax = sns.countplot(x="Late Delivery", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Late Delivery',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Hygiene Countplot
    plt.subplot(2,4,4)
    ax = sns.countplot(x="Poor Hygiene", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Poor Hygiene',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Past Countplot
    plt.subplot(2,4,5)
    ax = sns.countplot(x="Bad past experience", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Bad past experience',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Unavailability Countplot
    plt.subplot(2,4,6)
    ax = sns.countplot(x="Unavailability", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Unavailability',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count_Preference', fontsize = 12)
    plt.tight_layout()

    #Unaffordable Countplot
    plt.subplot(2,4,7)
    ax = sns.countplot(x="Unaffordable", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Unaffordable',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Setting up the frame
    plt.figure(figsize = (15, 7))
    plt.style.use('fivethirtyeight')


    #Long delivery time Countplot
    plt.subplot(2,3,1)
    ax = sns.countplot(x="Long delivery time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Long delivery time',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Delay of delivery person getting assigned Countplot
    plt.subplot(2,3,2)
    ax = sns.countplot(x="Delay of delivery person getting assigned", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Delay of delivery person getting assigned',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Delay of delivery person picking up food Countplot
    plt.subplot(2,3,3)
    ax = sns.countplot(x="Delay of delivery person picking up food", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Delay of delivery person picking up food',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Wrong order delivered Countplot
    plt.subplot(2,3,4)
    ax = sns.countplot(x="Wrong order delivered", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Wrong order delivered',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Missing item Countplot
    plt.subplot(2,3,5)
    ax = sns.countplot(x="Missing item", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Missing item',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Order placed by mistake Countplot
    plt.subplot(2,3,6)
    ax = sns.countplot(x="Order placed by mistake", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Order placed by mistake',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count_Preference', fontsize = 12)
    plt.tight_layout()

    #Setting up the frame
    plt.figure(figsize = (15, 7))
    plt.style.use('bmh')

    #Influence of time Countplot
    plt.subplot(2,4,1)
    ax = sns.countplot(x="Influence of time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Influence of time',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Order Time Countplot
    plt.subplot(2,4,2)
    ax = sns.countplot(x="Order Time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Order Time',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Maximum wait time Countplot
    plt.subplot(2,4,3)
    ax = sns.countplot(x="Maximum wait time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Maximum wait time',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Hygiene Countplot
    plt.subplot(2,4,4)
    ax = sns.countplot(x="Residence in busy location", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Residence in busy location',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Accuracy Countplot
    plt.subplot(2,4,5)
    ax = sns.countplot(x="Google Maps Accuracy", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Google Maps Accuracy',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Good Road Condition Countplot
    plt.subplot(2,4,6)
    ax = sns.countplot(x="Good Road Condition", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Good Road Condition',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Low quantity low time Countplot
    plt.subplot(2,4,7)
    ax = sns.countplot(x="Low quantity low time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Low quantity low time',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Delivery person ability Countplot
    plt.subplot(2,4,8)
    ax = sns.countplot(x="Delivery person ability", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Delivery person ability',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Setting up the frame
    plt.figure(figsize = (15, 7))
    plt.style.use('bmh')

    #Influence of rating Countplot
    plt.subplot(2,5,1)
    ax = sns.countplot(x="Influence of rating", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Influence of rating',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Less Delivery time Countplot
    plt.subplot(2,5,2)
    ax = sns.countplot(x="Less Delivery time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Less Delivery time',fontsize = 15)
    ax.set_xlabel('Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #High Quality of package Countplot
    plt.subplot(2,5,3)
    ax = sns.countplot(x="High Quality of package", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('High Quality of package',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Politeness Countplot
    plt.subplot(2,5,4)
    ax = sns.countplot(x="Politeness", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Politeness',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Number of calls Countplot
    plt.subplot(2,5,5)
    ax = sns.countplot(x="Number of calls", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Number of calls',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Freshness Countplot
    plt.subplot(2,5,6)
    ax = sns.countplot(x="Freshness ", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Freshness',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Temperature Countplot
    plt.subplot(2,5,7)
    ax = sns.countplot(x="Temperature", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Temperature',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Good Taste Contplot
    plt.subplot(2,5,8)
    ax = sns.countplot(x="Good Taste ", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Good Taste',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    #Good Quantity Countplot
    plt.subplot(2,5,9)
    ax = sns.countplot(x="Good Quantity", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax.set_title('Good Quantity',fontsize = 15)
    ax.set_xlabel( 'Types',fontsize = 12)
    ax.set_ylabel('Count', fontsize = 12)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))


    fig.suptitle('Bivariate Analysis-1')

    sns.boxplot(ax=axes[0], data=delivery, x='Influence of time', y='Age')
    sns.boxplot(ax=axes[1], data=delivery, x='Influence of time', y='Family size')
    sns.countplot(ax=axes[2],data=delivery,x="Occupation", hue="Influence of time")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))


    fig.suptitle('Bivariate Analysis-2')

    sns.countplot(ax=axes[0],data=delivery,x="Marital Status", hue="Maximum wait time")
    ax=sns.countplot(ax=axes[1],data=delivery,x="Monthly Income", hue="Influence of rating")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax=sns.countplot(ax=axes[2],data=delivery,x="Good Road Condition", hue="Long delivery time")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    sns.set_style("white")

    fig.suptitle('Bivariate Analysis-3')

    ax=sns.countplot(ax=axes[0],data=delivery,x="Order Time", hue="Late Delivery")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax=sns.countplot(ax=axes[1],data=delivery,x="Residence in busy location", hue="Long delivery time")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax=sns.countplot(ax=axes[2],data=delivery,x="Good Food quality", hue="Influence of rating")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))


    fig.suptitle('Bivariate Analysis-4')

    ax=sns.countplot(ax=axes[0],data=delivery,x="Good Quantity", hue="Influence of rating")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax=sns.countplot(ax=axes[1],data=delivery,x="Temperature", hue="Influence of rating")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)
    ax=sns.countplot(ax=axes[2],data=delivery,x="Good Taste ", hue="Influence of rating")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11,rotation=40)

    sns.set_style("dark")

    #Considering 3 variables
    fig = px.bar(delivery, x="Influence of time", y="Good Road Condition",
                 color="Good Road Condition", barmode="group",facet_col="Order Time")

    #Setting up the title
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="LightSteelBlue",
        title_text='Does Road condition and Order time has anything to do with influence in time?'
    )
    fig.show()

    #Considering 3 variables
    fig = px.bar(delivery, x="Influence of time", y="Number of calls",
                 color="Number of calls", barmode="group",facet_col="Late Delivery")

    #Setting up the title
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="LightSteelBlue",
        title_text='Does Number of calls and Late delivery has anything to do with influence in time?'
    )
    fig.show()

    #Considering 3 variables
    fig = px.bar(delivery, x="Influence of rating", y="High Quality of package",
                 color="High Quality of package", barmode="group",facet_col="Poor Hygiene")

    #Setting up the title
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="LightSteelBlue",
        title_text='Does Packaging and Hygiene has anything to do with influence in rating?'
    )
    fig.show()

    #Considering 3 variables
    fig = px.bar(delivery, x="Influence of rating", y="Good Taste ",
                 color="Good Taste ", barmode="group",facet_col="Temperature")

    #Setting up the title
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="LightSteelBlue",
        title_text='Does Taste and Temperature of food has anything to do with influence in rating?'
    )
    fig.show()

    #Considering 3 variables
    fig = px.bar(delivery, x="Influence of rating", y="Good Quantity",
                 color="Good Quantity", barmode="group",facet_col="Freshness ")

    #Setting up the title
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="LightSteelBlue",
        title_text='Does Quantity and Freshness of food has anything to do with influence in rating?'
    )
    fig.show()

    Age_band = delivery[(delivery.Age.isin(range(18,40)))]
    # Creating a map
    m_2 = folium.Map(location=[12.9716,77.5946], tiles='cartodbpositron', zoom_start=13)

    # Adding points to the map
    for idx, row in Age_band.iterrows():
        Marker([row['latitude'], row['longitude']]).add_to(m_2)

    # Displaying the map
    m_2

    # Creating the map
    m_3 = folium.Map(location=[12.9716,77.5946], tiles='cartodbpositron', zoom_start=13)

    # Adding points to the map
    mc = MarkerCluster()
    for idx, row in Age_band.iterrows():
        if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):
            mc.add_child(Marker([row['latitude'], row['longitude']]))
    m_3.add_child(mc)

    # Displaying the map
    m_3

    # Summary statistics
    df.describe(include='all')

    # Correlation matrix for numeric columns
    import matplotlib.pyplot as plt
    import seaborn as sns

    numeric_df = df.select_dtypes(include='number')
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Food Delivery Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
