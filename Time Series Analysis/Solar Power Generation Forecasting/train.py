#!/usr/bin/env python3
"""
Model training for Solar Power Generation Forecasting

Auto-generated from: code.ipynb
Project: Solar Power Generation Forecasting
Category: Time Series Analysis | Task: time_series
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
import datetime as dt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')
# Additional imports extracted from mixed cells
from pycaret.time_series import *

# ======================================================================
# TRAINING PIPELINE
# ======================================================================

def main():
    """Run the training pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    # --- DATA LOADING ────────────────────────────────────────

    gen_1=load_dataset('solar_power_generation_forecasting')
    gen_1.drop('PLANT_ID',1,inplace=True)
    sens_1= pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv')
    sens_1.drop('PLANT_ID',1,inplace=True)
    #format datetime
    gen_1['DATE_TIME']= pd.to_datetime(gen_1['DATE_TIME'],format='%d-%m-%Y %H:%M')
    sens_1['DATE_TIME']= pd.to_datetime(sens_1['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_gen=gen_1.groupby('DATE_TIME').sum().reset_index()
    df_gen['time']=df_gen['DATE_TIME'].dt.time

    fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))
    # daily yield plot
    df_gen.plot(x='DATE_TIME',y='DAILY_YIELD',color='navy',ax=ax[0])
    # AC & DC power plot
    df_gen.set_index('time').drop('DATE_TIME',1)[['AC_POWER','DC_POWER']].plot(style='o',ax=ax[1])

    ax[0].set_title('Daily yield',)
    ax[1].set_title('AC power & DC power during day hours')
    ax[0].set_ylabel('kW',color='navy',fontsize=17)
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    daily_gen=df_gen.copy()
    daily_gen['date']=daily_gen['DATE_TIME'].dt.date

    daily_gen=daily_gen.groupby('date').sum()

    fig,ax= plt.subplots(ncols=2,dpi=100,figsize=(20,5))
    daily_gen['DAILY_YIELD'].plot(ax=ax[0],color='navy')
    daily_gen['TOTAL_YIELD'].plot(kind='bar',ax=ax[1],color='navy')
    fig.autofmt_xdate(rotation=45)
    ax[0].set_title('Daily Yield')
    ax[1].set_title('Total Yield')
    ax[0].set_ylabel('kW',color='navy',fontsize=17)
    plt.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_sens=sens_1.groupby('DATE_TIME').sum().reset_index()
    df_sens['time']=df_sens['DATE_TIME'].dt.time

    fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))
    # daily yield plot
    df_sens.plot(x='time',y='IRRADIATION',ax=ax[0],style='o')
    # AC & DC power plot
    df_sens.set_index('DATE_TIME').drop('time',1)[['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].plot(ax=ax[1])

    ax[0].set_title('Irradiation during day hours',)
    ax[1].set_title('Ambient and Module temperature')
    ax[0].set_ylabel('W/m',color='navy',fontsize=17)
    ax[1].set_ylabel('°C',color='navy',fontsize=17)


    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    losses=gen_1.copy()
    losses['day']=losses['DATE_TIME'].dt.date
    losses=losses.groupby('day').sum()
    losses['losses']=losses['AC_POWER']/losses['DC_POWER']*100

    losses['losses'].plot(style='o--',figsize=(17,5),label='Real Power')

    plt.title('% of DC power converted in AC power',size=17)
    plt.ylabel('DC power converted (%)',fontsize=14,color='red')
    plt.axhline(losses['losses'].mean(),linestyle='--',color='gray',label='mean')
    plt.legend()
    plt.show()

    sources=gen_1.copy()
    sources['time']=sources['DATE_TIME'].dt.time
    sources.set_index('time').groupby('SOURCE_KEY')['DC_POWER'].plot(style='o',legend=True,figsize=(20,10))
    plt.title('DC Power during day for all sources',size=17)
    plt.ylabel('DC POWER ( kW )',color='navy',fontsize=17)
    plt.show()

    dc_gen=gen_1.copy()
    dc_gen['time']=dc_gen['DATE_TIME'].dt.time
    dc_gen=dc_gen.groupby(['time','SOURCE_KEY'])['DC_POWER'].mean().unstack()

    cmap = sns.color_palette("Spectral", n_colors=12)

    fig,ax=plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,6))
    dc_gen.iloc[:,0:11].plot(ax=ax[0],color=cmap)
    dc_gen.iloc[:,11:22].plot(ax=ax[1],color=cmap)

    ax[0].set_title('First 11 sources')
    ax[0].set_ylabel('DC POWER ( kW )',fontsize=17,color='navy')
    ax[1].set_title('Last 11 sources')
    plt.show()

    temp1_gen=gen_1.copy()

    temp1_gen['time']=temp1_gen['DATE_TIME'].dt.time
    temp1_gen['day']=temp1_gen['DATE_TIME'].dt.date


    temp1_sens=sens_1.copy()

    temp1_sens['time']=temp1_sens['DATE_TIME'].dt.time
    temp1_sens['day']=temp1_sens['DATE_TIME'].dt.date

    # just for columns
    cols=temp1_gen.groupby(['time','day'])['DC_POWER'].mean().unstack()

    ax =temp1_gen.groupby(['time','day'])['DC_POWER'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))
    temp1_gen.groupby(['time','day'])['DAILY_YIELD'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,20),style='-.',ax=ax)

    i=0
    for a in range(len(ax)):
        for b in range(len(ax[a])):
            ax[a,b].set_title(cols.columns[i],size=15)
            ax[a,b].legend(['DC_POWER','DAILY_YIELD'])
            i=i+1

    plt.tight_layout()
    plt.show()

    ax= temp1_sens.groupby(['time','day'])['MODULE_TEMPERATURE'].mean().unstack().plot(subplots=True,layout=(17,2),figsize=(20,30))
    temp1_sens.groupby(['time','day'])['AMBIENT_TEMPERATURE'].mean().unstack().plot(subplots=True,layout=(17,2),figsize=(20,40),style='-.',ax=ax)

    i=0
    for a in range(len(ax)):
        for b in range(len(ax[a])):
            ax[a,b].axhline(50)
            ax[a,b].set_title(cols.columns[i],size=15)
            ax[a,b].legend(['Module Temperature','Ambient Temperature'])
            i=i+1

    plt.tight_layout()
    plt.show()

    worst_source=gen_1[gen_1['SOURCE_KEY']=='bvBOhCH3iADSZry']
    worst_source['time']=worst_source['DATE_TIME'].dt.time
    worst_source['day']=worst_source['DATE_TIME'].dt.date

    ax=worst_source.groupby(['time','day'])['DC_POWER'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))
    worst_source.groupby(['time','day'])['DAILY_YIELD'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30),ax=ax,style='-.')

    i=0
    for a in range(len(ax)):
        for b in range(len(ax[a])):
            ax[a,b].set_title(cols.columns[i],size=15)
            ax[a,b].legend(['DC_POWER','DAILY_YIELD'])
            i=i+1

    plt.tight_layout()
    plt.show()



    # --- PYCARET AUTOML ──────────────────────────────────────

    from pycaret.time_series import *

    ts_setup = setup(data=gen_1, target='None', fh=12, session_id=42, verbose=False)

    # Compare models and select best
    best_model = compare_models()

    # Display comparison results
    print(best_model)

    # Plot forecast
    plot_model(best_model, plot='forecast')

    # Finalize the model
    final_model = finalize_model(best_model)

    # Make predictions
    predictions = predict_model(final_model)
    print(predictions)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model training for Solar Power Generation Forecasting")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
