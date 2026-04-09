#!/usr/bin/env python3
"""
Full pipeline for COVID-19 Global Data Analysis

Auto-generated from: code.ipynb
Project: COVID-19 Global Data Analysis
Category: Data Analysis | Task: data_analysis
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
import plotly.express as px
import plotly.graph_objs as go

import os
import math
# Additional imports extracted from mixed cells
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def add_commas(num):
    out = ""
    counter = 0
    for n in num[::-1]:
        counter += 1
        if counter == 4:
            counter = 1
            out = "," + out
        out = n + out
    return out

print(f"As of {df.date.max().strftime('%Y-%m-%d')}, here are the numbers:\n")

print(add_commas(str(int(df_summary.total_deaths.sum()))), "total deaths. That is more than the entire population of ", end="")
deaths_ref = df_summary[df_summary.population < df_summary.total_deaths.sum()].sort_values("population", ascending=False).iloc[:2]
print(deaths_ref.iloc[0].country, f"({add_commas(str(int(deaths_ref.iloc[0].population)))}) or",
      deaths_ref.iloc[1].country, f"({add_commas(str(int(deaths_ref.iloc[1].population)))})!")

print(add_commas(str(int(df_summary.active_cases.sum()))), "active cases. You can think of that as if the entire population of ", end="")
active_ref = df_summary[df_summary.population < df_summary.active_cases.sum()].sort_values("population", ascending=False).iloc[:2]
print(active_ref.iloc[0].country, f"({add_commas(str(int(active_ref.iloc[0].population)))}), or",
      active_ref.iloc[1].country, f"({add_commas(str(int(active_ref.iloc[1].population)))}), were sick right now!")

print(add_commas(str(int(df_summary.total_recovered.sum()))), "total recoveries. It's as if the entire population of ", end="")
recover_ref = df_summary[df_summary.population < df_summary.total_recovered.sum()].sort_values("population", ascending=False).iloc[:2]
print(recover_ref.iloc[0].country, f"({add_commas(str(int(recover_ref.iloc[0].population)))}), or",
      recover_ref.iloc[1].country, f"({add_commas(str(int(recover_ref.iloc[1].population)))}), went through and recovered from Covid-19!")
def plot_stats(country):
    if country in ["USA", "UK"]:
        country_prefix = "the "
    else:
        country_prefix = ""
    df_country = df[df.country == country]
    df_country.set_index('date', inplace=True)

    # Plot 1
    if not all(df_country.cumulative_total_cases.isna()):
        layout = go.Layout(
            yaxis={'range':[0, df_country.cumulative_total_cases[-1] * 1.05],
                  'title':'Coronavirus Confirmed Cases'},
            xaxis={'title':''},
            )

        fig = px.area(df_country, x=df_country.index, y="cumulative_total_cases",
                      title=f"<b>Cumulative Total Confirmed Cases in {country_prefix}{country}<br>from {df_country.index[0].strftime('%Y-%m-%d')} till {df_country.index[-1].strftime('%Y-%m-%d')}</b>",
                      template='plotly_dark')

        fig.update_traces(line={'width':5})

        fig.update_layout(layout)
        fig.show()

    # Plot 2
    if not all(df_country.daily_new_cases.isna()):
        layout = go.Layout(
            yaxis={'range':[0, df_country.daily_new_cases.max() * 1.05],
                  'title':'Daily New Coronavirus Confirmed Cases'},
            xaxis={'title':''},
            template='plotly_dark',
            title=f"<b>Daily New Cases in {country_prefix}{country}<br>from {df_country.index[0].strftime('%Y-%m-%d')} till {df_country.index[-1].strftime('%Y-%m-%d')}</b>",
            )

        MA7 = df_country.daily_new_cases.rolling(7).mean().dropna().astype(int)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Daily Cases", x=df_country.index, y=df_country.daily_new_cases))
        fig.add_trace(go.Scatter(name="7-Day Moving Average", x=df_country.index[df_country.shape[0] - MA7.shape[0]:], y=MA7, line=dict(width=3)))

        fig.update_layout(layout)
        fig.show()

    # Plot 3
    if not all(df_country.cumulative_total_deaths.isna()):
        layout = go.Layout(
            yaxis={'range':[0, df_country.cumulative_total_deaths[-1] * 1.05],
                  'title':'Coronavirus Deaths'},
            xaxis={'title':''},
            )

        fig = px.area(df_country, x=df_country.index, y="cumulative_total_deaths",
                      title=f"<b>Cumulative Total Deaths in {country_prefix}{country}<br>from {df_country.index[0].strftime('%Y-%m-%d')} till {df_country.index[-1].strftime('%Y-%m-%d')}</b>",
                      template='plotly_dark')

        fig.update_traces(line={'color':'red', 'width':5})

        fig.update_layout(layout)
        fig.show()

    # Plot 4
    if not all(df_country.daily_new_deaths.isna()):
        layout = go.Layout(
            yaxis={'range':[0, df_country.daily_new_deaths.max() * 1.05],
                  'title':'Daily New Coronavirus Deaths'},
            xaxis={'title':''},
            template='plotly_dark',
            title=f"<b>Daily Deaths in {country_prefix}{country}<br>from {df_country.index[0].strftime('%Y-%m-%d')} till {df_country.index[-1].strftime('%Y-%m-%d')}</b>",
            )

        MA7 = df_country.daily_new_deaths.rolling(7).mean().dropna().astype(int)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Daily Deaths", x=df_country.index, y=df_country.daily_new_deaths, marker_color='red'))
        fig.add_trace(go.Scatter(name="7-Day Moving Average", x=df_country.index[df_country.shape[0] - MA7.shape[0]:], y=MA7, line={'width':3, 'color':'white'}))

        fig.update_layout(layout)
        fig.show()

    # Plot 5
    if not all(df_country.active_cases.isna()):
        layout = go.Layout(
            yaxis={'range':[0, df_country.active_cases.max() * 1.05],
                  'title':'Active Coronavirus Cases'},
            xaxis={'title':''},
            )

        fig = px.line(df_country, x=df_country.index, y="active_cases",
                      title=f"<b>Active Cases in {country_prefix}{country}<br>from {df_country.index[0].strftime('%Y-%m-%d')} till {df_country.index[-1].strftime('%Y-%m-%d')}</b>",
                      template='plotly_dark')

        fig.update_traces(line={'color':'yellow', 'width':5})

        fig.update_layout(layout)
        fig.show()
def plot_continent(continent):
    df_continent = df[df.continent == continent]
    fig = px.line(df_continent, x="date", y="cumulative_total_cases", color="country", #log_y=True,
                  line_group="country", hover_name="country", template="plotly_dark")

    annotations = []
    # Adding labels
    ys = []
    for tr in fig.select_traces():
        ys.append(tr.y[-1])
    y_scale = 0.155 / max(ys)
    for tr in fig.select_traces():
        # labeling the right_side of the plot
        size = max(1, int(math.log(tr.y[-1], 1.1) * tr.y[-1] * y_scale))
        annotations.append(dict(x=tr.x[-1] + timedelta(hours=int((2 + size/5) * 24)), y=tr.y[-1],
                                xanchor='left', yanchor='middle',
                                text=tr.name,
                                font=dict(family='Arial',
                                          size=7+int(size/2)
                                         ),
                                showarrow=False))
        fig.add_trace(go.Scatter(
            x=[tr.x[-1]],
            y=[tr.y[-1]],
            mode='markers',
            name=tr.name,
            marker=dict(color=tr.line.color, size=size)
        ))
    fig.update_traces(line={'width':1})
    fig.update_layout(annotations=annotations, showlegend=False, uniformtext_mode='hide',
                      title=f"<b>Cumulative Total Coronavirus Cases in {continent}<br>between {df_continent.date.min().strftime('%Y-%m-%d')} and {df_continent.date.max().strftime('%Y-%m-%d')}</b>",
                      yaxis={'title':'Coronavirus Confirmed Cases'},
                      xaxis={'title':''}
                     )
    fig.show()

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

    from datetime import datetime, timedelta
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')



    # --- DATA LOADING ────────────────────────────────────────

    df = load_dataset('covid_19_global_data_analysis')

    df

    df_summary = pd.read_csv('data/worldometer_coronavirus_summary_data.csv')

    df_summary



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df['continent'] = df.apply(lambda row: df_summary[df_summary.country == row.country].iloc[0].continent, axis=1)

    df



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    trace = go.Pie(labels=['Total Recovered', 'Total Active', 'Total Deaths'],
                   values=[df_summary.total_recovered.sum(), df_summary.active_cases.sum(), df_summary.total_deaths.sum()],
                   title="<b>Coronavirus Cases</b>",
                   title_font_size=18,
                   hovertemplate="<b>%{label}</b><br>%{value}<br><i>%{percent}</i>",
                   #hoverinfo='percent+value+label',
                   textinfo='percent',
                   textposition='inside',
                   hole=0.6,
                   showlegend=True,
                   marker=dict(colors=["#8dd3c7", "ffffb3", "#fb8072"],
                               line=dict(color='#000000',
                                         width=2),
                              ),
                   name=""
                  )
    fig=go.Figure(data=[trace])
    fig.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_summary['log(Total Confirmed)'] = np.log2(df_summary['total_confirmed'])
    df_summary['Total Confirmed'] = df_summary['total_confirmed'].apply(lambda x: add_commas(str(x)))

    fig = px.choropleth(df_summary,
                        locations="country",
                        color="log(Total Confirmed)",
                        locationmode = 'country names',
                        hover_name='country',
                        hover_data=['Total Confirmed'],
                        color_continuous_scale='reds',
                        title = '<b>Coronavirus Confirmed Cases Around The Globe</b>')


    log_scale_vals = list(range(0,25,2))
    scale_vals = (np.exp2(log_scale_vals)).astype(int).astype(str)

    scale_vals = list(map(add_commas, scale_vals))

    fig.update_layout(title_font_size=22,
                      margin={"r":20, "l":30},
                      coloraxis={#"showscale":False,
                                "colorbar":dict(title="<b>Confirmed Cases</b><br>",
                                                #range=[np.log(50), np.log(6400)],
                                                titleside="top",
                                                tickmode="array",
                                                tickvals=log_scale_vals,
                                                ticktext=scale_vals
                                            )},
                     )

    fig.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig = px.treemap(df_summary, path=["country"], values="total_confirmed", height = 750,
                     title="<b>Total Coronavirus Confirmed Cases Breakdown by Country</b>",
                     color_discrete_sequence = px.colors.qualitative.Set3)

    fig.update_traces(textinfo = "label+text+value")
    fig.show()

    plot_stats('USA')

    plot_stats('China')

    plot_stats('UK')

    plot_stats('Italy')

    plot_stats("India")

    plot_stats("Australia")

    plot_stats("France")

    plot_continent("Asia")

    plot_continent("Europe")

    plot_continent("Africa")

    plot_continent("North America")

    plot_continent("South America")

    plot_continent("Australia/Oceania")



    # --- FEATURE ENGINEERING ─────────────────────────────────

    sorted_by_cases_per_1m = df_summary.sort_values(['total_cases_per_1m_population'])
    sorted_by_cases_per_1m['% of Population with Confirmed Cases'] = sorted_by_cases_per_1m['total_cases_per_1m_population']/1_000_000
    mean = sorted_by_cases_per_1m['% of Population with Confirmed Cases'].mean()
    sorted_by_cases_per_1m['color'] = sorted_by_cases_per_1m.apply(lambda row: "Red" if row['% of Population with Confirmed Cases'] > mean else "Blue", axis=1)
    fig = px.scatter(sorted_by_cases_per_1m, x='country', y='% of Population with Confirmed Cases',
                     size='% of Population with Confirmed Cases',
                     color='color',
                     title=f"<b>Coronavirus Infection-Rate by Country as of {df.date.max().strftime('%Y-%m-%d')}</b>",
                     height=650)
    fig.update_traces(marker_line_color='rgb(75,75,75)',
                      marker_line_width=1.5, opacity=0.8,
                      hovertemplate="<b>%{x}</b><br>%{y} of Population with Confirmed Cases<extra></extra>",)
    fig.update_layout(showlegend=False,
                     yaxis={"tickformat":".3%", "range":[0,sorted_by_cases_per_1m['% of Population with Confirmed Cases'].max() * 1.1]},
                     xaxis={"title": ""},
                     title_font_size=20)


    to_mention = ["China", "Australia", "India", "South Africa", "Russia", "Italy","Brazil", "UK", "France", "USA",  "Montenegro"]

    for i, country in enumerate(to_mention):
        ay = 30 if i%2 else -30
        ax = 20
        if country == "USA": ay, ax = -30, -20
        if country == "UK": ax = -20
        if country == "France": ay, ax = -60, -40
        if country == "Russia": ax = -20
        if country == "Australia": ay = -30
        if country == "Brazil": ax = -20
        fig.add_annotation(
                x=country,
                y=sorted_by_cases_per_1m['% of Population with Confirmed Cases'][sorted_by_cases_per_1m.index[sorted_by_cases_per_1m.country==country][0]],
                xref="x",
                yref="y",
                text=country,
                showarrow=True,
                font=dict(
                    family="Courier New, monospace",
                    size=14,
                    color="#ffffff"
                    ),
                align="center",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                ax=ax,
                ay=ay,
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor=sorted_by_cases_per_1m['color'][sorted_by_cases_per_1m.index[sorted_by_cases_per_1m.country==country][0]],
                opacity=0.6
                )

    fig.add_shape(type='line',
                  x0=sorted_by_cases_per_1m['country'].iloc[0], y0=mean,
                  x1=sorted_by_cases_per_1m['country'].iloc[-1], y1=mean,
                  line=dict(color='Green',width=1),
                  xref='x', yref='y'
                 )
    fig.add_annotation(x=sorted_by_cases_per_1m['country'].iloc[0], y=mean,
                       text=f"mean = {mean*100:.2f}%",
                       showarrow=False,
                       xanchor="left",
                       yanchor="bottom",
                       font={"color":"Green", "size":14}
                      )
    fig.show()



    # --- PREPROCESSING ───────────────────────────────────────

    sorted_by_deaths_per_1m = df_summary.sort_values(['total_deaths_per_1m_population'])
    sorted_by_deaths_per_1m = sorted_by_deaths_per_1m[sorted_by_deaths_per_1m['total_deaths_per_1m_population'].notna()]
    sorted_by_deaths_per_1m['% of Population with Coronavirus Death Cases'] = sorted_by_deaths_per_1m['total_deaths_per_1m_population']/1_000_000
    mean = sorted_by_deaths_per_1m['% of Population with Coronavirus Death Cases'].mean()
    sorted_by_deaths_per_1m['color'] = sorted_by_deaths_per_1m.apply(lambda row: "Red" if row['% of Population with Coronavirus Death Cases'] > mean else "Blue", axis=1)
    #sorted_by_deaths_per_1m.dropna(inplace=True)
    fig = px.scatter(sorted_by_deaths_per_1m, x='country', y='% of Population with Coronavirus Death Cases',
                     size='% of Population with Coronavirus Death Cases',
                     color='color',
                     title=f"<b>Coronavirus Death-Rate by Country as of {df.date.max().strftime('%Y-%m-%d')}</b>",
                     height=650)

    fig.update_traces(marker_line_color='rgb(75,75,75)',
                      marker_line_width=1.5, opacity=0.8,
                      hovertemplate="<b>%{x}</b><br>%{y} of Population with Death Cases<extra></extra>",)
    fig.update_layout(showlegend=False,
                     yaxis={"tickformat":".3%", "range":[0,sorted_by_deaths_per_1m['% of Population with Coronavirus Death Cases'].max() * 1.1]},
                     xaxis={"title": ""},
                     title_font_size=20)


    to_mention = ["China", "Australia", "India", "South Africa", "Russia", "Italy","Brazil", "UK", "France", "USA",  "Bulgaria", "Peru"]

    for i, country in enumerate(to_mention):
        print
        ay = 30 if i%2 else -30
        ax = 20
        if country == "Russia": ax = -20
        if country == "Czech Republic": ay, ax = -30, -60
        if country == "USA": ay = 50
        if country == "Italy": ay, ax = 30, -20
        if country == "UK": ay, ax = -30, 40
        if country == "Australia": ay = -30
        if country == "France": ay, ax = -60, -40
        if country == "Brazil": ax = -20
        if country == "Peru": ay = -30
        fig.add_annotation(
                x=country,
                y=sorted_by_deaths_per_1m['% of Population with Coronavirus Death Cases'][sorted_by_deaths_per_1m.index[sorted_by_deaths_per_1m.country==country][0]],
                xref="x",
                yref="y",
                text=country,
                showarrow=True,
                font=dict(
                    family="Courier New, monospace",
                    size=14,
                    color="#ffffff"
                    ),
                align="center",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                ax=ax,
                ay=ay,
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor=sorted_by_deaths_per_1m['color'][sorted_by_deaths_per_1m.index[sorted_by_deaths_per_1m.country==country][0]],
                opacity=0.6
                )

    fig.add_shape(type='line',
                  x0=sorted_by_deaths_per_1m['country'].iloc[0], y0=mean,
                  x1=sorted_by_deaths_per_1m['country'].iloc[-1], y1=mean,
                  line=dict(color='Green',width=1),
                  xref='x', yref='y'
                 )
    fig.add_annotation(x=sorted_by_deaths_per_1m['country'].iloc[0], y=mean,
                       text=f"mean = {mean*100:.2f}%",
                       showarrow=False,
                       xanchor="left",
                       yanchor="bottom",
                       font={"color":"Green", "size":14}
                      )

    fig.show()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df_summary["Coronavirus Deaths/Confirmed Cases"] = df_summary["total_deaths"] / df_summary["total_confirmed"]
    sorted_by_deaths_per_confirmed = df_summary.sort_values(['Coronavirus Deaths/Confirmed Cases'])
    sorted_by_deaths_per_confirmed = sorted_by_deaths_per_confirmed[sorted_by_deaths_per_confirmed['Coronavirus Deaths/Confirmed Cases'].notna()]
    mean = sorted_by_deaths_per_confirmed['Coronavirus Deaths/Confirmed Cases'].mean()
    sorted_by_deaths_per_confirmed['color'] = sorted_by_deaths_per_confirmed.apply(lambda row: "Red" if row['Coronavirus Deaths/Confirmed Cases'] > mean else "Blue", axis=1)
    fig = px.scatter(sorted_by_deaths_per_confirmed, x='country', y='Coronavirus Deaths/Confirmed Cases',
                     size='Coronavirus Deaths/Confirmed Cases',
                     color='color',
                     title=f"<b>Coronavirus severity by Country as of {df.date.max().strftime('%Y-%m-%d')}</b>",
                     height=650)

    fig.update_traces(marker_line_color='rgb(75,75,75)',
                      marker_line_width=1.5, opacity=0.8,
                      hovertemplate="<b>%{x}</b><br>%{y} of Cases Leading to Death Cases<extra></extra>",)
    fig.update_layout(showlegend=False,
                     yaxis={"tickformat":".3%", "range":[0,sorted_by_deaths_per_confirmed['Coronavirus Deaths/Confirmed Cases'].max() * 1.1]},
                     xaxis={"title": ""},
                     title_font_size=20)


    to_mention = ["China", "Australia", "India", "South Africa", "Russia", "Italy","Brazil", "UK", "France", "USA", "Yemen", "Vanuatu"]

    for i, country in enumerate(to_mention):
        print
        ay = 30 if i%2 else -30
        ax = 20
        if country in ["India", "USA", "Russia"]: ax = -20
        if country == "Yemen": ay = 30
        if country == "UK": ay, ax = -60, -40
        if country == "Belgium": ay, ax = -30, -60
        if country == "USA": ay, ax = -30, 40
        if country == "Italy": ax = -40
        if country == "Australia": ay = -30
        if country == "France": ay, ax = -60, 40
        if country == "Brazil": ay, ax = -60, -20
        fig.add_annotation(
                x=country,
                y=sorted_by_deaths_per_confirmed['Coronavirus Deaths/Confirmed Cases'][sorted_by_deaths_per_confirmed.index[sorted_by_deaths_per_confirmed.country==country][0]],
                xref="x",
                yref="y",
                text=country,
                showarrow=True,
                font=dict(
                    family="Courier New, monospace",
                    size=14,
                    color="#ffffff"
                    ),
                align="center",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                ax=ax,
                ay=ay,
                bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor=sorted_by_deaths_per_confirmed['color'][sorted_by_deaths_per_confirmed.index[sorted_by_deaths_per_confirmed.country==country][0]],
                opacity=0.6
                )

    fig.add_shape(type='line',
                  x0=sorted_by_deaths_per_confirmed['country'].iloc[0], y0=mean,
                  x1=sorted_by_deaths_per_confirmed['country'].iloc[-1], y1=mean,
                  line=dict(color='Green',width=1),
                  xref='x', yref='y'
                 )
    fig.add_annotation(x=sorted_by_deaths_per_confirmed['country'].iloc[0], y=mean,
                       text=f"mean = {mean*100:.2f}%",
                       showarrow=False,
                       xanchor="left",
                       yanchor="bottom",
                       font={"color":"Green", "size":14}
                      )

    fig.show()



    # --- PREPROCESSING ───────────────────────────────────────

    active_cases_df = df[['date', 'country', 'active_cases']].dropna().sort_values('date')
    active_cases_df = active_cases_df[active_cases_df.active_cases > 0]
    active_cases_df['log2(active_cases)'] = np.log2(active_cases_df['active_cases'])
    active_cases_df['date'] = active_cases_df['date'].dt.strftime('%m/%d/%Y')

    fig = px.choropleth(active_cases_df, locations="country", locationmode='country names',
                        color="log2(active_cases)", hover_name="country", hover_data=['active_cases'],
                        projection="natural earth", animation_frame="date",
                        title='<b>Coronavirus Global Active Cases Over Time</b>',
                        color_continuous_scale="reds",
                       )

    fig.update_layout(coloraxis={"colorbar": {"title":"<b>Active Cases</b><br>",
                                              "titleside":"top",
                                              "tickmode":"array",
                                              "tickvals":log_scale_vals,
                                              "ticktext":scale_vals}
                                }
                     )

    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 10
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 2

    fig.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig = px.area(df[df.country.isin(df[df.date == df.date.max()].sort_values("active_cases", ascending=False).iloc[:20].country)].sort_values("active_cases", ascending=False),
                  x="date", y="active_cases", color="country", template="plotly_dark")#, groupnorm='percent')

    fig.update_traces(line={"width":1.25})
    fig.update_layout(title = f"Top 20 Countries with Most Active Cases on {df.date.max().strftime('%Y-%m-%d')}",
                      xaxis={"title": ""},
                      yaxis={"title":"Active Cases"})

    fig = px.treemap(df_summary, path=["country"], values="active_cases", height = 750,
                     title=f"<b>Active Cases Breakdown on {df.date.max().strftime('%Y-%m-%d')}</b>",
                     color_discrete_sequence = px.colors.qualitative.Set3)

    fig.update_traces(textinfo = "label+text+value")
    fig.show()

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
    _parser = _ap.ArgumentParser(description="Full pipeline for COVID-19 Global Data Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
