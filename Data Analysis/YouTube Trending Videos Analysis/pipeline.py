#!/usr/bin/env python3
"""
Full pipeline for YouTube Trending Videos Analysis

Auto-generated from: code.ipynb
Project: YouTube Trending Videos Analysis
Category: Data Analysis | Task: data_analysis
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
from collections import Counter
import datetime
import wordcloud
import json
# Hiding warnings for cleaner display
warnings.filterwarnings('ignore')

# Configuring some options
# If you want interactive plots, uncomment the next line
# %matplotlib notebook
# Additional imports extracted from mixed cells
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def contains_capitalized_word(s):
    for w in s.split():
        if w.isupper():
            return True
    return False


df["contains_capitalized"] = df["title"].apply(contains_capitalized_word)

value_counts = df["contains_capitalized"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'],
           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'}, startangle=45)
_ = ax.axis('equal')
_ = ax.set_title('Title Contains Capitalized Word?')

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

    # --- DATA LOADING ────────────────────────────────────────

    df = load_dataset('youtube_trending_videos_analysis')



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    PLOT_COLORS = ["#268bd2", "#0052CC", "#FF5722", "#b58900", "#003f5c"]
    pd.options.display.float_format = '{:.2f}'.format
    sns.set(style="ticks")
    plt.rc('figure', figsize=(8, 5), dpi=100)
    plt.rc('axes', labelpad=20, facecolor="#ffffff", linewidth=0.4, grid=True, labelsize=14)
    plt.rc('patch', linewidth=0)
    plt.rc('xtick.major', width=0.2)
    plt.rc('ytick.major', width=0.2)
    plt.rc('grid', color='#9E9E9E', linewidth=0.4)
    plt.rc('font', family='Arial', weight='400', size=10)
    plt.rc('text', color='#282828')
    plt.rc('savefig', pad_inches=0.3, dpi=300)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.head()

    df.info()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df[df["description"].apply(lambda x: pd.isna(x))].head(3)



    # --- PREPROCESSING ───────────────────────────────────────

    df["description"] = df["description"].fillna(value="")



    # --- FEATURE ENGINEERING ─────────────────────────────────

    cdf = df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts() \
                .to_frame().reset_index() \
                .rename(columns={"index": "year", "trending_date": "No_of_videos"})

    fig, ax = plt.subplots()
    _ = sns.barplot(x="year", y="No_of_videos", data=cdf,
                    palette=sns.color_palette(['#ff764a', '#ffa600'], n_colors=7), ax=ax)
    _ = ax.set(xlabel="Year", ylabel="No. of videos")

    df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts(normalize=True)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.describe()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig, ax = plt.subplots()
    _ = sns.distplot(df["views"], kde=False, color=PLOT_COLORS[4],
                     hist_kws={'alpha': 1}, bins=np.linspace(0, 2.3e8, 47), ax=ax)
    _ = ax.set(xlabel="Views", ylabel="No. of videos", xticks=np.arange(0, 2.4e8, 1e7))
    _ = ax.set_xlim(right=2.5e8)
    _ = plt.xticks(rotation=90)

    fig, ax = plt.subplots()
    _ = sns.distplot(df[df["views"] < 25e6]["views"], kde=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
    _ = ax.set(xlabel="Views", ylabel="No. of videos")

    df[df['views'] < 1e6]['views'].count() / df['views'].count() * 100

    plt.rc('figure.subplot', wspace=0.9)
    fig, ax = plt.subplots()
    _ = sns.distplot(df["likes"], kde=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1},
                     bins=np.linspace(0, 6e6, 61), ax=ax)
    _ = ax.set(xlabel="Likes", ylabel="No. of videos")
    _ = plt.xticks(rotation=90)

    fig, ax = plt.subplots()
    _ = sns.distplot(df[df["likes"] <= 1e5]["likes"], kde=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
    _ = ax.set(xlabel="Likes", ylabel="No. of videos")

    df[df['likes'] < 4e4]['likes'].count() / df['likes'].count() * 100

    fig, ax = plt.subplots()
    _ = sns.distplot(df["comment_count"], kde=False, rug=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
    _ = ax.set(xlabel="Comment Count", ylabel="No. of videos")

    fig, ax = plt.subplots()
    _ = sns.distplot(df[df["comment_count"] < 200000]["comment_count"], kde=False, rug=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1},
                     bins=np.linspace(0, 2e5, 49), ax=ax)
    _ = ax.set(xlabel="Comment Count", ylabel="No. of videos")

    df[df['comment_count'] < 4000]['comment_count'].count() / df['comment_count'].count() * 100

    df.describe(include = ['O'])

    grouped = df.groupby("video_id")
    groups = []
    wanted_groups = []
    for key, item in grouped:
        groups.append(grouped.get_group(key))

    for g in groups:
        if len(g['title'].unique()) != 1:
            wanted_groups.append(g)

    wanted_groups[0]



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df["contains_capitalized"].value_counts(normalize=True)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df["title_length"] = df["title"].apply(lambda x: len(x))

    fig, ax = plt.subplots()
    _ = sns.distplot(df["title_length"], kde=False, rug=False,
                     color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
    _ = ax.set(xlabel="Title Length", ylabel="No. of videos", xticks=range(0, 110, 10))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig, ax = plt.subplots()
    _ = ax.scatter(x=df['views'], y=df['title_length'], color=PLOT_COLORS[2], edgecolors="#000000", linewidths=0.5)
    _ = ax.set(xlabel="Views", ylabel="Title Length")



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.corr()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    h_labels = [x.replace('_', ' ').title() for x in
                list(df.select_dtypes(include=['number', 'bool']).columns.values)]

    fig, ax = plt.subplots(figsize=(10,6))
    _ = sns.heatmap(df.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    fig, ax = plt.subplots()
    _ = plt.scatter(x=df['views'], y=df['likes'], color=PLOT_COLORS[2], edgecolors="#000000", linewidths=0.5)
    _ = ax.set(xlabel="Views", ylabel="Likes")



    # --- FEATURE ENGINEERING ─────────────────────────────────

    title_words = list(df["title"].apply(lambda x: x.split()))
    title_words = [x for y in title_words for x in y]
    Counter(title_words).most_common(25)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # wc = wordcloud.WordCloud(width=1200, height=600, collocations=False, stopwords=None, background_color="white", colormap="tab20b").generate_from_frequencies(dict(Counter(title_words).most_common(150)))
    wc = wordcloud.WordCloud(width=1200, height=500,
                             collocations=False, background_color="white",
                             colormap="tab20b").generate(" ".join(title_words))
    plt.figure(figsize=(15,10))
    plt.imshow(wc, interpolation='bilinear')
    _ = plt.axis("off")

    cdf = df.groupby("channel_title").size().reset_index(name="video_count") \
        .sort_values("video_count", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8,8))
    _ = sns.barplot(x="video_count", y="channel_title", data=cdf,
                    palette=sns.cubehelix_palette(n_colors=20, reverse=True), ax=ax)
    _ = ax.set(xlabel="No. of videos", ylabel="Channel")



    # --- FEATURE ENGINEERING ─────────────────────────────────

    with open("data/US_category_id.json") as f:
        categories = json.load(f)["items"]
    cat_dict = {}
    for cat in categories:
        cat_dict[int(cat["id"])] = cat["snippet"]["title"]
    df['category_name'] = df['category_id'].map(cat_dict)

    cdf = df["category_name"].value_counts().to_frame().reset_index()
    cdf.rename(columns={"index": "category_name", "category_name": "No_of_videos"}, inplace=True)
    fig, ax = plt.subplots()
    _ = sns.barplot(x="category_name", y="No_of_videos", data=cdf,
                    palette=sns.cubehelix_palette(n_colors=16, reverse=True), ax=ax)
    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    _ = ax.set(xlabel="Category", ylabel="No. of videos")

    df["publishing_day"] = df["publish_time"].apply(
        lambda x: datetime.datetime.strptime(x[:10], "%Y-%m-%d").date().strftime('%a'))
    df["publishing_hour"] = df["publish_time"].apply(lambda x: x[11:13])
    df.drop(labels='publish_time', axis=1, inplace=True)

    cdf = df["publishing_day"].value_counts()\
            .to_frame().reset_index().rename(columns={"index": "publishing_day", "publishing_day": "No_of_videos"})
    fig, ax = plt.subplots()
    _ = sns.barplot(x="publishing_day", y="No_of_videos", data=cdf,
                    palette=sns.color_palette(['#003f5c', '#374c80', '#7a5195',
                                               '#bc5090', '#ef5675', '#ff764a', '#ffa600'], n_colors=7), ax=ax)
    _ = ax.set(xlabel="Publishing Day", ylabel="No. of videos")

    cdf = df["publishing_hour"].value_counts().to_frame().reset_index()\
            .rename(columns={"index": "publishing_hour", "publishing_hour": "No_of_videos"})
    fig, ax = plt.subplots()
    _ = sns.barplot(x="publishing_hour", y="No_of_videos", data=cdf,
                    palette=sns.cubehelix_palette(n_colors=24), ax=ax)
    _ = ax.set(xlabel="Publishing Hour", ylabel="No. of videos")



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    value_counts = df["video_error_or_removed"].value_counts().to_dict()
    fig, ax = plt.subplots()
    _ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'],
            colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
    _ = ax.axis('equal')
    _ = ax.set_title('Video Error or Removed?')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df["video_error_or_removed"].value_counts()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    value_counts = df["comments_disabled"].value_counts().to_dict()
    fig, ax = plt.subplots()
    _ = ax.pie(x=[value_counts[False], value_counts[True]], labels=['No', 'Yes'],
               colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
    _ = ax.axis('equal')
    _ = ax.set_title('Comments Disabled?')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df["comments_disabled"].value_counts(normalize=True)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    value_counts = df["ratings_disabled"].value_counts().to_dict()
    fig, ax = plt.subplots()
    _ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'],
                colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'})
    _ = ax.axis('equal')
    _ = ax.set_title('Ratings Disabled?')



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df["ratings_disabled"].value_counts()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    len(df[(df["comments_disabled"] == True) & (df["ratings_disabled"] == True)].index)

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
    _parser = _ap.ArgumentParser(description="Full pipeline for YouTube Trending Videos Analysis")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
