#!/usr/bin/env python3
"""
Full pipeline for Movie Genre Classification

Auto-generated from: movies_genre_classification.ipynb
Project: Movie Genre Classification
Category: Classification | Task: NLP
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
# Importing essential libraries
import numpy as np
import pandas as pd
# Importing essential libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Additional imports extracted from mixed cells
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

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

    # Loading the dataset
    df = load_dataset('movie_genre_classification')

    # Validation
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    df.head()



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.columns

    df.shape

    df.head(10)

    # Visualizing the count of 'genre' column from the dataset
    plt.figure(figsize=(12,12))
    sns.countplot(x='genre', data=df)
    plt.xlabel('Movie Genres')
    plt.ylabel('Count')
    plt.title('Genre Plot')
    plt.show()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Finding unique genres
    movie_genre = list(df['genre'].unique())
    movie_genre.sort()
    movie_genre



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Mapping the genres to values
    genre_mapper = {'other': 0, 'action': 1, 'adventure': 2, 'comedy':3, 'drama':4, 'horror':5, 'romance':6, 'sci-fi':7, 'thriller': 8}
    df['genre'] = df['genre'].map(genre_mapper)
    df.head(10)



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    # Finding any NaN values
    df.isna().any()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Removing the 'id' column
    df.drop('id', axis=1, inplace=True)
    df.columns



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Importing essential libraries for performing Natural Language Processing on given dataset
    import nltk
    import re
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer



    # --- EXPLORATORY DATA ANALYSIS ───────────────────────────

    df.shape



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # Cleaning the text
    corpus = []
    ps = PorterStemmer()

    for i in range(0, df.shape[0]):

      # Cleaning special character from the dialog/script
      dialog = re.sub(pattern='[^a-zA-Z]', repl=' ', string=df['text'][i])

      # Converting the entire dialog/script into lower case
      dialog = dialog.lower()

      # Tokenizing the dialog/script by words
      words = dialog.split()

      # Removing the stop words
      dialog_words = [word for word in words if word not in set(stopwords.words('english'))]

      # Stemming the words
      words = [ps.stem(word) for word in dialog_words]

      # Joining the stemmed words
      dialog = ' '.join(words)

      # Creating a corpus
      corpus.append(dialog)

    corpus[0:10]

    df[df['genre']==4].index

    len(corpus)

    drama_words = []
    for i in list(df[df['genre']==4].index):
      drama_words.append(corpus[i])

    action_words = []
    for i in list(df[df['genre']==1].index):
      action_words.append(corpus[i])

    comedy_words = []
    for i in list(df[df['genre']==3].index):
      comedy_words.append(corpus[i])

    drama = ''
    action = ''
    comedy = ''
    for i in range(0, 3):
      drama += drama_words[i]
      action += action_words[i]
      comedy += comedy_words[i]

    # Creating wordcloud for drama genre
    from wordcloud import WordCloud
    wordcloud1 = WordCloud(background_color='white', width=3000, height=2500).generate(drama)
    plt.figure(figsize=(8,8))
    plt.imshow(wordcloud1)
    plt.axis('off')
    plt.title("Words which indicate 'DRAMA' genre ")
    plt.show()

    # Creating wordcloud for action genre
    wordcloud2 = WordCloud(background_color='white', width=3000, height=2500).generate(action)
    plt.figure(figsize=(8,8))
    plt.imshow(wordcloud2)
    plt.axis('off')
    plt.title("Words which indicate 'ACTION' genre ")
    plt.show()

    # Creating wordcloud for comedy genre
    wordcloud3 = WordCloud(background_color='white', width=3000, height=2500).generate(comedy)
    plt.figure(figsize=(8,8))
    plt.imshow(wordcloud3)
    plt.axis('off')
    plt.title("Words which indicate 'COMEDY' genre ")
    plt.show()



    # --- PREPROCESSING ───────────────────────────────────────

    # Creating the Bag of Words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=10000, ngram_range=(1,2))
    X = cv.fit_transform(corpus).toarray()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    y = df['genre'].values


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full pipeline for Movie Genre Classification")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
