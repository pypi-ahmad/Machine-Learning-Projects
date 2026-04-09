#!/usr/bin/env python3
"""
Model evaluation for Drug Classification

Auto-generated from: Drug_Classification.ipynb
Project: Drug Classification
Category: Classification | Task: classification
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
# Importing Libraries for the Machine Learning Model
from xgboost import XGBClassifier
from lightgbm import LGBMModel,LGBMClassifier, plot_importance
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
# Additional imports extracted from mixed cells
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from textblob import TextBlob
from nltk.corpus import stopwords
from collections import Counter
import warnings; warnings.simplefilter('ignore')
import nltk
import string
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from lazypredict.Supervised import LazyClassifier
from pycaret.classification import *

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def review_clean(review): 
    # changing to lower case
    lower = review.str.lower()
    
    # Replacing the repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")
    
    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
    
    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')
    
    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')
    
    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+',' ')
    
    # Replacing Two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')
    
    return dataframe
def sentiment(review):
    # Sentiment polarity of the reviews
    pol = []
    for i in review:
        analysis = TextBlob(i)
        pol.append(analysis.sentiment.polarity)
    return pol

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

    df = load_dataset('drug_classification')
    test = pd.read_csv('../../data/drug_classification/drugsComTest_raw.csv')
    df.head()



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # as both the dataset contains same columns we can combine them for better analysis

    data = pd.concat([df, test])
    data.head()



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    # let's see the words cloud for the reviews 

    # most popular drugs

    from wordcloud import WordCloud
    from wordcloud import STOPWORDS

    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(background_color = 'orange', stopwords = stopwords, width = 1200, height = 800).generate(str(data['drugName']))

    plt.rcParams['figure.figsize'] = (15, 15)
    plt.title('Word Cloud - Drug Names', fontsize = 25)
    print(wordcloud)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

    # This barplot shows the top 20 drugs with the 10/10 rating

    # Setting the Parameter
    sns.set(font_scale = 1.2, style = 'darkgrid')
    plt.rcParams['figure.figsize'] = [15, 8]

    rating = dict(data.loc[data.rating == 10, "drugName"].value_counts())
    drugname = list(rating.keys())
    drug_rating = list(rating.values())

    sns_rating = sns.barplot(x = drugname[0:20], y = drug_rating[0:20])

    sns_rating.set_title('Top 20 drugs with 10/10 rating')
    sns_rating.set_ylabel("Number of Ratings")
    sns_rating.set_xlabel("Drug Names")
    plt.setp(sns_rating.get_xticklabels(), rotation=90);

    # This barplot shows the Top 20 drugs with the 1/10 rating

    # Setting the Parameter
    sns.set(font_scale = 1.2, style = 'darkgrid')
    plt.rcParams['figure.figsize'] = [15, 8]

    rating = dict(data.loc[data.rating == 1, "drugName"].value_counts())
    drugname = list(rating.keys())
    drug_rating = list(rating.values())

    sns_rating = sns.barplot(x = drugname[0:20], y = drug_rating[0:20], palette = 'winter')

    sns_rating.set_title('Top 20 drugs with 1/10 rating')
    sns_rating.set_ylabel("Number of Ratings")
    sns_rating.set_xlabel("Drug Names")
    plt.setp(sns_rating.get_xticklabels(), rotation=90);

    # making a donut chart to represent share of each ratings

    size = [68005, 46901, 36708, 25046, 12547, 10723, 8462, 6671]
    colors = ['pink', 'cyan', 'maroon',  'magenta', 'orange', 'navy', 'lightgreen', 'yellow']
    labels = "10", "1", "9", "8", "7", "5", "6", "4"

    my_circle = plt.Circle((0, 0), 0.7, color = 'white')

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.pie(size, colors = colors, labels = labels, autopct = '%.2f%%')
    plt.axis('off')
    plt.title('Pie Chart Representation of Ratings', fontsize = 25)
    p = plt.gcf()
    plt.gca().add_artist(my_circle)
    plt.legend()
    plt.show()

    # A countplot of the ratings so we can see the distribution of the ratings
    plt.rcParams['figure.figsize'] = [20,8]
    sns.set(font_scale = 1.4, style = 'darkgrid')
    fig, ax = plt.subplots(1, 2)

    sns_1 = sns.countplot(data['rating'], palette = 'spring', order = list(range(10, 0, -1)), ax = ax[0])
    sns_2 = sns.distplot(data['rating'], ax = ax[1])
    sns_1.set_title('Count of Ratings')
    sns_1.set_xlabel("Rating")

    sns_2.set_title('Distribution of Ratings')
    sns_2.set_xlabel("Rating")

    # This barplot show the top 10 conditions the people are suffering.
    cond = dict(data['condition'].value_counts())
    top_condition = list(cond.keys())[0:10]
    values = list(cond.values())[0:10]
    sns.set(style = 'darkgrid', font_scale = 1.3)
    plt.rcParams['figure.figsize'] = [18, 7]

    sns_ = sns.barplot(x = top_condition, y = values, palette = 'winter')
    sns_.set_title("Top 10 conditions")
    sns_.set_xlabel("Conditions")
    sns_.set_ylabel("Count");

    # Top 10 drugs which are used for the top condition, that is Birth Control
    df1 = data[data['condition'] == 'Birth Control']['drugName'].value_counts()[0: 10]
    sns.set(font_scale = 1.2, style = 'darkgrid')

    sns_ = sns.barplot(x = df1.index, y = df1.values, palette = 'summer')
    sns_.set_xlabel('Drug Names')
    sns_.set_title("Top 10 Drugs used for Birth Control")
    plt.setp(sns_.get_xticklabels(), rotation = 90);

    # let's see the words cloud for the reviews 

    # most popular drugs

    from wordcloud import WordCloud
    from wordcloud import STOPWORDS

    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(background_color = 'lightblue', stopwords = stopwords, width = 1200, height = 800).generate(str(data['review']))

    plt.rcParams['figure.figsize'] = (15, 15)
    plt.title('WORD CLOUD OF REVIEWS', fontsize = 25)
    print(wordcloud)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

    # feature engineering 
    # let's make a new column review sentiment 

    data.loc[(data['rating'] >= 5), 'Review_Sentiment'] = 1
    data.loc[(data['rating'] < 5), 'Review_Sentiment'] = 0

    data['Review_Sentiment'].value_counts()

    # a pie chart to represent the sentiments of the patients

    size = [161491, 53572]
    colors = ['lightblue', 'navy']
    labels = "Positive Sentiment","Negative Sentiment"
    explode = [0, 0.1]

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.pie(size, colors = colors, labels = labels, explode = explode, autopct = '%.2f%%')
    plt.axis('off')
    plt.title('Pie Chart Representation of Sentiments', fontsize = 25)
    plt.legend()
    plt.show()

    # making Words cloud for the postive sentiments

    positive_sentiments = " ".join([text for text in data['review'][data['Review_Sentiment'] == 1]])

    from wordcloud import WordCloud
    from wordcloud import STOPWORDS

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color = 'magenta', stopwords = stopwords, width = 1200, height = 800).generate(positive_sentiments)

    plt.rcParams['figure.figsize'] = (15, 15)
    plt.title('Word Cloud of Positive Reviews', fontsize = 30)
    print(wordcloud)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

    # making wordscloud for the Negative sentiments

    negative_sentiments = " ".join([text for text in data['review'][data['Review_Sentiment'] == 0]])

    from wordcloud import WordCloud
    from wordcloud import STOPWORDS

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color = 'cyan', stopwords = stopwords, width = 1200, height = 800).generate(negative_sentiments)

    plt.rcParams['figure.figsize'] = (15, 15)
    plt.title('Word Cloud of Negative Reviews', fontsize = 30)
    print(wordcloud)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

    # converting the date into datetime format
    data['date'] = pd.to_datetime(data['date'], errors = 'coerce')

    # now extracting year from date
    data['Year'] = data['date'].dt.year

    # extracting the month from the date
    data['month'] = data['date'].dt.month

    # extracting the days from the date
    data['day'] = data['date'].dt.day

    data['review_clean'] = review_clean(data['review'])



    # --- PREPROCESSING ───────────────────────────────────────

    from textblob import TextBlob
    from nltk.corpus import stopwords
    from collections import Counter
    import warnings; warnings.simplefilter('ignore')
    import nltk
    import string
    from nltk import ngrams
    from nltk.tokenize import word_tokenize 
    from nltk.stem import SnowballStemmer

    # Removing the stopwords
    stop_words = set(stopwords.words('english'))
    data['review_clean'] = data['review_clean'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))



    # --- FEATURE ENGINEERING ─────────────────────────────────

    # Removing the word stems using the Snowball Stemmer
    Snow_ball = SnowballStemmer("english")
    data['review_clean'] = data['review_clean'].apply(lambda x: " ".join(Snow_ball.stem(word) for word in x.split()))



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    data['sentiment'] = sentiment(data['review'])

    data['sentiment_clean'] = sentiment(data['review_clean'])

    # Cleaning the reviews without removing the stop words and using snowball stemmer
    data['review_clean_ss'] = review_clean(data['review'])
    data['sentiment_clean_ss'] = sentiment(data['review_clean_ss'])



    # --- PREPROCESSING ───────────────────────────────────────

    data = data.dropna(how="any", axis=0)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    #Word count in each review
    data['count_word']=data["review_clean_ss"].apply(lambda x: len(str(x).split()))

    #Unique word count 
    data['count_unique_word']=data["review_clean_ss"].apply(lambda x: len(set(str(x).split())))

    #Letter count
    data['count_letters']=data["review_clean_ss"].apply(lambda x: len(str(x)))

    #punctuation count
    data["count_punctuations"] = data["review"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    #upper case words count
    data["count_words_upper"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    #title case words count
    data["count_words_title"] = data["review"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    #Number of stopwords
    data["count_stopwords"] = data["review"].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

    #Average length of the words
    data["mean_word_len"] = data["review_clean_ss"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



    # --- PREPROCESSING ───────────────────────────────────────

    # Label Encoding Drugname and Conditions
    from sklearn.preprocessing import LabelEncoder
    label_encoder_feat = {}
    for feature in ['drugName', 'condition']:
        label_encoder_feat[feature] = LabelEncoder()
        data[feature] = label_encoder_feat[feature].fit_transform(data[feature])

    # Defining Features and splitting the data as train and test set

    features = data[['condition', 'usefulCount', 'sentiment', 'day', 'month', 'Year',
                       'sentiment_clean_ss', 'count_word', 'count_unique_word', 'count_letters',
                       'count_punctuations', 'count_words_upper', 'count_words_title',
                       'count_stopwords', 'mean_word_len']]

    target = data['Review_Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)
    print ("The Train set size ", X_train.shape)
    print ("The Test set size ", X_test.shape)



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

            clf_setup = setup(data=df, target='Drug', session_id=42, verbose=False)

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
    _parser = _ap.ArgumentParser(description="Model evaluation for Drug Classification")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
