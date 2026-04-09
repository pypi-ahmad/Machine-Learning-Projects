#!/usr/bin/env python3
"""
Model evaluation for Cyberbullying Classification

Auto-generated from: Cyberbullying_classification.ipynb
Project: Cyberbullying Classification
Category: Classification | Task: NLP
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from core.data_loader import load_dataset
import demoji
# Additional imports extracted from mixed cells
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, string
import emoji
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from collections import Counter
from gensim.models import Word2Vec
import transformers
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
import random
import time

# ======================================================================
# HELPER FUNCTIONS (from notebook)
# ======================================================================
def conf_matrix(y, y_pred, title, labels):
    fig, ax =plt.subplots(figsize=(7.5,7.5))
    ax=sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Purples", fmt='g', cbar=False, annot_kws={"size":30})
    plt.title(title, fontsize=25)
    ax.xaxis.set_ticklabels(labels, fontsize=16) 
    ax.yaxis.set_ticklabels(labels, fontsize=14.5)
    ax.set_ylabel('Test', fontsize=25)
    ax.set_xlabel('Predicted', fontsize=25)
    plt.show()
def strip_emoji(text):
    return demoji.replace(text, '')
##CUSTOM DEFINED FUNCTIONS TO CLEAN THE TWEETS

#Clean emojis from text
#def strip_emoji(text):
#    return re.sub(emoji.get_emoji_regexp(), r"", text) #remove emoji

#Remove punctuations, links, stopwords, mentions and \r\n new line characters
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    text =' '.join(word for word in text.split() if len(word) < 14) # remove words longer than 14 characters
    return text

#remove contractions
def decontract(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the "#" symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split(r'#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

#Filter special characters such as "&" and "$" present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

#Remove multiple sequential spaces
def remove_mult_spaces(text):
    return re.sub(r"\s\s+" , " ", text)

#Stemming
def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

#Lemmatization 
#NOTE:Stemming seems to work better for this dataset
def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized])

#Then we apply all the defined functions in the following order
def deep_clean(text):
    text = strip_emoji(text)
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_chars(text)
    text = remove_mult_spaces(text)
    text = stemmer(text)
    return text

# ======================================================================
# EVALUATION PIPELINE
# ======================================================================

def main():
    """Run the evaluation pipeline."""
    # --- REPRODUCIBILITY ─────────────────────────────────────
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- ADDITIONAL PROCESSING ───────────────────────────────

    import os
    for dirname, _, filenames in os.walk('./archive/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))



    # --- EVALUATION ──────────────────────────────────────────

    #Libraries for general purpose
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    #Text cleaning
    import re, string
    import emoji
    import nltk
    from nltk.stem import WordNetLemmatizer,PorterStemmer
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    #Data preprocessing
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import RandomOverSampler

    #Naive Bayes
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB


    #PyTorch LSTM
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

    #Tokenization for LSTM
    from collections import Counter
    from gensim.models import Word2Vec

    #Transformers library for BERT
    import transformers
    from transformers import BertModel
    from transformers import BertTokenizer
    from transformers import AdamW, get_linear_schedule_with_warmup

    from sklearn.metrics import classification_report, confusion_matrix

    #Seed for reproducibility
    import random

    seed_value=42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    import time

    #set style for plots
    sns.set_style("whitegrid")
    sns.despine()
    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)



    # --- DATA LOADING ────────────────────────────────────────

    df = load_dataset('cyberbullying_classification')



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    df.duplicated().sum()

    df = df[~df.duplicated()]

    texts_new = []
    for t in df.text:
        texts_new.append(deep_clean(t))

    df['text_clean'] = texts_new

    df["text_clean"].duplicated().sum()

    df.drop_duplicates("text_clean", inplace=True)

    df = df[df["sentiment"]!="other_cyberbullying"]

    sentiments = ["religion","age","ethnicity","gender","not bullying"]

    text_len = []
    for text in df.text_clean:
        tweet_len = len(text.split())
        text_len.append(tweet_len)

    df['text_len'] = text_len

    plt.figure(figsize=(7,5))
    ax = sns.countplot(x='text_len', data=df[df['text_len']<10], palette='mako')
    plt.title('Count of tweets with less than 10 words', fontsize=20)
    plt.yticks([])
    ax.bar_label(ax.containers[0])
    plt.ylabel('count')
    plt.xlabel('')
    plt.show()

    df = df[df['text_len'] > 3]

    df.sort_values(by=['text_len'], ascending=False)

    plt.figure(figsize=(16,5))
    ax = sns.countplot(x='text_len', data=df[(df['text_len']<=1000) & (df['text_len']>10)], palette='Blues_r')
    plt.title('Count of tweets with high number of words', fontsize=25)
    plt.yticks([])
    ax.bar_label(ax.containers[0])
    plt.ylabel('count')
    plt.xlabel('')
    plt.show()

    df = df[df['text_len'] < 100]

    max_len = np.max(df['text_len'])
    max_len

    df.sort_values(by=["text_len"], ascending=False)



    # --- FEATURE ENGINEERING ─────────────────────────────────

    df['sentiment'] = df['sentiment'].replace({'religion':0,'age':1,'ethnicity':2,'gender':3,'not_cyberbullying':4})



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X = df['text_clean']
    y = df['sentiment']



    # --- PREPROCESSING ───────────────────────────────────────

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_value)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=seed_value)



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    (unique, counts) = np.unique(y_train, return_counts=True)
    np.asarray((unique, counts)).T



    # --- PREPROCESSING ───────────────────────────────────────

    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1));
    train_os = pd.DataFrame(list(zip([x[0] for x in X_train], y_train)), columns = ['text_clean', 'sentiment']);



    # --- ADDITIONAL PROCESSING ───────────────────────────────

    X_train = train_os['text_clean'].values
    y_train = train_os['sentiment'].values

    (unique, counts) = np.unique(y_train, return_counts=True)
    np.asarray((unique, counts)).T


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Model evaluation for Cyberbullying Classification")
    _parser.add_argument("--reproduce", action="store_true", default=True,
                         help="Force deterministic behaviour (default: True)")
    _parser.add_argument("--seed", type=int, default=42,
                         help="Global random seed (default: 42)")
    _args = _parser.parse_args()
    main()
