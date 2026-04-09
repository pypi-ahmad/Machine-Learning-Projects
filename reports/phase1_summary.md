# Phase 1 — Summary Report

Generated: 2026-03-03 01:18 UTC

**Projects**: 21  
**Datasets ready**: 21 / 21  
**Notebooks patched**: 21 (+ 0 already done)

---

| # | Project Slug | Task | Dataset Status | Source | Text Col | Target Col | Baseline Ready |
|---|---|---|---|---|---|---|---|
| 1 | `automated-image-captioning` | image_captioning | READY | data_dir (previously prepared) | `caption` | `-` | yes |
| 2 | `bbc-articles-summarization` | summarization | READY | data_dir (previously prepared) | `article` | `summary` | yes |
| 3 | `cyberbullying-classification` | classification | READY | data_dir (previously prepared) | `tweet_text` | `cyberbullying_type` | yes |
| 4 | `e-commerce-clothing-reviews` | analysis | READY | data_dir (previously prepared) | `Review Text` | `-` | yes |
| 5 | `e-commerce-product-classification` | classification | READY | data_dir (previously prepared) | `__col1__` | `__col0__` | yes |
| 6 | `economic-news-articles` | classification | READY | data_dir (previously prepared) | `text` | `positivity` | yes |
| 7 | `english-to-french-translation` | translation | READY | data_dir (previously prepared) | `english` | `french` | yes |
| 8 | `fake-news-detection` | classification | READY | data_dir (previously prepared) | `text` | `label` | yes |
| 9 | `kaggle-survey-questions-clustering` | clustering | READY | data_dir (previously prepared) | `-` | `-` | yes |
| 10 | `medium-articles-clustering` | clustering | READY | data_dir (previously prepared) | `text` | `-` | yes |
| 11 | `name-generate-from-languages` | generation | READY | data_dir (previously prepared) | `name` | `language` | yes |
| 12 | `news-headline-classification` | classification | READY | data_dir (previously prepared) | `headline` | `category` | yes |
| 13 | `newsgroups-posts-clustering` | clustering | READY | data_dir (previously prepared) | `text` | `newsgroup` | yes |
| 14 | `paper-subject-prediction` | classification | READY | data_dir (previously prepared) | `summaries` | `terms` | yes |
| 15 | `review-classification` | classification | READY | data_dir (previously prepared) | `text` | `label` | yes |
| 16 | `spam-message-detection` | classification | READY | data_dir (previously prepared) | `text` | `label` | yes |
| 17 | `stories-clustering` | clustering | READY | data_dir (previously prepared) | `story` | `-` | yes |
| 18 | `toxic-comment-classification` | classification_multilabel | READY | data_dir (previously prepared) | `comment_text` | `toxic, severe_toxic...` | yes |
| 19 | `trip-advisor-hotel-reviews` | analysis | READY | data_dir (previously prepared) | `Review` | `Rating` | yes |
| 20 | `twitter-sentiment-analysis` | classification | READY | data_dir (previously prepared) | `OriginalTweet` | `Sentiment` | yes |
| 21 | `world-war-i-letters` | analysis | READY | data_dir (previously prepared) | `-` | `-` | yes |

---

## Per-Project Details

### automated-image-captioning

- **Task:** image_captioning
- **Directory:** `3. Text Generation/3. Text Generation/Automated Image Captioning`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['captions.txt', 'images']
- **Official links:**
  - https://www.kaggle.com/datasets/adityajn105/flickr8k
  - https://forms.illinois.edu/sec/1713398
- **License:** Research use (Flickr terms)
- **Notebook patch:** 4 Phase-1 cells inserted after pos 1

### bbc-articles-summarization

- **Task:** summarization
- **Directory:** `3. Text Generation/3. Text Generation/BBC Articles Summarization`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['BBC News Summary']
- **Official links:**
  - https://www.kaggle.com/datasets/pariza/bbc-news-summary
  - http://mlg.ucd.ie/datasets/bbc.html
- **License:** Academic research only
- **Notebook patch:** 4 Phase-1 cells inserted after pos 4
- **Note:** Dataset NOT bundled; must download from Kaggle

### cyberbullying-classification

- **Task:** classification
- **Directory:** `2. Text Classification/2. Text Classification/Cyberbullying Classification`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['data.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification
- **License:** CC BY 4.0
- **Notebook patch:** 4 Phase-1 cells inserted after pos 7

### e-commerce-clothing-reviews

- **Task:** analysis
- **Directory:** `1. Text Processing and Analysis/1. Text Processing and Analysis/E-Commerce Clothing Reviews`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['data.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
- **License:** CC0: Public Domain
- **Notebook patch:** 4 Phase-1 cells inserted after pos 6

### e-commerce-product-classification

- **Task:** classification
- **Directory:** `2. Text Classification/2. Text Classification/E-commerce Product Classification`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['data.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/sumedhdataaspirant/e-commerce-text-dataset
- **License:** Unknown
- **Notebook patch:** 4 Phase-1 cells inserted after pos 2
- **Note:** CSV has no header; col0=category, col1=description

### economic-news-articles

- **Task:** classification
- **Directory:** `2. Text Classification/2. Text Classification/Economic News Articles`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['data.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/adhamelkomy/news-classification-and-analysis-using-nlp
  - https://www.figure-eight.com/
- **License:** Open (crowdsourced annotations)
- **Notebook patch:** 4 Phase-1 cells inserted after pos 3

### english-to-french-translation

- **Task:** translation
- **Directory:** `3. Text Generation/3. Text Generation/English to French Translation`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['eng-fra.txt']
- **Official links:**
  - https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset
  - https://www.manythings.org/anki/
- **License:** CC BY 2.0 (Tatoeba)
- **Notebook patch:** 4 Phase-1 cells inserted after pos 2

### fake-news-detection

- **Task:** classification
- **Directory:** `2. Text Classification/2. Text Classification/Fake News Detection`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['Fake.csv', 'True.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/vishakhdapat/fake-news-detection
  - https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php
- **License:** Academic use
- **Notebook patch:** 4 Phase-1 cells inserted after pos 3
- **Note:** True.csv and Fake.csv must be merged with a label column (1=True, 0=Fake)

### kaggle-survey-questions-clustering

- **Task:** clustering
- **Directory:** `4. Text Clustering & Topic Modelling/4. Text Clustering _ Topic Modelling/Kaggle Survey Questions Clustering`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['questions.csv']
- **Official links:**
  - https://www.kaggle.com/competitions/kaggle-survey-2021
- **License:** Competition rules (Kaggle)
- **Notebook patch:** 4 Phase-1 cells inserted after pos 1

### medium-articles-clustering

- **Task:** clustering
- **Directory:** `4. Text Clustering & Topic Modelling/4. Text Clustering _ Topic Modelling/Medium Articles Clustering`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['articles.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/arnabchaki/medium-articles-dataset
- **License:** Unknown (scraped data)
- **Notebook patch:** 4 Phase-1 cells inserted after pos 1

### name-generate-from-languages

- **Task:** generation
- **Directory:** `3. Text Generation/3. Text Generation/Name Generate From Languages`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['names']
- **Official links:**
  - https://www.kaggle.com/datasets/davidam9/international-names
  - https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
- **License:** Public Domain
- **Notebook patch:** 4 Phase-1 cells inserted after pos 2
- **Note:** Local names/*.txt files bundled in repo

### news-headline-classification

- **Task:** classification
- **Directory:** `2. Text Classification/2. Text Classification/News Headline Classification`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['data.json']
- **Official links:**
  - https://www.kaggle.com/datasets/rmisra/news-category-dataset
  - https://rishabhmisra.github.io/publications/
- **License:** CC BY 4.0
- **Notebook patch:** 4 Phase-1 cells inserted after pos 4

### newsgroups-posts-clustering

- **Task:** clustering
- **Directory:** `4. Text Clustering & Topic Modelling/4. Text Clustering _ Topic Modelling/Newsgroups Posts Clustering`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['alt.atheism.txt', 'comp.graphics.txt', 'comp.os.ms-windows.misc.txt', 'comp.sys.ibm.pc.hardware.txt', 'comp.sys.mac.hardware.txt', 'comp.windows.x.txt', 'list.csv', 'misc.forsale.txt', 'rec.autos.txt', 'rec.motorcycles.txt', 'rec.sport.baseball.txt', 'rec.sport.hockey.txt', 'sci.crypt.txt', 'sci.electronics.txt', 'sci.med.txt', 'sci.space.txt', 'soc.religion.christian.txt', 'talk.politics.guns.txt', 'talk.politics.mideast.txt', 'talk.politics.misc.txt', 'talk.religion.misc.txt']
- **Official links:**
  - https://www.kaggle.com/datasets/crawford/20-newsgroups
  - http://qwone.com/~jason/20Newsgroups/
- **License:** Public Domain
- **Notebook patch:** 4 Phase-1 cells inserted after pos 3
- **Note:** Uses sklearn.datasets.fetch_20newsgroups; data not bundled

### paper-subject-prediction

- **Task:** classification
- **Directory:** `2. Text Classification/2. Text Classification/Paper Subject Prediction`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['arxiv_data.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset
  - https://arxiv.org/
- **License:** arXiv Terms of Use
- **Notebook patch:** 4 Phase-1 cells inserted after pos 4

### review-classification

- **Task:** classification
- **Directory:** `2. Text Classification/2. Text Classification/Review Classification`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['amazon.txt', 'imdb.txt', 'yelp.txt']
- **Official links:**
  - https://www.kaggle.com/datasets/ndrianahani/imdb-yelp-and-amazon-reviews
- **License:** Various (Amazon ToS, IMDb ToS, Yelp ToS)
- **Notebook patch:** 4 Phase-1 cells inserted after pos 3
- **Note:** Tab-separated files: text<TAB>label (0/1). Three sources.

### spam-message-detection

- **Task:** classification
- **Directory:** `2. Text Classification/2. Text Classification/Spam Message Detection`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['data.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
  - https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- **License:** CC BY 4.0 (UCI)
- **Notebook patch:** 4 Phase-1 cells inserted after pos 2

### stories-clustering

- **Task:** clustering
- **Directory:** `4. Text Clustering & Topic Modelling/4. Text Clustering _ Topic Modelling/Stories Clustering`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['1k_stories_100_genre.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/fareedkhan557/1000-stories-100-genres
- **License:** CC0: Public Domain
- **Notebook patch:** 4 Phase-1 cells inserted after pos 2
- **Note:** Dataset NOT bundled; must download from Kaggle

### toxic-comment-classification

- **Task:** classification_multilabel
- **Directory:** `2. Text Classification/2. Text Classification/Toxic Comment Classification`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['test.csv', 'train.csv']
- **Official links:**
  - https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
  - https://www.perspectiveapi.com/research/
- **License:** Competition rules (Kaggle)
- **Notebook patch:** 4 Phase-1 cells inserted after pos 2

### trip-advisor-hotel-reviews

- **Task:** analysis
- **Directory:** `1. Text Processing and Analysis/1. Text Processing and Analysis/Trip Advisor Hotel Reviews`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['data.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/thedevastator/tripadvisor-hotel-reviews
- **License:** Open Data Commons
- **Notebook patch:** 4 Phase-1 cells inserted after pos 3

### twitter-sentiment-analysis

- **Task:** classification
- **Directory:** `2. Text Classification/2. Text Classification/Twitter Sentiment Analysis`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['test.csv', 'train.csv']
- **Official links:**
  - https://www.kaggle.com/datasets/milobele/sentiment140-dataset-1600000-tweets
  - http://help.sentiment140.com/for-students/
- **License:** Research use
- **Notebook patch:** 4 Phase-1 cells inserted after pos 7

### world-war-i-letters

- **Task:** analysis
- **Directory:** `1. Text Processing and Analysis/1. Text Processing and Analysis/World War I Letters`
- **Dataset status:** READY
- **Source:** data_dir (previously prepared)
- **Files:** ['data.csv', 'letters.json']
- **Official links:**
  - https://www.kaggle.com/datasets/anthaus/world-war-i-letters
- **License:** CC BY 4.0
- **Notebook patch:** 4 Phase-1 cells inserted after pos 2
