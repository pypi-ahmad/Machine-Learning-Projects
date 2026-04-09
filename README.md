# Natural Language Processing Projects

> **33 NLP projects** — sentiment analysis, text classification, named entity recognition, summarization, keyword extraction, autocorrect, fake news detection, and more.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-358%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Project Catalog

| # | Project | Type | Key Technique | AutoML |
|---|---------|------|---------------|:------:|
| 1 | [Resume Screening](NLP%20Projecct%201.ResumeScreening/) | Classification | TF-IDF + LazyPredict / PyCaret | ✅ |
| 2 | [Named Entity Recognition](NLP%20Projecct%202.Named%20Entity%20Recognition/) | Sequence Labeling | BiLSTM + LSTM (Keras) | |
| 3 | [Sentiment Analysis](NLP%20Projecct%203.Sentiment%20Analysis/) | Classification | TF-IDF + LazyPredict / PyCaret | ✅ |
| 4 | [Keyword Extraction](NLP%20Projecct%204.Keyword%20Extraction/) | NLP Utility | TF-IDF + CountVectorizer | |
| 5 | [Correct Spelling](NLP%20Projecct%205.correct%20Spelling/) | NLP Utility | TextBlob | |
| 6 | [Autocorrect](NLP%20Projecct%206.Autocorrect/) | NLP Utility | Jaccard Distance + N-grams | |
| 7 | [Predict US Election](NLP%20Projecct%207.Predict_US_election/) | EDA / Sentiment | TextBlob Polarity | |
| 8 | [NLP for Other Languages](NLP%20Projecct%208.NLP%20for%20other%20languages/) | NLP Utility | spaCy + WordCloud (Hindi) | |
| 9 | [Text Classification](NLP%20Projecct%209.Textclassification/) | Classification | Keras Dense NN | |
| 10 | [Text Summarization](NLP%20Projecct%2010.TextSummarization/) | Summarization | TextRank + GloVe | |
| 11 | [Hate Speech Detection](NLP%20Projecct%2011.HateSpeechDetection/) | Classification | TF-IDF + LazyPredict / PyCaret | ✅ |
| 12 | [Keyword Research](NLP%20Projecct%2012.KeywordResearch/) | EDA | Google Trends (pytrends) | |
| 13 | [WhatsApp Group Chat Analysis](NLP%20Projecct%2013.WhatsApp%20Group%20Chat%20Analysis/) | EDA | Regex Parsing + WordCloud | |
| 14 | [Next Word Prediction](NLP%20Projecct%2014.Next%20Word%20prediction%20Model/) | Language Model | LSTM (word-level) | |
| 15 | [Fake News Detection](NLP%20Projecct%2015.FakeNews%20Detection%20Model/) | Classification | CountVectorizer + LazyPredict / PyCaret | ✅ |
| 16 | [NLP for WhatsApp Chats](NLP%20Projecct%2016.NLP%20for%20whatsapp%20chats/) | EDA / Sentiment | TextBlob + WordCloud | |
| 17 | [Twitter Sentiment Analysis](NLP%20Projecct%2017.Twitter%20Sentiment%20Analysis/) | Classification | NLTK NaiveBayes + LazyPredict / PyCaret | ✅ |
| 18 | [SMS Spam Detection](NLP%20Projecct%2018.SMS%20spam%20detection/) | Classification | TF-IDF + LazyPredict / PyCaret | ✅ |
| 19 | [Movie Review Sentiments](NLP%20Projecct%2019.%20MoviesReviewSentiments/) | Classification | CountVectorizer + LazyPredict / PyCaret | ✅ |
| 20 | [Amazon Sentiment Analysis](NLP%20Projecct%2020.Amazon%20Sentiment%20Analysis/) | EDA | StratifiedShuffleSplit + LazyPredict / PyCaret | ✅ |
| 21 | [Dow Jones Stock Sentiment](NLP%20Project%2021.%20Sentiment%20Analysis%20-%20Dow%20Jones%20(DJIA)%20Stock%20using%20News%20Headlines/) | Classification | CountVectorizer (bigrams) + LazyPredict / PyCaret | ✅ |
| 22 | [Restaurant Reviews](NLP%20Project%2022.%20-%20Sentiment%20Analysis%20-%20Restaurant%20Reviews/) | Classification | CountVectorizer + LazyPredict / PyCaret | ✅ |
| 23 | [Spam SMS Classification](NLP%20Project%2023.%20-%20Spam%20SMS%20Classification/) | Classification | TfidfVectorizer + LazyPredict / PyCaret | ✅ |
| 24 | [Clinton / Trump Tweets](NLP%20Projects%2024%20-%20Hillary%20Clinton%20and%20Donald%20Trump%20Tweets/) | EDA / NLP | simpletransformers (GPT-2 init) | |
| 25 | [Airline Sentiment](NLP%20Projects%2025%20-%20Twitter%20Us%20Airline%20Sentiment%20Analysis/) | EDA | Pandas + Matplotlib | |
| 26 | [IMDB Deep Learning](NLP%20Projects%2026%20-%20IMDB%20Sentiment%20Analysis%20using%20Deep%20Learning/) | Classification | TF-IDF + LazyPredict / PyCaret | ✅ |
| 27 | [Amazon Alexa Reviews](NLP%20Projects%2027%20-%20Amazon%20Alexa%20Review%20Sentiment%20Analysis/) | Classification | CountVectorizer + LazyPredict / PyCaret | ✅ |
| 28 | [Flask Sentiment App](NLP%20Projects%2028%20-%20Build_Sentiment_Analysis_Flask_Web_App/) | Deep Learning | Keras LSTM | |
| 31 | [SMS Spam Analysis](NLP%20Projects%2031%20-%20SMS%20Spam%20Detection%20Analysis/) | Classification | TF-IDF + LazyPredict / PyCaret | ✅ |
| 32 | [Text Summarization (Freq)](NLP%20Projects%2032%20-%20Text%20Summarization%20using%20Word%20Frequency/) | Summarization | Word Frequency + heapq | |
| 33 | [GitHub Bugs Prediction](NLP%20Projects%2033%20-%20GitHub%20Bugs%20Prediction/) | Classification | DistilBERT / ALBERT / XLM-RoBERTa / GPT-2 | |
| 34 | [Hindi Stop Words](NLP%20Projects%2034%20-%20Stop%20words%20in%2028%20Languages/) | NLP Utility | Word2Vec + TensorFlow v1 | |
| 35 | [Word Clouds](NLP%20Projects%2035%20-%20Text%20Similarity/) | EDA / Visualization | WordCloud | |

> Projects 29 and 30 were removed as duplicates of 21 and 22.

---

## Categories

| Category | Projects |
|----------|----------|
| **Sentiment Analysis** | 1, 3, 7, 16, 17, 19, 20, 21, 22, 25, 26, 27 |
| **Text Classification** | 9, 11, 15, 18, 23, 31, 33 |
| **NLP Utilities** | 4, 5, 6, 8, 34 |
| **Deep Learning / Transformers** | 2, 9, 14, 28, 33 |
| **Summarization** | 10, 32 |
| **EDA / Visualization** | 7, 12, 13, 24, 25, 35 |

---

## Repository Structure

```
├── data/                        # Centralized datasets (per project)
├── artifacts/                   # Trained model artifacts (per project)
├── NLP Projecct 1.…/            # Project folders (each has notebook + README + test)
├── NLP Projecct 2.…/
├── …
├── config.py                    # Centralized path resolution
├── inference_engine.py          # Model loading and prediction
├── api.py                       # FastAPI serving endpoint
├── leaderboard.py               # Cross-project model comparison
├── stress_test.py               # API load testing
├── conftest.py                  # Shared pytest fixtures
├── pyproject.toml               # Project metadata
├── .gitattributes               # Git LFS tracking rules
└── WORKSPACE_OVERVIEW.md        # Detailed project catalog
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/pypi-ahmad/Natural-Language-Processing-Projects.git
cd Natural-Language-Processing-Projects

# Set up environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Testing

358 tests cover data loading, preprocessing, model training, and prediction across all projects.

```bash
pytest                              # run all tests
pytest -q --tb=short                # compact output
pytest NLP\ Projecct\ 1.*/         # run tests for a single project
```

---

## API

```bash
uvicorn api:app --reload            # start FastAPI server
```

```
POST /predict  {"project": "resume_screening", "text": "Senior Python Developer..."}
GET  /health
GET  /projects
```

---

## Leaderboard

Compare model performance across all AutoML projects:

```bash
python leaderboard.py
```

---

## Git LFS

Large files (>50 MB) are tracked with [Git LFS](https://git-lfs.github.com/):

| File | Size |
|------|------|
| `glove.6B.100d.txt` | 331 MB |
| `database.sqlite` | 207 MB |
| `papers.csv` | 201 MB |
| `embold_train_extra.json` | 187 MB |
| `embold_train.json` | 94 MB |

Make sure Git LFS is installed before cloning:

```bash
git lfs install
git clone https://github.com/pypi-ahmad/Natural-Language-Processing-Projects.git
```

---

## License

MIT
