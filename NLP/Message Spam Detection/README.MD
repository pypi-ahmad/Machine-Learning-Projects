# Message Spam Detection

> A machine learning pipeline that cleans SMS data and classifies messages as spam or not spam using a Random Forest classifier.

## Overview

This project consists of two scripts: one for dataset preprocessing/cleaning and one for training a Random Forest model and performing real-time spam classification on user-provided text. It uses the [SMS Spam Collection Dataset from Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset), applies NLP preprocessing (tokenization, stopword removal, lemmatization), and classifies messages using TF-IDF features with a `RandomForestClassifier`.

## Features

- Dataset cleaning pipeline with punctuation/digit removal, tokenization, stopword filtering, and lemmatization
- Exports cleaned dataset to CSV for reuse
- TF-IDF vectorization with `CountVectorizer` (max 5000 features) and `TfidfTransformer`
- Random Forest classification with 100 estimators
- Interactive user input for real-time spam detection
- Full NLP preprocessing applied to user input before prediction

## Project Structure

```
Message_Spam_Detection/
├── Dataset_Cleaning.py          # Data preprocessing and cleaning script
├── Message_Spam_Detection.py    # Model training and prediction script
├── dataset.csv                  # Raw SMS Spam Collection dataset
└── Cleaned_Dataset.csv          # Preprocessed output from cleaning script
```

## Requirements

- Python 3.x
- pandas
- nltk
- scikit-learn

## Installation

```bash
cd Message_Spam_Detection
pip install pandas nltk scikit-learn
```

NLTK data downloads are handled automatically by `Dataset_Cleaning.py` (downloads `stopwords` and `wordnet`).

## Usage

**Step 1: Clean the dataset** (generates `Cleaned_Dataset.csv`):

```bash
python Dataset_Cleaning.py
```

**Step 2: Run spam detection**:

```bash
python Message_Spam_Detection.py
```

When prompted, enter a text message. The script outputs either `Not Spam` or `Spam`.

**Example:**

```
Enter text: Winner! Winner! Congrats! Call at xyz to claim your prize!
Spam

Enter text: Hello, I am James Bond
Not Spam
```

## How It Works

### Dataset Cleaning (`Dataset_Cleaning.py`)
1. Reads `dataset.csv` (Kaggle SMS Spam Collection), drops unnamed columns, renames `v1`→`label`, `v2`→`text`
2. Maps labels: `ham`→0, `spam`→1
3. Removes duplicate rows
4. Removes punctuation and digits from text
5. Tokenizes and converts to lowercase
6. Removes English stopwords using NLTK
7. Lemmatizes tokens using `WordNetLemmatizer`
8. Joins tokens back into strings and saves as `Cleaned_Dataset.csv`

### Spam Detection (`Message_Spam_Detection.py`)
1. Loads `Cleaned_Dataset.csv`
2. Applies `CountVectorizer` (max 5000 features) and `TfidfTransformer` on the text
3. Trains a `RandomForestClassifier` with 100 estimators
4. Takes user input, applies the same preprocessing pipeline
5. Transforms input using the fitted vectorizer/transformer and predicts spam/not-spam

## Configuration

- The dataset path is hardcoded as `./Message_Spam_Detection/dataset.csv` — adjust if running from a different directory
- `CountVectorizer` max features: 5000
- `RandomForestClassifier`: 100 estimators, `random_state=0`

## Limitations

- Dataset path is hardcoded with a relative path that assumes execution from the parent directory
- The model is retrained from scratch on every run — no model persistence (despite `pickle` being imported, it is unused)
- No train/test split — the model trains and predicts on the full dataset, so accuracy metrics are not evaluated
- Uses `SettingWithCopyWarning`-prone pandas patterns (chained indexing)
- `warnings.filterwarnings("ignore")` suppresses all warnings silently
- The cleaning script downloads NLTK data on every run

## License

Not specified.
