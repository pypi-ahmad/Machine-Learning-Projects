"""
PHASE 3b — NOTEBOOK PATH PATCHER
Fixes data loading paths in ALL blocked project notebooks.
Uses JSON-level manipulation of .ipynb files for precision.
"""

import json
import re
import shutil
from pathlib import Path

ROOT = Path(r"d:\Workspace\Github\Machine-Learning-Projects")
DATA_ROOT = ROOT / "data"

FIXES_APPLIED = []
FIXES_FAILED = []

def backup_and_load(nb_path):
    """Backup notebook and load JSON."""
    nb_path = Path(nb_path)
    backup = nb_path.with_suffix('.ipynb.bak')
    if not backup.exists():
        shutil.copy2(nb_path, backup)
    with open(nb_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb_path, nb_data):
    """Save modified notebook."""
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)

def replace_in_cell(cell, old_pattern, new_text, is_regex=False):
    """Replace text in a cell's source. Returns True if changed."""
    source = cell.get('source', [])
    if isinstance(source, list):
        full_text = ''.join(source)
    else:
        full_text = source
    
    if is_regex:
        new_full = re.sub(old_pattern, new_text, full_text)
    else:
        new_full = full_text.replace(old_pattern, new_text)
    
    if new_full != full_text:
        # Convert back to list of lines
        lines = new_full.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
        return True
    return False

def insert_cell_at_top(nb_data, code_text):
    """Insert a new code cell at position 1 (after first markdown if exists)."""
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in code_text.split('\n')[:-1]] + [code_text.split('\n')[-1]]
    }
    
    # Find first code cell position
    insert_pos = 0
    for i, cell in enumerate(nb_data['cells']):
        if cell['cell_type'] == 'code':
            insert_pos = i
            break
    
    nb_data['cells'].insert(insert_pos, new_cell)
    return nb_data

def prepend_to_first_code_cell(nb_data, code_text):
    """Prepend code to the first code cell."""
    for cell in nb_data['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                old = ''.join(source)
            else:
                old = source
            new = code_text + '\n' + old
            lines = new.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]
            return True
    return False


def fix_project(proj_name, nb_name, slug, replacements, data_dir_setup=None,
                insert_imports=None):
    """Fix a single project notebook."""
    proj_dir = ROOT / proj_name
    nb_path = proj_dir / nb_name
    
    if not nb_path.exists():
        FIXES_FAILED.append((proj_name, f"Notebook not found: {nb_name}"))
        return False
    
    print(f"\n  Fixing: {proj_name}")
    print(f"    Notebook: {nb_name}")
    
    nb = backup_and_load(nb_path)
    changed = False
    
    # Insert DATA_DIR setup at top if specified
    if data_dir_setup:
        # Check if already patched
        full_nb = json.dumps(nb)
        if 'DATA_DIR' not in full_nb:
            if insert_imports:
                # Prepend import + DATA_DIR to first code cell
                prepend_to_first_code_cell(nb, data_dir_setup)
                changed = True
                print(f"    + Added DATA_DIR setup")
            else:
                insert_cell_at_top(nb, data_dir_setup)
                changed = True
                print(f"    + Inserted DATA_DIR cell")
        else:
            print(f"    ~ DATA_DIR already present")
    
    # Apply replacements
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        for old, new, is_regex in replacements:
            if replace_in_cell(cell, old, new, is_regex):
                changed = True
                old_short = old[:60] if not is_regex else f"regex:{old[:50]}"
                print(f"    ✓ Replaced: {old_short}")
    
    if changed:
        save_notebook(nb_path, nb)
        FIXES_APPLIED.append((proj_name, nb_name))
        print(f"    ★ Saved")
    else:
        print(f"    (no changes needed)")
    
    return changed


# ═══════════════════════════════════════════════════════════════════════════════
# FIX DEFINITIONS — one per project
# ═══════════════════════════════════════════════════════════════════════════════

def fix_all():
    print("=" * 60)
    print("PHASE 3b — NOTEBOOK PATH PATCHER")
    print("=" * 60)
    
    # ── Project 11: Credit Card Fraud (Imbalanced) ──────────────────────
    fix_project(
        "Machine Learning Project 11 - Creadi Card Fraud Problem Handling Imbalanced Dataset",
        "Hanling_Imbalanced_Data.ipynb",
        "creadi_card_fraud_problem_handling_imbalanced_dataset",
        replacements=[
            (r"pd\.read_csv\(r?['\"]C:[^'\"]+creditcard\.csv['\"]\)",
             "pd.read_csv(DATA_DIR / 'creditcard.csv')", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'creadi_card_fraud_problem_handling_imbalanced_dataset'",
        insert_imports=True
    )
    
    # ── Project 25: Advanced Credit Card Fraud ──────────────────────────
    fix_project(
        "Machine Learning Project 25 - Advanced Credit Card fraud Detection",
        "Handling_Imbalanced_Data-Under_Sampling.ipynb",
        "advanced_credit_card_fraud_detection",
        replacements=[
            (r"pd\.read_csv\(r?['\"]C:[^'\"]+creditcard\.csv['\"]\)",
             "pd.read_csv(DATA_DIR / 'creditcard.csv')", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'advanced_credit_card_fraud_detection'",
        insert_imports=True
    )
    
    # ── Project 24: Dog vs Cat ──────────────────────────────────────────
    fix_project(
        "Machine Learning Project 24 - Dog Vs Cat Classification",
        "main.ipynb",
        "dog_vs_cat_classification",
        replacements=[
            # Replace all absolute training_set/test_set paths
            (r"r?['\"]C:\\\\?Users\\\\?[^'\"]+training_set['\"]",
             "str(DATA_DIR / 'PetImages' / 'Cat') + ',' + str(DATA_DIR / 'PetImages' / 'Dog')", True),
            (r"r?['\"]C:\\\\?Users\\\\?[^'\"]+test_set['\"]",
             "str(DATA_DIR / 'PetImages')", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'dog_vs_cat_classification'",
        insert_imports=True
    )
    
    # ── Project 30: Pneumonia Classification ────────────────────────────
    fix_project(
        "Machine Learning Project 30 - Pneumonia Classification",
        "pneumonia.ipynb",
        "pneumonia_classification",
        replacements=[
            (r"r?['\"]C:\\\\?Users\\\\?[^'\"]*chest_xray\\\\?train['\"]",
             "str(DATA_DIR / 'chest_xray' / 'train')", True),
            (r"r?['\"]C:\\\\?Users\\\\?[^'\"]*chest_xray\\\\?val['\"]",
             "str(DATA_DIR / 'chest_xray' / 'val')", True),
            (r"r?['\"]C:\\\\?Users\\\\?[^'\"]*chest_xray\\\\?test['\"]",
             "str(DATA_DIR / 'chest_xray' / 'test')", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'pneumonia_classification'",
        insert_imports=True
    )
    
    # ── Project 4: Indian Classical Dance ───────────────────────────────
    fix_project(
        "Machine Learning Project 4 - Indian-classical-dance problem using Machine Learning",
        "main.ipynb",
        "indian-classical-dance_problem_using_machine_learning",
        replacements=[
            (r"r?['\"]C:\\\\?Users\\\\?[^'\"]*train['\"]",
             "str(DATA_DIR / 'train')", True),
            (r"r?['\"]C:\\\\?Users\\\\?[^'\"]*validation['\"]",
             "str(DATA_DIR / 'train')", True),  # Use train as fallback
            (r"r?['\"]C:\\\\?Users\\\\?[^'\"]*test['\"]",
             "str(DATA_DIR / 'test')", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'indian-classical-dance_problem_using_machine_learning'",
        insert_imports=True
    )
    
    # ── Project 55: Million Songs ───────────────────────────────────────
    fix_project(
        "Machine Learning Project 55 - Million Songs Dataset - Recommendation Engine",
        "Million Songs Data - Recommendation Engine.ipynb",
        "million_songs_dataset_-_recommendation_engine",
        replacements=[
            ("pd.read_csv('triplets_file.csv')",
             "pd.read_csv(DATA_DIR / 'triplets_file.csv')", False),
            ("pd.read_csv('song_data.csv')",
             "pd.read_csv(DATA_DIR / 'song_data.csv')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'million_songs_dataset_-_recommendation_engine'",
        insert_imports=True
    )
    
    # ── Project 90: SMS Spam Detection ──────────────────────────────────
    fix_project(
        "Machine Learning Projects  90 - SMS Spam Detection Analysis - NLP",
        "SMS Spam Detection Analysis - NLP.ipynb",
        "sms_spam_detection_analysis_-_nlp",
        replacements=[
            ("pd.read_csv('spam.csv')",
             "pd.read_csv(DATA_DIR / 'spam.csv')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'sms_spam_detection_analysis_-_nlp'",
        insert_imports=True
    )
    
    # ── Project 109: Mercari Price ──────────────────────────────────────
    fix_project(
        "Machine Learning Projects 109 - Mercari Price Suggestion Lightgbm",
        "Mercari Price Suggestion Lightgbm.ipynb",
        "mercari_price_suggestion_lightgbm",
        replacements=[
            ("pd.read_csv('train.tsv', sep='\\t')",
             "pd.read_csv(DATA_DIR / 'train.tsv', sep='\\t')", False),
            (r"pd\.read_csv\(['\"]train\.tsv['\"],\s*sep=['\"]\\t['\"]\)",
             "pd.read_csv(DATA_DIR / 'train.tsv', sep='\\t')", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'mercari_price_suggestion_lightgbm'",
        insert_imports=True
    )
    
    # ── Project 124: Text Classification Keras ──────────────────────────
    fix_project(
        "Machine Learning Projects 124 - (Conceptual) Text Classification Keras",
        "Text Classification Keras.ipynb",
        "conceptual_text_classification_keras",
        replacements=[
            (r"pd\.read_csv\(['\"]Consumer_Complaints\.csv['\"]",
             "pd.read_csv(DATA_DIR / 'consumer_complaints.csv'", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'conceptual_text_classification_keras'",
        insert_imports=True
    )
    
    # ── Project 125: Text Classification keras_consumer ─────────────────
    fix_project(
        "Machine Learning Projects 125 - Text Classification keras_consumer_complaints",
        "Text Classification keras_consumer_complaints.ipynb",
        "text_classification_keras_consumer_complaints",
        replacements=[
            (r"pd\.read_csv\(['\"]Consumer_Complaints\.csv['\"]",
             "pd.read_csv(DATA_DIR / 'consumer_complaints.csv'", True),
            ("gzip.open('glove.6B.50d.txt.gz')",
             "gzip.open(DATA_DIR / 'glove.6B.50d.txt.gz')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'text_classification_keras_consumer_complaints'",
        insert_imports=True
    )
    
    # ── Project 137: Consumer Complaints ────────────────────────────────
    fix_project(
        "Machine Learning Projects 137 - Consumer_complaints",
        "Consumer_complaints.ipynb",
        "consumer_complaints",
        replacements=[
            (r"pd\.read_csv\(['\"]Consumer_Complaints\.csv['\"]\)",
             "pd.read_csv(DATA_DIR / 'consumer_complaints.csv')", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'consumer_complaints'",
        insert_imports=True
    )
    
    # ── Project 144: LSTM Power Consumption ─────────────────────────────
    fix_project(
        "Machine Learning Projects 144 - LSTM Time Series Power Consumption",
        "LSTM Time Series Power Consumption.ipynb",
        "lstm_time_series_power_consumption",
        replacements=[
            (r"pd\.read_csv\(['\"]household_power_consumption\.txt['\"]",
             "pd.read_csv(DATA_DIR / 'household_power_consumption.txt'", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'lstm_time_series_power_consumption'",
        insert_imports=True
    )
    
    # ── Project 149: Twitter Sentiment ──────────────────────────────────
    fix_project(
        "Machine Learning Projects 149 - Twitter Sentiment Analysis - NLP",
        "Twitter Sentiment Analysis - NLP.ipynb",
        "twitter_sentiment_analysis_-_nlp",
        replacements=[
            (r"pd\.read_csv\(['\"]Twitter Sentiments?\.csv['\"]\)",
             "pd.read_csv(DATA_DIR / 'training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None, names=['target','ids','date','flag','user','text'])", True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'twitter_sentiment_analysis_-_nlp'",
        insert_imports=True
    )
    
    # ── Project 117: Recommender Systems ────────────────────────────────
    fix_project(
        "Machine Learning Projects 117 - Recommender Systems - The Fundamentals",
        "Recommender Systems - The Fundamentals.ipynb",
        "recommender_systems_-_the_fundamentals",
        replacements=[
            ("'BX-Books.csv'", "str(DATA_DIR / 'BX-Books.csv')", False),
            ("'BX-Users.csv'", "str(DATA_DIR / 'BX-Users.csv')", False),
            ("'BX-Book-Ratings.csv'", "str(DATA_DIR / 'BX-Book-Ratings.csv')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'recommender_systems_-_the_fundamentals'",
        insert_imports=True
    )
    
    # ── Project 147: Language Translation ───────────────────────────────
    fix_project(
        "Machine Learning Projects 147 - Language Translation Model using ML",
        "english-rnn-attempt.ipynb",
        "language_translation_model_using_ml",
        replacements=[
            ("'../input/stop-words-in-28-languages/english.txt'",
             "str(DATA_DIR / 'english.txt')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'language_translation_model_using_ml'",
        insert_imports=True
    )
    
    # ── Project 148: US Gasoline Prices ─────────────────────────────────
    fix_project(
        "Machine Learning Projects 148 -U.S. Gasoline and Diesel Retail Prices 1995-2021",
        "gasoline-price-predictions-97-acc-0-overfitting.ipynb",
        "us_gasoline_and_diesel_retail_prices_1995-2021",
        replacements=[
            (r'"/kaggle/input/[^"]+/PET_PRI_GND_DCUS_NUS_W\.csv"',
             'str(DATA_DIR / "PET_PRI_GND_DCUS_NUS_W.csv")', True),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'us_gasoline_and_diesel_retail_prices_1995-2021'",
        insert_imports=True
    )
    
    # ── Project 93: Spam Email Classification ───────────────────────────
    fix_project(
        "Machine Learning Projects 93 - Spam Email Classification",
        "spam_email_classification.ipynb",
        "spam_email_classification",
        replacements=[
            ("'./data/enron1/spam'", "str(DATA_DIR / 'enron1' / 'spam')", False),
            ("'./data/enron1/ham'", "str(DATA_DIR / 'enron1' / 'ham')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'spam_email_classification'",
        insert_imports=True
    )
    
    # ── Project 94: Traffic Sign Recognizer ─────────────────────────────
    fix_project(
        "Machine Learning Projects 94 - Traffic Sign Recognizer",
        "Traffic-Sign-Recognition.ipynb",
        "traffic_sign_recognizer",
        replacements=[
            ("'data/traffic_sign_dataset/Final_Training/Images/'",
             "str(DATA_DIR / 'Train') + '/'", False),
            ("'data/traffic_sign_dataset/Final_Test/Images/'",
             "str(DATA_DIR / 'Test') + '/'", False),
            ("'data/traffic_sign_dataset/GT-final_test.csv'",
             "str(DATA_DIR / 'Test.csv')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'traffic_sign_recognizer'",
        insert_imports=True
    )
    
    # ── Project 20: Face Mask Detection ─────────────────────────────────
    fix_project(
        "Machine Learning Project 20 - Face Mask Detection using ML",
        "notebookf9ab511482.ipynb",
        "face_mask_detection_using_ml",
        replacements=[
            (r"'../input/face-mask-12k-images-dataset/Face Mask Dataset/Train'",
             "str(DATA_DIR / 'Face Mask Dataset' / 'Train')", False),
            (r"'../input/face-mask-12k-images-dataset/Face Mask Dataset/Test'",
             "str(DATA_DIR / 'Face Mask Dataset' / 'Test')", False),
            (r"'../input/face-mask-12k-images-dataset/Face Mask Dataset/Validation'",
             "str(DATA_DIR / 'Face Mask Dataset' / 'Validation')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'face_mask_detection_using_ml'",
        insert_imports=True
    )
    
    # ── Project 21: Face Expression ─────────────────────────────────────
    fix_project(
        "Machine Learning Project 21 - Face Expression Identifier",
        "face-exp-resnet.ipynb",
        "face_expression_identifier",
        replacements=[
            (r"'../input/facial-expression-recog-image-ver-of-fercdataset/Dataset'",
             "str(DATA_DIR)", False),
            (r"data_dir\+'/train'", "str(DATA_DIR / 'train')", False),
            (r"data_dir\+'/test'", "str(DATA_DIR / 'test')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'face_expression_identifier'",
        insert_imports=True
    )
    
    # ── Project 29: Plant Diseases ──────────────────────────────────────
    fix_project(
        "Machine Learning Project 29 - Plant Diseases Recognition",
        "recognition-plant-diseases-by-leaf.ipynb",
        "plant_diseases_recognition",
        replacements=[
            (r"'../input/new-plant-diseases-dataset/New Plant Diseases Dataset\(Augmented\)/New Plant Diseases Dataset\(Augmented\)/train'",
             "str(DATA_DIR / 'New Plant Diseases Dataset(Augmented)' / 'New Plant Diseases Dataset(Augmented)' / 'train')", True),
            (r"'../input/new-plant-diseases-dataset/New Plant Diseases Dataset\(Augmented\)/New Plant Diseases Dataset\(Augmented\)/valid'",
             "str(DATA_DIR / 'New Plant Diseases Dataset(Augmented)' / 'New Plant Diseases Dataset(Augmented)' / 'valid')", True),
            (r"'../input/new-plant-diseases-dataset/test'",
             "str(DATA_DIR / 'test')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'plant_diseases_recognition'",
        insert_imports=True
    )
    
    # ── Project 42: Traffic Sign Recognition ────────────────────────────
    fix_project(
        "Machine Learning Project 42 - Traffic_Sign_Recognition",
        "rgb_classifier_recognition.ipynb",
        "traffic_sign_recognition",
        replacements=[
            # The notebook uses os.path.join(cwd, 'Train', str(i)) pattern
            # We need to add DATA_DIR and fix the path references
        ],
        data_dir_setup="from pathlib import Path\nimport os\nDATA_DIR = Path.cwd().parent / 'data' / 'traffic_sign_recognition'\nos.chdir(str(DATA_DIR))",
        insert_imports=True
    )
    
    # ── Project 129: Weather Clustering ─────────────────────────────────
    fix_project(
        "Machine Learning Projects 129 - Weather Data Clustering using k-Means",
        "Weather Data Clustering using k-Means.ipynb",
        "weather_data_clustering_using_k-means",
        replacements=[
            ("pd.read_csv('./weather/minute_weather.csv')",
             "pd.read_csv(DATA_DIR / 'minute_weather.csv')", False),
        ],
        data_dir_setup="from pathlib import Path\nDATA_DIR = Path.cwd().parent / 'data' / 'weather_data_clustering_using_k-means'",
        insert_imports=True
    )
    
    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"NOTEBOOK PATCHER COMPLETE")
    print(f"{'=' * 60}")
    print(f"  FIXED:  {len(FIXES_APPLIED)}")
    print(f"  FAILED: {len(FIXES_FAILED)}")
    
    if FIXES_FAILED:
        print("\n  Failed projects:")
        for name, reason in FIXES_FAILED:
            print(f"    - {name}: {reason}")
    
    return FIXES_APPLIED, FIXES_FAILED


if __name__ == '__main__':
    fix_all()
