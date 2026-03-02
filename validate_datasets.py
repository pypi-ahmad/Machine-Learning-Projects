#!/usr/bin/env python3
"""
Phase 3b — Step 6: Dataset Validation Script
Checks every previously-blocked project for:
  1. Data directory exists
  2. Key files / subdirectories are present
  3. Notebook is patched (contains DATA_DIR or built-in loader)
Outputs a summary table and JSON report.
"""

from pathlib import Path
import json, re, sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ── registry: project_dir_name → { slug, key_files[], notebook, loader_type } ──
PROJECTS = {
    # ── PATCHED NOTEBOOKS (external datasets) ──
    "Machine Learning Project 11 - Creadi Card Fraud Problem Handling Imbalanced Dataset": {
        "slug": "creadi_card_fraud_problem_handling_imbalanced_dataset",
        "key_files": ["creditcard.csv"],
        "notebook": "Hanling_Imbalanced_Data.ipynb",
        "loader": "csv",
    },
    "Machine Learning Project 25 - Advanced Credit Card fraud Detection": {
        "slug": "advanced_credit_card_fraud_detection",
        "key_files": ["creditcard.csv"],
        "notebook": "Handling_Imbalanced_Data-Under_Sampling.ipynb",
        "loader": "csv",
    },
    "Machine Learning Project 24 - Dog Vs Cat Classification": {
        "slug": "dog_vs_cat_classification",
        "key_files": ["PetImages/Cat", "PetImages/Dog"],
        "notebook": "main.ipynb",
        "loader": "image_dir",
    },
    "Machine Learning Project 30 - Pneumonia Classification": {
        "slug": "pneumonia_classification",
        "key_files": ["chest_xray/train", "chest_xray/test", "chest_xray/val"],
        "notebook": "pneumonia.ipynb",
        "loader": "image_dir",
    },
    "Machine Learning Project 4 - Indian-classical-dance problem using Machine Learning": {
        "slug": "indian-classical-dance_problem_using_machine_learning",
        "key_files": ["train", "test", "train.csv", "test.csv"],
        "notebook": "main.ipynb",
        "loader": "image_dir",
    },
    "Machine Learning Project 55 - Million Songs Dataset - Recommendation Engine": {
        "slug": "million_songs_dataset_-_recommendation_engine",
        "key_files": ["triplets_file.csv", "song_data.csv"],
        "notebook": "Million Songs Data - Recommendation Engine.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects  90 - SMS Spam Detection Analysis - NLP": {
        "slug": "sms_spam_detection_analysis_-_nlp",
        "key_files": ["spam.csv"],
        "notebook": "SMS Spam Detection Analysis - NLP.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 109 - Mercari Price Suggestion Lightgbm": {
        "slug": "mercari_price_suggestion_lightgbm",
        "key_files": ["train.tsv"],
        "notebook": "Mercari Price Suggestion Lightgbm.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 124 - (Conceptual) Text Classification Keras": {
        "slug": "conceptual_text_classification_keras",
        "key_files": ["consumer_complaints.csv"],
        "notebook": "Text Classification Keras.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 125 - Text Classification keras_consumer_complaints": {
        "slug": "text_classification_keras_consumer_complaints",
        "key_files": ["consumer_complaints.csv"],
        "notebook": "Text Classification keras_consumer_complaints.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 137 - Consumer_complaints": {
        "slug": "consumer_complaints",
        "key_files": ["consumer_complaints.csv"],
        "notebook": "Consumer_complaints.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 144 - LSTM Time Series Power Consumption": {
        "slug": "lstm_time_series_power_consumption",
        "key_files": ["household_power_consumption.txt"],
        "notebook": "LSTM Time Series Power Consumption.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 149 - Twitter Sentiment Analysis - NLP": {
        "slug": "twitter_sentiment_analysis_-_nlp",
        "key_files": ["training.1600000.processed.noemoticon.csv"],
        "notebook": "Twitter Sentiment Analysis - NLP.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 117 - Recommender Systems - The Fundamentals": {
        "slug": "recommender_systems_-_the_fundamentals",
        "key_files": ["BX-Books.csv", "BX-Users.csv", "BX-Book-Ratings.csv"],
        "notebook": "Recommender Systems - The Fundamentals.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 147 - Language Translation Model using ML": {
        "slug": "language_translation_model_using_ml",
        "key_files": ["english.txt"],
        "notebook": "english-rnn-attempt.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 148 -U.S. Gasoline and Diesel Retail Prices 1995-2021": {
        "slug": "us_gasoline_and_diesel_retail_prices_1995-2021",
        "key_files": ["PET_PRI_GND_DCUS_NUS_W.csv"],
        "notebook": "gasoline-price-predictions-97-acc-0-overfitting.ipynb",
        "loader": "csv",
    },
    "Machine Learning Projects 93 - Spam Email Classification": {
        "slug": "spam_email_classification",
        "key_files": ["enron1/spam", "enron1/ham"],
        "notebook": "spam_email_classification.ipynb",
        "loader": "file_dir",
    },
    "Machine Learning Projects 94 - Traffic Sign Recognizer": {
        "slug": "traffic_sign_recognizer",
        "key_files": ["Train", "Test", "Test.csv"],
        "notebook": "Traffic-Sign-Recognition.ipynb",
        "loader": "image_dir",
    },
    "Machine Learning Project 20 - Face Mask Detection using ML": {
        "slug": "face_mask_detection_using_ml",
        "key_files": ["Face Mask Dataset/Train", "Face Mask Dataset/Test", "Face Mask Dataset/Validation"],
        "notebook": "notebookf9ab511482.ipynb",
        "loader": "image_dir",
    },
    "Machine Learning Project 21 - Face Expression Identifier": {
        "slug": "face_expression_identifier",
        "key_files": ["train/angry", "test/angry"],
        "notebook": "face-exp-resnet.ipynb",
        "loader": "image_dir",
    },
    "Machine Learning Project 29 - Plant Diseases Recognition": {
        "slug": "plant_diseases_recognition",
        "key_files": ["New Plant Diseases Dataset(Augmented)"],
        "notebook": "recognition-plant-diseases-by-leaf.ipynb",
        "loader": "image_dir",
    },
    "Machine Learning Project 42 - Traffic_Sign_Recognition": {
        "slug": "traffic_sign_recognition",
        "key_files": ["Train", "Test", "Test.csv"],
        "notebook": "rgb_classifier_recognition.ipynb",
        "loader": "image_dir",
    },
    "Machine Learning Projects 129 - Weather Data Clustering using k-Means": {
        "slug": "weather_data_clustering_using_k-means",
        "key_files": ["minute_weather.csv"],
        "notebook": "Weather Data Clustering using k-Means.ipynb",
        "loader": "csv",
    },
    # ── MANUALLY FIXED NOTEBOOKS ──
    "Machine Learning Project 7 - Advandced Hyperparameter Tunning": {
        "slug": "advandced_hyperparameter_tunning",
        "key_files": ["diabetes.csv"],
        "notebook": "Hyper_Parameter_Tunning.ipynb",
        "loader": "csv",
    },
    "Machine Learning Project 34 - Live Smile Detector": {
        "slug": None,  # uses cv2.data.haarcascades built-in
        "key_files": [],
        "notebook": "Smile.py",
        "loader": "builtin",
    },
    "Machine Learning Projects 92 - Movie Recommendation Engine": {
        "slug": "movie_recommendation_engine",
        "key_files": ["movies.csv", "ratings.csv"],
        "notebook": "Movie_Recommendation_Engine.ipynb",
        "loader": "csv",
    },
    # ── BUILT-IN LOADERS (no external data needed) ──
    "Machine Learning Project 6 - Imdb sentiment review Analysis using ML": {
        "slug": None,
        "key_files": [],
        "notebook": "Untitled.ipynb",
        "loader": "builtin",  # tfds.load('imdb_reviews')
    },
    "Machine Learning Project 12 - Hand Digit Recognition Using ML": {
        "slug": None,
        "key_files": [],
        "notebook": "Untitled.ipynb",
        "loader": "builtin",  # sklearn.datasets.load_digits()
    },
    "Machine Learning Project 31 - InDetailed Principal Component Analysis": {
        "slug": None,
        "key_files": [],
        "notebook": "Principal Component Analysis.ipynb",
        "loader": "builtin",  # sklearn.datasets.load_breast_cancer()
    },
    # ── SELF-DOWNLOADING  ──
    "Machine Learning Project 49 - Cifar 10": {
        "slug": "cifar10",
        "key_files": [],  # downloaded at runtime by download_url
        "notebook": "04_image_classification_with_CNN(Colab).ipynb",
        "loader": "self_download",  # download_url + tarfile
    },
    "Machine Learning Project 19 - Fashion Mnist Data Analysis Using ML": {
        "slug": "fashion_mnist",
        "key_files": [],  # downloaded at runtime by FashionMNIST(download=True) - dir created on first run
        "notebook": "F_Mnist_model.ipynb",
        "loader": "self_download",
    },
}


def check_project(name: str, spec: dict) -> dict:
    """Return a result dict for one project."""
    result = {
        "project": name,
        "slug": spec["slug"],
        "notebook": spec["notebook"],
        "loader": spec["loader"],
        "data_dir_exists": None,
        "missing_files": [],
        "notebook_patched": None,
        "status": "UNKNOWN",
        "notes": "",
    }

    # ── 1. Check data directory ──
    if spec["slug"] is None:
        result["data_dir_exists"] = "N/A"
    elif spec["loader"] == "self_download":
        # Data is auto-downloaded at notebook runtime; dir may not exist yet
        data_dir = DATA / spec["slug"]
        result["data_dir_exists"] = data_dir.is_dir() or "RUNTIME"
    else:
        data_dir = DATA / spec["slug"]
        result["data_dir_exists"] = data_dir.is_dir()
        if not data_dir.is_dir():
            result["status"] = "FAIL"
            result["notes"] = f"Data dir missing: data/{spec['slug']}"
            return result

        # ── 2. Check key files / subdirectories ──
        for kf in spec["key_files"]:
            p = data_dir / kf
            if not p.exists():
                result["missing_files"].append(kf)

    # ── 3. Check notebook is patched ──
    proj_dir = ROOT / name
    if not proj_dir.is_dir():
        # Try to find it
        candidates = list(ROOT.glob(name + "*"))
        if candidates:
            proj_dir = candidates[0]
        else:
            result["notebook_patched"] = "DIR_NOT_FOUND"
            result["status"] = "FAIL"
            result["notes"] = "Project directory not found"
            return result

    nb_path = proj_dir / spec["notebook"]
    if not nb_path.exists():
        result["notebook_patched"] = "FILE_NOT_FOUND"
        result["status"] = "FAIL"
        result["notes"] = f"Notebook not found: {spec['notebook']}"
        return result

    nb_text = nb_path.read_text(encoding="utf-8", errors="replace")

    # For .ipynb: only check source cells, not output cells
    if nb_path.suffix == ".ipynb":
        import json as _json
        try:
            nb_data = _json.loads(nb_text)
            source_text = ""
            for cell in nb_data.get("cells", []):
                if cell.get("cell_type") == "code":
                    source_text += "".join(cell.get("source", []))
            nb_text_for_check = source_text
        except Exception:
            nb_text_for_check = nb_text
    else:
        nb_text_for_check = nb_text

    if spec["loader"] == "builtin":
        has_hardcoded = bool(re.search(r"C:\\\\?Users\\\\?", nb_text_for_check))
        result["notebook_patched"] = not has_hardcoded
    elif spec["loader"] == "self_download":
        has_hardcoded = bool(re.search(r"C:\\\\?Users\\\\?", nb_text_for_check))
        result["notebook_patched"] = not has_hardcoded
    else:
        has_data_dir = bool(re.search(r"DATA_DIR|data_dir", nb_text_for_check))
        has_hardcoded = bool(re.search(r"C:\\\\?Users\\\\?", nb_text_for_check))
        result["notebook_patched"] = has_data_dir and not has_hardcoded

    # ── Final status ──
    if result["missing_files"]:
        result["status"] = "WARN"
        result["notes"] = f"Missing: {', '.join(result['missing_files'])}"
    elif result["notebook_patched"] is False:
        result["status"] = "WARN"
        result["notes"] = "Notebook may still have hardcoded paths"
    elif result["notebook_patched"] is True or result["notebook_patched"] == "N/A":
        result["status"] = "OK"
    else:
        result["status"] = "WARN"

    return result


def main():
    print("=" * 70)
    print("PHASE 3b — DATASET VALIDATION")
    print("=" * 70)

    results = []
    ok = warn = fail = 0

    for name, spec in PROJECTS.items():
        r = check_project(name, spec)
        results.append(r)
        tag = r["status"]
        if tag == "OK":
            ok += 1
            mark = "✓"
        elif tag == "WARN":
            warn += 1
            mark = "⚠"
        else:
            fail += 1
            mark = "✗"
        notes = f"  ({r['notes']})" if r['notes'] else ""
        print(f"  {mark} {name}{notes}")

    print()
    print(f"  OK: {ok}  |  WARN: {warn}  |  FAIL: {fail}  |  Total: {len(results)}")
    print("=" * 70)

    # ── Write JSON report ──
    report_path = ROOT / "audit_phase3" / "dataset_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Report saved → {report_path.relative_to(ROOT)}")

    return 1 if fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
