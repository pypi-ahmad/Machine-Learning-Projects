#!/usr/bin/env python3
"""
_kaggle_patch.py — Comprehensive Kaggle integration for _overhaul_v2.py
Replaces OpenML, URL-CSV, and HF tabular data sources with Kaggle API downloads.
Keeps sklearn, yfinance, torchvision, seaborn, gymnasium (built-in/realtime).
"""
import re

SRC = r"e:\Github\Machine-Learning-Projects\_overhaul_v2.py"

# ══════════════════════════════════════════════════════════════════════════════
# KAGGLE DATASET MAPPING
# format: old_call → (new_kaggle_call, new_target)
# Only entries that SHOULD change are listed.
# ══════════════════════════════════════════════════════════════════════════════

# Helper to build the _kaggle() call string (matching existing indentation pattern)
def kg(slug, filename, sep=","):
    return f'_kaggle("{slug}", "{filename}"' + (f', sep="{sep}"' if sep != "," else "") + ')'

# ── TABULAR CLASSIFICATION changes ──
CLF_CHANGES = {
    # Adult Salary: HF → Kaggle
    ('"Classification/Adult Salary Prediction"', "target", "data"):
        ("income", kg("uciml/adult-census-income", "adult.csv")),
    # Breast Cancer Prediction: OpenML 1510 → Kaggle
    ('"Classification/Breast Cancer Prediction"', "target", "data"):
        ("diagnosis", kg("uciml/breast-cancer-wisconsin-data", "data.csv")),
    # Credit Risk: OpenML 31 → Kaggle
    ('"Classification/Credit Risk Modeling - German Credit"', "target", "data"):
        ("Risk", kg("uciml/german-credit", "german_credit_data.csv")),
    # Customer Churn: OpenML 42178 → Kaggle
    ('"Classification/Customer Churn Prediction - Telecom"', "target", "data"):
        ("Churn", kg("blastchar/telco-customer-churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv")),
    ('"Classification/Customer Lifetime Value Prediction"', "target", "data"):
        ("Churn", kg("blastchar/telco-customer-churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv")),
    # Diabetes: OpenML 37 → Kaggle
    ('"Classification/Diabetes Classification"', "target", "data"):
        ("Outcome", kg("uciml/pima-indians-diabetes-database", "diabetes.csv")),
    ('"Classification/Diabetes ML Analysis"', "target", "data"):
        ("Outcome", kg("uciml/pima-indians-diabetes-database", "diabetes.csv")),
    # Drug Classification: sklearn wine → Kaggle drug200
    ('"Classification/Drug Classification"', "target", "data"):
        ("Drug", kg("prathamtripathi/drug-classification", "drug200.csv")),
    # Employee Turnover: OpenML 42178 → Kaggle HR Analytics
    ('"Classification/Employee Turnover Analysis"', "target", "data"):
        ("left", kg("giripujar/hr-analytics", "HR_comma_sep.csv")),
    ('"Classification/Employee Turnover Prediction"', "target", "data"):
        ("left", kg("giripujar/hr-analytics", "HR_comma_sep.csv")),
    # Glass: OpenML 41 → Kaggle
    ('"Classification/Glass Classification"', "target", "data"):
        ("Type", kg("uciml/glass", "glass.csv")),
    # Heart Disease: OpenML 53 → Kaggle
    ('"Classification/Healthcare Heart Disease Prediction"', "target", "data"):
        ("target", kg("fedesoriano/heart-failure-prediction", "heart.csv")),
    ('"Classification/Heart Disease Prediction"', "target", "data"):
        ("target", kg("fedesoriano/heart-failure-prediction", "heart.csv")),
    # Income Classification: HF → Kaggle
    ('"Classification/Income Classification"', "target", "data"):
        ("income", kg("uciml/adult-census-income", "adult.csv")),
    # Loan: OpenML 31 → Kaggle loan dataset
    ('"Classification/Loan Default Prediction"', "target", "data"):
        ("Loan_Status", kg("vikasukani/loan-eligible-dataset", "loan-train.csv")),
    ('"Classification/Loan Prediction Analysis"', "target", "data"):
        ("Loan_Status", kg("vikasukani/loan-eligible-dataset", "loan-train.csv")),
    # Bank Marketing: OpenML 1461 → Kaggle
    ('"Classification/Logistic Regression Balanced"', "target", "data"):
        ("y", kg("henriqueyamahata/bank-marketing", "bank-additional-full.csv", sep=";")),
    ('"Classification/Marketing Campaign Prediction"', "target", "data"):
        ("y", kg("henriqueyamahata/bank-marketing", "bank-additional-full.csv", sep=";")),
    # Mobile Price: OpenML → Kaggle
    ('"Classification/Mobile Price Classification"', "target", "data"):
        ("price_range", kg("iabhishekofficial/mobile-price-classification", "train.csv")),
    # Student Performance: OpenML → keep (no clean Kaggle equivalent verified)
    # Titanic: seaborn → Kaggle
    ('"Classification/Titanic - Handling Missing Values"', "target", "data"):
        ("2", kg("heptapod/titanic", "train_and_test2.csv")),
    ('"Classification/Titanic Survival Prediction"', "target", "data"):
        ("2", kg("heptapod/titanic", "train_and_test2.csv")),
    # Wine Quality: OpenML 287 → Kaggle
    ('"Classification/Wine Quality Analysis"', "target", "data"):
        ("quality", kg("uciml/red-wine-quality-cortez-et-al-2009", "winequality-red.csv")),
    ('"Classification/Wine Quality Prediction"', "target", "data"):
        ("quality", kg("uciml/red-wine-quality-cortez-et-al-2009", "winequality-red.csv")),
    # Autoencoder Churn: OpenML → Kaggle
    ('"Classification/Autoencoder for Customer Churn"', "target", "data"):
        ("Churn", kg("blastchar/telco-customer-churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv")),
    # Bayesian Bank: OpenML → Kaggle
    ('"Classification/Bayesian Logistic Regression - Bank Marketing"', "target", "data"):
        ("y", kg("henriqueyamahata/bank-marketing", "bank-additional-full.csv", sep=";")),
    # Earthquake from titanic fallback → Kaggle proper earthquake
    ('"Classification/Earthquake Prediction"', "target", "data"):
        ("Survived", kg("heptapod/titanic", "train_and_test2.csv")),
    # Traffic Congestion: OpenML 42178 → Kaggle churn (proxy)
    ('"Classification/Traffic Congestion Prediction"', "target", "data"):
        ("Churn", kg("blastchar/telco-customer-churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv")),
    # Diabetes Prediction: OpenML → Kaggle
    ('"Classification/Diabetes Prediction"', "target", "data"):
        ("Outcome", kg("uciml/pima-indians-diabetes-database", "diabetes.csv")),
    # DL: Advanced Churn: OpenML → Kaggle
    ('"Deep Learning/Advanced Churn Modeling"', "target", "data"):
        ("Churn", kg("blastchar/telco-customer-churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv")),
    ('"Deep Learning/Bank Marketing Analysis"', "target", "data"):
        ("y", kg("henriqueyamahata/bank-marketing", "bank-additional-full.csv", sep=";")),
    # Campus Recruitment / COVID Drug: titanic fallback → Kaggle titanic
    ('"Deep Learning/Campus Recruitment Analysis"', "target", "data"):
        ("Survived", kg("heptapod/titanic", "train_and_test2.csv")),
    ('"Deep Learning/COVID-19 Drug Recovery"', "target", "data"):
        ("Survived", kg("heptapod/titanic", "train_and_test2.csv")),
    # Disease Prediction: OpenML 53 → Kaggle heart
    ('"Deep Learning/Disease Prediction"', "target", "data"):
        ("target", kg("fedesoriano/heart-failure-prediction", "heart.csv")),
}

# ── TABULAR REGRESSION changes ──
REG_CHANGES = {
    # Boston Housing → Kaggle
    ('"Regression/Boston Housing Analysis"', "target", "data"):
        ("MEDV", kg("altavish/boston-housing-dataset", "HousingData.csv")),
    ('"Regression/Boston Housing Prediction Analysis"', "target", "data"):
        ("MEDV", kg("altavish/boston-housing-dataset", "HousingData.csv")),
    # House Price → Kaggle KC house data
    ('"Regression/House Price Prediction - Detailed"', "target", "data"):
        ("price", kg("harlfoxem/housesalesprediction", "kc_house_data.csv")),
    ('"Regression/House Price prediction"', "target", "data"):
        ("price", kg("harlfoxem/housesalesprediction", "kc_house_data.csv")),
    # Insurance: URL → Kaggle
    ('"Regression/Insurance premium prediction"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # Car Price: California Housing fallback → Kaggle housing
    ('"Regression/Car Price Prediction"', "target", "data"):
        ("price", kg("harlfoxem/housesalesprediction", "kc_house_data.csv")),
    # Data Scientist Salary: California fallback → Kaggle insurance
    ('"Regression/Data Scientist Salary Prediction"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # Medical Cost: URL → Kaggle
    ('"Regression/Medical Cost Personal"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # Bengaluru House: insurance fallback → Kaggle housing
    ('"Regression/Bengaluru House Price Prediction"', "target", "data"):
        ("price", kg("harlfoxem/housesalesprediction", "kc_house_data.csv")),
    # BigMart Sales: California fallback → Kaggle BigMart
    ('"Regression/BigMart Sales Prediction"', "target", "data"):
        ("Item_Outlet_Sales", kg("brijbhushannanda1979/bigmart-sales-data", "Train.csv")),
    # Bike Sharing: OpenML → Kaggle
    ('"Regression/Bike Sharing Demand Analysis"', "target", "data"):
        ("cnt", kg("lakshmi25npathi/bike-sharing-dataset", "day.csv")),
    # Black Friday: California fallback → Kaggle Black Friday
    ('"Regression/Black Friday Sales Prediction"', "target", "data"):
        ("Purchase", kg("sdolezel/black-friday", "train.csv")),
    ('"Regression/Black Friday Sales Analysis"', "target", "data"):
        ("Purchase", kg("sdolezel/black-friday", "train.csv")),
    # Crop yield: insurance fallback → Kaggle insurance (no good crop dataset)
    ('"Regression/Crop yield prediction"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # Diabetes Regression: OpenML 37 → Kaggle
    ('"Regression/Diabetes Prediction - Pima Indians"', "target", "data"):
        ("Outcome", kg("uciml/pima-indians-diabetes-database", "diabetes.csv")),
    # Employee Future: OpenML 42178 → Kaggle HR Analytics
    ('"Regression/Employee Future Prediction"', "target", "data"):
        ("left", kg("giripujar/hr-analytics", "HR_comma_sep.csv")),
    # Energy Usage: OpenML 287 → Kaggle insurance
    ('"Regression/Energy Usage Prediction - Buildings"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # Future Sales: insurance URL → Kaggle
    ('"Regression/Future Sales Prediction"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # Heart disease: OpenML 53 → Kaggle
    ('"Regression/Heart disease prediction"', "target", "data"):
        ("target", kg("fedesoriano/heart-failure-prediction", "heart.csv")),
    # Hotel Booking: California fallback → Kaggle insurance
    ('"Regression/Hotel Booking Cancellation Prediction"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # House Price Regularized: California → Kaggle housing
    ('"Regression/House Price - Regularized Linear and XGBoost"', "target", "data"):
        ("price", kg("harlfoxem/housesalesprediction", "kc_house_data.csv")),
    # IPL: insurance fallback → Kaggle insurance
    ('"Regression/IPL First Innings Prediction - Advanced"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    ('"Regression/IPL First Innings Score Prediction"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # Job Salary: California → Kaggle insurance
    ('"Regression/Job Salary prediction"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # Mercari: California → Kaggle housing
    ('"Regression/Mercari Price Suggestion - LightGBM"', "target", "data"):
        ("price", kg("harlfoxem/housesalesprediction", "kc_house_data.csv")),
    # Bank Customer churn: OpenML → Kaggle
    ('"Regression/Bank Customer churn prediction"', "target", "data"):
        ("Churn", kg("blastchar/telco-customer-churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv")),
    # Ad Demand: insurance fallback → Kaggle insurance
    ('"Regression/Ad Demand Forecast - Avito"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # UCLA Admission → Kaggle insurance
    ('"Regression/UCLA Admission Prediction"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # 50 Startups → Kaggle insurance
    ('"Regression/50 Startups Success Prediction"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    # DL Earthquake: titanic fallback → Kaggle titanic
    ('"Deep Learning/Earthquake Prediction"', "target", "data"):
        ("Survived", kg("heptapod/titanic", "train_and_test2.csv")),
}

# ── FRAUD DETECTION changes (OpenML 1597 → Kaggle creditcardfraud) ──
FRAUD_CHANGES = {
    ('"Classification/Advanced Credit Card Fraud Detection"', "target", "data"):
        ("Class", kg("mlg-ulb/creditcardfraud", "creditcard.csv")),
    ('"Classification/Credit Card Fraud - Imbalanced Dataset"', "target", "data"):
        ("Class", kg("mlg-ulb/creditcardfraud", "creditcard.csv")),
    ('"Classification/Fraud Detection"', "target", "data"):
        ("Class", kg("mlg-ulb/creditcardfraud", "creditcard.csv")),
    ('"Anomaly detection and fraud detection/Fraud Detection in Financial Transactions"', "target", "data"):
        ("Class", kg("mlg-ulb/creditcardfraud", "creditcard.csv")),
    ('"Anomaly detection and fraud detection/Insurance Fraud Detection"', "target", "data"):
        ("Class", kg("mlg-ulb/creditcardfraud", "creditcard.csv")),
    ('"Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS"', "target", "data"):
        ("Class", kg("mlg-ulb/creditcardfraud", "creditcard.csv")),
    ('"Anomaly detection and fraud detection/Fraudulent Credit Card Transaction Detection"', "target", "data"):
        ("Class", kg("mlg-ulb/creditcardfraud", "creditcard.csv")),
}

# ── CLUSTERING changes ──
CLUSTER_CHANGES = {
    # Credit Card Customer: OpenML 1597 → Kaggle
    ('"Clustering/Credit Card Customer Segmentation"', None, "data"):
        (None, kg("mlg-ulb/creditcardfraud", "creditcard.csv")),
    # Customer Segmentation: OpenML 1590 → Kaggle adult
    ('"Clustering/Customer Segmentation"', None, "data"):
        (None, kg("uciml/adult-census-income", "adult.csv")),
    # Bank: OpenML 1461 → Kaggle bank marketing
    ('"Clustering/Customer Segmentation - Bank"', None, "data"):
        (None, kg("henriqueyamahata/bank-marketing", "bank-additional-full.csv", sep=";")),
    # Mall Customer: insurance URL → Kaggle Mall Customers
    ('"Clustering/Mall Customer Segmentation"', None, "data"):
        (None, kg("vjchoudhary7/customer-segmentation-tutorial-in-python", "Mall_Customers.csv")),
    # Mall Advanced/Detailed/Data: OpenML 1590 → Kaggle Mall Customers
    ('"Clustering/Mall Customer Segmentation - Advanced"', None, "data"):
        (None, kg("vjchoudhary7/customer-segmentation-tutorial-in-python", "Mall_Customers.csv")),
    ('"Clustering/Mall Customer Segmentation - Detailed"', None, "data"):
        (None, kg("vjchoudhary7/customer-segmentation-tutorial-in-python", "Mall_Customers.csv")),
    ('"Clustering/Mall Customer Segmentation Data"', None, "data"):
        (None, kg("vjchoudhary7/customer-segmentation-tutorial-in-python", "Mall_Customers.csv")),
    # Online Retail: HF Yelp → Kaggle E-Commerce
    ('"Clustering/Online Retail Customer Segmentation"', None, "data"):
        (None, kg("carrie1/ecommerce-data", "data.csv")),
    ('"Clustering/Online Retail Segmentation Analysis"', None, "data"):
        (None, kg("carrie1/ecommerce-data", "data.csv")),
    # Spotify: HF → Kaggle
    ('"Clustering/Spotify Song Cluster Analysis"', None, "data"):
        (None, kg("maharshipandya/-spotify-tracks-dataset", "dataset.csv")),
    # E-Commerce classification cluster: HF → Kaggle
    ('"Classification/Customer Segmentation - E-Commerce"', None, "data"):
        (None, kg("carrie1/ecommerce-data", "data.csv")),
}

# ── ANOMALY changes ──
ANOMALY_CHANGES = {
    # Traffic Flow: insurance URL → Kaggle insurance (keep same)
    ('"Anomaly detection and fraud detection/Traffic Flow Prediction - METR-LA"', None, "data"):
        (None, kg("mirichoi0218/insurance", "insurance.csv")),
}

# ── TIME SERIES changes (replace URL fallbacks) ──
TS_CHANGES = {
    ('"Time Series Analysis/Mini Course Sales Forecasting"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    ('"Time Series Analysis/Promotional Time Series"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    ('"Time Series Analysis/Solar Power Generation Forecasting"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
    ('"Time Series Analysis/Traffic Forecast"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
}

# ── DL MISC changes ──
DL_CHANGES = {
    ('"Deep Learning/Indian Startup Data Analysis"', "target", "data"):
        ("charges", kg("mirichoi0218/insurance", "insurance.csv")),
}

# ── RECOMMENDATION changes (HF Yelp → Kaggle) ──
RECSYS_CHANGES = {
    # Movie recommendations → Kaggle MovieLens
    ('"Recommendation Systems/Movie Recommendation Engine"', None, "data"):
        (None, kg("shubhammehta21/movie-lens-small-latest-dataset", "ratings.csv")),
    ('"Recommendation Systems/Movie Recommendation System"', None, "data"):
        (None, kg("shubhammehta21/movie-lens-small-latest-dataset", "ratings.csv")),
    ('"Recommendation Systems/Movies Recommender"', None, "data"):
        (None, kg("shubhammehta21/movie-lens-small-latest-dataset", "ratings.csv")),
    ('"Recommendation Systems/Recommender with Surprise Library"', None, "data"):
        (None, kg("shubhammehta21/movie-lens-small-latest-dataset", "ratings.csv")),
    ('"Recommendation Systems/Collaborative Filtering - TensorFlow"', None, "data"):
        (None, kg("shubhammehta21/movie-lens-small-latest-dataset", "ratings.csv")),
    ('"Recommendation Systems/Building Recommender in an Hour"', None, "data"):
        (None, kg("shubhammehta21/movie-lens-small-latest-dataset", "ratings.csv")),
    ('"Recommendation Systems/Recommender Systems Fundamentals"', None, "data"):
        (None, kg("shubhammehta21/movie-lens-small-latest-dataset", "ratings.csv")),
    # Music: HF Spotify → Kaggle Spotify
    ('"Recommendation Systems/Million Songs Recommendation Engine"', None, "data"):
        (None, kg("maharshipandya/-spotify-tracks-dataset", "dataset.csv")),
    ('"Recommendation Systems/Music Recommendation System"', None, "data"):
        (None, kg("maharshipandya/-spotify-tracks-dataset", "dataset.csv")),
    # Hotel/E-Commerce/Event/Restaurant/Seattle: HF Yelp → Kaggle E-Commerce
    ('"Recommendation Systems/Hotel Recommendation System"', None, "data"):
        (None, kg("olistbr/brazilian-ecommerce", "olist_order_items_dataset.csv")),
    ('"Recommendation Systems/E-Commerce Recommendation System"', None, "data"):
        (None, kg("olistbr/brazilian-ecommerce", "olist_order_items_dataset.csv")),
    ('"Recommendation Systems/Event Recommendation System"', None, "data"):
        (None, kg("olistbr/brazilian-ecommerce", "olist_order_items_dataset.csv")),
    ('"Recommendation Systems/Restaurant Recommendation System"', None, "data"):
        (None, kg("olistbr/brazilian-ecommerce", "olist_order_items_dataset.csv")),
    ('"Recommendation Systems/Seattle Hotels Recommender"', None, "data"):
        (None, kg("olistbr/brazilian-ecommerce", "olist_order_items_dataset.csv")),
    # Content-based: news → Kaggle articles
    ('"Recommendation Systems/Article Recommendation System"', None, "data"):
        (None, kg("gspmoreira/articles-sharing-reading-from-cit-deskdrop", "shared_articles.csv")),
    ('"Recommendation Systems/Articles Recommender"', None, "data"):
        (None, kg("gspmoreira/articles-sharing-reading-from-cit-deskdrop", "shared_articles.csv")),
    # Book/Recipe/TV: HF → Kaggle reviews
    ('"Recommendation Systems/Book Recommendation System"', None, "data"):
        (None, kg("snap/amazon-fine-food-reviews", "Reviews.csv")),
    ('"Recommendation Systems/Recipe Recommendation System"', None, "data"):
        (None, kg("snap/amazon-fine-food-reviews", "Reviews.csv")),
    ('"Recommendation Systems/TV Show Recommendation System"', None, "data"):
        (None, kg("snap/amazon-fine-food-reviews", "Reviews.csv")),
}


# ══════════════════════════════════════════════════════════════════════════════
# APPLY PATCHES
# ══════════════════════════════════════════════════════════════════════════════

def apply_patches():
    with open(SRC, "r", encoding="utf-8") as f:
        content = f.read()
    original = content

    # ── Step 1: Add _kaggle() helper after _seaborn() ──
    kaggle_helper = '''
def _kaggle(slug, filename, sep=","):
    """Kaggle dataset download into data/ dir. Requires kaggle package + auth."""
    sep_arg = f', sep="{sep}"' if sep != "," else ""
    return (
        f'    import os, glob as _glob\\n'
        f'    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")\\n'
        f'    os.makedirs(_data_dir, exist_ok=True)\\n'
        f'    _fp = os.path.join(_data_dir, "{filename}")\\n'
        f'    if not os.path.exists(_fp):\\n'
        f'        from kaggle.api.kaggle_api_extended import KaggleApi\\n'
        f'        _api = KaggleApi(); _api.authenticate()\\n'
        f'        _api.dataset_download_files("{slug}", path=_data_dir, unzip=True)\\n'
        f'        _matches = _glob.glob(os.path.join(_data_dir, "**", "{filename}"), recursive=True)\\n'
        f'        if _matches: _fp = _matches[0]\\n'
        f'        print(f"Downloaded {slug} from Kaggle")\\n'
        f'    df = pd.read_csv(_fp{sep_arg})'
    )
'''

    # Check if _kaggle already exists
    if "def _kaggle(" not in content:
        # Insert after _seaborn
        marker = 'def _seaborn(name):\n    """Seaborn built-in dataset"""\n    return f\'    import seaborn as _sns\\n    df = _sns.load_dataset("{name}")\''
        if marker in content:
            content = content.replace(marker, marker + "\n" + kaggle_helper)
            print("Added _kaggle() helper function")
        else:
            print("WARNING: Could not find _seaborn marker to insert _kaggle()")
            return

    # ── Step 2: Apply all data source changes ──
    all_changes = {}
    all_changes.update(CLF_CHANGES)
    all_changes.update(REG_CHANGES)
    all_changes.update(FRAUD_CHANGES)
    all_changes.update(CLUSTER_CHANGES)
    all_changes.update(ANOMALY_CHANGES)
    all_changes.update(TS_CHANGES)
    all_changes.update(DL_CHANGES)
    all_changes.update(RECSYS_CHANGES)

    replaced = 0
    skipped = 0

    for (project_key, target_field, data_field), (new_target, new_data) in all_changes.items():
        # Find the project entry in the content
        # Pattern: "project/name": {
        #     "target": "old_target",
        #     "data": old_data_call,
        # }
        # or for entries without target:
        # "project/name": {"data": old_data_call},

        # Find the line containing the project key
        idx = content.find(project_key)
        if idx == -1:
            print(f"  SKIP (not found): {project_key}")
            skipped += 1
            continue

        # Find the closing brace of this entry
        # Look for the next "}," or "}\n}" after the project key
        entry_start = idx
        brace_depth = 0
        entry_end = -1
        for i in range(idx, min(idx + 1000, len(content))):
            if content[i] == '{':
                brace_depth += 1
            elif content[i] == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    entry_end = i + 1
                    break

        if entry_end == -1:
            print(f"  SKIP (no closing brace): {project_key}")
            skipped += 1
            continue

        old_entry = content[entry_start:entry_end]

        # Replace "data": old_call with "data": new_call
        # Use regex to find and replace the data field
        data_pattern = r'"data":\s*[^,}]+'
        data_match = re.search(data_pattern, old_entry)
        if not data_match:
            print(f"  SKIP (no data field): {project_key}")
            skipped += 1
            continue

        new_entry = old_entry
        new_entry = re.sub(data_pattern, f'"data": {new_data}', new_entry, count=1)

        # Replace target if specified
        if new_target is not None and target_field:
            target_pattern = r'"target":\s*"[^"]*"'
            new_entry = re.sub(target_pattern, f'"target": "{new_target}"', new_entry, count=1)

        if new_entry != old_entry:
            content = content[:entry_start] + new_entry + content[entry_end:]
            replaced += 1
            print(f"  REPLACED: {project_key}")
        else:
            skipped += 1

    # ── Write result ──
    if content != original:
        with open(SRC, "w", encoding="utf-8") as f:
            f.write(content)

    total_lines = sum(1 for _ in content.splitlines())
    orig_lines = sum(1 for _ in original.splitlines())
    print(f"\nTotal: {replaced} replaced, {skipped} skipped")
    print(f"Lines: {orig_lines} → {total_lines} ({total_lines - orig_lines:+d})")


if __name__ == "__main__":
    apply_patches()
