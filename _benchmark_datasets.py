#!/usr/bin/env python3
"""
Multi-Dataset Benchmark System for ML Projects
================================================
For each project, tests 3-5 candidate datasets, benchmarks with
Random Forest (quick) + XGBoost, selects the best by metric score.

Outputs:
  - dataset_benchmark.json per project directory
  - benchmark_summary.json (global summary)
"""
import json, os, sys, time, warnings, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import make_scorer, f1_score, r2_score
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Dataset loading helpers ──────────────────────────────────────────

def _load_kaggle(slug, filename, sep=","):
    """Download from Kaggle API."""
    import glob as _glob
    _dir = os.path.join(ROOT, ".benchmark_data", slug.replace("/", "_"))
    os.makedirs(_dir, exist_ok=True)
    _fp = os.path.join(_dir, filename)
    if not os.path.exists(_fp):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.dataset_download_files(slug, path=_dir, unzip=True)
        matches = _glob.glob(os.path.join(_dir, "**", filename), recursive=True)
        if matches:
            _fp = matches[0]
    return pd.read_csv(_fp, sep=sep)

def _load_sklearn(name):
    """Load sklearn built-in dataset."""
    import sklearn.datasets as skd
    loader = getattr(skd, name)
    bunch = loader()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names if hasattr(bunch, 'feature_names') else [f"f{i}" for i in range(bunch.data.shape[1])])
    df["target"] = bunch.target
    return df

def _load_openml(did):
    """Load from OpenML by dataset ID."""
    from sklearn.datasets import fetch_openml
    bunch = fetch_openml(data_id=did, as_frame=True, parser="auto")
    df = bunch.frame
    if df is None:
        df = pd.DataFrame(bunch.data)
        df["target"] = bunch.target
    return df

def _load_url(url, sep=","):
    """Load CSV from URL."""
    return pd.read_csv(url, sep=sep)


# ── Candidate Dataset Registry ───────────────────────────────────────
# Each entry: {"source": "kaggle"|"sklearn"|"openml"|"url",
#              "slug/name": ..., "file": ..., "target": ..., "sep": ...}

CANDIDATES = {
    # ─── CLASSIFICATION TASKS ───
    "adult_income": [
        {"id": "kaggle_adult", "source": "kaggle", "slug": "uciml/adult-census-income",
         "file": "adult.csv", "target": "income", "task": "clf"},
        {"id": "openml_adult", "source": "openml", "did": 1590, "target": "class", "task": "clf"},
        {"id": "sklearn_adult", "source": "openml", "did": 2, "target": "class", "task": "clf"},
    ],
    "breast_cancer": [
        {"id": "sklearn_bc", "source": "sklearn", "name": "load_breast_cancer",
         "target": "target", "task": "clf"},
        {"id": "kaggle_bc", "source": "kaggle", "slug": "uciml/breast-cancer-wisconsin-data",
         "file": "data.csv", "target": "diagnosis", "task": "clf"},
        {"id": "openml_bc", "source": "openml", "did": 15, "target": "class", "task": "clf"},
    ],
    "diabetes": [
        {"id": "kaggle_pima", "source": "kaggle", "slug": "uciml/pima-indians-diabetes-database",
         "file": "diabetes.csv", "target": "Outcome", "task": "clf"},
        {"id": "openml_diabetes", "source": "openml", "did": 37, "target": "class", "task": "clf"},
        {"id": "kaggle_stroke", "source": "kaggle", "slug": "fedesoriano/stroke-prediction-dataset",
         "file": "healthcare-dataset-stroke-data.csv", "target": "stroke", "task": "clf"},
    ],
    "heart_disease": [
        {"id": "kaggle_heart", "source": "kaggle", "slug": "fedesoriano/heart-failure-prediction",
         "file": "heart.csv", "target": "HeartDisease", "task": "clf"},
        {"id": "kaggle_hf_clinical", "source": "kaggle", "slug": "andrewmvd/heart-failure-clinical-data",
         "file": "heart_failure_clinical_records_dataset.csv", "target": "DEATH_EVENT", "task": "clf"},
        {"id": "openml_heart", "source": "openml", "did": 53, "target": "class", "task": "clf"},
    ],
    "churn": [
        {"id": "kaggle_telco", "source": "kaggle", "slug": "blastchar/telco-customer-churn",
         "file": "WA_Fn-UseC_-Telco-Customer-Churn.csv", "target": "Churn", "task": "clf"},
        {"id": "kaggle_ibm_hr", "source": "kaggle", "slug": "pavansubhasht/ibm-hr-analytics-attrition-dataset",
         "file": "WA_Fn-UseC_-HR-Employee-Attrition.csv", "target": "Attrition", "task": "clf"},
        {"id": "openml_churn", "source": "openml", "did": 40701, "target": "class", "task": "clf"},
    ],
    "wine_quality": [
        {"id": "kaggle_wine_red", "source": "kaggle", "slug": "uciml/red-wine-quality-cortez-et-al-2009",
         "file": "winequality-red.csv", "target": "quality", "task": "clf"},
        {"id": "kaggle_wine_merged", "source": "kaggle", "slug": "yasserh/wine-quality-dataset",
         "file": "WineQT.csv", "target": "quality", "task": "clf"},
        {"id": "sklearn_wine", "source": "sklearn", "name": "load_wine",
         "target": "target", "task": "clf"},
    ],
    "bank_marketing": [
        {"id": "kaggle_bank", "source": "kaggle", "slug": "henriqueyamahata/bank-marketing",
         "file": "bank-additional-full.csv", "target": "y", "sep": ";", "task": "clf"},
        {"id": "openml_bank", "source": "openml", "did": 1461, "target": "class", "task": "clf"},
        {"id": "kaggle_credit_default", "source": "kaggle", "slug": "uciml/default-of-credit-card-clients-dataset",
         "file": "UCI_Credit_Card.csv", "target": "default.payment.next.month", "task": "clf"},
    ],
    "loan_prediction": [
        {"id": "kaggle_loan", "source": "kaggle", "slug": "vikasukani/loan-eligible-dataset",
         "file": "loan-train.csv", "target": "Loan_Status", "task": "clf"},
        {"id": "kaggle_credit_default", "source": "kaggle", "slug": "uciml/default-of-credit-card-clients-dataset",
         "file": "UCI_Credit_Card.csv", "target": "default.payment.next.month", "task": "clf"},
        {"id": "kaggle_german_credit", "source": "kaggle", "slug": "kabure/german-credit-data-with-risk",
         "file": "german_credit_data.csv", "target": "Risk", "task": "clf"},
    ],
    "employee_turnover": [
        {"id": "kaggle_hr", "source": "kaggle", "slug": "giripujar/hr-analytics",
         "file": "HR_comma_sep.csv", "target": "left", "task": "clf"},
        {"id": "kaggle_ibm_attrition", "source": "kaggle", "slug": "pavansubhasht/ibm-hr-analytics-attrition-dataset",
         "file": "WA_Fn-UseC_-HR-Employee-Attrition.csv", "target": "Attrition", "task": "clf"},
        {"id": "openml_employee", "source": "openml", "did": 1016, "target": "class", "task": "clf"},
    ],
    "fraud_detection": [
        {"id": "kaggle_creditcard", "source": "kaggle", "slug": "mlg-ulb/creditcardfraud",
         "file": "creditcard.csv", "target": "Class", "task": "clf"},
        {"id": "openml_creditcard", "source": "openml", "did": 1597, "target": "Class", "task": "clf"},
        {"id": "kaggle_credit_risk", "source": "kaggle", "slug": "uciml/default-of-credit-card-clients-dataset",
         "file": "UCI_Credit_Card.csv", "target": "default.payment.next.month", "task": "clf"},
    ],
    "titanic": [
        {"id": "kaggle_titanic", "source": "kaggle", "slug": "heptapod/titanic",
         "file": "train_and_test2.csv", "target": "2", "task": "clf"},
        {"id": "openml_titanic", "source": "openml", "did": 40945, "target": "survived", "task": "clf"},
        {"id": "url_titanic", "source": "url",
         "url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
         "target": "Survived", "task": "clf"},
    ],

    # ─── REGRESSION TASKS ───
    "house_prices": [
        {"id": "kaggle_kc_house", "source": "kaggle", "slug": "harlfoxem/housesalesprediction",
         "file": "kc_house_data.csv", "target": "price", "task": "reg"},
        {"id": "sklearn_california", "source": "sklearn", "name": "load_california_housing",
         "target": "target", "task": "reg"},
        {"id": "kaggle_california", "source": "kaggle", "slug": "camnugent/california-housing-prices",
         "file": "housing.csv", "target": "median_house_value", "task": "reg"},
    ],
    "boston_housing": [
        {"id": "kaggle_boston", "source": "kaggle", "slug": "altavish/boston-housing-dataset",
         "file": "HousingData.csv", "target": "MEDV", "task": "reg"},
        {"id": "sklearn_california", "source": "sklearn", "name": "load_california_housing",
         "target": "target", "task": "reg"},
        {"id": "kaggle_diamonds", "source": "kaggle", "slug": "shivam2503/diamonds",
         "file": "diamonds.csv", "target": "price", "task": "reg"},
    ],
    "insurance": [
        {"id": "kaggle_insurance", "source": "kaggle", "slug": "mirichoi0218/insurance",
         "file": "insurance.csv", "target": "charges", "task": "reg"},
        {"id": "kaggle_diamonds", "source": "kaggle", "slug": "shivam2503/diamonds",
         "file": "diamonds.csv", "target": "price", "task": "reg"},
        {"id": "openml_insurance", "source": "openml", "did": 44065, "target": "charges", "task": "reg"},
    ],
    "bike_sharing": [
        {"id": "kaggle_bike", "source": "kaggle", "slug": "lakshmi25npathi/bike-sharing-dataset",
         "file": "day.csv", "target": "cnt", "task": "reg"},
        {"id": "openml_bike", "source": "openml", "did": 42712, "target": "cnt", "task": "reg"},
        {"id": "kaggle_supermarket", "source": "kaggle", "slug": "aungpyaeap/supermarket-sales",
         "file": "supermarket_sales - Sheet1.csv", "target": "Total", "task": "reg"},
    ],
    "bigmart_sales": [
        {"id": "kaggle_bigmart", "source": "kaggle", "slug": "brijbhushannanda1979/bigmart-sales-data",
         "file": "Train.csv", "target": "Item_Outlet_Sales", "task": "reg"},
        {"id": "kaggle_avocado", "source": "kaggle", "slug": "neuromusic/avocado-prices",
         "file": "avocado.csv", "target": "AveragePrice", "task": "reg"},
        {"id": "kaggle_insurance", "source": "kaggle", "slug": "mirichoi0218/insurance",
         "file": "insurance.csv", "target": "charges", "task": "reg"},
    ],
    "black_friday": [
        {"id": "kaggle_bf", "source": "kaggle", "slug": "sdolezel/black-friday",
         "file": "train.csv", "target": "Purchase", "task": "reg"},
        {"id": "kaggle_bigmart", "source": "kaggle", "slug": "brijbhushannanda1979/bigmart-sales-data",
         "file": "Train.csv", "target": "Item_Outlet_Sales", "task": "reg"},
        {"id": "kaggle_diamonds", "source": "kaggle", "slug": "shivam2503/diamonds",
         "file": "diamonds.csv", "target": "price", "task": "reg"},
    ],

    # ─── CLUSTERING TASKS ───
    "customer_segmentation": [
        {"id": "kaggle_mall", "source": "kaggle",
         "slug": "vjchoudhary7/customer-segmentation-tutorial-in-python",
         "file": "Mall_Customers.csv", "target": None, "task": "cluster"},
        {"id": "sklearn_iris_cluster", "source": "sklearn", "name": "load_iris",
         "target": None, "task": "cluster"},
        {"id": "kaggle_wholesale", "source": "openml", "did": 1511,
         "target": None, "task": "cluster"},
    ],
    "ecommerce_segmentation": [
        {"id": "kaggle_ecommerce", "source": "kaggle", "slug": "carrie1/ecommerce-data",
         "file": "data.csv", "target": None, "task": "cluster"},
        {"id": "kaggle_mall", "source": "kaggle",
         "slug": "vjchoudhary7/customer-segmentation-tutorial-in-python",
         "file": "Mall_Customers.csv", "target": None, "task": "cluster"},
        {"id": "openml_wholesale", "source": "openml", "did": 1511,
         "target": None, "task": "cluster"},
    ],
}

# ── Map projects to candidate groups ─────────────────────────────────
PROJECT_MAP = {
    # Classification
    "Classification/Adult Salary Prediction": "adult_income",
    "Classification/Income Classification": "adult_income",
    "Classification/Breast Cancer Detection": "breast_cancer",
    "Classification/Breast Cancer Prediction": "breast_cancer",
    "Classification/Diabetes Classification": "diabetes",
    "Classification/Diabetes ML Analysis": "diabetes",
    "Classification/Diabetes Prediction": "diabetes",
    "Classification/Healthcare Heart Disease Prediction": "heart_disease",
    "Classification/Heart Disease Prediction": "heart_disease",
    "Classification/Customer Churn Prediction - Telecom": "churn",
    "Classification/Customer Lifetime Value Prediction": "churn",
    "Classification/Autoencoder for Customer Churn": "churn",
    "Classification/Traffic Congestion Prediction": "churn",
    "Classification/Wine Quality Analysis": "wine_quality",
    "Classification/Wine Quality Prediction": "wine_quality",
    "Classification/Logistic Regression Balanced": "bank_marketing",
    "Classification/Marketing Campaign Prediction": "bank_marketing",
    "Classification/Bayesian Logistic Regression - Bank Marketing": "bank_marketing",
    "Classification/Loan Default Prediction": "loan_prediction",
    "Classification/Loan Prediction Analysis": "loan_prediction",
    "Classification/Credit Risk Modeling - German Credit": "loan_prediction",
    "Classification/Employee Turnover Analysis": "employee_turnover",
    "Classification/Employee Turnover Prediction": "employee_turnover",
    "Classification/Titanic - Handling Missing Values": "titanic",
    "Classification/Titanic Survival Prediction": "titanic",
    "Classification/Advanced Credit Card Fraud Detection": "fraud_detection",
    "Classification/Credit Card Fraud - Imbalanced Dataset": "fraud_detection",
    "Classification/Fraud Detection": "fraud_detection",
    # Deep Learning CLF
    "Deep Learning/Advanced Churn Modeling": "churn",
    "Deep Learning/Bank Marketing Analysis": "bank_marketing",
    "Deep Learning/Disease Prediction": "heart_disease",
    # Anomaly/Fraud
    "Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS": "fraud_detection",
    "Anomaly detection and fraud detection/Fraud Detection in Financial Transactions": "fraud_detection",
    "Anomaly detection and fraud detection/Fraudulent Credit Card Transaction Detection": "fraud_detection",
    "Anomaly detection and fraud detection/Insurance Fraud Detection": "fraud_detection",
    # Regression
    "Regression/Boston Housing Analysis": "boston_housing",
    "Regression/Boston Housing Prediction Analysis": "boston_housing",
    "Regression/House Price Prediction - Detailed": "house_prices",
    "Regression/House Price prediction": "house_prices",
    "Regression/Car Price Prediction": "house_prices",
    "Regression/Bengaluru House Price Prediction": "house_prices",
    "Regression/House Price - Regularized Linear and XGBoost": "house_prices",
    "Regression/Mercari Price Suggestion - LightGBM": "house_prices",
    "Regression/Insurance premium prediction": "insurance",
    "Regression/Medical Cost Personal": "insurance",
    "Regression/Crop yield prediction": "insurance",
    "Regression/Data Scientist Salary Prediction": "insurance",
    "Regression/Energy Usage Prediction - Buildings": "insurance",
    "Regression/Future Sales Prediction": "insurance",
    "Regression/Hotel Booking Cancellation Prediction": "insurance",
    "Regression/IPL First Innings Prediction - Advanced": "insurance",
    "Regression/IPL First Innings Score Prediction": "insurance",
    "Regression/Job Salary prediction": "insurance",
    "Regression/UCLA Admission Prediction": "insurance",
    "Regression/50 Startups Success Prediction": "insurance",
    "Regression/Ad Demand Forecast - Avito": "insurance",
    "Regression/Bike Sharing Demand Analysis": "bike_sharing",
    "Regression/BigMart Sales Prediction": "bigmart_sales",
    "Regression/Black Friday Sales Prediction": "black_friday",
    "Regression/Black Friday Sales Analysis": "black_friday",
    # Clustering
    "Clustering/Mall Customer Segmentation": "customer_segmentation",
    "Clustering/Mall Customer Segmentation - Advanced": "customer_segmentation",
    "Clustering/Mall Customer Segmentation - Detailed": "customer_segmentation",
    "Clustering/Mall Customer Segmentation Data": "customer_segmentation",
    "Clustering/Customer Segmentation": "customer_segmentation",
    "Clustering/Online Retail Customer Segmentation": "ecommerce_segmentation",
    "Clustering/Online Retail Segmentation Analysis": "ecommerce_segmentation",
    "Classification/Customer Segmentation - E-Commerce": "ecommerce_segmentation",
}


# ── Benchmark engine ──────────────────────────────────────────────────

def preprocess_for_benchmark(df, target_col, task):
    """Quick preprocessing: drop NaN, encode categoricals, return X, y."""
    df = df.copy()
    # Drop ID-like columns
    for c in list(df.columns):
        cs = str(c).lower()
        if cs in ("id", "unnamed: 0", "unnamed:_0", "customerid", "customer_id"):
            df.drop(columns=[c], inplace=True)

    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    elif target_col is None and task == "cluster":
        X = df.select_dtypes(include=[np.number])
        y = None
    else:
        raise ValueError(f"Target '{target_col}' not in columns: {list(df.columns)[:10]}")

    # Encode target if needed
    if y is not None and (y.dtype == "object" or y.dtype.name == "category"):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)

    # Drop non-numeric, encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        for c in cat_cols:
            X[c] = X[c].astype(str)
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])

    # Drop remaining non-numeric
    X = X.select_dtypes(include=[np.number])

    # Drop NaN
    if y is not None:
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
    else:
        X = X.dropna()

    # Subsample for speed (max 20K rows for benchmark)
    if len(X) > 20000:
        idx = np.random.RandomState(42).choice(len(X), 20000, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        if y is not None:
            y = y.iloc[idx].reset_index(drop=True)

    return X, y


def benchmark_one(candidate):
    """Benchmark a single candidate dataset. Returns metrics dict."""
    t0 = time.time()
    result = {
        "id": candidate["id"],
        "source": candidate["source"],
        "task": candidate["task"],
        "status": "failed",
        "error": "",
        "shape": None,
        "n_features": None,
        "missing_pct": None,
        "metric_name": None,
        "metric_rf": None,
        "metric_xgb": None,
        "metric_best": None,
        "time_s": 0,
    }

    try:
        # Load
        src = candidate["source"]
        if src == "kaggle":
            df = _load_kaggle(candidate["slug"], candidate["file"],
                              candidate.get("sep", ","))
        elif src == "sklearn":
            df = _load_sklearn(candidate["name"])
        elif src == "openml":
            df = _load_openml(candidate["did"])
        elif src == "url":
            df = _load_url(candidate["url"], candidate.get("sep", ","))
        else:
            raise ValueError(f"Unknown source: {src}")

        result["shape"] = list(df.shape)
        result["missing_pct"] = round(df.isnull().mean().mean() * 100, 2)

        target = candidate.get("target")
        task = candidate["task"]

        X, y = preprocess_for_benchmark(df, target, task)
        result["n_features"] = X.shape[1]

        if task == "cluster":
            # Silhouette score
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler
            Xs = StandardScaler().fit_transform(X)
            labels = KMeans(n_clusters=min(5, len(X)//10), random_state=42, n_init=5).fit_predict(Xs)
            sil = silhouette_score(Xs, labels)
            result["metric_name"] = "silhouette"
            result["metric_rf"] = round(sil, 4)
            result["metric_best"] = round(sil, 4)
            result["status"] = "success"
        elif task == "clf":
            result["metric_name"] = "f1_weighted"
            n_classes = len(np.unique(y))
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scorer = make_scorer(f1_score, average="weighted")

            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_scores = cross_val_score(rf, X, y, cv=cv, scoring=scorer)
            result["metric_rf"] = round(rf_scores.mean(), 4)

            # XGBoost (if available)
            try:
                from xgboost import XGBClassifier
                xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False,
                                    eval_metric="mlogloss", verbosity=0)
                xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring=scorer)
                result["metric_xgb"] = round(xgb_scores.mean(), 4)
            except Exception:
                result["metric_xgb"] = None

            result["metric_best"] = max(filter(None, [result["metric_rf"], result["metric_xgb"]]))
            result["status"] = "success"
        elif task == "reg":
            result["metric_name"] = "r2"
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")
            result["metric_rf"] = round(rf_scores.mean(), 4)

            try:
                from xgboost import XGBRegressor
                xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                xgb_scores = cross_val_score(xgb, X, y, cv=cv, scoring="r2")
                result["metric_xgb"] = round(xgb_scores.mean(), 4)
            except Exception:
                result["metric_xgb"] = None

            result["metric_best"] = max(filter(None, [result["metric_rf"], result["metric_xgb"]]))
            result["status"] = "success"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        result["status"] = "failed"

    result["time_s"] = round(time.time() - t0, 1)
    return result


def run_benchmarks():
    """Run all benchmarks for all candidate groups."""
    results = {}
    total_groups = len(CANDIDATES)
    out_file = ROOT / "benchmark_summary.json"

    print(f"╔══ Multi-Dataset Benchmark System ══╗")
    print(f"║ {total_groups} candidate groups to benchmark ║")
    print(f"╚════════════════════════════════════╝\n")

    for idx, (group_name, candidates) in enumerate(CANDIDATES.items(), 1):
        print(f"\n[{idx}/{total_groups}] GROUP: {group_name} ({len(candidates)} candidates)")
        print("-" * 60)

        group_results = []
        for cand in candidates:
            print(f"  Testing {cand['id']}...", end=" ", flush=True)
            res = benchmark_one(cand)
            group_results.append(res)
            if res["status"] == "success":
                print(f"OK  {res['metric_name']}={res['metric_best']:.4f}  "
                      f"shape={res['shape']}  ({res['time_s']}s)")
            else:
                print(f"FAIL  {res['error'][:80]}")

        # Pick winner
        successful = [r for r in group_results if r["status"] == "success"]
        if successful:
            winner = max(successful, key=lambda r: r["metric_best"])
            print(f"\n  ★ WINNER: {winner['id']}  "
                  f"({winner['metric_name']}={winner['metric_best']:.4f})")
        else:
            winner = None
            print(f"\n  ✗ NO SUCCESSFUL CANDIDATES")

        results[group_name] = {
            "candidates": group_results,
            "winner": winner["id"] if winner else None,
            "winner_metric": winner["metric_best"] if winner else None,
        }

    # Save results
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Group':<30} {'Winner':<25} {'Metric':<10} {'Score':<8}")
    print("=" * 80)
    for gname, gdata in results.items():
        w = gdata["winner"] or "NONE"
        m = gdata.get("winner_metric", 0) or 0
        # Find metric name
        mn = ""
        for c in gdata["candidates"]:
            if c["status"] == "success":
                mn = c["metric_name"]
                break
        print(f"{gname:<30} {w:<25} {mn:<10} {m:.4f}")
    print("=" * 80)

    # Also save per-project benchmark files
    for proj_path, group_name in PROJECT_MAP.items():
        if group_name in results:
            proj_dir = ROOT / proj_path
            if proj_dir.exists():
                bench_file = proj_dir / "dataset_benchmark.json"
                with open(bench_file, "w") as f:
                    json.dump({
                        "project": proj_path,
                        "group": group_name,
                        **results[group_name]
                    }, f, indent=2)

    return results


if __name__ == "__main__":
    run_benchmarks()
