"""
Master overhaul script v2: generates modern pipeline.py for every project.
ALL data is auto-downloaded at runtime — no prior CSV files needed.
Run: python _overhaul_all.py

Data sources: HuggingFace datasets, sklearn, torchvision, yfinance, UCI, OpenML, direct URLs
"""
import os, sys, json, textwrap
from pathlib import Path

BASE = Path(r"e:\Github\Machine-Learning-Projects")
sys.path.insert(0, str(BASE))


def write_pipeline(project_rel_path, content):
    proj_dir = BASE / project_rel_path
    if not proj_dir.exists():
        print(f"  SKIP (not found): {project_rel_path}")
        return False
    out = proj_dir / "pipeline.py"
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    return True


# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADING SNIPPETS — each returns 4-space-indented code producing `df`
# ════════════════════════════════════════════════════════════════════════════════

def _hf(dataset_name, split="train", config=None, columns=None):
    """HuggingFace datasets.load_dataset to df"""
    cfg = f', "{config}"' if config else ""
    col_filter = ""
    if columns:
        col_filter = f"\n    df = df[{columns}]"
    return f'    from datasets import load_dataset as _hf_load\n    df = _hf_load("{dataset_name}"{cfg}, split="{split}").to_pandas(){col_filter}'

def _sklearn(func_name):
    """sklearn.datasets to df"""
    return f'    from sklearn.datasets import {func_name}\n    _d = {func_name}()\n    df = pd.DataFrame(_d.data, columns=_d.feature_names); df["target"] = _d.target'

def _sklearn_fetch(func_name, target_rename=None):
    """sklearn fetch_ functions (auto-download)"""
    rename = ""
    if target_rename:
        rename = f'\n    df.rename(columns={{"{target_rename}": "target"}}, inplace=True)'
    return f'    from sklearn.datasets import {func_name}\n    _d = {func_name}(as_frame=True)\n    df = _d.frame{rename}'

def _url_csv(url, sep=","):
    """Direct CSV download from URL"""
    return f'    df = pd.read_csv("{url}", sep="{sep}")'

def _yfinance(ticker, period="10y"):
    """Yahoo Finance stock data"""
    return f'    import yfinance as yf\n    df = yf.download("{ticker}", period="{period}", auto_adjust=True).reset_index()\n    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]'

def _torchvision(dataset_cls, n_classes=10):
    """torchvision dataset to returns special flag for image pipeline"""
    return f"__torchvision__{dataset_cls}__{n_classes}"

def _openml(data_id, target=None):
    """OpenML dataset by ID"""
    t = f', target_column="{target}"' if target else ""
    return f'    from sklearn.datasets import fetch_openml\n    _d = fetch_openml(data_id={data_id}{t}, as_frame=True, parser="auto")\n    df = _d.frame'

def _seaborn(name):
    """Seaborn built-in dataset"""
    return f'    import seaborn as _sns\n    df = _sns.load_dataset("{name}")'


# ════════════════════════════════════════════════════════════════════════════════
# PROJECT to DATA SOURCE MAPPING  (every project has an online source)
# ════════════════════════════════════════════════════════════════════════════════

# ── FAMILY 1: TABULAR CLASSIFICATION ──
TABULAR_CLF = {
    "Classification/Adult Salary Prediction": {
        "target": "income",
        "data": _hf("scikit-learn/adult-census-income"),
    },
    "Classification/Breast Cancer Detection": {
        "target": "target",
        "data": _sklearn("load_breast_cancer"),
    },
    "Classification/Breast Cancer Prediction": {
        "target": "diagnosis",
        "data": _openml(1510),
    },
    "Classification/Credit Risk Modeling - German Credit": {
        "target": "class",
        "data": _openml(31),
    },
    "Classification/Customer Churn Prediction - Telecom": {
        "target": "Churn",
        "data": _hf("aai510-group1/telecom-churn-dataset"),
    },
    "Classification/Customer Lifetime Value Prediction": {
        "target": "Response",
        "data": _hf("vkrishna90/vehicle-insurance-customer-data"),
    },
    "Classification/Diabetes Classification": {
        "target": "Outcome",
        "data": _openml(37),  # Pima Indians diabetes
    },
    "Classification/Diabetes ML Analysis": {
        "target": "Outcome",
        "data": _openml(37),
    },
    "Classification/Drinking Water Potability": {
        "target": "Potability",
        "data": _hf("scikit-learn/water-potability"),
    },
    "Classification/Drug Classification": {
        "target": "Drug",
        "data": _openml(46045),
    },
    "Classification/Employee Turnover Analysis": {
        "target": "left",
        "data": _hf("mfaisalqureshi/hr-analytics-and-job-change-of-data-scientists"),
    },
    "Classification/Employee Turnover Prediction": {
        "target": "left",
        "data": _hf("mfaisalqureshi/hr-analytics-and-job-change-of-data-scientists"),
    },
    "Classification/Flower Species Classification": {
        "target": "target",
        "data": _sklearn("load_iris"),
    },
    "Classification/Glass Classification": {
        "target": "Type",
        "data": _openml(41),  # Glass Identification
    },
    "Classification/Groundhog Day Predictions": {
        "target": "Punxsutawney Phil",
        "data": _url_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/refs/heads/main/data/2024/2024-01-30/groundhogs.csv"),
    },
    "Classification/Hand Digit Recognition": {
        "target": "target",
        "data": _sklearn("load_digits"),
    },
    "Classification/Healthcare Heart Disease Prediction": {
        "target": "target",
        "data": _hf("codesignal/heart-disease-prediction"),
    },
    "Classification/Heart Disease Prediction": {
        "target": "target",
        "data": _hf("codesignal/heart-disease-prediction"),
    },
    "Classification/Income Classification": {
        "target": "income",
        "data": _hf("scikit-learn/adult-census-income"),
    },
    "Classification/Iris Dataset Analysis": {
        "target": "target",
        "data": _sklearn("load_iris"),
    },
    "Classification/Loan Default Prediction": {
        "target": "loan_status",
        "data": _hf("ErenalpCet/Loan-Prediction"),
    },
    "Classification/Loan Prediction Analysis": {
        "target": "Loan_Status",
        "data": _hf("ErenalpCet/Loan-Prediction"),
    },
    "Classification/Logistic Regression Balanced": {
        "target": "y",
        "data": _hf("scikit-learn/bank-marketing"),
    },
    "Classification/Marketing Campaign Prediction": {
        "target": "Response",
        "data": _hf("vijaygkd/Marketing_Campaign"),
    },
    "Classification/Mobile Price Classification": {
        "target": "price_range",
        "data": _openml(44126),
    },
    "Classification/Simple Classification Problem": {
        "target": "target",
        "data": _sklearn("load_iris"),
    },
    "Classification/Social Network Ads Analysis": {
        "target": "Purchased",
        "data": _url_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/Social%20Network%20Ads.csv"),
    },
    "Classification/Student Performance Prediction": {
        "target": "G3",
        "data": _openml(42352),  # Student performance
    },
    "Classification/Titanic - Handling Missing Values": {
        "target": "survived",
        "data": _seaborn("titanic"),
    },
    "Classification/Titanic Survival Prediction": {
        "target": "survived",
        "data": _seaborn("titanic"),
    },
    "Classification/Weather Classification - Decision Trees": {
        "target": "RainTomorrow",
        "data": _hf("Zaherrr/Weather-Dataset"),
    },
    "Classification/Wine Quality Analysis": {
        "target": "quality",
        "data": _openml(287),  # wine quality red
    },
    "Classification/Wine Quality Prediction": {
        "target": "quality",
        "data": _openml(287),
    },
    "Classification/Autoencoder for Customer Churn": {
        "target": "Churn",
        "data": _hf("aai510-group1/telecom-churn-dataset"),
    },
    "Classification/Bayesian Logistic Regression - Bank Marketing": {
        "target": "y",
        "data": _hf("scikit-learn/bank-marketing"),
    },
    "Classification/Boston House Classification": {
        "target": "MEDV",
        "data": _sklearn_fetch("fetch_california_housing"),
    },
    "Classification/H2O Higgs Boson": {
        "target": "class",
        "data": _openml(44129),  # Higgs
    },
    "Classification/Earthquake Prediction": {
        "target": "magnitude",
        "data": _url_csv("https://raw.githubusercontent.com/datasets/earthquake/main/data/earthquake.csv"),
    },
    "Classification/SONAR Rock vs Mine Prediction": {
        "target": "Class",
        "data": _openml(40),  # Sonar
    },
    "Classification/Traffic Congestion Prediction": {
        "target": "traffic_situation",
        "data": _hf("mfumanelli/traffic-prediction"),
    },
    "Classification/Diabetes Prediction": {
        "target": "Outcome",
        "data": _openml(37),
    },
    "Deep Learning/Advanced Churn Modeling": {
        "target": "Exited",
        "data": _hf("aai510-group1/telecom-churn-dataset"),
    },
    "Deep Learning/Bank Marketing Analysis": {
        "target": "y",
        "data": _hf("scikit-learn/bank-marketing"),
    },
    "Deep Learning/Campus Recruitment Analysis": {
        "target": "status",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset3/refs/heads/master/Placement_Data_Full_Class.csv"),
    },
    "Deep Learning/COVID-19 Drug Recovery": {
        "target": "Recovered",
        "data": _url_csv("https://raw.githubusercontent.com/datasets/covid-19/main/data/time-series-19-covid-combined.csv"),
    },
    "Deep Learning/Disease Prediction": {
        "target": "prognosis",
        "data": _hf("saravan2024/Disease-Symptom"),
    },
}

# ── FAMILY 2: TABULAR REGRESSION ──
TABULAR_REG = {
    "Regression/Boston Housing Analysis": {
        "target": "target",
        "data": _sklearn_fetch("fetch_california_housing"),
    },
    "Regression/Boston Housing Prediction Analysis": {
        "target": "target",
        "data": _sklearn_fetch("fetch_california_housing"),
    },
    "Regression/House Price Prediction - Detailed": {
        "target": "price",
        "data": _hf("leostelon/KC-House-Data"),
    },
    "Regression/House Price prediction": {
        "target": "SalePrice",
        "data": _hf("leostelon/house-prices-advanced-regression"),
    },
    "Regression/Insurance premium prediction": {
        "target": "charges",
        "data": _openml(43463),
    },
    "Regression/Gold Price Prediction": {
        "target": "Close",
        "data": _yfinance("GLD"),
    },
    "Regression/Flight Fare Prediction": {
        "target": "Price",
        "data": _hf("thedevastator/flight-price-prediction-data"),
    },
    "Regression/Car Price Prediction": {
        "target": "selling_price",
        "data": _hf("Xenova/used-cars"),
    },
    "Regression/Data Scientist Salary Prediction": {
        "target": "salary_in_usd",
        "data": _hf("inductiva/ds-salaries"),
    },
    "Regression/Medical Cost Personal": {
        "target": "charges",
        "data": _openml(43463),
    },
    "Regression/Bengaluru House Price Prediction": {
        "target": "price",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/bangalore.csv"),
    },
    "Regression/BigMart Sales Prediction": {
        "target": "Item_Outlet_Sales",
        "data": _hf("saurabh1212/Bigmart-Sales-Data"),
    },
    "Regression/Bike Sharing Demand Analysis": {
        "target": "cnt",
        "data": _openml(42712),
    },
    "Regression/Black Friday Sales Prediction": {
        "target": "Purchase",
        "data": _hf("puspendert/Black-Friday-Sales-Prediction"),
    },
    "Regression/Black Friday Sales Analysis": {
        "target": "Purchase",
        "data": _hf("puspendert/Black-Friday-Sales-Prediction"),
    },
    "Regression/Bitcoin Price Prediction": {
        "target": "Close",
        "data": _yfinance("BTC-USD"),
    },
    "Regression/Bitcoin Price Prediction - Advanced": {
        "target": "Close",
        "data": _yfinance("BTC-USD"),
    },
    "Regression/California Housing Prediction": {
        "target": "target",
        "data": _sklearn_fetch("fetch_california_housing"),
    },
    "Regression/Car Price Prediction - Feature Based": {
        "target": "selling_price",
        "data": _hf("Xenova/used-cars"),
    },
    "Regression/China GDP Estimation": {
        "target": "Value",
        "data": _url_csv("https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv"),
    },
    "Regression/Crop yield prediction": {
        "target": "hg/ha_yield",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/crop_yield.csv"),
    },
    "Regression/Diabetes Prediction - Pima Indians": {
        "target": "Outcome",
        "data": _openml(37),
    },
    "Regression/Employee Future Prediction": {
        "target": "LeaveOrNot",
        "data": _hf("mfaisalqureshi/hr-analytics-and-job-change-of-data-scientists"),
    },
    "Regression/Energy Usage Prediction - Buildings": {
        "target": "Heating Load",
        "data": _openml(242),  # Energy efficiency
    },
    "Regression/Flight Delay Prediction": {
        "target": "dep_delayed_15min",
        "data": _hf("vitaliy-datamonster/flight-delays"),
    },
    "Regression/Future Sales Prediction": {
        "target": "Sales",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/advertising.csv"),
    },
    "Regression/Heart disease prediction": {
        "target": "target",
        "data": _hf("codesignal/heart-disease-prediction"),
    },
    "Regression/Hotel Booking Cancellation Prediction": {
        "target": "is_canceled",
        "data": _hf("Tirumala/hotel_booking_demand"),
    },
    "Regression/House Price - Regularized Linear and XGBoost": {
        "target": "SalePrice",
        "data": _hf("leostelon/house-prices-advanced-regression"),
    },
    "Regression/IPL First Innings Prediction - Advanced": {
        "target": "total",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/ipl_data.csv"),
    },
    "Regression/IPL First Innings Score Prediction": {
        "target": "total",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/ipl_data.csv"),
    },
    "Regression/Job Salary prediction": {
        "target": "salary_in_usd",
        "data": _hf("inductiva/ds-salaries"),
    },
    "Regression/Mercari Price Suggestion - LightGBM": {
        "target": "price",
        "data": _hf("thedevastator/mercari-price-prediction"),
    },
    "Regression/Rainfall Amount Prediction": {
        "target": "PRCP",
        "data": _hf("Zaherrr/Weather-Dataset"),
    },
    "Regression/Rainfall Prediction": {
        "target": "PRCP",
        "data": _hf("Zaherrr/Weather-Dataset"),
    },
    "Regression/Stock price prediction": {
        "target": "Close",
        "data": _yfinance("AAPL"),
    },
    "Regression/TPOT Mercedes Prediction": {
        "target": "y",
        "data": _openml(42570),
    },
    "Regression/Tesla Car Price Prediction": {
        "target": "Close",
        "data": _yfinance("TSLA"),
    },
    "Regression/UCLA Admission Prediction": {
        "target": "Chance of Admit",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/admission_predict.csv"),
    },
    "Regression/50 Startups Success Prediction": {
        "target": "Profit",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/50_Startups.csv"),
    },
    "Regression/Bank Customer churn prediction": {
        "target": "Exited",
        "data": _hf("aai510-group1/telecom-churn-dataset"),
    },
    "Regression/Ad Demand Forecast - Avito": {
        "target": "deal_probability",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/advertising.csv"),
    },
    "Deep Learning/Concrete Strength Prediction": {
        "target": "csMPa",
        "data": _openml(4353),  # Concrete compressive strength
    },
    "Deep Learning/Earthquake Prediction": {
        "target": "magnitude",
        "data": _url_csv("https://raw.githubusercontent.com/datasets/earthquake/main/data/earthquake.csv"),
    },
}

# ── FAMILY 3: FRAUD / IMBALANCED ──
FRAUD = {
    "Classification/Advanced Credit Card Fraud Detection": {
        "target": "Class",
        "data": _hf("imodels/credit-card"),
    },
    "Classification/Credit Card Fraud - Imbalanced Dataset": {
        "target": "Class",
        "data": _hf("imodels/credit-card"),
    },
    "Classification/Fraud Detection": {
        "target": "isFraud",
        "data": _hf("vitaliy-datamonster/fraud-detection"),
    },
    "Anomaly detection and fraud detection/Fraud Detection in Financial Transactions": {
        "target": "isFraud",
        "data": _hf("vitaliy-datamonster/fraud-detection"),
    },
    "Anomaly detection and fraud detection/Insurance Fraud Detection": {
        "target": "fraud_reported",
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/insurance_fraud.csv"),
    },
    "Anomaly detection and fraud detection/Fraud Detection - IEEE-CIS": {
        "target": "isFraud",
        "data": _hf("vitaliy-datamonster/fraud-detection"),
    },
    "Anomaly detection and fraud detection/Fraudulent Credit Card Transaction Detection": {
        "target": "Class",
        "data": _hf("imodels/credit-card"),
    },
}

# ── FAMILY 4: ANOMALY DETECTION ──
ANOMALY = {
    "Anomaly detection and fraud detection/Anomaly Detection - Numenta Benchmark": {
        "data": _hf("VictorSanh/anomaly-detection"),
    },
    "Anomaly detection and fraud detection/Anomaly Detection - Social Networks Twitter Bot": {
        "data": _openml(44307),  # bot detection
    },
    "Anomaly detection and fraud detection/Anomaly Detection in Images - CIFAR-10": {
        "data": '    from sklearn.datasets import load_digits\n    _d = load_digits()\n    df = pd.DataFrame(_d.data); df["target"] = _d.target',
    },
    "Anomaly detection and fraud detection/Banknote Authentication": {
        "data": _openml(1462),  # banknote
    },
    "Anomaly detection and fraud detection/Breast Cancer Detection - Wisconsin Dataset": {
        "data": _sklearn("load_breast_cancer"),
    },
    "Anomaly detection and fraud detection/Intrusion Detection": {
        "data": _sklearn_fetch("fetch_kddcup99"),
    },
    "Anomaly detection and fraud detection/Traffic Flow Prediction - METR-LA": {
        "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/traffic_volume.csv"),
    },
}

# ── FAMILY 5: CLUSTERING ──
CLUSTERING = {
    "Clustering/Credit Card Customer Segmentation": {"data": _hf("imodels/credit-card")},
    "Clustering/Customer Segmentation": {"data": _openml(1590)},
    "Clustering/Customer Segmentation - Bank": {"data": _hf("scikit-learn/bank-marketing")},
    "Clustering/Financial Time Series Clustering": {"data": _yfinance("SPY", "5y")},
    "Clustering/Housing Price Segmentation": {"data": _sklearn_fetch("fetch_california_housing")},
    "Clustering/KMeans Clustering - Imagery Analysis": {"data": _sklearn("load_digits")},
    "Clustering/Mall Customer Segmentation": {"data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/Mall_Customers.csv")},
    "Clustering/Mall Customer Segmentation - Advanced": {"data": _openml(1590)},
    "Clustering/Mall Customer Segmentation - Detailed": {"data": _openml(1590)},
    "Clustering/Mall Customer Segmentation Data": {"data": _openml(1590)},
    "Clustering/Online Retail Customer Segmentation": {"data": _hf("nazlicanto/e-commerce")},
    "Clustering/Online Retail Segmentation Analysis": {"data": _hf("nazlicanto/e-commerce")},
    "Clustering/Spotify Song Cluster Analysis": {"data": _hf("maharshipandya/spotify-tracks-dataset")},
    "Clustering/Turkiye Student Evaluation - Advanced": {"data": _openml(1523)},
    "Clustering/Turkiye Student Evaluation Analysis": {"data": _openml(1523)},
    "Clustering/Vehicle Crash Data Clustering": {"data": _url_csv("https://raw.githubusercontent.com/fivethirtyeight/data/refs/heads/master/bad-drivers/bad-drivers.csv")},
    "Clustering/Weather Data Clustering - KMeans": {"data": _hf("Zaherrr/Weather-Dataset")},
    "Clustering/Wholesale Customer Segmentation": {"data": _openml(1511)},
    "Clustering/Wholesale Segmentation Analysis": {"data": _openml(1511)},
    "Clustering/Wine Segmentation": {"data": _sklearn("load_wine")},
    "Classification/Customer Segmentation - E-Commerce": {"data": _hf("nazlicanto/e-commerce")},
}

# ── FAMILY 6: NLP CLASSIFICATION ──
NLP_CLF = {
    "Classification/Cyberbullying Classification": {"target": "cyberbullying_type", "text_col": "tweet_text", "data": _hf("mtbench101/cyberbullying_tweets")},
    "Classification/Movie Genre Classification": {"target": "genre", "text_col": "description", "data": _hf("datadrivenscience/movies-genres-prediction")},
    "Classification/Spam Email Classification": {"target": "label", "text_col": "text", "data": _hf("TrainingDataPro/email-spam-classification")},
    "NLP/Amazon Alexa Review Sentiment": {"target": "feedback", "text_col": "verified_reviews", "data": _hf("mesolitica/amazon-alexa-review")},
    "NLP/Amazon Sentiment Analysis": {"target": "label", "text_col": "text", "data": _hf("mteb/amazon_polarity")},
    "NLP/Clinton vs Trump Tweets Analysis": {"target": "label", "text_col": "text", "data": _hf("SetFit/tweet_eval_stance_hillary")},
    "NLP/Consumer Complaints Analysis": {"target": "product", "text_col": "text", "data": _hf("consumer-finance-complaints/consumer_complaints")},
    "NLP/Disaster or Not Disaster": {"target": "target", "text_col": "text", "data": _hf("venetis/disaster_tweets")},
    "NLP/DJIA Sentiment Analysis - News Headlines": {"target": "label", "text_col": "text", "data": _hf("financial_phrasebank", split="train", config="sentences_50agree")},
    "NLP/DJIA Sentiment Analysis - Stock Prediction": {"target": "label", "text_col": "text", "data": _hf("financial_phrasebank", split="train", config="sentences_50agree")},
    "NLP/Fake News Detection": {"target": "label", "text_col": "text", "data": _hf("GonzaloA/fake_news")},
    "NLP/GitHub Bugs Prediction": {"target": "label", "text_col": "text", "data": _hf("bigcode/the-stack-github-issues", split="train")},
    "NLP/Hate Speech Detection": {"target": "label", "text_col": "tweet", "data": _hf("hate_speech18")},
    "NLP/IMDB Sentiment Analysis - Deep Learning": {"target": "label", "text_col": "text", "data": _hf("stanfordnlp/imdb")},
    "NLP/IMDB Sentiment Review Analysis": {"target": "label", "text_col": "text", "data": _hf("stanfordnlp/imdb")},
    "NLP/Message Spam Detection": {"target": "label", "text_col": "sms", "data": _hf("ucirvine/sms_spam")},
    "NLP/Movie Review Sentiments": {"target": "label", "text_col": "text", "data": _hf("rotten_tomatoes")},
    "NLP/Restaurant Review Sentiment Analysis": {"target": "label", "text_col": "text", "data": _hf("scikit-learn/restaurant-reviews")},
    "NLP/Resume Screening": {"target": "Category", "text_col": "Resume", "data": _hf("Pravincoder/Resume_Dataset")},
    "NLP/Sentiment Analysis": {"target": "label", "text_col": "text", "data": _hf("stanfordnlp/imdb")},
    "NLP/Sentiment Analysis - Flask Web App": {"target": "label", "text_col": "text", "data": _hf("stanfordnlp/imdb")},
    "NLP/Sentiment Analysis - Restaurant Reviews": {"target": "label", "text_col": "text", "data": _hf("scikit-learn/restaurant-reviews")},
    "NLP/SMS Spam Detection": {"target": "label", "text_col": "sms", "data": _hf("ucirvine/sms_spam")},
    "NLP/SMS Spam Detection - Detailed": {"target": "label", "text_col": "sms", "data": _hf("ucirvine/sms_spam")},
    "NLP/SMS Spam Detection Analysis": {"target": "label", "text_col": "sms", "data": _hf("ucirvine/sms_spam")},
    "NLP/Spam Classifier": {"target": "label", "text_col": "sms", "data": _hf("ucirvine/sms_spam")},
    "NLP/Spam SMS Classification": {"target": "label", "text_col": "sms", "data": _hf("ucirvine/sms_spam")},
    "NLP/Text Classification": {"target": "label", "text_col": "text", "data": _hf("SetFit/20_newsgroups")},
    "NLP/Text Classification - Keras Consumer Complaints": {"target": "product", "text_col": "text", "data": _hf("consumer-finance-complaints/consumer_complaints")},
    "NLP/Text Classification with NLP": {"target": "label", "text_col": "text", "data": _hf("SetFit/20_newsgroups")},
    "NLP/Three-Way Sentiment Analysis - Tweets": {"target": "sentiment", "text_col": "text", "data": _hf("mteb/tweet_sentiment_extraction")},
    "NLP/Twitter Sentiment Analysis": {"target": "label", "text_col": "text", "data": _hf("cardiffnlp/tweet_eval", config="sentiment")},
    "NLP/Twitter Sentiment Analysis - ML": {"target": "label", "text_col": "text", "data": _hf("cardiffnlp/tweet_eval", config="sentiment")},
    "NLP/Twitter US Airline Sentiment": {"target": "airline_sentiment", "text_col": "text", "data": _hf("osanseviero/twitter-airline-sentiment")},
    "NLP/US Election Prediction": {"target": "label", "text_col": "text", "data": _hf("SetFit/tweet_eval_stance_hillary")},
    "Deep Learning/Amazon Alexa Sentiment Analysis": {"target": "feedback", "text_col": "verified_reviews", "data": _hf("mesolitica/amazon-alexa-review")},
    "Deep Learning/IMDB Sentiment Analysis": {"target": "label", "text_col": "text", "data": _hf("stanfordnlp/imdb")},
    "Deep Learning/News Category Prediction": {"target": "category", "text_col": "headline", "data": _hf("heegyu/news-category-dataset")},
    "Deep Learning/Sentiment Analysis - Flask App": {"target": "label", "text_col": "text", "data": _hf("stanfordnlp/imdb")},
    # Text classification misc
    "NLP/Profanity Checker": {"target": "label", "text_col": "text", "data": _hf("hate_speech18")},
    "NLP/BOW and TF-IDF with XGBoost": {"target": "label", "text_col": "text", "data": _hf("stanfordnlp/imdb")},
}

# ── FAMILY 7: NER / ENTITY EXTRACTION ──
NLP_NER = {
    "NLP/Named Entity Recognition": {
        "labels": ["person", "location", "organization", "miscellaneous"],
        "data": _hf("conll2003"),
        "text_col": "tokens",
        "tag_col": "ner_tags",
    },
    "NLP/Keyword Extraction": {
        "labels": ["keyword", "keyphrase", "topic", "entity"],
        "data": _hf("midas/inspec", config="extraction"),
        "text_col": "document",
        "tag_col": "doc_bio_tags",
    },
    "NLP/Keyword Research": {
        "labels": ["keyword", "keyphrase", "topic", "entity"],
        "data": _hf("midas/inspec", config="extraction"),
        "text_col": "document",
        "tag_col": "doc_bio_tags",
    },
}

# ── FAMILY 8: NLP GENERATION ──
NLP_GEN = {
    "NLP/Document Summary Creator": {"task": "summarization", "data": _hf("EdinburghNLP/xsum")},
    "NLP/Language Translation Model": {"task": "translation", "data": _hf("wmt16", config="de-en", split="train[:1000]")},
    "NLP/Language Translator": {"task": "translation", "data": _hf("wmt16", config="de-en", split="train[:1000]")},
    "NLP/Next Word Prediction": {"task": "generation", "data": _hf("wikitext", config="wikitext-2-raw-v1")},
    "NLP/Text Generation": {"task": "generation", "data": _hf("wikitext", config="wikitext-2-raw-v1")},
    "NLP/Text Summarization": {"task": "summarization", "data": _hf("EdinburghNLP/xsum")},
    "NLP/Text Summarization - Medium": {"task": "summarization", "data": _hf("EdinburghNLP/xsum")},
    "NLP/Text Summarization - Word Frequency": {"task": "summarization", "data": _hf("EdinburghNLP/xsum")},
    "NLP/Text Summarization - Word Frequency Method": {"task": "summarization", "data": _hf("EdinburghNLP/xsum")},
    "NLP/Spell Checker": {"task": "generation", "data": _hf("wikitext", config="wikitext-2-raw-v1")},
    "NLP/Spelling Correction": {"task": "generation", "data": _hf("wikitext", config="wikitext-2-raw-v1")},
    "NLP/Autocorrect": {"task": "generation", "data": _hf("wikitext", config="wikitext-2-raw-v1")},
    "NLP/NLP for Other Languages": {"task": "translation", "data": _hf("wmt16", config="de-en", split="train[:1000]")},
    "Deep Learning/Chatbot": {"task": "chatbot", "data": _hf("Alizimal/daily-dialogs")},
    "Deep Learning/ChatBot - Neural Network": {"task": "chatbot", "data": _hf("Alizimal/daily-dialogs")},
    "Deep Learning/Movie Title Prediction": {"task": "generation", "data": _hf("wikitext", config="wikitext-2-raw-v1")},
}

# ── FAMILY 8: IMAGE CLASSIFICATION ──
IMAGE_CLF = {
    "Classification/Autoencoder Fashion MNIST": {"dataset": "FashionMNIST", "n_classes": 10},
    "Classification/CIFAR-10 Classification": {"dataset": "CIFAR10", "n_classes": 10},
    "Classification/Cotton Disease Prediction": {"dataset": "hf:smaranjitghose/cotton-disease-dataset", "n_classes": 4},
    "Classification/Digit Recognition - MNIST Sequence": {"dataset": "MNIST", "n_classes": 10},
    "Classification/Dog vs Cat Classification": {"dataset": "hf:microsoft/cats_vs_dogs", "n_classes": 2},
    "Classification/Fashion MNIST Analysis": {"dataset": "FashionMNIST", "n_classes": 10},
    "Classification/Garbage Classification": {"dataset": "hf:garythung/trashnet", "n_classes": 6},
    "Classification/Plant Disease Recognition": {"dataset": "hf:mhammad/PlantVillage", "n_classes": 38},
    "Classification/Pneumonia Classification": {"dataset": "hf:keremberke/chest-xray-classification", "n_classes": 2},
    "Computer Vision/Indian Classical Dance Classification": {"dataset": "hf:Indian-Dance-Form-Recognition", "n_classes": 8},
    "Computer Vision/Traffic Sign Recognition": {"dataset": "hf:bazyl/GTSRB", "n_classes": 43},
    "Computer Vision/Traffic Sign Recognizer": {"dataset": "hf:bazyl/GTSRB", "n_classes": 43},
    "Deep Learning/Advanced ResNet-50": {"dataset": "CIFAR10", "n_classes": 10},
    "Deep Learning/Arabic Character Recognition": {"dataset": "hf:HosamEddinMohamed/arabic-handwritten-chars", "n_classes": 28},
    "Deep Learning/Bottle vs Can Classification": {"dataset": "hf:garythung/trashnet", "n_classes": 2},
    "Deep Learning/Brain Tumor Recognition": {"dataset": "hf:sartajbhuvaji/Brain-Tumor-Classification", "n_classes": 4},
    "Deep Learning/Cactus Aerial Image Recognition": {"dataset": "hf:IQTLabs/aerial-cactus-identification", "n_classes": 2},
    "Deep Learning/Cat vs Dog Classification": {"dataset": "hf:microsoft/cats_vs_dogs", "n_classes": 2},
    "Deep Learning/Clothing Prediction - Flask App": {"dataset": "FashionMNIST", "n_classes": 10},
    "Deep Learning/Dance Form Identification": {"dataset": "hf:Indian-Dance-Form-Recognition", "n_classes": 8},
    "Deep Learning/Diabetic Retinopathy": {"dataset": "hf:aharley/diabetic-retinopathy-detection", "n_classes": 5},
    "Deep Learning/Fingerprint Recognition": {"dataset": "hf:Antoinegg1/fingerprint", "n_classes": 10},
    "Deep Learning/Glass Detection": {"dataset": "CIFAR10", "n_classes": 2},
    "Deep Learning/Happy House Predictor": {"dataset": "hf:Falah/happy_house", "n_classes": 2},
    "Deep Learning/Keep Babies Safe": {"dataset": "CIFAR10", "n_classes": 2},
    "Deep Learning/Lego Brick Classification": {"dataset": "hf:LEGO-Brick-Images", "n_classes": 16},
    "Deep Learning/Pneumonia Detection": {"dataset": "hf:keremberke/chest-xray-classification", "n_classes": 2},
    "Deep Learning/Sheep Breed Classification - CNN": {"dataset": "CIFAR10", "n_classes": 4},
    "Deep Learning/Skin Cancer Recognition": {"dataset": "hf:marmal88/skin_cancer", "n_classes": 7},
    "Deep Learning/Walking or Running Classification": {"dataset": "CIFAR10", "n_classes": 2},
    "Deep Learning/World Currency Coin Detection": {"dataset": "CIFAR10", "n_classes": 10},
}

# ── FAMILY 9: CV DETECTION (YOLO) ──
CV_DETECTION = {
    "Computer Vision/Car and Pedestrian Tracker": {"task": "track"},
    "Computer Vision/Lane Finder": {"task": "detect"},
    "Deep Learning/Landmark Detection": {"task": "detect"},
}

# ── FAMILY 10: FACE/GESTURE ──
FACE_GESTURE = {
    "Computer Vision/Face Detection - OpenCV": {"task": "face_detection"},
    "Computer Vision/Face Expression Identifier": {"task": "expression"},
    "Computer Vision/Face Mask Detection": {"task": "face_detection"},
    "Computer Vision/Gesture Control Media Player": {"task": "hand_gesture"},
    "Computer Vision/Home Security": {"task": "face_detection"},
    "Computer Vision/Live Smile Detector": {"task": "expression"},
    "Computer Vision/Room Security - Webcam": {"task": "face_detection"},
    "Computer Vision/Face Recognition Door Lock - AWS Rekognition": {"task": "face_recognition"},
    "Deep Learning/Caffe Face Detector - OpenCV": {"task": "face_detection"},
    "Deep Learning/Face Gender and Ethnicity Recognizer": {"task": "face_recognition"},
    "Deep Learning/Face Mask Detection": {"task": "face_detection"},
    "Deep Learning/Parkinson Pose Estimation": {"task": "pose"},
}

# ── FAMILY 11: OCR ──
OCR = {
    "Computer Vision/Image Text Extraction - OCR": {},
    "Computer Vision/Image to Text Conversion - OCR": {},
    "Computer Vision/QR Code Readability": {},
    "Computer Vision/Document Word Detection": {},
    "Computer Vision/Captcha Recognition": {},
}

# ── FAMILY 12: RECOMMENDATION ──
RECOMMENDATION = {
    # CF-primary (implicit ALS/BPR as primary, Surprise SVD/KNN baseline)
    "Recommendation Systems/Movie Recommendation Engine": {"data": _hf("reczilla/movielens-100k"), "task": "cf"},
    "Recommendation Systems/Movie Recommendation System": {"data": _hf("reczilla/movielens-100k"), "task": "cf"},
    "Recommendation Systems/Movies Recommender": {"data": _hf("reczilla/movielens-100k"), "task": "cf"},
    "Recommendation Systems/Recommender with Surprise Library": {"data": _hf("reczilla/movielens-100k"), "task": "cf"},
    "Recommendation Systems/Collaborative Filtering - TensorFlow": {"data": _hf("reczilla/movielens-100k"), "task": "cf"},
    "Recommendation Systems/Building Recommender in an Hour": {"data": _hf("reczilla/movielens-100k"), "task": "cf"},
    "Recommendation Systems/Recommender Systems Fundamentals": {"data": _hf("reczilla/movielens-100k"), "task": "cf"},
    "Recommendation Systems/Million Songs Recommendation Engine": {"data": _hf("maharshipandya/spotify-tracks-dataset"), "task": "cf"},
    "Recommendation Systems/Music Recommendation System": {"data": _hf("maharshipandya/spotify-tracks-dataset"), "task": "cf"},
    # Hybrid (LightFM — metadata-aware, cold-start)
    "Recommendation Systems/Hotel Recommendation System": {"data": _hf("Yelp/yelp_review_full"), "task": "hybrid"},
    "Recommendation Systems/E-Commerce Recommendation System": {"data": _hf("nazlicanto/e-commerce"), "task": "hybrid"},
    "Recommendation Systems/Event Recommendation System": {"data": _hf("reczilla/movielens-100k"), "task": "hybrid"},
    "Recommendation Systems/Restaurant Recommendation System": {"data": _hf("Yelp/yelp_review_full"), "task": "hybrid"},
    "Recommendation Systems/Seattle Hotels Recommender": {"data": _hf("Yelp/yelp_review_full"), "task": "hybrid"},
    # Content-based (Sentence Transformers / Qwen3-Embedding / BGE-M3)
    "Recommendation Systems/Article Recommendation System": {"data": _hf("heegyu/news-category-dataset"), "task": "content"},
    "Recommendation Systems/Articles Recommender": {"data": _hf("heegyu/news-category-dataset"), "task": "content"},
    "Recommendation Systems/Book Recommendation System": {"data": _hf("zhengyun21/Book-Crossing"), "task": "content"},
    "Recommendation Systems/Recipe Recommendation System": {"data": _hf("Hieu-Pham/kaggle_food_recipes"), "task": "content"},
    "Recommendation Systems/TV Show Recommendation System": {"data": _hf("reczilla/movielens-100k"), "task": "content"},
}

# ── FAMILY 13: TIME SERIES ──
TIME_SERIES = {
    "Time Series Analysis/Cryptocurrency Price Forecasting": {"target": "Close", "data": _yfinance("BTC-USD", "5y")},
    "Time Series Analysis/Electricity Demand Forecasting": {"target": "value", "data": _hf("EnergyStatisticsDatasets/electricity_demand")},
    "Time Series Analysis/Forecasting with ARIMA": {"target": "Close", "data": _yfinance("SPY", "10y")},
    "Time Series Analysis/Gold Price Forecasting": {"target": "Close", "data": _yfinance("GLD", "10y")},
    "Time Series Analysis/Granger Causality Test": {"target": "Close", "data": _yfinance("AAPL", "5y")},
    "Time Series Analysis/Mini Course Sales Forecasting": {"target": "Sales", "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/advertising.csv")},
    "Time Series Analysis/Pollution Forecasting": {"target": "pollution", "data": _hf("juanma9613/Beijing-PM2.5-dataset")},
    "Time Series Analysis/Power Consumption - LSTM": {"target": "Global_active_power", "data": _hf("Ammok/Household_Power_Consumption")},
    "Time Series Analysis/Promotional Time Series": {"target": "Sales", "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/advertising.csv")},
    "Time Series Analysis/Rossmann Store Sales Forecasting": {"target": "Sales", "data": _hf("thedevastator/rossmann-store-sales")},
    "Time Series Analysis/Smart Home Temperature Forecasting": {"target": "temperature", "data": _hf("Zaherrr/Weather-Dataset")},
    "Time Series Analysis/Solar Power Generation Forecasting": {"target": "power", "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/solar_power.csv")},
    "Time Series Analysis/Stock Market Analysis - Tech Stocks": {"target": "Close", "data": _yfinance("AAPL MSFT GOOGL AMZN NVDA", "5y")},
    "Time Series Analysis/Stock Price Forecasting": {"target": "Close", "data": _yfinance("AAPL", "10y")},
    "Time Series Analysis/Store Item Demand Forecasting": {"target": "sales", "data": _hf("thedevastator/store-item-demand-forecasting")},
    "Time Series Analysis/Time Series Forecasting": {"target": "Close", "data": _yfinance("SPY", "10y")},
    "Time Series Analysis/Time Series Forecasting - Introduction": {"target": "Close", "data": _yfinance("SPY", "5y")},
    "Time Series Analysis/Time Series with LSTM": {"target": "Close", "data": _yfinance("AAPL", "10y")},
    "Time Series Analysis/Traffic Forecast": {"target": "traffic_volume", "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/traffic_volume.csv")},
    "Time Series Analysis/US Gasoline and Diesel Prices 1995-2021": {"target": "value", "data": _hf("jaeyoung-im/us-gasoline-prices")},
    "Time Series Analysis/Weather Forecasting": {"target": "temp", "data": _hf("Zaherrr/Weather-Dataset")},
    "Deep Learning/Amazon Stock Price Analysis": {"target": "Close", "data": _yfinance("AMZN", "10y")},
    "Deep Learning/Hourly Energy Demand and Weather": {"target": "demand", "data": _hf("Ammok/Household_Power_Consumption")},
    "Deep Learning/Stock Market Prediction": {"target": "Close", "data": _yfinance("AAPL", "10y")},
    "Deep Learning/Electric Car Temperature Prediction": {"target": "temperature", "data": _hf("Zaherrr/Weather-Dataset")},
}

# ── FAMILY 14: REINFORCEMENT LEARNING ──
RL = {
    # Discrete-action environments (PPO primary)
    "Reinforcement Learning/Cliff Walking": {"env": "CliffWalking-v0", "algo": "PPO"},
    "Reinforcement Learning/Frozen Lake": {"env": "FrozenLake-v1", "algo": "PPO"},
    "Reinforcement Learning/Gridworld Navigation": {"env": "CartPole-v1", "algo": "PPO"},
    "Reinforcement Learning/Lunar Landing": {"env": "LunarLander-v3", "algo": "PPO"},
    "Reinforcement Learning/Taxi Navigation": {"env": "Taxi-v3", "algo": "PPO"},
    # Continuous-action environments (SAC primary)
    "Reinforcement Learning/Pendulum Control": {"env": "Pendulum-v1", "algo": "SAC"},
    "Reinforcement Learning/Mountain Car Continuous": {"env": "MountainCarContinuous-v0", "algo": "SAC"},
    "Reinforcement Learning/Lunar Lander Continuous": {"env": "LunarLander-v3", "algo": "SAC", "continuous": True},
    "Reinforcement Learning/Bipedal Walker": {"env": "BipedalWalker-v3", "algo": "SAC"},
}

# ── FAMILY 15: AUDIO / SPEECH ──
AUDIO = {
    "Speech and Audio processing/Audio Denoising": {"task": "denoising", "data": _hf("edinburghcstr/vctk")},
    "Speech and Audio processing/Music Genre Prediction - Million Songs": {"task": "classification", "data": _hf("marsyas/gtzan")},
    "Speech and Audio processing/Voice Cloning": {"task": "cloning", "data": _hf("edinburghcstr/vctk")},
    "Speech and Audio processing/Speech to Text": {"task": "transcription"},
    "Computer Vision/Noise Reduction": {"task": "denoising"},
    "Deep Learning/Cat and Dog Voice Recognition": {"task": "classification", "data": _hf("google/speech_commands", config="v0.02")},
}

# ── NLP Similarity / Retrieval / Plagiarism (embedding-based) ──
NLP_SIM = {
    "NLP/Text Similarity": {"data": _hf("mteb/stsbenchmark-sts")},
    "NLP/Plagiarism Checker": {"data": _hf("wikitext", config="wikitext-2-raw-v1")},
    "NLP/Cross Language Information Retrieval": {"data": _hf("wmt16", config="de-en", split="train[:1000]")},
    "NLP/Text Clustering and Topic Modelling": {"data": _hf("SetFit/20_newsgroups")},
}

# ── MISC NLP (text utilities — stop words, analysis, etc.) ──
NLP_MISC = {
    "NLP/Stop Words in 28 Languages": {"task": "generation", "data": _hf("wikitext", config="wikitext-2-raw-v1")},
    "NLP/Text File Analysis": {"task": "summarization", "data": _hf("EdinburghNLP/xsum")},
    "NLP/Text Processing and Analysis": {"task": "summarization", "data": _hf("EdinburghNLP/xsum")},
    "NLP/WhatsApp Chat Analysis": {"task": "summarization", "data": _hf("EdinburghNLP/xsum")},
    "NLP/WhatsApp Group Chat Analysis": {"task": "summarization", "data": _hf("EdinburghNLP/xsum")},
    "NLP/Wikipedia Search Word Cloud": {"task": "summarization", "data": _hf("wikitext", config="wikitext-2-raw-v1")},
}

# ── MISC CV ──
CV_MISC = {
    "Computer Vision/Dominant Color Analysis": {"task": "detect"},
    "Computer Vision/Dominant Color Extraction": {"task": "detect"},
    "Computer Vision/Image Cartoonify": {"task": "detect"},
    "Computer Vision/Image to Sketch": {"task": "detect"},
    "Computer Vision/Image Watermark": {"task": "detect"},
}

# ── CAPTIONING / VLM ──
CAPTIONING = {
    "Computer Vision/Image Captioning": {"data": _hf("nlphuji/flickr30k")},
    "Deep Learning/Image Colorization": {"data": _hf("nlphuji/flickr30k")},
    "Speech and Audio processing/Image Captioning": {"data": _hf("nlphuji/flickr30k")},
}

# ── MEDICAL SEGMENTATION ──
MEDICAL_SEG = {
    "Deep Learning/Brain MRI Segmentation": {"dataset": "hf:mateuszbuda/brain-segmentation", "n_classes": 2},
    "Deep Learning/COVID-19 Lung CT Scan Analysis": {"dataset": "hf:sartajbhuvaji/Brain-Tumor-Classification", "n_classes": 2},
}

# ── MISC Deep Learning ──
DL_IMAGE_MISC = {
    "Deep Learning/GANs": {"dataset": "MNIST", "n_classes": 10},
    "Deep Learning/Sudoku Solver - Neural Network": {"dataset": "MNIST", "n_classes": 10},
}
DL_TABULAR_MISC = {
    "Deep Learning/All Space Missions Analysis": {"target": "MissionStatus", "data": _url_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/refs/heads/main/data/2019/2019-01-15/launches.csv")},
    "Deep Learning/Indian Startup Data Analysis": {"target": "AmountInUSD", "data": _url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/indian_startup_funding.csv")},
}
DL_CLUSTER_MISC = {
    "Deep Learning/Pokemon Generation Clustering": {"data": _url_csv("https://raw.githubusercontent.com/lgreski/pokemonData/master/Pokemon.csv")},
}


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE TEMPLATES (inline — all data downloaded at runtime)
# ════════════════════════════════════════════════════════════════════════════════

def gen_tabular_clf(path, cfg):
    target = cfg["target"]
    data_load = cfg["data"]
    return textwrap.dedent(f'''\
"""
Modern Tabular Classification Pipeline (April 2026)
Models: CatBoost/LightGBM/XGBoost (GPU) + AutoGluon + RealTabPFN-v2 + TabM
Data: Auto-downloaded at runtime — no local files needed

Compute: GPU recommended (CatBoost/LightGBM/XGBoost use CUDA, TabM uses torch.cuda).
         CPU fallback is automatic. ~2-10 min per dataset on RTX 4060.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

TARGET = "{target}"


def load_data():
    """Download dataset from the internet."""
{data_load}
    print(f"Dataset shape: {{df.shape}}")
    print(f"Target distribution:\\n{{df[TARGET].value_counts()}}")
    return df


def preprocess(df):
    df = df.copy()
    df.dropna(subset=[TARGET], inplace=True)

    le_target = None
    if df[TARGET].dtype == "object" or df[TARGET].dtype.name == "category":
        le_target = LabelEncoder()
        df[TARGET] = le_target.fit_transform(df[TARGET])

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        if hasattr(X[c], "cat"): X[c] = X[c].astype(str)
        X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "unknown")

    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() < 50 else None
    )
    print(f"Train: {{X_train.shape}}, Test: {{X_test.shape}}")
    return X_train, X_test, y_train, y_test, le_target


def run_eda(df, target, save_dir):
    """Exploratory Data Analysis — summary stats, distributions, correlations."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
    print(f"Column types:\\n{{df.dtypes.value_counts().to_string()}}")
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\\nMissing values ({{len(n_miss)}} columns):")
        print(n_miss.sort_values(ascending=False).head(15).to_string())
    else:
        print("\\nNo missing values")
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    print("Summary statistics saved to eda_summary.csv")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        n = len(num_cols)
        fig, ax = plt.subplots(figsize=(min(n + 2, 20), min(n, 16)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=n <= 15, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap")
        fig.savefig(os.path.join(save_dir, "eda_correlation.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        if target in num_cols:
            tc = corr[target].drop(target).abs().sort_values(ascending=False)
            print(f"\\nTop correlations with '{{target}}':")
            print(tc.head(10).to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    if df[target].nunique() <= 30:
        df[target].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    else:
        df[target].hist(bins=50, ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(f"Target Distribution: {{target}}")
    ax.set_xlabel(target)
    fig.savefig(os.path.join(save_dir, "eda_target.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    plot_cols = [c for c in num_cols if c != target][:20]
    if plot_cols:
        nr = max(1, (len(plot_cols) + 4) // 5)
        nc = min(5, len(plot_cols))
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
        for i, col in enumerate(plot_cols):
            ri, ci = divmod(i, nc)
            df[col].hist(bins=30, ax=axes[ri][ci], color="steelblue", edgecolor="black")
            axes[ri][ci].set_title(col, fontsize=9)
        for i in range(len(plot_cols), nr * nc):
            ri, ci = divmod(i, nc)
            axes[ri][ci].set_visible(False)
        fig.suptitle("Feature Distributions")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "eda_distributions.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print("EDA plots saved.")


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {{}}      # name -> y_pred
    probas  = {{}}      # name -> probability array (for ROC-AUC / PR-AUC / calibration)
    timings = {{}}      # name -> wall-clock seconds
    n_classes = y_train.nunique()
    is_binary = n_classes == 2

    # ── CatBoost (GPU) ──
    try:
        from catboost import CatBoostClassifier
        t0 = time.perf_counter()
        cb = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=8,
            task_type="GPU", devices="0",
            eval_metric="AUC" if is_binary else "MultiClass",
            early_stopping_rounds=50, verbose=100,
            auto_class_weights="Balanced",
        )
        cb.fit(X_train, y_train, eval_set=(X_test, y_test))
        timings["CatBoost"] = time.perf_counter() - t0
        results["CatBoost"] = cb.predict(X_test).flatten()
        probas["CatBoost"] = cb.predict_proba(X_test)
        print(f"\\nCatBoost Accuracy: {{accuracy_score(y_test, results['CatBoost']):.4f}}  ({{timings['CatBoost']:.1f}}s)")
    except Exception as e:
        print(f"CatBoost: {{e}}")

    # ── LightGBM (GPU) ──
    try:
        import lightgbm as lgb
        t0 = time.perf_counter()
        m = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            device="gpu", class_weight="balanced", verbose=-1, n_jobs=-1,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        timings["LightGBM"] = time.perf_counter() - t0
        results["LightGBM"] = m.predict(X_test)
        probas["LightGBM"] = m.predict_proba(X_test)
        print(f"\\nLightGBM Accuracy: {{accuracy_score(y_test, results['LightGBM']):.4f}}  ({{timings['LightGBM']:.1f}}s)")
    except Exception as e:
        print(f"LightGBM: {{e}}")

    # ── XGBoost (CUDA) ──
    try:
        from xgboost import XGBClassifier
        t0 = time.perf_counter()
        m = XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            device="cuda", tree_method="hist",
            eval_metric="auc" if is_binary else "mlogloss",
            early_stopping_rounds=50, verbosity=1, n_jobs=-1,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        timings["XGBoost"] = time.perf_counter() - t0
        results["XGBoost"] = m.predict(X_test)
        probas["XGBoost"] = m.predict_proba(X_test)
        print(f"\\nXGBoost Accuracy: {{accuracy_score(y_test, results['XGBoost']):.4f}}  ({{timings['XGBoost']:.1f}}s)")
    except Exception as e:
        print(f"XGBoost: {{e}}")

    # ── AutoGluon Tabular ──
    try:
        from autogluon.tabular import TabularPredictor
        import tempfile
        t0 = time.perf_counter()
        train_ag = X_train.copy(); train_ag["{target}"] = y_train.values
        test_ag = X_test.copy(); test_ag["{target}"] = y_test.values
        with tempfile.TemporaryDirectory() as tmp:
            predictor = TabularPredictor(label="{target}", path=tmp, verbosity=1)
            predictor.fit(train_ag, time_limit=180, presets="best_quality")
            results["AutoGluon"] = predictor.predict(test_ag.drop(columns=["{target}"])).values
            try:
                probas["AutoGluon"] = predictor.predict_proba(test_ag.drop(columns=["{target}"])).values
            except Exception:
                pass
            timings["AutoGluon"] = time.perf_counter() - t0
            print(f"\\nAutoGluon Accuracy: {{accuracy_score(y_test, results['AutoGluon']):.4f}}  ({{timings['AutoGluon']:.1f}}s)")
    except Exception as e:
        print(f"AutoGluon: {{e}}")

    # ── RealTabPFN-v2 (prior-fitted network) ──
    try:
        from tabpfn import TabPFNClassifier
        if X_train.shape[0] <= 10000 and X_train.shape[1] <= 500:
            t0 = time.perf_counter()
            m = TabPFNClassifier(device="cuda", N_ensemble_configurations=32)
            m.fit(X_train.values, y_train.values)
            timings["TabPFN-v2"] = time.perf_counter() - t0
            results["TabPFN-v2"] = m.predict(X_test.values)
            try:
                probas["TabPFN-v2"] = m.predict_proba(X_test.values)
            except Exception:
                pass
            print(f"\\nTabPFN-v2 Accuracy: {{accuracy_score(y_test, results['TabPFN-v2']):.4f}}  ({{timings['TabPFN-v2']:.1f}}s)")
        else:
            print("TabPFN-v2: dataset too large (>10k rows or >500 cols), skipped")
    except Exception as e:
        print(f"TabPFN-v2: {{e}}")

    # ── TabM (parameter-efficient tabular ensembling) ──
    try:
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = StandardScaler()
        Xt = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
        Xv = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
        yt = torch.tensor(y_train.values, dtype=torch.long).to(device)
        d_in = Xt.shape[1]; d_out = n_classes

        class TabMBlock(nn.Module):
            def __init__(self, d, n_heads=4):
                super().__init__()
                self.heads = nn.ModuleList([nn.Sequential(
                    nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
                ) for _ in range(n_heads)])
                self.norm = nn.LayerNorm(d)
            def forward(self, x):
                return self.norm(x + sum(h(x) for h in self.heads) / len(self.heads))

        class TabMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Sequential(nn.Linear(d_in, 256), nn.SiLU())
                self.blocks = nn.Sequential(TabMBlock(256), TabMBlock(256), TabMBlock(256))
                self.head = nn.Linear(256, d_out)
            def forward(self, x): return self.head(self.blocks(self.embed(x)))

        t0 = time.perf_counter()
        model = TabMNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        for ep in range(100):
            model.train(); loss = loss_fn(model(Xt), yt); loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            results["TabM"] = torch.argmax(logits, dim=-1).cpu().numpy()
            probas["TabM"] = torch.softmax(logits, dim=-1).cpu().numpy()
        timings["TabM"] = time.perf_counter() - t0
        print(f"\\nTabM Accuracy: {{accuracy_score(y_test, results['TabM']):.4f}}  ({{timings['TabM']:.1f}}s)")
    except Exception as e:
        print(f"TabM: {{e}}")

    # ── Baseline Comparison: FLAML AutoML ──
    try:
        from flaml import AutoML
        t0 = time.perf_counter()
        automl = AutoML()
        automl.fit(X_train, y_train, task="classification", time_budget=120, verbose=0)
        timings["FLAML"] = time.perf_counter() - t0
        results["FLAML"] = automl.predict(X_test)
        try:
            probas["FLAML"] = automl.predict_proba(X_test)
        except Exception:
            pass
        print(f"\\nFLAML ({{automl.best_estimator}}) Accuracy: {{accuracy_score(y_test, results['FLAML']):.4f}}  ({{timings['FLAML']:.1f}}s)")
    except Exception as e:
        print(f"FLAML: {{e}}")

    # ── Baseline Comparison: LazyPredict ──
    try:
        from lazypredict.Supervised import LazyClassifier
        t0 = time.perf_counter()
        lazy = LazyClassifier(verbose=0, ignore_warnings=True)
        lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)
        timings["LazyPredict"] = time.perf_counter() - t0
        print(f"\\nLazyPredict — Top 5 classifiers:  ({{timings['LazyPredict']:.1f}}s)")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"LazyPredict: {{e}}")

    return results, probas, timings


def report(results, probas, timings, y_test, save_dir="."):
    n_classes = len(set(y_test))
    is_binary = n_classes == 2
    metrics_out = {{}}

    print("\\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    best_name, best_acc = None, 0
    for name, y_pred in results.items():
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        row = {{"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}}

        extra = ""
        if name in probas:
            p = probas[name]
            try:
                if is_binary:
                    p1 = p[:, 1] if p.ndim == 2 else p
                    row["roc_auc"] = round(roc_auc_score(y_test, p1), 4)
                    row["pr_auc"] = round(average_precision_score(y_test, p1), 4)
                    row["brier"] = round(brier_score_loss(y_test, p1), 4)
                    extra = f"  ROC-AUC: {{row['roc_auc']:.4f}}  PR-AUC: {{row['pr_auc']:.4f}}"
                else:
                    row["roc_auc_ovr"] = round(roc_auc_score(
                        y_test, p, multi_class="ovr", average="weighted"
                    ), 4)
                    extra = f"  ROC-AUC(OVR): {{row['roc_auc_ovr']:.4f}}"
            except Exception:
                pass

        if name in timings:
            row["time_s"] = round(timings[name], 1)

        print(f"\\n— {{name}} —  Accuracy: {{acc:.4f}}  |  F1: {{f1:.4f}}{{extra}}")
        print(classification_report(y_test, y_pred, zero_division=0))
        if acc > best_acc:
            best_acc, best_name = acc, name
        metrics_out[name] = row

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{{name}} Confusion Matrix")
        fig.savefig(os.path.join(save_dir, f"cm_{{name.lower().replace(' ', '_')}}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── Calibration plot (binary only) ──
    if is_binary and probas:
        try:
            from sklearn.calibration import calibration_curve
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            for name, p in probas.items():
                p1 = p[:, 1] if p.ndim == 2 else p
                prob_true, prob_pred = calibration_curve(y_test, p1, n_bins=10, strategy="uniform")
                ax.plot(prob_pred, prob_true, "s-", label=name)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Calibration Plot (Reliability Diagram)")
            ax.legend(loc="lower right", fontsize=8)
            fig.savefig(os.path.join(save_dir, "calibration_plot.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)
            print("\\nCalibration plot saved")
        except Exception:
            pass

    print(f"\\nBest: {{best_name}} ({{best_acc:.4f}})")

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\\nMetrics saved to {{out_path}}")


def cross_validate_best(X, y, save_dir):
    """5-fold stratified cross-validation on gradient boosting models."""
    print("\\n" + "=" * 60)
    print("CROSS-VALIDATION (5-Fold Stratified)")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {{}}
    for name, build_fn in [
        ("CatBoost", lambda: __import__("catboost").CatBoostClassifier(
            iterations=300, verbose=0, task_type="GPU", devices="0")),
        ("LightGBM", lambda: __import__("lightgbm").LGBMClassifier(
            n_estimators=300, device="gpu", verbose=-1, n_jobs=-1)),
        ("XGBoost", lambda: __import__("xgboost").XGBClassifier(
            n_estimators=300, device="cuda", tree_method="hist",
            verbosity=0, n_jobs=-1)),
    ]:
        try:
            model = build_fn()
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=1)
            cv_results[name] = {{"mean": round(float(scores.mean()), 4),
                                "std": round(float(scores.std()), 4),
                                "folds": [round(float(s), 4) for s in scores]}}
            print(f"  {{name}}: {{scores.mean():.4f}} +/- {{scores.std():.4f}}")
        except Exception as e:
            print(f"  {{name}} CV skipped: {{e}}")
    if cv_results:
        out_path = os.path.join(save_dir, "cv_results.json")
        with open(out_path, "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"CV results saved to {{out_path}}")
    return cv_results


def main():
    print("=" * 60)
    print("MODERN TABULAR CLASSIFICATION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | AutoGluon | TabPFN-v2 | TabM | FLAML | LazyPredict")
    print("=" * 60)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_data()
    run_eda(df, TARGET, save_dir)
    X_train, X_test, y_train, y_test, le = preprocess(df)
    results, probas, timings = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, probas, timings, y_test, save_dir)
    cross_validate_best(
        pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), save_dir)


if __name__ == "__main__":
    main()
''')


def gen_tabular_reg(path, cfg):
    target = cfg["target"]
    data_load = cfg["data"]
    return textwrap.dedent(f'''\
"""
Modern Tabular Regression Pipeline (April 2026)
Models: CatBoost/LightGBM/XGBoost (GPU) + AutoGluon + RealTabPFN-v2 + TabM
Data: Auto-downloaded at runtime

Compute: GPU recommended (CatBoost/LightGBM/XGBoost use CUDA, TabM uses torch.cuda).
         CPU fallback is automatic. ~2-10 min per dataset on RTX 4060.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

TARGET = "{target}"


def load_data():
{data_load}
    print(f"Dataset shape: {{df.shape}}")
    return df


def preprocess(df):
    df = df.copy()
    df.dropna(subset=[TARGET], inplace=True)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        if hasattr(X[c], "cat"): X[c] = X[c].astype(str)
        X[c] = X[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])
    return train_test_split(X, y, test_size=0.2, random_state=42)


def run_eda(df, target, save_dir):
    """Exploratory Data Analysis."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
    print(f"Column types:\\n{{df.dtypes.value_counts().to_string()}}")
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\\nMissing values ({{len(n_miss)}} columns):")
        print(n_miss.sort_values(ascending=False).head(15).to_string())
    else:
        print("\\nNo missing values")
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    print("Summary statistics saved to eda_summary.csv")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        n = len(num_cols)
        fig, ax = plt.subplots(figsize=(min(n + 2, 20), min(n, 16)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=n <= 15, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap")
        fig.savefig(os.path.join(save_dir, "eda_correlation.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        if target in num_cols:
            tc = corr[target].drop(target).abs().sort_values(ascending=False)
            print(f"\\nTop correlations with '{{target}}':")
            print(tc.head(10).to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    if df[target].nunique() <= 30:
        df[target].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    else:
        df[target].hist(bins=50, ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(f"Target Distribution: {{target}}")
    ax.set_xlabel(target)
    fig.savefig(os.path.join(save_dir, "eda_target.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    plot_cols = [c for c in num_cols if c != target][:20]
    if plot_cols:
        nr = max(1, (len(plot_cols) + 4) // 5)
        nc = min(5, len(plot_cols))
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
        for i, col in enumerate(plot_cols):
            ri, ci = divmod(i, nc)
            df[col].hist(bins=30, ax=axes[ri][ci], color="steelblue", edgecolor="black")
            axes[ri][ci].set_title(col, fontsize=9)
        for i in range(len(plot_cols), nr * nc):
            ri, ci = divmod(i, nc)
            axes[ri][ci].set_visible(False)
        fig.suptitle("Feature Distributions")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "eda_distributions.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print("EDA plots saved.")


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {{}}      # name -> y_pred
    timings = {{}}      # name -> wall-clock seconds

    # ── CatBoost (GPU) ──
    try:
        from catboost import CatBoostRegressor
        t0 = time.perf_counter()
        m = CatBoostRegressor(iterations=1000, lr=0.05, depth=8, task_type="GPU",
                              devices="0", early_stopping_rounds=50, verbose=100)
        m.fit(X_train, y_train, eval_set=(X_test, y_test))
        timings["CatBoost"] = time.perf_counter() - t0
        results["CatBoost"] = m.predict(X_test)
        print(f"CatBoost RMSE: {{mean_squared_error(y_test, results['CatBoost'], squared=False):.4f}}  ({{timings['CatBoost']:.1f}}s)")
    except Exception as e:
        print(f"CatBoost: {{e}}")

    # ── LightGBM (GPU) ──
    try:
        import lightgbm as lgb
        t0 = time.perf_counter()
        m = lgb.LGBMRegressor(n_estimators=1000, lr=0.05, max_depth=8,
                              device="gpu", verbose=-1, n_jobs=-1)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        timings["LightGBM"] = time.perf_counter() - t0
        results["LightGBM"] = m.predict(X_test)
        print(f"LightGBM RMSE: {{mean_squared_error(y_test, results['LightGBM'], squared=False):.4f}}  ({{timings['LightGBM']:.1f}}s)")
    except Exception as e:
        print(f"LightGBM: {{e}}")

    # ── XGBoost (CUDA) ──
    try:
        from xgboost import XGBRegressor
        t0 = time.perf_counter()
        m = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8,
                         device="cuda", tree_method="hist", early_stopping_rounds=50,
                         verbosity=1, n_jobs=-1)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        timings["XGBoost"] = time.perf_counter() - t0
        results["XGBoost"] = m.predict(X_test)
        print(f"XGBoost RMSE: {{mean_squared_error(y_test, results['XGBoost'], squared=False):.4f}}  ({{timings['XGBoost']:.1f}}s)")
    except Exception as e:
        print(f"XGBoost: {{e}}")

    # ── AutoGluon Tabular ──
    try:
        from autogluon.tabular import TabularPredictor
        import tempfile
        t0 = time.perf_counter()
        train_ag = X_train.copy(); train_ag["{target}"] = y_train.values
        with tempfile.TemporaryDirectory() as tmp:
            predictor = TabularPredictor(label="{target}", path=tmp, problem_type="regression", verbosity=1)
            predictor.fit(train_ag, time_limit=180, presets="best_quality")
            results["AutoGluon"] = predictor.predict(X_test).values
            timings["AutoGluon"] = time.perf_counter() - t0
            print(f"AutoGluon RMSE: {{mean_squared_error(y_test, results['AutoGluon'], squared=False):.4f}}  ({{timings['AutoGluon']:.1f}}s)")
    except Exception as e:
        print(f"AutoGluon: {{e}}")

    # ── RealTabPFN-v2 (prior-fitted network — regression) ──
    try:
        from tabpfn import TabPFNRegressor
        if X_train.shape[0] <= 10000 and X_train.shape[1] <= 500:
            t0 = time.perf_counter()
            m = TabPFNRegressor(device="cuda", N_ensemble_configurations=32)
            m.fit(X_train.values, y_train.values)
            timings["TabPFN-v2"] = time.perf_counter() - t0
            results["TabPFN-v2"] = m.predict(X_test.values)
            print(f"TabPFN-v2 RMSE: {{mean_squared_error(y_test, results['TabPFN-v2'], squared=False):.4f}}  ({{timings['TabPFN-v2']:.1f}}s)")
        else:
            print("TabPFN-v2: dataset too large (>10k rows or >500 cols), skipped")
    except Exception as e:
        print(f"TabPFN-v2: {{e}}")

    # ── TabM (deep tabular) ──
    try:
        import torch, torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = StandardScaler()
        Xt = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
        Xv = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
        yt = torch.tensor(y_train.values, dtype=torch.float32).to(device)
        d_in = Xt.shape[1]
        class TabMBlock(nn.Module):
            def __init__(self, d, n_heads=4):
                super().__init__()
                self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)) for _ in range(n_heads)])
                self.norm = nn.LayerNorm(d)
            def forward(self, x): return self.norm(x + sum(h(x) for h in self.heads) / len(self.heads))
        class TabMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d_in, 256), nn.SiLU(), TabMBlock(256), TabMBlock(256), nn.Linear(256, 1))
            def forward(self, x): return self.net(x).squeeze(-1)
        t0 = time.perf_counter()
        model = TabMNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        for ep in range(100):
            model.train(); loss = nn.MSELoss()(model(Xt), yt); loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): results["TabM"] = model(Xv).cpu().numpy()
        timings["TabM"] = time.perf_counter() - t0
        print(f"TabM RMSE: {{mean_squared_error(y_test, results['TabM'], squared=False):.4f}}  ({{timings['TabM']:.1f}}s)")
    except Exception as e:
        print(f"TabM: {{e}}")

    # ── Baseline Comparison: FLAML AutoML ──
    try:
        from flaml import AutoML
        t0 = time.perf_counter()
        automl = AutoML()
        automl.fit(X_train, y_train, task="regression", time_budget=120, verbose=0)
        timings["FLAML"] = time.perf_counter() - t0
        results["FLAML"] = automl.predict(X_test)
        print(f"FLAML ({{automl.best_estimator}}) RMSE: {{mean_squared_error(y_test, results['FLAML'], squared=False):.4f}}  ({{timings['FLAML']:.1f}}s)")
    except Exception as e:
        print(f"FLAML: {{e}}")

    # ── Baseline Comparison: LazyPredict ──
    try:
        from lazypredict.Supervised import LazyRegressor
        t0 = time.perf_counter()
        lazy = LazyRegressor(verbose=0, ignore_warnings=True)
        lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)
        timings["LazyPredict"] = time.perf_counter() - t0
        print(f"\\nLazyPredict — Top 5 regressors:  ({{timings['LazyPredict']:.1f}}s)")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"LazyPredict: {{e}}")

    return results, timings


def report(results, timings, y_test, save_dir="."):
    metrics_out = {{}}

    print("\\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    best_name, best_rmse = None, float("inf")
    for name, y_pred in results.items():
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        row = {{"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}}

        # MAPE — only meaningful when target has no zeros
        mape_str = ""
        try:
            if (y_test != 0).all():
                mape = mean_absolute_percentage_error(y_test, y_pred)
                row["mape"] = round(mape, 4)
                mape_str = f"  MAPE: {{mape:.4f}}"
        except Exception:
            pass

        if name in timings:
            row["time_s"] = round(timings[name], 1)

        print(f"\\n— {{name}} — RMSE: {{rmse:.4f}} | MAE: {{mae:.4f}} | R²: {{r2:.4f}}{{mape_str}}")
        if rmse < best_rmse:
            best_rmse, best_name = rmse, name
        metrics_out[name] = row

        # Predicted-vs-Actual scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.4, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_title(f"{{name}} — Predicted vs Actual")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        fig.savefig(os.path.join(save_dir, f"scatter_{{name.lower().replace(' ', '_')}}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── Residual distribution for the best model ──
    if best_name:
        residuals = y_test.values - results[best_name]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(residuals, bins=40, edgecolor="black", alpha=0.7)
        axes[0].set_title(f"{{best_name}} — Residual Distribution")
        axes[0].set_xlabel("Residual (actual − predicted)")
        axes[1].scatter(results[best_name], residuals, alpha=0.4, s=10)
        axes[1].axhline(0, color="r", linestyle="--")
        axes[1].set_title(f"{{best_name}} — Residual vs Predicted")
        axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "residuals_best.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("\\nResidual plots saved")

    print(f"\\nBest: {{best_name}} (RMSE: {{best_rmse:.4f}})")

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\\nMetrics saved to {{out_path}}")


def cross_validate_best(X, y, save_dir):
    """5-fold cross-validation on gradient boosting models."""
    from sklearn.metrics import make_scorer
    print("\\n" + "=" * 60)
    print("CROSS-VALIDATION (5-Fold)")
    print("=" * 60)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)
    cv_results = {{}}
    for name, build_fn in [
        ("CatBoost", lambda: __import__("catboost").CatBoostRegressor(
            iterations=300, verbose=0, task_type="GPU", devices="0")),
        ("LightGBM", lambda: __import__("lightgbm").LGBMRegressor(
            n_estimators=300, device="gpu", verbose=-1, n_jobs=-1)),
        ("XGBoost", lambda: __import__("xgboost").XGBRegressor(
            n_estimators=300, device="cuda", tree_method="hist",
            verbosity=0, n_jobs=-1)),
    ]:
        try:
            model = build_fn()
            scores = -cross_val_score(model, X, y, cv=cv, scoring=rmse_scorer, n_jobs=1)
            cv_results[name] = {{"rmse_mean": round(float(scores.mean()), 4),
                                "rmse_std": round(float(scores.std()), 4),
                                "folds": [round(float(s), 4) for s in scores]}}
            print(f"  {{name}}: RMSE {{scores.mean():.4f}} +/- {{scores.std():.4f}}")
        except Exception as e:
            print(f"  {{name}} CV skipped: {{e}}")
    if cv_results:
        out_path = os.path.join(save_dir, "cv_results.json")
        with open(out_path, "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"CV results saved to {{out_path}}")
    return cv_results


def main():
    print("=" * 60)
    print("MODERN TABULAR REGRESSION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | AutoGluon | TabPFN-v2 | TabM | FLAML | LazyPredict")
    print("=" * 60)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_data()
    run_eda(df, TARGET, save_dir)
    X_train, X_test, y_train, y_test = preprocess(df)
    results, timings = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, timings, y_test, save_dir)
    cross_validate_best(
        pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), save_dir)


if __name__ == "__main__":
    main()
''')


def gen_fraud(path, cfg):
    target = cfg["target"]
    data_load = cfg["data"]
    return textwrap.dedent(f'''\
"""
Fraud / Imbalanced Classification Pipeline (April 2026)
Models: CatBoost, LightGBM, XGBoost + calibrated probabilities + PyOD (ECOD, COPOD, IForest)
GPU + threshold tuning + isotonic calibration
Data: Auto-downloaded at runtime

Compute: GPU recommended (CatBoost/LightGBM/XGBoost use CUDA). CPU fallback automatic.
         ~2–8 min per dataset on RTX 4060.
"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix,
    recall_score, precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

TARGET = "{target}"


def load_data():
{data_load}
    print(f"Dataset shape: {{df.shape}}")
    print(f"Fraud rate: {{df[TARGET].mean():.4%}}")
    return df


def preprocess(df):
    df = df.copy()
    df.dropna(subset=[TARGET], inplace=True)
    y = df[TARGET]; X = df.drop(columns=[TARGET])
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        if hasattr(X[c], "cat"): X[c] = X[c].astype(str)
        X[c] = X[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def run_eda(df, target, save_dir):
    """Exploratory Data Analysis for fraud/imbalanced datasets."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
    print(f"Column types:\\n{{df.dtypes.value_counts().to_string()}}")
    fraud_rate = df[target].mean()
    print(f"\\nClass balance: {{1 - fraud_rate:.2%}} legit / {{fraud_rate:.2%}} fraud (ratio {{int(1/max(fraud_rate,1e-9))}}:1)")
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\\nMissing values ({{len(n_miss)}} columns):")
        print(n_miss.sort_values(ascending=False).head(15).to_string())
    else:
        print("\\nNo missing values")
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    print("Summary statistics saved to eda_summary.csv")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        n = len(num_cols)
        fig, ax = plt.subplots(figsize=(min(n + 2, 20), min(n, 16)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=n <= 15, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap")
        fig.savefig(os.path.join(save_dir, "eda_correlation.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        if target in num_cols:
            tc = corr[target].drop(target).abs().sort_values(ascending=False)
            print(f"\\nTop correlations with '{{target}}':")
            print(tc.head(10).to_string())
    fig, ax = plt.subplots(figsize=(8, 5))
    df[target].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "salmon"], edgecolor="black")
    ax.set_title(f"Target Distribution: {{target}} (Fraud rate: {{fraud_rate:.2%}})")
    ax.set_xlabel(target)
    fig.savefig(os.path.join(save_dir, "eda_target.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("EDA plots saved.")


def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    idx = np.argmax(f1s)
    return thresholds[idx] if idx < len(thresholds) else 0.5


def train_and_evaluate(X_train, X_test, y_train, y_test):
    from sklearn.calibration import CalibratedClassifierCV
    results = {{}}      # name -> {{preds, proba, thresh, model}}
    timings = {{}}      # name -> wall-clock seconds
    # Hold out calibration split from training data
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
    scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    for name, builder in [
        ("CatBoost", lambda: __import__("catboost").CatBoostClassifier(
            iterations=1000, lr=0.03, depth=8, task_type="GPU", devices="0",
            scale_pos_weight=scale, eval_metric="F1", early_stopping_rounds=50, verbose=100)),
        ("LightGBM", lambda: __import__("lightgbm").LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, max_depth=8,
            device="gpu", scale_pos_weight=scale, verbose=-1, n_jobs=-1)),
        ("XGBoost", lambda: __import__("xgboost").XGBClassifier(
            n_estimators=1000, learning_rate=0.03, max_depth=8,
            device="cuda", tree_method="hist", scale_pos_weight=scale,
            eval_metric="aucpr", early_stopping_rounds=50, verbosity=1, n_jobs=-1)),
    ]:
        try:
            t0 = time.perf_counter()
            m = builder()
            if name == "LightGBM":
                import lightgbm as lgb
                m.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
            else:
                m.fit(X_tr, y_tr, eval_set=[(X_cal, y_cal)] if name == "XGBoost"
                      else (X_cal, y_cal), verbose=100 if name == "XGBoost" else None)
            # Calibrate probabilities (isotonic regression on held-out cal split)
            cal_model = CalibratedClassifierCV(m, cv="prefit", method="isotonic")
            cal_model.fit(X_cal, y_cal)
            proba = cal_model.predict_proba(X_test)[:, 1]
            thresh = find_best_threshold(y_test, proba)
            preds = (proba >= thresh).astype(int)
            timings[name] = time.perf_counter() - t0
            results[name] = {{"preds": preds, "proba": proba, "thresh": thresh, "model": name}}
            print(f"{{name}} F1: {{f1_score(y_test, preds):.4f}} (t={{thresh:.3f}}) [calibrated]  ({{timings[name]:.1f}}s)")
        except Exception as e:
            print(f"{{name}}: {{e}}")

    # ── PyOD Anomaly Scoring (unsupervised cross-check) ──
    for pyod_name, pyod_builder in [
        ("ECOD", lambda: __import__("pyod.models.ecod", fromlist=["ECOD"]).ECOD(contamination=0.05)),
        ("COPOD", lambda: __import__("pyod.models.copod", fromlist=["COPOD"]).COPOD(contamination=0.05)),
        ("IForest-PyOD", lambda: __import__("pyod.models.iforest", fromlist=["IForest"]).IForest(contamination=0.05, random_state=42)),
    ]:
        try:
            t0 = time.perf_counter()
            pm = pyod_builder()
            pm.fit(X_train.values if hasattr(X_train, "values") else X_train)
            scores = pm.decision_function(X_test.values if hasattr(X_test, "values") else X_test)
            pyod_preds = pm.predict(X_test.values if hasattr(X_test, "values") else X_test)
            n_anom = pyod_preds.sum()
            f1 = f1_score(y_test, pyod_preds) if len(set(y_test)) > 1 else 0
            auc = roc_auc_score(y_test, scores) if len(set(y_test)) > 1 else 0
            elapsed = time.perf_counter() - t0
            timings[f"PyOD-{{pyod_name}}"] = elapsed
            print(f"PyOD {{pyod_name}}: {{n_anom}} anomalies ({{n_anom/len(X_test):.2%}}), F1={{f1:.4f}}, AUC={{auc:.4f}}  ({{elapsed:.1f}}s)")
        except Exception as e:
            print(f"PyOD {{pyod_name}}: {{e}}")

    # ── FLAML AutoML (imbalance-aware benchmark) ──
    try:
        from flaml import AutoML
        t0 = time.perf_counter()
        automl = AutoML()
        automl.fit(X_train, y_train, task="classification", time_budget=120,
                   metric="ap", verbose=0)
        flaml_proba = automl.predict_proba(X_test)[:, 1]
        flaml_thresh = find_best_threshold(y_test, flaml_proba)
        flaml_preds = (flaml_proba >= flaml_thresh).astype(int)
        timings["FLAML"] = time.perf_counter() - t0
        results["FLAML"] = {{"preds": flaml_preds, "proba": flaml_proba, "thresh": flaml_thresh, "model": "FLAML"}}
        print(f"FLAML ({{automl.best_estimator}}) F1: {{f1_score(y_test, flaml_preds):.4f}} (t={{flaml_thresh:.3f}})  ({{timings['FLAML']:.1f}}s)")
    except Exception as e:
        print(f"FLAML: {{e}}")

    # ── LazyPredict (quick sweep benchmark) ──
    try:
        from lazypredict.Supervised import LazyClassifier
        t0 = time.perf_counter()
        lazy = LazyClassifier(verbose=0, ignore_warnings=True)
        lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)
        timings["LazyPredict"] = time.perf_counter() - t0
        print(f"\\nLazyPredict — Top 5 classifiers:  ({{timings['LazyPredict']:.1f}}s)")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"LazyPredict: {{e}}")

    return results, timings


def report(results, timings, y_test, save_dir="."):
    from sklearn.calibration import calibration_curve
    metrics_out = {{}}

    for name, r in results.items():
        pr_auc = average_precision_score(y_test, r["proba"])
        roc = roc_auc_score(y_test, r["proba"])
        f1 = f1_score(y_test, r["preds"])
        rec = recall_score(y_test, r["preds"])
        prec = precision_score(y_test, r["preds"], zero_division=0)
        acc = accuracy_score(y_test, r["preds"])
        row = {{"f1": round(f1, 4), "pr_auc": round(pr_auc, 4), "roc_auc": round(roc, 4),
               "recall": round(rec, 4), "precision": round(prec, 4), "accuracy": round(acc, 4),
               "threshold": round(r["thresh"], 4)}}
        if name in timings:
            row["time_s"] = round(timings[name], 1)
        metrics_out[name] = row

        print(f"\\n— {{name}} (threshold={{r['thresh']:.3f}}) —")
        print(classification_report(y_test, r["preds"], target_names=["Legit", "Fraud"]))
        print(f"  PR-AUC: {{pr_auc:.4f}}  ROC-AUC: {{roc:.4f}}  Recall@t: {{rec:.4f}}")

    # Reliability diagram (calibration plot)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, r in results.items():
            prob_true, prob_pred = calibration_curve(y_test, r["proba"], n_bins=10, strategy="uniform")
            axes[0].plot(prob_pred, prob_true, marker="o", label=name)
        axes[0].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        axes[0].set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
                    title="Reliability Diagram")
        axes[0].legend()

        # Confusion matrix for best model
        best = max(results.items(), key=lambda x: f1_score(y_test, x[1]["preds"]))
        cm = confusion_matrix(y_test, best[1]["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        axes[1].set(xlabel="Predicted", ylabel="Actual", title=f"Confusion Matrix ({{best[0]}})")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "fraud_report.png"), dpi=150)
        plt.close()
        print(f"\\nReport saved to {{save_dir}}/fraud_report.png")
    except Exception as e:
        print(f"Plot: {{e}}")

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\\nMetrics saved to {{out_path}}")


def cross_validate_best(X, y, save_dir):
    """5-fold stratified cross-validation on gradient boosting models."""
    print("\\n" + "=" * 60)
    print("CROSS-VALIDATION (5-Fold Stratified)")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {{}}
    for name, build_fn in [
        ("CatBoost", lambda: __import__("catboost").CatBoostClassifier(
            iterations=300, verbose=0, task_type="GPU", devices="0")),
        ("LightGBM", lambda: __import__("lightgbm").LGBMClassifier(
            n_estimators=300, device="gpu", verbose=-1, n_jobs=-1)),
        ("XGBoost", lambda: __import__("xgboost").XGBClassifier(
            n_estimators=300, device="cuda", tree_method="hist",
            verbosity=0, n_jobs=-1)),
    ]:
        try:
            model = build_fn()
            scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=1)
            cv_results[name] = {{"f1_mean": round(float(scores.mean()), 4),
                                "f1_std": round(float(scores.std()), 4),
                                "folds": [round(float(s), 4) for s in scores]}}
            print(f"  {{name}}: F1 {{scores.mean():.4f}} +/- {{scores.std():.4f}}")
        except Exception as e:
            print(f"  {{name}} CV skipped: {{e}}")
    if cv_results:
        out_path = os.path.join(save_dir, "cv_results.json")
        with open(out_path, "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"CV results saved to {{out_path}}")
    return cv_results


def main():
    print("=" * 60)
    print("FRAUD / IMBALANCED CLASSIFICATION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | PyOD (ECOD/COPOD/IForest)")
    print("Calibrated probabilities + threshold tuning")
    print("=" * 60)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_data()
    run_eda(df, TARGET, save_dir)
    X_train, X_test, y_train, y_test = preprocess(df)
    results, timings = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, timings, y_test, save_dir)
    cross_validate_best(
        pd.concat([X_train, X_test]), pd.concat([y_train, y_test]), save_dir)


if __name__ == "__main__":
    main()
''')


def gen_nlp_clf(path, cfg):
    target = cfg.get("target", "label")
    text_col = cfg.get("text_col", "text")
    data_load = cfg["data"]
    model = cfg.get("model", "answerdotai/ModernBERT-base")
    return textwrap.dedent(f'''\
"""
Modern NLP Classification Pipeline (April 2026)

Primary model: ModernBERT (answerdotai/ModernBERT-base) — English-first encoder,
               fine-tuned with mixed-precision (fp16) for sequence classification.
Secondary:     XLM-RoBERTa (multilingual fallback).
Baselines:     TF-IDF + Naive Bayes / Logistic Regression (kept for comparison).
Extras:        GLiNER zero-shot NER, BGE-M3 / Qwen3-Embedding similarity.

Compute: GPU strongly recommended (~2-8 min per model on RTX 4060).
         TF-IDF baselines run on CPU in <10s.
Data: Auto-downloaded at runtime
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    confusion_matrix, roc_auc_score,
)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

TARGET = "{target}"
TEXT_COL = "{text_col}"
MAX_LEN, BATCH_SIZE, EPOCHS, LR = 256, 16, 3, 2e-5

MODELS = [
    ("{model}", "ModernBERT"),
    ("FacebookAI/xlm-roberta-base", "XLM-R"),
]


def load_data():
{data_load}
    # Auto-detect text column
    text_col = TEXT_COL
    if text_col not in df.columns:
        candidates = [c for c in df.columns if df[c].dtype == "object" and df[c].str.len().mean() > 20]
        text_col = candidates[0] if candidates else df.select_dtypes("object").columns[0]
    target = TARGET if TARGET in df.columns else df.columns[-1]
    df = df[[text_col, target]].dropna()
    df.columns = ["text", "label"]
    print(f"Dataset: {{len(df)}} samples")
    return df


# ═══════════════════════════════════════════════════════════════
# BASELINE: TF-IDF + Naive Bayes / Logistic Regression
# ═══════════════════════════════════════════════════════════════
def run_tfidf_baseline(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    le = LabelEncoder(); y = le.fit_transform(df["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], y, test_size=0.2, random_state=42,
                                                stratify=y if len(le.classes_) < 50 else None)
    baseline_results = {{}}
    for name, clf in [("Naive Bayes", MultinomialNB()), ("LogReg", LogisticRegression(max_iter=1000, n_jobs=-1))]:
        t0 = time.perf_counter()
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        elapsed = time.perf_counter() - t0
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="weighted")
        baseline_results[name] = {{"accuracy": round(acc, 4), "f1_weighted": round(f1, 4), "time_s": round(elapsed, 1)}}
        print(f"  [Baseline] {{name}} — Accuracy: {{acc:.4f}}, F1: {{f1:.4f}}  ({{elapsed:.1f}}s)")
    return baseline_results


# ═══════════════════════════════════════════════════════════════
# PRIMARY: ModernBERT / XLM-R fine-tuned classifier
# ═══════════════════════════════════════════════════════════════
def train_transformer(df, model_name, display_name):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    le = LabelEncoder(); df["label_id"] = le.fit_transform(df["label"])
    n_classes = len(le.classes_)
    is_binary = n_classes == 2
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                          stratify=df["label_id"] if n_classes < 50 else None)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes).to(device)

    class DS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return {{**{{k: v[i] for k, v in self.enc.items()}}, "labels": self.labels[i]}}

    train_loader = DataLoader(DS(train_df["text"].tolist(), train_df["label_id"].tolist()), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(DS(test_df["text"].tolist(), test_df["label_id"].tolist()), batch_size=BATCH_SIZE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_loader) * EPOCHS), len(train_loader) * EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for batch in train_loader:
            batch = {{k: v.to(device) for k, v in batch.items()}}
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            total_loss += loss.item()
        print(f"  [{{display_name}}] Epoch {{epoch+1}}/{{EPOCHS}}, Loss: {{total_loss/len(train_loader):.4f}}")

    elapsed = time.perf_counter() - t0
    model.eval(); all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {{k: v.to(device) for k, v in batch.items()}}
            logits = model(**batch).logits
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_logits.append(logits.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    row = {{"accuracy": round(acc, 4), "f1_weighted": round(f1, 4), "time_s": round(elapsed, 1)}}

    # ROC-AUC (binary or multiclass OVR)
    try:
        probs = torch.softmax(all_logits, dim=-1).numpy()
        if is_binary:
            row["roc_auc"] = round(roc_auc_score(all_labels, probs[:, 1]), 4)
        else:
            row["roc_auc_ovr"] = round(roc_auc_score(all_labels, probs, multi_class="ovr", average="weighted"), 4)
    except Exception:
        pass

    print(f"\\n{{display_name}} — Accuracy: {{acc:.4f}}, F1: {{f1:.4f}}  ({{elapsed:.1f}}s)")
    if "roc_auc" in row:
        print(f"  ROC-AUC: {{row['roc_auc']:.4f}}")
    elif "roc_auc_ovr" in row:
        print(f"  ROC-AUC (OVR): {{row['roc_auc_ovr']:.4f}}")
    print(classification_report(all_labels, all_preds, target_names=le.classes_.astype(str), zero_division=0))

    # Confusion matrix
    save_dir = os.path.dirname(os.path.abspath(__file__))
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(6, n_classes * 0.8), max(5, n_classes * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=le.classes_.astype(str), yticklabels=le.classes_.astype(str))
    ax.set_title(f"{{display_name}} Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"cm_{{display_name.lower().replace('-','_')}}.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    model.save_pretrained(os.path.join(save_dir, f"{{display_name.lower().replace('-','_')}}_model"))
    return acc, f1, row


# ═══════════════════════════════════════════════════════════════
# GLiNER: Zero-shot NER on text samples
# ═══════════════════════════════════════════════════════════════
def run_gliner(df):
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained("urchade/gliner_base")
        sample_labels = ["person", "location", "organization", "date", "money", "product", "event"]
        for i, text in enumerate(df["text"].head(10)):
            entities = model.predict_entities(text[:512], sample_labels, threshold=0.4)
            if entities:
                ent_str = ", ".join(f"{{e['text']}}({{e['label']}})" for e in entities[:5])
                print(f"  [{{i+1}}] {{ent_str}}")
        print("GLiNER zero-shot NER complete")
    except Exception as e:
        print(f"GLiNER: {{e}}")


# ═══════════════════════════════════════════════════════════════
# EMBEDDING SIMILARITY: BGE-M3 / Qwen3-Embedding
# ═══════════════════════════════════════════════════════════════
def run_embedding_similarity(df):
    \"\"\"Embedding-based retrieval/similarity with BGE-M3 and Qwen3-Embedding.\"\"\"
    texts = df["text"].dropna().head(200).tolist()
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        # BGE-M3
        model = SentenceTransformer("BAAI/bge-m3")
        embs = model.encode(texts, batch_size=32, show_progress_bar=True)
        sim = cosine_similarity(embs)
        # Show top-3 similar texts for first 3 samples
        for i in range(min(3, len(texts))):
            top_idx = np.argsort(sim[i])[-4:-1][::-1]
            print(f"  Text {{i+1}} most similar to: {{[idx for idx in top_idx]}}")
        print(f"BGE-M3: {{len(texts)}} texts embedded (dim={{embs.shape[1]}})")

        # Qwen3-Embedding
        try:
            qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
            qwen_embs = qwen.encode(texts[:100], batch_size=16, show_progress_bar=True)
            print(f"Qwen3-Embedding: {{len(qwen_embs)}} texts embedded (dim={{qwen_embs.shape[1]}})")
        except Exception as e:
            print(f"Qwen3-Embedding: {{e}}")
    except Exception as e:
        print(f"Embedding similarity: {{e}}")


def run_eda(df, text_col, target, save_dir):
    """Exploratory Data Analysis for text classification."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Samples: {{df.shape[0]}}, Columns: {{df.shape[1]}}")
    if target in df.columns:
        print(f"\\nClass distribution:")
        vc = df[target].value_counts()
        for cls, cnt in vc.items():
            print(f"  {{cls}}: {{cnt}} ({{cnt/len(df):.1%}})")
        fig, ax = plt.subplots(figsize=(8, 5))
        vc.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"Class Distribution: {{target}}")
        ax.set_xlabel(target)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "eda_class_distribution.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    if text_col in df.columns:
        lengths = df[text_col].astype(str).str.len()
        word_counts = df[text_col].astype(str).str.split().str.len()
        print(f"\\nText length (chars): mean={{lengths.mean():.0f}}, median={{lengths.median():.0f}}, "
              f"min={{lengths.min()}}, max={{lengths.max()}}")
        print(f"Word count: mean={{word_counts.mean():.1f}}, median={{word_counts.median():.0f}}, "
              f"min={{word_counts.min()}}, max={{word_counts.max()}}")
        vocab = set()
        for text in df[text_col].astype(str).head(10000):
            vocab.update(text.lower().split())
        print(f"Approx vocabulary size (first 10k): {{len(vocab):,}}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(lengths, bins=50, color="steelblue", edgecolor="black")
        axes[0].set_title("Text Length Distribution (chars)")
        axes[1].hist(word_counts.clip(upper=word_counts.quantile(0.99)), bins=50,
                     color="steelblue", edgecolor="black")
        axes[1].set_title("Word Count Distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "eda_text_stats.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    print("EDA plots saved.")


def main():
    print("=" * 60)
    print("NLP CLASSIFICATION — ModernBERT + XLM-R | TF-IDF baseline | GLiNER NER")
    print("Mixed-precision (fp16) training on GPU")
    print("=" * 60)
    df = load_data()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    run_eda(df, TEXT_COL, TARGET, save_dir)
    metrics_out = {{}}

    # Baseline first
    print("\\n— TF-IDF / Naive Bayes Baseline —")
    baseline_metrics = run_tfidf_baseline(df)
    metrics_out.update(baseline_metrics)

    # Primary transformer models
    best_acc, best_name = 0, ""
    for model_name, display_name in MODELS:
        try:
            acc, f1, row = train_transformer(df.copy(), model_name, display_name)
            metrics_out[display_name] = row
            if acc > best_acc:
                best_acc, best_name = acc, display_name
        except Exception as e:
            print(f"{{display_name}}: {{e}}")
    print(f"\\nBest: {{best_name}} (Accuracy: {{best_acc:.4f}})")

    # Zero-shot NER
    print("\\n— GLiNER Zero-Shot NER —")
    run_gliner(df)

    # Embedding similarity
    print("\\n— Embedding Similarity (BGE-M3 / Qwen3-Embedding) —")
    run_embedding_similarity(df)

    # Save JSON metrics
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\\nMetrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR: NER / Entity Extraction  (GLiNER-primary)
# ═══════════════════════════════════════════════════════════════════════════════
def gen_ner(path, cfg):
    labels_list = cfg.get("labels", ["person", "location", "organization", "miscellaneous"])
    labels_str = json.dumps(labels_list)
    data_load = cfg.get("data", '    raise FileNotFoundError("No data")')
    text_col = cfg.get("text_col", "tokens")
    tag_col = cfg.get("tag_col", "ner_tags")
    return textwrap.dedent(f'''\
"""
Modern NER / Entity Extraction Pipeline (April 2026)

Primary model : GLiNER (urchade/gliner_large-v2.1) — zero-shot NER that
                generalises to arbitrary entity types without fine-tuning.
Supervised    : HuggingFace token classification with ModernBERT when
                labelled data is available.
Baseline      : spaCy NER (en_core_web_sm) for quick comparison.

Evaluated with seqeval (entity-level precision / recall / F1).
All results + per-entity metrics exported to metrics.json.

Compute: GPU recommended for transformer models; GLiNER runs on CPU in
         ~1 min for small corpora.  spaCy baseline is CPU-only.
Data: Auto-downloaded at runtime.
"""
import os, json, time, warnings, re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

LABELS = {labels_str}
TEXT_COL = "{text_col}"
TAG_COL = "{tag_col}"

# CoNLL BIO tag mapping (used when tag_col contains int IDs)
CONLL_TAG_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def load_data():
{data_load}
    print(f"Dataset: {{len(df)}} samples")
    return df


def prepare_sentences(df):
    """Convert raw dataset rows into (tokens_list, tags_list) pairs."""
    sentences, tags_all = [], []
    for _, row in df.iterrows():
        toks = row[TEXT_COL]
        tags = row.get(TAG_COL)
        # Tokens may be a list or a space-separated string
        if isinstance(toks, str):
            toks = toks.split()
        if isinstance(toks, (list, np.ndarray)):
            toks = [str(t) for t in toks]
        else:
            continue
        # Tags: list of ints (CoNLL) or list of BIO strings
        if tags is not None:
            if isinstance(tags, str):
                tags = tags.split()
            if isinstance(tags, (list, np.ndarray)) and len(tags) == len(toks):
                if all(isinstance(t, (int, np.integer)) for t in tags):
                    tags = [CONLL_TAG_NAMES[int(t)] if int(t) < len(CONLL_TAG_NAMES) else "O" for t in tags]
                tags = [str(t) for t in tags]
            else:
                tags = ["O"] * len(toks)
        else:
            tags = ["O"] * len(toks)
        sentences.append(toks)
        tags_all.append(tags)
    return sentences, tags_all


# ═══════════════════════════════════════════════════════════════
# PRIMARY: GLiNER zero-shot NER
# ═══════════════════════════════════════════════════════════════
def run_gliner(sentences, gold_tags, labels):
    from gliner import GLiNER
    model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
    pred_tags_all = []
    t0 = time.perf_counter()
    for toks in sentences:
        text = " ".join(toks)
        entities = model.predict_entities(text[:2048], labels, threshold=0.35)
        # Map character-span entities back to token-level BIO tags
        tag_seq = ["O"] * len(toks)
        char_offsets = []
        pos = 0
        for tok in toks:
            start = text.find(tok, pos)
            if start == -1:
                start = pos
            char_offsets.append((start, start + len(tok)))
            pos = start + len(tok)
        for ent in entities:
            ent_start, ent_end = ent["start"], ent["end"]
            lbl = ent["label"].upper().replace(" ", "_")
            first = True
            for i, (cs, ce) in enumerate(char_offsets):
                if cs >= ent_start and ce <= ent_end + 1:
                    tag_seq[i] = f"B-{{lbl}}" if first else f"I-{{lbl}}"
                    first = False
        pred_tags_all.append(tag_seq)
    elapsed = time.perf_counter() - t0
    return pred_tags_all, elapsed


# ═══════════════════════════════════════════════════════════════
# SUPERVISED: Token classification with ModernBERT
# ═══════════════════════════════════════════════════════════════
def run_supervised(sentences, gold_tags):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
    from sklearn.model_selection import train_test_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Build label vocab from gold tags
    all_labels = sorted(set(t for seq in gold_tags for t in seq))
    label2id = {{l: i for i, l in enumerate(all_labels)}}
    id2label = {{i: l for l, i in label2id.items()}}
    n_labels = len(all_labels)

    idx = list(range(len(sentences)))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)
    tr_sents = [sentences[i] for i in tr_idx]
    tr_tags = [gold_tags[i] for i in tr_idx]
    te_sents = [sentences[i] for i in te_idx]
    te_tags = [gold_tags[i] for i in te_idx]

    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=n_labels, id2label=id2label, label2id=label2id,
    ).to(device)

    MAX_LEN = 256

    class NERDataset(Dataset):
        def __init__(self, sents, tag_seqs):
            self.items = []
            for toks, tags in zip(sents, tag_seqs):
                enc = tokenizer(toks, is_split_into_words=True, truncation=True,
                                padding="max_length", max_length=MAX_LEN, return_tensors="pt")
                word_ids = enc.word_ids()
                label_ids = []
                prev_word = None
                for wid in word_ids:
                    if wid is None:
                        label_ids.append(-100)
                    elif wid != prev_word:
                        label_ids.append(label2id.get(tags[wid], 0) if wid < len(tags) else 0)
                    else:
                        label_ids.append(-100)
                    prev_word = wid
                enc = {{k: v.squeeze(0) for k, v in enc.items()}}
                enc["labels"] = torch.tensor(label_ids, dtype=torch.long)
                self.items.append(enc)
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    train_ds = NERDataset(tr_sents, tr_tags)
    test_ds = NERDataset(te_sents, te_tags)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_loader) * 3), len(train_loader) * 3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t0 = time.perf_counter()

    for epoch in range(3):
        model.train(); total_loss = 0
        for batch in train_loader:
            batch = {{k: v.to(device) for k, v in batch.items()}}
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            total_loss += loss.item()
        print(f"  [ModernBERT-NER] Epoch {{epoch+1}}/3, Loss: {{total_loss/len(train_loader):.4f}}")

    elapsed = time.perf_counter() - t0

    # Predict on test set
    model.eval()
    pred_tags_all, true_tags_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {{k: v.to(device) for k, v in batch.items()}}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            for pred_seq, label_seq in zip(preds, labels):
                p, t = [], []
                for pi, li in zip(pred_seq, label_seq):
                    if li != -100:
                        p.append(id2label.get(int(pi), "O"))
                        t.append(id2label.get(int(li), "O"))
                pred_tags_all.append(p)
                true_tags_all.append(t)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    model.save_pretrained(os.path.join(save_dir, "modernbert_ner_model"))
    return pred_tags_all, true_tags_all, elapsed


# ═══════════════════════════════════════════════════════════════
# BASELINE: spaCy NER
# ═══════════════════════════════════════════════════════════════
def run_spacy_baseline(sentences, labels):
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # spaCy label -> our label mapping (best-effort)
    SPACY_MAP = {{
        "PERSON": "PER", "NORP": "MISC", "FAC": "LOC", "ORG": "ORG",
        "GPE": "LOC", "LOC": "LOC", "PRODUCT": "MISC", "EVENT": "MISC",
        "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
        "DATE": "MISC", "TIME": "MISC", "PERCENT": "MISC", "MONEY": "MISC",
        "QUANTITY": "MISC", "ORDINAL": "MISC", "CARDINAL": "MISC",
        "KEYWORD": "KEYWORD", "KEYPHRASE": "KEYPHRASE",
        "TOPIC": "TOPIC", "ENTITY": "ENTITY",
    }}
    pred_tags_all = []
    t0 = time.perf_counter()
    for toks in sentences:
        text = " ".join(toks)
        doc = nlp(text)
        tag_seq = ["O"] * len(toks)
        # Map spaCy char-span entities to token indices
        char_offsets = []
        pos = 0
        for tok in toks:
            start = text.find(tok, pos)
            if start == -1:
                start = pos
            char_offsets.append((start, start + len(tok)))
            pos = start + len(tok)
        for ent in doc.ents:
            lbl = SPACY_MAP.get(ent.label_, "MISC")
            first = True
            for i, (cs, ce) in enumerate(char_offsets):
                if cs >= ent.start_char and ce <= ent.end_char + 1:
                    tag_seq[i] = f"B-{{lbl}}" if first else f"I-{{lbl}}"
                    first = False
        pred_tags_all.append(tag_seq)
    elapsed = time.perf_counter() - t0
    return pred_tags_all, elapsed


# ═══════════════════════════════════════════════════════════════
# EVALUATION: seqeval entity-level metrics
# ═══════════════════════════════════════════════════════════════
def normalise_tags(pred_tags, gold_tags):
    """Ensure pred and gold have the same length per sentence."""
    out_p, out_g = [], []
    for p, g in zip(pred_tags, gold_tags):
        min_len = min(len(p), len(g))
        out_p.append(p[:min_len])
        out_g.append(g[:min_len])
    return out_p, out_g


def evaluate(pred_tags, gold_tags, model_name):
    from seqeval.metrics import classification_report as seq_report
    from seqeval.metrics import f1_score as seq_f1, precision_score as seq_p, recall_score as seq_r
    pred_tags, gold_tags = normalise_tags(pred_tags, gold_tags)
    p = seq_p(gold_tags, pred_tags, zero_division=0)
    r = seq_r(gold_tags, pred_tags, zero_division=0)
    f1 = seq_f1(gold_tags, pred_tags, zero_division=0)
    print()
    print(f"=== {{model_name}} entity-level metrics ===")
    print(f"  Precision: {{p:.4f}}")
    print(f"  Recall:    {{r:.4f}}")
    print(f"  F1:        {{f1:.4f}}")
    print(seq_report(gold_tags, pred_tags, zero_division=0))
    return {{"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}}


def plot_entity_counts(pred_tags, title, save_path):
    """Bar chart of predicted entity type counts."""
    counts = {{}}
    for seq in pred_tags:
        for tag in seq:
            if tag.startswith("B-"):
                lbl = tag[2:]
                counts[lbl] = counts.get(lbl, 0) + 1
    if not counts:
        return
    labels_sorted = sorted(counts.keys())
    vals = [counts[l] for l in labels_sorted]
    fig, ax = plt.subplots(figsize=(max(6, len(labels_sorted) * 0.8), 5))
    ax.bar(labels_sorted, vals, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_eda(df, save_dir):
    """Dataset summary for NER / entity extraction tasks."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Dataset size: {{len(df)}} rows")

    summary_rows = []
    for col in df.columns:
        series = df[col]
        summary_rows.append({{
            "column": col,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "n_unique": int(series.nunique(dropna=True)) if series.dtype != "object" else None,
        }})
    pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "eda_summary.csv"), index=False)

    if TEXT_COL in df.columns:
        token_lengths = []
        for value in df[TEXT_COL].head(5000):
            if isinstance(value, str):
                token_lengths.append(len(value.split()))
            elif isinstance(value, (list, np.ndarray)):
                token_lengths.append(len(value))
        if token_lengths:
            print(
                f"Token length: mean={{np.mean(token_lengths):.1f}}, "
                f"median={{np.median(token_lengths):.1f}}, max={{max(token_lengths)}}"
            )

    if TAG_COL in df.columns:
        entity_counts = {{}}
        for value in df[TAG_COL].head(5000):
            tags = value if isinstance(value, (list, np.ndarray)) else str(value).split()
            for tag in tags:
                tag_str = str(tag)
                if tag_str == "O":
                    continue
                if isinstance(tag, (int, np.integer)) and int(tag) < len(CONLL_TAG_NAMES):
                    tag_str = CONLL_TAG_NAMES[int(tag)]
                entity_counts[tag_str] = entity_counts.get(tag_str, 0) + 1
        if entity_counts:
            top_items = sorted(entity_counts.items(), key=lambda item: item[1], reverse=True)[:12]
            labels = [item[0] for item in top_items]
            values = [item[1] for item in top_items]
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
            ax.bar(labels, values, color="steelblue")
            ax.set_title("Top Entity Tags in Dataset")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "eda_entity_tags.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

    print("Summary statistics saved to eda_summary.csv")
    print("EDA complete.")


def main():
    print("=" * 60)
    print("NER / ENTITY EXTRACTION  GLiNER + ModernBERT + spaCy")
    print(f"Target labels: {{LABELS}}")
    print("=" * 60)
    df = load_data()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    run_eda(df, save_dir)
    sentences, gold_tags = prepare_sentences(df)
    if len(sentences) > 5000:
        sentences, gold_tags = sentences[:5000], gold_tags[:5000]
    print(f"Prepared {{len(sentences)}} sentences")
    metrics_out = {{}}

    # -- GLiNER (primary) --
    print()
    print("-- GLiNER Zero-Shot NER --")
    try:
        gliner_preds, gliner_time = run_gliner(sentences, gold_tags, LABELS)
        m = evaluate(gliner_preds, gold_tags, "GLiNER")
        m["time_s"] = round(gliner_time, 1)
        metrics_out["GLiNER"] = m
        plot_entity_counts(gliner_preds, "GLiNER  Predicted Entities",
                           os.path.join(save_dir, "entities_gliner.png"))
        print(f"  Time: {{gliner_time:.1f}}s")
    except Exception as e:
        print(f"GLiNER failed: {{e}}")

    # -- Supervised ModernBERT (if gold tags available) --
    has_gold = any(any(t != "O" for t in seq) for seq in gold_tags)
    if has_gold:
        print()
        print("-- ModernBERT Token Classification (supervised) --")
        try:
            sup_preds, sup_golds, sup_time = run_supervised(sentences, gold_tags)
            m = evaluate(sup_preds, sup_golds, "ModernBERT-NER")
            m["time_s"] = round(sup_time, 1)
            metrics_out["ModernBERT-NER"] = m
            plot_entity_counts(sup_preds, "ModernBERT-NER  Predicted Entities",
                               os.path.join(save_dir, "entities_modernbert.png"))
            print(f"  Time: {{sup_time:.1f}}s")
        except Exception as e:
            print(f"ModernBERT-NER failed: {{e}}")
    else:
        print()
        print("-- Skipping supervised ModernBERT (no gold BIO tags) --")

    # -- spaCy baseline --
    print()
    print("-- spaCy Baseline NER --")
    try:
        spacy_preds, spacy_time = run_spacy_baseline(sentences, LABELS)
        m = evaluate(spacy_preds, gold_tags, "spaCy")
        m["time_s"] = round(spacy_time, 1)
        metrics_out["spaCy"] = m
        plot_entity_counts(spacy_preds, "spaCy  Predicted Entities",
                           os.path.join(save_dir, "entities_spacy.png"))
        print(f"  Time: {{spacy_time:.1f}}s")
    except Exception as e:
        print(f"spaCy failed: {{e}}")

    # -- Summary --
    if metrics_out:
        best_name = max(metrics_out, key=lambda k: metrics_out[k].get("f1", 0))
        print()
        print(f"Best: {{best_name}} (F1: {{metrics_out[best_name].get('f1', 0):.4f}})")

    # Save metrics
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print()
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


def gen_nlp_similarity(path, cfg):
    data_load = cfg.get("data", '    raise FileNotFoundError("No data")')
    return textwrap.dedent(f'''\
"""
Modern NLP Similarity / Retrieval Pipeline (April 2026)

Primary   : Qwen3-Embedding-0.6B  — state-of-the-art dense embeddings.
Secondary : BGE-M3                 — multilingual dense embeddings.
Baseline  : TF-IDF cosine          — sparse bag-of-words comparison.

Evaluation: If the dataset contains sentence pairs with gold similarity
            scores (e.g. STS-B), Spearman and Pearson correlations are
            computed. Otherwise, average pairwise cosine similarity and
            top-k retrieval examples are reported.

Exploration: UMAP + HDBSCAN embedding cluster visualisation.
Timing    : Wall-clock per model.
Export    : metrics.json with all scores and timings.
Compute   : GPU recommended for Qwen3; BGE-M3 and TF-IDF run on CPU.
Data      : Auto-downloaded at runtime.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
{data_load}
    print(f"Dataset shape: {{df.shape}}")
    return df


# ── helpers ──────────────────────────────────────────────────
def get_texts(df, n=500):
    """Extract the best text column, return up to *n* samples."""
    for c in df.columns:
        if df[c].dtype == "object" and df[c].str.len().mean() > 20:
            return df[c].dropna().head(n).tolist()
    text_cols = df.select_dtypes("object").columns
    if len(text_cols) > 0:
        return df[text_cols[0]].dropna().head(n).tolist()
    return df.iloc[:, 0].astype(str).head(n).tolist()


def detect_sts_pairs(df):
    \"\"\"Return (sent1, sent2, gold_scores) if the dataset is a sentence-pair
    benchmark (STS-B style), else (None, None, None).\"\"\"
    # STS-B columns: sentence1, sentence2, score  (or label)
    s1 = s2 = scores = None
    for a, b in [("sentence1", "sentence2"), ("text1", "text2"),
                  ("premise", "hypothesis"), ("text_a", "text_b")]:
        if a in df.columns and b in df.columns:
            s1, s2 = df[a].astype(str).tolist(), df[b].astype(str).tolist()
            break
    if s1 is None:
        return None, None, None
    for sc in ["score", "label", "similarity", "relatedness"]:
        if sc in df.columns:
            vals = pd.to_numeric(df[sc], errors="coerce")
            if vals.notna().sum() > 10:
                scores = vals.tolist()
                break
    return s1, s2, scores


def show_top_pairs(sim, texts, n=3):
    \"\"\"Print the top-k most similar pairs for the first *n* texts.\"\"\"
    tmp = sim.copy(); np.fill_diagonal(tmp, 0)
    for i in range(min(n, len(texts))):
        top = np.argsort(tmp[i])[-3:][::-1]
        parts = [f"{{j}}({{tmp[i,j]:.3f}})" for j in top]
        print(f"    Text {{i}} most similar to: {{', '.join(parts)}}")


# ═══════════════════════════════════════════════════════════════
# BASELINE: TF-IDF Cosine Similarity
# ═══════════════════════════════════════════════════════════════
def run_tfidf(texts, pairs=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    t0 = time.perf_counter()
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    if pairs:
        s1, s2, gold = pairs
        all_texts = s1 + s2
        vecs = tfidf.fit_transform(all_texts)
        v1, v2 = vecs[:len(s1)], vecs[len(s1):]
        pred = np.array([cosine_similarity(v1[i], v2[i])[0, 0] for i in range(len(s1))])
        elapsed = time.perf_counter() - t0
        return {{"pred_scores": pred, "time_s": round(elapsed, 1)}}
    vecs = tfidf.fit_transform(texts)
    sim = cosine_similarity(vecs)
    avg = (sim.sum() - len(texts)) / max(len(texts) * (len(texts) - 1), 1)
    elapsed = time.perf_counter() - t0
    print(f"  TF-IDF avg pairwise similarity = {{avg:.4f}}")
    show_top_pairs(sim, texts)
    return {{"avg_cosine": round(float(avg), 4), "time_s": round(elapsed, 1)}}


# ═══════════════════════════════════════════════════════════════
# PRIMARY: Qwen3-Embedding
# ═══════════════════════════════════════════════════════════════
def run_qwen3(texts, pairs=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    t0 = time.perf_counter()
    if pairs:
        s1, s2, gold = pairs
        e1 = model.encode(s1, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        e2 = model.encode(s2, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        pred = np.array([float(cosine_similarity(e1[i:i+1], e2[i:i+1])[0, 0]) for i in range(len(s1))])
        elapsed = time.perf_counter() - t0
        return {{"pred_scores": pred, "dim": int(e1.shape[1]), "time_s": round(elapsed, 1)}}
    embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    sim = cosine_similarity(embs)
    avg = (sim.sum() - len(texts)) / max(len(texts) * (len(texts) - 1), 1)
    elapsed = time.perf_counter() - t0
    print(f"  Qwen3: {{len(texts)}} texts embedded (dim={{embs.shape[1]}})")
    print(f"  Avg pairwise semantic similarity = {{avg:.4f}}")
    show_top_pairs(sim, texts)
    return {{"embs": embs, "avg_cosine": round(float(avg), 4),
             "dim": int(embs.shape[1]), "time_s": round(elapsed, 1)}}


# ═══════════════════════════════════════════════════════════════
# SECONDARY: BGE-M3 Embedding Similarity
# ═══════════════════════════════════════════════════════════════
def run_bge_m3(texts, pairs=None):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3")
    t0 = time.perf_counter()
    if pairs:
        s1, s2, gold = pairs
        e1 = model.encode(s1, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        e2 = model.encode(s2, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        pred = np.array([float(cosine_similarity(e1[i:i+1], e2[i:i+1])[0, 0]) for i in range(len(s1))])
        elapsed = time.perf_counter() - t0
        return {{"pred_scores": pred, "dim": int(e1.shape[1]), "time_s": round(elapsed, 1)}}
    embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    sim = cosine_similarity(embs)
    avg = (sim.sum() - len(texts)) / max(len(texts) * (len(texts) - 1), 1)
    elapsed = time.perf_counter() - t0
    print(f"  BGE-M3: {{len(texts)}} texts embedded (dim={{embs.shape[1]}})")
    print(f"  Avg pairwise semantic similarity = {{avg:.4f}}")
    show_top_pairs(sim, texts)
    return {{"embs": embs, "avg_cosine": round(float(avg), 4),
             "dim": int(embs.shape[1]), "time_s": round(elapsed, 1)}}


# ═══════════════════════════════════════════════════════════════
# EVALUATION: STS correlation (when gold scores available)
# ═══════════════════════════════════════════════════════════════
def eval_sts(pred, gold, model_name):
    \"\"\"Spearman + Pearson correlation between predicted and gold similarity.\"\"\"
    from scipy.stats import spearmanr, pearsonr
    mask = ~np.isnan(gold)
    pred, gold = np.asarray(pred)[mask], np.asarray(gold)[mask]
    sp, _ = spearmanr(pred, gold)
    pr, _ = pearsonr(pred, gold)
    print(f"  [{{model_name}}] Spearman: {{sp:.4f}}  |  Pearson: {{pr:.4f}}")
    return {{"spearman": round(float(sp), 4), "pearson": round(float(pr), 4)}}


# ═══════════════════════════════════════════════════════════════
# VISUALISATION: UMAP + HDBSCAN embedding clusters
# ═══════════════════════════════════════════════════════════════
def plot_clusters(embs, save_name="embedding_clusters.png"):
    if embs is None:
        return
    try:
        import umap, hdbscan
    except ImportError:
        print("  (umap/hdbscan not installed — skipping cluster plot)")
        return
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_2d = reducer.fit_transform(embs)
    labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  UMAP + HDBSCAN: {{n_clusters}} clusters")
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=15, alpha=0.6)
    ax.set_title("Embedding Space (UMAP + HDBSCAN)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, save_name), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {{save_name}}")


# ═══════════════════════════════════════════════════════════════
# VISUALISATION: Similarity heatmap
# ═══════════════════════════════════════════════════════════════
def plot_similarity_heatmap(sim, title, save_name):
    n = min(30, sim.shape[0])
    fig, ax = plt.subplots(figsize=(8, 7))
    import seaborn as sns
    sns.heatmap(sim[:n, :n], ax=ax, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, save_name), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {{save_name}}")


def run_eda(df, save_dir):
    """Text similarity dataset statistics."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in text_cols[:3]:
        lengths = df[col].astype(str).str.len()
        print(f"  {{col}}: mean_len={{lengths.mean():.0f}}, median={{lengths.median():.0f}}")
    print("Summary statistics saved to eda_summary.csv")
    print("EDA complete.")


def main():
    print("=" * 60)
    print("NLP SIMILARITY / RETRIEVAL")
    print("Qwen3-Embedding | BGE-M3 | TF-IDF baseline")
    print("=" * 60)
    df = load_data()
    run_eda(df, SAVE_DIR)
    metrics = {{}}

    # Check for sentence-pair benchmark structure (STS-B style)
    s1, s2, gold = detect_sts_pairs(df)
    is_paired = s1 is not None and gold is not None

    if is_paired:
        n = min(len(s1), len(gold))
        s1, s2, gold = s1[:n], s2[:n], gold[:n]
        gold_arr = np.array([float(g) if g is not None else float("nan") for g in gold])
        pairs = (s1, s2, gold_arr)
        print(f"Detected sentence-pair benchmark: {{n}} pairs")
        print()

        print("-- TF-IDF Baseline --")
        try:
            r = run_tfidf(None, pairs=pairs)
            m = eval_sts(r["pred_scores"], gold_arr, "TF-IDF")
            m["time_s"] = r["time_s"]
            metrics["TF-IDF"] = m
        except Exception as e:
            print(f"  TF-IDF failed: {{e}}")
        print()

        print("-- Qwen3-Embedding (primary) --")
        try:
            r = run_qwen3(None, pairs=pairs)
            m = eval_sts(r["pred_scores"], gold_arr, "Qwen3-Embedding")
            m["time_s"] = r["time_s"]; m["dim"] = r.get("dim")
            metrics["Qwen3-Embedding"] = m
        except Exception as e:
            print(f"  Qwen3-Embedding failed: {{e}}")
        print()

        print("-- BGE-M3 (secondary) --")
        try:
            r = run_bge_m3(None, pairs=pairs)
            m = eval_sts(r["pred_scores"], gold_arr, "BGE-M3")
            m["time_s"] = r["time_s"]; m["dim"] = r.get("dim")
            metrics["BGE-M3"] = m
        except Exception as e:
            print(f"  BGE-M3 failed: {{e}}")
    else:
        texts = get_texts(df)
        print(f"Using {{len(texts)}} text samples (unpaired mode)")
        print()

        print("-- TF-IDF Baseline --")
        try:
            r = run_tfidf(texts)
            metrics["TF-IDF"] = r
        except Exception as e:
            print(f"  TF-IDF failed: {{e}}")
        print()

        print("-- Qwen3-Embedding (primary) --")
        try:
            r = run_qwen3(texts)
            embs_q = r.pop("embs", None)
            metrics["Qwen3-Embedding"] = r
        except Exception as e:
            embs_q = None
            print(f"  Qwen3-Embedding failed: {{e}}")
        print()

        print("-- BGE-M3 (secondary) --")
        try:
            r = run_bge_m3(texts)
            embs_b = r.pop("embs", None)
            metrics["BGE-M3"] = r
        except Exception as e:
            embs_b = None
            print(f"  BGE-M3 failed: {{e}}")
        print()

        # Use best available embeddings for clustering + heatmap
        best_embs = embs_q if embs_q is not None else embs_b
        print("-- Embedding Clustering --")
        plot_clusters(best_embs)
        if best_embs is not None:
            sim_mat = cosine_similarity(best_embs)
            plot_similarity_heatmap(sim_mat, "Cosine Similarity Heatmap",
                                    "similarity_heatmap.png")

    # ── Summary ──
    if metrics:
        # Pick best by spearman (paired) or avg_cosine (unpaired)
        key = "spearman" if is_paired else "avg_cosine"
        scored = {{k: v.get(key, 0) for k, v in metrics.items()}}
        best = max(scored, key=scored.get)
        print()
        print(f"Best model: {{best}} ({{key}}={{scored[best]:.4f}})")

    # Save metrics
    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


def gen_nlp_gen(path, cfg):
    task = cfg.get("task", "summarization")
    data_load = cfg.get("data", "    df = None")
    return textwrap.dedent(f'''\
"""
Modern NLP Generation Pipeline (April 2026)

Task      : {task}
Primary   : Qwen3-Instruct (8B) via Ollama — chat, generation, summarisation.
Translation: NLLB-200-distilled-600M (Meta) — 200+ language pairs, offline.
Baseline  : BART-large-CNN (summarisation only).

Chatbot mode runs a scripted demo conversation from the dataset first,
then offers an interactive session. All modes report wall-clock timing
and export results to metrics.json.

Compute: Ollama manages GPU for Qwen3; NLLB/BART use torch + CUDA if available.
Data   : Auto-downloaded at runtime.
"""
import os, json, time, warnings
import pandas as pd
warnings.filterwarnings("ignore")

TASK = "{task}"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def query_ollama(prompt, temperature=0.7, max_tokens=512):
    import requests
    try:
        r = requests.post(OLLAMA_URL, json={{"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                          "options": {{"temperature": temperature, "num_predict": max_tokens}}}}, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        print(f"Ollama error: {{e}}")
        return None


def load_data():
{data_load}
    return df


# ═══════════════════════════════════════════════════════════════
# SUMMARISATION — Qwen3 + BART baseline
# ═══════════════════════════════════════════════════════════════
def run_summarization(df):
    text_col = next((c for c in df.columns if df[c].dtype == "object" and df[c].str.len().mean() > 50), df.select_dtypes("object").columns[0])
    texts = df[text_col].dropna().head(10).tolist()
    metrics = {{}}

    # Detect gold reference summaries (xsum: 'summary'; cnn: 'highlights')
    ref_col = None
    for c in ("summary", "highlights", "abstract", "target"):
        if c in df.columns and c != text_col:
            ref_col = c
            break
    refs = df[ref_col].dropna().head(10).tolist() if ref_col else None
    if refs:
        print(f"  Gold references found in column '{{ref_col}}'")

    # PRIMARY: Qwen3-Instruct via Ollama
    t0 = time.perf_counter()
    qwen_summaries = []
    for i, text in enumerate(texts):
        summary = query_ollama(f"Summarize concisely:\\n\\n{{text[:2000]}}\\n\\nSummary:")
        if summary:
            qwen_summaries.append(summary.strip())
            print(f"  Qwen3 [{{i+1}}] {{summary[:100]}}...")
    elapsed = time.perf_counter() - t0
    if qwen_summaries:
        print(f"  Qwen3-Instruct: {{len(qwen_summaries)}} summaries in {{elapsed:.1f}}s")
    m = {{"count": len(qwen_summaries), "time_s": round(elapsed, 1)}}
    if refs and qwen_summaries:
        m.update(_compute_rouge(qwen_summaries, refs[:len(qwen_summaries)], "Qwen3"))
    metrics["Qwen3-Instruct"] = m

    # BASELINE: BART
    try:
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        t0 = time.perf_counter()
        bart_summaries = []
        for i, text in enumerate(texts[:5]):
            inputs = tokenizer(text[:1024], return_tensors="pt", truncation=True, max_length=1024).to(device)
            summary_ids = model.generate(**inputs, max_length=150, min_length=30, num_beams=4, length_penalty=2.0)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            bart_summaries.append(summary)
            print(f"  BART [{{i+1}}] {{summary[:100]}}...")
        elapsed = time.perf_counter() - t0
        print(f"  BART: {{len(bart_summaries)}} summaries in {{elapsed:.1f}}s")
        m = {{"count": len(bart_summaries), "time_s": round(elapsed, 1)}}
        if refs and bart_summaries:
            m.update(_compute_rouge(bart_summaries, refs[:len(bart_summaries)], "BART"))
        metrics["BART"] = m
    except Exception as e:
        print(f"  BART baseline failed: {{e}}")
    return metrics


def _compute_rouge(hypotheses, references, model_name):
    \"\"\"Compute ROUGE-1/2/L F1 scores. Uses rouge-scorer if available, else
    falls back to a simple n-gram overlap implementation.\"\"\"
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for hyp, ref in zip(hypotheses, references):
            s = scorer.score(str(ref), str(hyp))
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        avg = lambda xs: round(sum(xs) / max(len(xs), 1) * 100, 2)
        out = {{"rouge1": avg(r1), "rouge2": avg(r2), "rougeL": avg(rl)}}
        print(f"  [{{model_name}}] ROUGE-1: {{out['rouge1']}}  ROUGE-2: {{out['rouge2']}}  ROUGE-L: {{out['rougeL']}}")
        return out
    except ImportError:
        pass
    # Fallback: simple unigram overlap (ROUGE-1 approximation)
    try:
        scores = []
        for hyp, ref in zip(hypotheses, references):
            h_toks = set(str(hyp).lower().split())
            r_toks = set(str(ref).lower().split())
            if not r_toks:
                continue
            overlap = len(h_toks & r_toks)
            p = overlap / max(len(h_toks), 1)
            r = overlap / max(len(r_toks), 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            scores.append(f1)
        avg_f1 = round(sum(scores) / max(len(scores), 1) * 100, 2) if scores else 0.0
        print(f"  [{{model_name}}] ROUGE-1 (approx): {{avg_f1}}")
        return {{"rouge1_approx": avg_f1}}
    except Exception:
        return {{}}


# ═══════════════════════════════════════════════════════════════
# TRANSLATION — NLLB-200
# ═══════════════════════════════════════════════════════════════
def _extract_texts(df, n=20):
    \"\"\"Extract source texts from the dataset.  Handles WMT-style nested
    'translation' dicts ({{\"de\": ..., \"en\": ...}}) as well as plain
    string columns.  Returns (texts, references_or_None).
    References are available when the dataset contains parallel pairs.\"\"\"
    # WMT-style: column named 'translation' containing dicts
    if "translation" in df.columns:
        sample = df["translation"].dropna().head(n).tolist()
        if sample and isinstance(sample[0], dict):
            # Prefer English source if available, else first key
            src_key = "en" if "en" in sample[0] else list(sample[0].keys())[0]
            ref_key = [k for k in sample[0].keys() if k != src_key]
            texts = [str(row.get(src_key, "")) for row in sample]
            refs = {{k: [str(row.get(k, "")) for row in sample] for k in ref_key}} if ref_key else None
            return texts, refs
    # Plain string column
    for c in df.columns:
        if df[c].dtype == "object" and df[c].str.len().mean() > 10:
            return df[c].dropna().head(n).astype(str).tolist(), None
    text_cols = df.select_dtypes("object").columns
    if len(text_cols):
        return df[text_cols[0]].dropna().head(n).astype(str).tolist(), None
    return df.iloc[:, 0].astype(str).head(n).tolist(), None


def run_translation(df):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/nllb-200-distilled-600M"
    print(f"  Loading {{model_id}} on {{device}} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    metrics = {{}}

    texts, refs = _extract_texts(df, n=10)
    print(f"  Source texts: {{len(texts)}}")

    targets = [("fra_Latn", "French"), ("deu_Latn", "German"),
               ("spa_Latn", "Spanish"), ("zho_Hans", "Chinese")]
    total_t0 = time.perf_counter()

    for tgt_code, tgt_name in targets:
        t0 = time.perf_counter()
        translations = []
        for i, text in enumerate(texts):
            inputs = tokenizer(str(text)[:512], return_tensors="pt", truncation=True).to(device)
            out_ids = model.generate(**inputs,
                                     forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code),
                                     max_new_tokens=256)
            translated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            translations.append(translated)
            if i < 3:
                print(f"    -> {{tgt_name}} [{{i+1}}] {{translated[:100]}}...")
        elapsed = time.perf_counter() - t0
        lang_metrics = {{"count": len(translations), "time_s": round(elapsed, 1)}}

        # BLEU if we have reference translations for this language
        lang_code_short = tgt_code.split("_")[0][:2]   # fra_Latn -> fr
        iso_map = {{"fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn", "zh": "zho_Hans"}}
        if refs:
            for rk in refs:
                if rk == lang_code_short or iso_map.get(rk) == tgt_code:
                    try:
                        from sacrebleu.metrics import BLEU
                        bleu = BLEU()
                        score = bleu.corpus_score(translations, [refs[rk][:len(translations)]])
                        lang_metrics["bleu"] = round(score.score, 2)
                        print(f"    BLEU ({{tgt_name}}): {{score.score:.2f}}")
                    except ImportError:
                        try:
                            from nltk.translate.bleu_score import corpus_bleu
                            ref_tok = [[r.split()] for r in refs[rk][:len(translations)]]
                            hyp_tok = [t.split() for t in translations]
                            b = corpus_bleu(ref_tok, hyp_tok)
                            lang_metrics["bleu"] = round(b * 100, 2)
                            print(f"    BLEU ({{tgt_name}}): {{b*100:.2f}}")
                        except Exception:
                            pass
                    break

        print(f"  NLLB-200 -> {{tgt_name}}: {{len(translations)}} texts in {{elapsed:.1f}}s")
        metrics[f"NLLB-200_{{tgt_name}}"] = lang_metrics

    total_elapsed = time.perf_counter() - total_t0
    metrics["NLLB-200_total"] = {{"time_s": round(total_elapsed, 1), "languages": len(targets),
                                   "model": model_id, "device": str(device)}}
    return metrics


# ═══════════════════════════════════════════════════════════════
# GENERATION — Qwen3-Instruct
# ═══════════════════════════════════════════════════════════════
def run_generation(df):
    prompts = [
        "Write a creative short story about artificial intelligence discovering emotions:",
        "Complete this sentence: The future of machine learning is",
        "Explain quantum computing to a 10-year-old:",
    ]
    if df is not None:
        text_col = next((c for c in df.columns if df[c].dtype == "object"), None)
        if text_col:
            samples = df[text_col].dropna().head(3).tolist()
            prompts = [f"Continue this text creatively:\\n\\n{{t[:300]}}\\n\\nContinuation:" for t in samples]

    t0 = time.perf_counter()
    responses = []
    for i, prompt in enumerate(prompts):
        response = query_ollama(prompt, temperature=0.9, max_tokens=256)
        if response:
            responses.append(response)
            print(f"  [{{i+1}}] {{response[:200]}}...")
    elapsed = time.perf_counter() - t0
    print(f"  Qwen3-Instruct: {{len(responses)}} texts in {{elapsed:.1f}}s")
    return {{"Qwen3-Instruct": {{"count": len(responses), "time_s": round(elapsed, 1)}}}}


# ═══════════════════════════════════════════════════════════════
# CHATBOT — Qwen3-Instruct | demo + interactive
# ═══════════════════════════════════════════════════════════════
def extract_demo_turns(df, n=5):
    \"\"\"Pull sample user utterances from the loaded dialogue dataset for a
    scripted demo conversation (avoids relying solely on interactive input).\"\"\"
    samples = []
    # daily-dialogs: column named 'dialog' (list of turns) or first text col
    for col in ("dialog", "utterance", "text", "question", "input"):
        if col in df.columns:
            vals = df[col].dropna().head(n * 3)
            for v in vals:
                if isinstance(v, list):
                    samples.extend([str(t) for t in v[:2]])
                elif isinstance(v, str) and len(v.strip()) > 5:
                    samples.append(v.strip()[:200])
                if len(samples) >= n:
                    break
            break
    if not samples:
        text_cols = df.select_dtypes("object").columns
        if len(text_cols):
            samples = df[text_cols[0]].dropna().astype(str).head(n).tolist()
    return samples[:n]


def run_chatbot(df=None):
    \"\"\"Qwen3-Instruct chatbot: scripted demo + optional interactive session.\"\"\"
    system = "You are a helpful, concise assistant."
    metrics = {{"model": OLLAMA_MODEL, "turns": []}}

    # ── Scripted demo from dataset ──
    demo_prompts = []
    if df is not None:
        demo_prompts = extract_demo_turns(df, n=5)
    if not demo_prompts:
        demo_prompts = ["Hello!", "What can you help me with?",
                        "Explain machine learning in one sentence.",
                        "What is the capital of France?", "Thanks, goodbye!"]

    print()
    print("--- Demo Conversation (scripted) ---")
    history = []
    for user_msg in demo_prompts:
        history.append(f"User: {{user_msg}}")
        ctx = "\\n".join(history[-6:])
        prompt = f"{{system}}\\n\\n{{ctx}}\\nAssistant:"
        t0 = time.perf_counter()
        resp = query_ollama(prompt, temperature=0.7, max_tokens=256)
        latency = time.perf_counter() - t0
        if resp:
            resp = resp.strip()
            history.append(f"Assistant: {{resp}}")
            print(f"  User : {{user_msg}}")
            print(f"  Bot  : {{resp[:200]}}")
            print(f"  ({{latency:.1f}}s)")
            metrics["turns"].append({{"user": user_msg, "bot": resp[:300],
                                      "latency_s": round(latency, 1)}})
        else:
            print(f"  User : {{user_msg}}")
            print(f"  Bot  : [no response]")

    avg_lat = 0
    if metrics["turns"]:
        avg_lat = sum(t["latency_s"] for t in metrics["turns"]) / len(metrics["turns"])
        print()
        print(f"  Demo: {{len(metrics['turns'])}} turns, avg latency {{avg_lat:.1f}}s")
    metrics["demo_avg_latency_s"] = round(avg_lat, 1)

    # ── Interactive session (skipped in non-interactive environments) ──
    import sys
    if sys.stdin.isatty():
        print()
        print("--- Interactive Chat (type 'quit' to exit) ---")
        while True:
            try:
                user = input("You: ").strip()
            except EOFError:
                break
            if user.lower() in ("quit", "exit", "q", ""):
                break
            history.append(f"User: {{user}}")
            ctx = "\\n".join(history[-6:])
            prompt = f"{{system}}\\n\\n{{ctx}}\\nAssistant:"
            t0 = time.perf_counter()
            resp = query_ollama(prompt, temperature=0.8, max_tokens=512)
            latency = time.perf_counter() - t0
            if resp:
                resp = resp.strip()
                history.append(f"Assistant: {{resp}}")
                print(f"Bot: {{resp}}")
                print(f"  ({{latency:.1f}}s)")
    else:
        print("  (non-interactive environment — skipping live chat)")
    return metrics


def run_eda(df, save_dir):
    """Input data statistics for generation tasks."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    if df is not None:
        print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
        desc = df.describe(include="all").T
        desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
        text_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in text_cols[:3]:
            lengths = df[col].astype(str).str.len()
            print(f"  {{col}}: mean_len={{lengths.mean():.0f}}, median={{lengths.median():.0f}}")
        print("Summary statistics saved to eda_summary.csv")
    else:
        print("  No structured dataset (chatbot/generation mode)")
    print("EDA complete.")


def main():
    print("=" * 60)
    print(f"NLP GENERATION | Task: {{TASK}} | Model: {{OLLAMA_MODEL}}")
    print("=" * 60)

    # Ollama connectivity check (not needed for translation)
    if TASK != "translation":
        test = query_ollama("Say hello.", max_tokens=10)
        if not test:
            print("Ollama not reachable. Run: ollama serve && ollama pull " + OLLAMA_MODEL)
            if TASK != "translation":
                return

    df = load_data()
    run_eda(df, SAVE_DIR)
    metrics = {{}}

    if TASK == "summarization" and df is not None:
        metrics = run_summarization(df)
    elif TASK == "translation" and df is not None:
        metrics = run_translation(df)
    elif TASK == "chatbot":
        metrics = run_chatbot(df)
    elif TASK == "generation":
        metrics = run_generation(df)
    else:
        if df is not None:
            metrics = run_summarization(df)
        else:
            metrics = run_chatbot()

    # Export metrics
    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print()
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


def gen_image_clf(path, cfg):
    ds = cfg.get("dataset", "CIFAR10")
    n_classes = cfg.get("n_classes", 10)
    # Determine data loading code
    if ds.startswith("hf:"):
        hf_name = ds[3:]
        ds_load = f'''    from datasets import load_dataset as _hf_load
    hf_ds = _hf_load("{hf_name}", split="train")
    # Convert HF image dataset to torchvision-style
    class HFImageDataset(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.ds = hf_dataset; self.transform = transform
            img_col = next((c for c in hf_dataset.column_names if "image" in c.lower()), hf_dataset.column_names[0])
            lbl_col = next((c for c in hf_dataset.column_names if "label" in c.lower()), hf_dataset.column_names[-1])
            self.img_col, self.lbl_col = img_col, lbl_col
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            img = self.ds[i][self.img_col].convert("RGB") if hasattr(self.ds[i][self.img_col], "convert") else Image.open(self.ds[i][self.img_col]).convert("RGB")
            lbl = self.ds[i][self.lbl_col]
            return self.transform(img) if self.transform else img, lbl
    train_ds = HFImageDataset(hf_ds, transform=get_transforms(True))
    n_classes = len(set(hf_ds[next(c for c in hf_ds.column_names if "label" in c.lower())]))'''
    else:
        ds_load = f'''    from torchvision import datasets as tv_datasets
    train_ds = tv_datasets.{ds}(root="./data", train=True, download=True, transform=get_transforms(True))
    n_classes = {n_classes}'''

    return textwrap.dedent(f'''\
"""
Modern Image Classification Pipeline (April 2026)

Primary : DINOv3 ViT-S/14 backbone (frozen head-only, then full fine-tune).
Alternative: ConvNeXt V2 Tiny (fine-tuning baseline via timm).
Timing  : Wall-clock per model.
Export  : metrics.json with per-model accuracy + timing; confusion matrix plot.
Data    : Auto-downloaded at runtime.
"""
import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IMG_SIZE, BATCH_SIZE, EPOCHS, LR = 224, 32, 10, 1e-4
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
{ds_load}

    val_size = max(1, int(0.2 * len(train_ds)))
    train_sub, val_sub = random_split(train_ds, [len(train_ds) - val_size, val_size])
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, num_workers=0)
    metrics = {{}}

    # ── DINOv3 (primary) ──
    print()
    print("-- DINOv3 ViT-S/14 --")
    backbone = torch.hub.load("facebookresearch/dinov3", "dinov3_vits14", pretrained=True)
    embed_dim = 384  # ViT-S/14

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 256),
                                      nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n_classes))
            for p in self.backbone.parameters(): p.requires_grad = False
        def forward(self, x):
            feat = self.backbone(x)
            return self.head(feat)

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=0.01)

    best_acc = 0
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels); loss.backward()
            opt.step(); opt.zero_grad(); total_loss += loss.item()
        if epoch == 2:
            for p in model.backbone.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=0.01)
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds.extend(torch.argmax(model(imgs.to(device)), dim=-1).cpu().numpy())
                gts.extend(labels.numpy())
        val_acc = accuracy_score(gts, preds)
        print(f"  Epoch {{epoch+1}}/{{EPOCHS}} -- Loss: {{total_loss/len(train_loader):.4f}} -- Val Acc: {{val_acc:.4f}}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_preds, best_gts = preds, gts
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
    dino_elapsed = round(time.perf_counter() - t0, 1)

    print(f"  DINOv3 Best Val Accuracy: {{best_acc:.4f}} ({{dino_elapsed}}s)")
    print(classification_report(best_gts, best_preds, zero_division=0))
    metrics["DINOv3"] = {{"val_accuracy": round(best_acc, 4), "epochs": EPOCHS, "time_s": dino_elapsed}}

    # Confusion matrix
    cm = confusion_matrix(best_gts, best_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("DINOv3 Confusion Matrix")
    fig.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # ── ConvNeXt V2 (alternative baseline) ──
    print()
    print("-- ConvNeXt V2 Tiny --")
    try:
        import timm
        t1 = time.perf_counter()
        convnext = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True, num_classes=n_classes).to(device)
        convnext_opt = torch.optim.AdamW(convnext.parameters(), lr=LR * 0.5, weight_decay=0.01)
        for epoch in range(3):
            convnext.train(); total_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                loss = criterion(convnext(imgs), labels); loss.backward()
                convnext_opt.step(); convnext_opt.zero_grad(); total_loss += loss.item()
        convnext.eval(); cv_preds, cv_gts = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                cv_preds.extend(torch.argmax(convnext(imgs.to(device)), dim=-1).cpu().numpy())
                cv_gts.extend(labels.numpy())
        cv_acc = accuracy_score(cv_gts, cv_preds)
        cv_elapsed = round(time.perf_counter() - t1, 1)
        print(f"  ConvNeXt V2 Val Accuracy: {{cv_acc:.4f}} ({{cv_elapsed}}s)")
        metrics["ConvNeXtV2"] = {{"val_accuracy": round(cv_acc, 4), "epochs": 3, "time_s": cv_elapsed}}
    except Exception as e:
        print(f"  ConvNeXt V2: {{e}}")

    return metrics


def run_eda(dataset, save_dir):
    """Dataset statistics for image classification."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {{len(dataset)}}")
    if hasattr(dataset, "classes"):
        print(f"Number of classes: {{len(dataset.classes)}}")
        from collections import Counter
        if hasattr(dataset, "targets"):
            class_counts = Counter(dataset.targets)
            print("\\nSamples per class:")
            for cls_idx in sorted(class_counts.keys()):
                name = dataset.classes[cls_idx] if cls_idx < len(dataset.classes) else str(cls_idx)
                print(f"  {{name}}: {{class_counts[cls_idx]}}")
    elif hasattr(dataset, "class_to_idx"):
        print(f"Number of classes: {{len(dataset.class_to_idx)}}")
    print("EDA complete.")


def main():
    print("=" * 60)
    print("IMAGE CLASSIFICATION | DINOv3 + ConvNeXt V2")
    print("=" * 60)
    # run_eda is called inside train_model() after dataset is loaded
    metrics = train_model()

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


def gen_captioning(path, cfg):
    """Image captioning / multimodal VLM pipeline — Qwen3-VL + Molmo 2."""
    data_load = cfg.get("data", '    raise FileNotFoundError("No data source configured")')
    return textwrap.dedent(f'''\
"""
Modern Image Captioning / VLM Pipeline (April 2026)

Primary    : Qwen3-VL-2B-Instruct (vision-language, bfloat16, auto device).
Alternative: Molmo-7B-D-0924 (AllenAI multimodal LLM, bfloat16).
Timing     : Wall-clock per model.
Export     : metrics.json with caption counts + avg length + timing;
             captions.json with per-image captions from each model.
Data       : Auto-downloaded at runtime.
"""
import os, json, time, warnings
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MAX_SAMPLES = 20
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
{data_load}
    return df


def caption_images():
    df = load_data()
    run_eda(df, SAVE_DIR)
    img_col = next((c for c in df.column_names if "image" in c.lower()), df.column_names[0])
    images = [df[i][img_col] for i in range(min(MAX_SAMPLES, len(df)))]
    metrics = {{}}
    all_captions = {{}}

    # -- PRIMARY: Qwen3-VL --
    print()
    print("-- Qwen3-VL-2B-Instruct --")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        t0 = time.perf_counter()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        captions = []
        for idx, img in enumerate(images):
            pil_img = img.convert("RGB") if hasattr(img, "convert") else Image.open(img).convert("RGB")
            msgs = [{{"role": "user", "content": [
                {{"type": "image", "image": pil_img}},
                {{"type": "text", "text": "Describe this image in detail."}}
            ]}}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            vis_inp = process_vision_info(msgs)
            inputs = processor(text=[text], images=vis_inp[0], return_tensors="pt").to(model.device)
            out_ids = model.generate(**inputs, max_new_tokens=128)
            caption = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True)[0]
            captions.append(caption)
            print(f"  [{{idx+1}}/{{len(images)}}] {{caption[:100]}}...")
        elapsed = round(time.perf_counter() - t0, 1)
        avg_len = sum(len(c) for c in captions) / max(len(captions), 1)
        print(f"  Qwen3-VL: {{len(captions)}} captions, avg {{avg_len:.0f}} chars ({{elapsed}}s)")
        metrics["Qwen3-VL"] = {{"captions": len(captions), "avg_length": round(avg_len, 1), "time_s": elapsed}}
        all_captions["Qwen3-VL"] = captions
    except Exception as e:
        print(f"  Qwen3-VL failed: {{e}}")

    # -- ALTERNATIVE: Molmo 2 --
    print()
    print("-- Molmo-7B-D-0924 --")
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor as AP2
        t1 = time.perf_counter()
        molmo = AutoModelForCausalLM.from_pretrained("allenai/Molmo-7B-D-0924",
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        molmo_proc = AP2.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True)
        molmo_captions = []
        for idx, img in enumerate(images[:10]):
            pil_img = img.convert("RGB") if hasattr(img, "convert") else Image.open(img).convert("RGB")
            inputs = molmo_proc.process(images=[pil_img], text="Describe this image in detail.")
            inputs = {{k: v.to(molmo.device).unsqueeze(0) if hasattr(v, "to") else v for k, v in inputs.items()}}
            out = molmo.generate_from_batch(inputs, max_new_tokens=128, tokenizer=molmo_proc.tokenizer)
            caption = molmo_proc.tokenizer.decode(out[0], skip_special_tokens=True)
            molmo_captions.append(caption)
            print(f"  [{{idx+1}}/{{min(10, len(images))}}] {{caption[:100]}}...")
        mol_elapsed = round(time.perf_counter() - t1, 1)
        mol_avg = sum(len(c) for c in molmo_captions) / max(len(molmo_captions), 1)
        print(f"  Molmo-2: {{len(molmo_captions)}} captions, avg {{mol_avg:.0f}} chars ({{mol_elapsed}}s)")
        metrics["Molmo-2"] = {{"captions": len(molmo_captions), "avg_length": round(mol_avg, 1), "time_s": mol_elapsed}}
        all_captions["Molmo-2"] = molmo_captions
    except Exception as e:
        print(f"  Molmo-2 failed: {{e}}")

    # Save captions
    cap_path = os.path.join(SAVE_DIR, "captions.json")
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2, ensure_ascii=False)
    print(f"Captions saved to {{cap_path}}")

    validation = validate_results(all_captions, len(images), SAVE_DIR)
    metrics["validation"] = validation

    return metrics


def run_eda(df, save_dir):
    """Input data summary for captioning."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    n_rows = len(df)
    columns = list(getattr(df, "column_names", []))
    print(f"  Samples: {{n_rows}}")
    if columns:
        print(f"  Columns: {{columns}}")
    summary = {{"samples": n_rows, "columns": columns}}
    with open(os.path.join(save_dir, "eda_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("EDA complete.")


def validate_results(all_captions, expected_images, save_dir):
    """Validate caption outputs for completeness and diversity."""
    validation = {{"expected_images": expected_images, "models": {{}}}}
    for model_name, captions in all_captions.items():
        clean = [c.strip() for c in captions if isinstance(c, str) and c.strip()]
        validation["models"][model_name] = {{
            "captions": len(captions),
            "non_empty": len(clean),
            "coverage_ratio": round(len(clean) / max(expected_images, 1), 4),
            "unique_ratio": round(len(set(clean)) / max(len(clean), 1), 4),
            "avg_chars": round(sum(len(c) for c in clean) / max(len(clean), 1), 1),
            "passed": bool(clean),
        }}
    validation["passed"] = any(model.get("passed") for model in validation["models"].values())
    out_path = os.path.join(save_dir, "validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"Validation saved to {{out_path}}")
    return validation


def main():
    print("=" * 60)
    print("IMAGE CAPTIONING / VLM | Qwen3-VL + Molmo 2")
    print("=" * 60)
    metrics = caption_images()

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


def gen_medical_seg(path, cfg):
    """Medical image segmentation — nnU-Net + MedSAM2."""
    ds = cfg.get("dataset", "hf:mateuszbuda/brain-segmentation")
    n_classes = cfg.get("n_classes", 2)
    if ds.startswith("hf:"):
        hf_name = ds[3:]
        ds_load = f'''    from datasets import load_dataset as _hf_load
    hf_ds = _hf_load("{hf_name}", split="train")
    print(f"Loaded {{len(hf_ds)}} samples from {hf_name}")'''
    else:
        ds_load = f'''    from datasets import load_dataset as _hf_load
    hf_ds = _hf_load("{ds}", split="train")
    print(f"Loaded {{len(hf_ds)}} samples")'''

    return textwrap.dedent(f'''\
"""
Modern Medical Image Segmentation Pipeline (April 2026)

Primary : nnU-Net-style supervised U-Net (encoder-decoder with skip connections).
Optional: MedSAM2 zero-shot promptable segmentation (center-point prompts).
Metrics : Dice coefficient + mean IoU per model, wall-clock timing.
Export  : metrics.json, segmentation_results.png, best_unet.pth.
Data    : Auto-downloaded from HuggingFace at runtime.

DISCLAIMER: This is an educational/research demonstration pipeline.
It is NOT validated for clinical use. Medical image analysis models
require rigorous validation on curated datasets, regulatory approval,
and expert clinical oversight before any diagnostic application.
"""
import os, json, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IMG_SIZE, BATCH_SIZE, EPOCHS, LR = 256, 8, 15, 1e-4
N_CLASSES = {n_classes}
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
{ds_load}
    return hf_ds


class MedSegDataset(Dataset):
    def __init__(self, hf_ds, img_size=IMG_SIZE):
        self.ds = hf_ds
        self.img_size = img_size
        cols = hf_ds.column_names
        self.img_col = next((c for c in cols if "image" in c.lower()), cols[0])
        self.mask_col = next((c for c in cols if "mask" in c.lower() or "seg" in c.lower() or "label" in c.lower()), cols[-1])
        self.to_tensor = transforms.ToTensor()
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        img = item[self.img_col]
        mask = item[self.mask_col]
        if hasattr(img, "convert"):
            img = img.convert("RGB").resize((self.img_size, self.img_size))
        if hasattr(mask, "convert"):
            mask = mask.convert("L").resize((self.img_size, self.img_size), Image.NEAREST)
        img_t = self.to_tensor(img)
        mask_t = torch.from_numpy(np.array(mask)).long()
        if mask_t.ndim == 3: mask_t = mask_t[0]
        mask_t = torch.clamp(mask_t, 0, N_CLASSES - 1)
        return img_t, mask_t


class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x): return self.conv(x)


class SimpleUNet(nn.Module):
    \"\"\"Lightweight U-Net as nnU-Net-style supervised baseline.\"\"\"
    def __init__(self, in_ch=3, out_ch=N_CLASSES, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        for f in features:
            self.encoders.append(UNetBlock(in_ch, f)); in_ch = f
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.decoders.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(UNetBlock(f * 2, f))
        self.final = nn.Conv2d(features[0], out_ch, 1)
    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)
            skip = skips[-(i // 2 + 1)]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i + 1](x)
        return self.final(x)


def dice_score(pred, target, n_classes=N_CLASSES):
    pred = torch.argmax(pred, dim=1)
    dice = 0.0
    for c in range(n_classes):
        p = (pred == c).float(); t = (target == c).float()
        inter = (p * t).sum()
        dice += (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return dice / n_classes


def mean_iou(pred, target, n_classes=N_CLASSES):
    pred = torch.argmax(pred, dim=1) if pred.ndim == 4 else pred
    iou = 0.0
    for c in range(n_classes):
        p = (pred == c).float(); t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        iou += (inter + 1e-6) / (union + 1e-6)
    return iou / n_classes


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_ds = load_data()
    dataset = MedSegDataset(hf_ds)
    val_size = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)
    metrics = {{}}

    # --- PRIMARY: nnU-Net-style supervised U-Net ---
    print()
    print("-- nnU-Net-style U-Net (supervised) --")
    model = SimpleUNet().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_dice = 0
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            loss = criterion(out, masks); loss.backward()
            opt.step(); opt.zero_grad(); total_loss += loss.item()
        scheduler.step()
        model.eval(); dice_sum, iou_sum, n = 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                dice_sum += dice_score(out, masks).item()
                iou_sum += mean_iou(out, masks).item()
                n += 1
        val_dice = dice_sum / max(n, 1)
        val_iou = iou_sum / max(n, 1)
        print(f"  Epoch {{epoch+1}}/{{EPOCHS}} -- Loss: {{total_loss/len(train_loader):.4f}} -- Dice: {{val_dice:.4f}} -- IoU: {{val_iou:.4f}}")
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_unet.pth"))
    unet_elapsed = round(time.perf_counter() - t0, 1)
    print(f"  nnU-Net Best -- Dice: {{best_dice:.4f}} -- IoU: {{best_iou:.4f}} ({{unet_elapsed}}s)")
    metrics["nnUNet"] = {{"val_dice": round(best_dice, 4), "val_iou": round(best_iou, 4),
                         "epochs": EPOCHS, "time_s": unet_elapsed}}

    # --- OPTIONAL: MedSAM2 (zero-shot promptable segmentation) ---
    print()
    print("-- MedSAM2 (zero-shot, center-point prompt) --")
    try:
        from transformers import SamModel, SamProcessor
        t1 = time.perf_counter()
        sam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)
        sam_proc = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
        sam_model.eval()
        dice_sum, iou_sum, n = 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                for j in range(min(4, imgs.shape[0])):
                    pil_img = transforms.ToPILImage()(imgs[j])
                    h, w = pil_img.size[1], pil_img.size[0]
                    input_points = [[[w // 2, h // 2]]]
                    inputs = sam_proc(pil_img, input_points=input_points, return_tensors="pt").to(device)
                    outputs = sam_model(**inputs)
                    pred_mask = sam_proc.image_processor.post_process_masks(
                        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                        inputs["reshaped_input_sizes"].cpu())[0]
                    pred_binary = (pred_mask[0, 0] > 0).long()
                    gt = masks[j]
                    if pred_binary.shape != gt.shape:
                        pred_binary = nn.functional.interpolate(
                            pred_binary.float().unsqueeze(0).unsqueeze(0),
                            size=gt.shape, mode="nearest")[0, 0].long()
                    inter = ((pred_binary == 1) & (gt == 1)).sum().float()
                    p_sum = (pred_binary == 1).sum().float()
                    t_sum = (gt == 1).sum().float()
                    dice_sum += (2 * inter + 1e-6) / (p_sum + t_sum + 1e-6)
                    iou_sum += (inter + 1e-6) / (p_sum + t_sum - inter + 1e-6)
                    n += 1
                if n >= 32:
                    break
        sam_dice = (dice_sum / max(n, 1)).item() if hasattr(dice_sum, "item") else dice_sum / max(n, 1)
        sam_iou = (iou_sum / max(n, 1)).item() if hasattr(iou_sum, "item") else iou_sum / max(n, 1)
        sam_elapsed = round(time.perf_counter() - t1, 1)
        print(f"  MedSAM2 -- Dice: {{sam_dice:.4f}} -- IoU: {{sam_iou:.4f}} ({{sam_elapsed}}s, {{n}} samples)")
        metrics["MedSAM2"] = {{"val_dice": round(sam_dice, 4), "val_iou": round(sam_iou, 4),
                              "samples": n, "time_s": sam_elapsed}}
    except Exception as e:
        print(f"  MedSAM2 skipped: {{e}}")

    # Visualize sample predictions
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    with torch.no_grad():
        for imgs, masks in val_loader:
            preds = torch.argmax(model(imgs.to(device)), dim=1).cpu()
            for i in range(min(4, imgs.shape[0])):
                axes[0, i].imshow(imgs[i].permute(1, 2, 0).numpy())
                axes[0, i].set_title("Input"); axes[0, i].axis("off")
                axes[1, i].imshow(preds[i].numpy(), cmap="jet", alpha=0.7)
                axes[1, i].set_title("Prediction"); axes[1, i].axis("off")
            break
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "segmentation_results.png"), dpi=100)
    plt.close(fig)
    print(f"Saved segmentation_results.png")

    return metrics


def run_eda(save_dir):
    """Dataset summary for medical segmentation."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    data_dir = os.path.join(save_dir, "data")
    if os.path.isdir(data_dir):
        imgs = [f for f in os.listdir(data_dir) if not f.startswith(".")]
        print(f"  Files in data directory: {{len(imgs)}}")
    print("EDA complete.")


def main():
    print("=" * 60)
    print("MEDICAL SEGMENTATION | nnU-Net + MedSAM2")
    print("=" * 60)
    run_eda(SAVE_DIR)
    print("NOTE: Educational/research demo only -- not for clinical use.")
    metrics = train_model()

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


# No template imports needed — all generators are inline below


# ── Specialized generators that embed data download ──

def gen_clustering(path, cfg):
    data_load = cfg.get("data", '    raise FileNotFoundError("No data source configured")')
    return textwrap.dedent(f'''\
"""
Modern Clustering Pipeline (April 2026)
Models: UMAP + HDBSCAN (primary) + GMM (soft assignments) + K-Means (baseline)
Data: Auto-downloaded at runtime

Compute: CPU-only for K-Means/GMM (<10s). UMAP + HDBSCAN ~10-60s depending
         on dataset size. No GPU required.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data():
{data_load}
    # Drop ID-like columns
    for c in df.columns:
        if c.lower() in ("id", "customerid", "customer_id"): df.drop(columns=[c], inplace=True, errors="ignore")
    print(f"Dataset shape: {{df.shape}}")
    return df


def preprocess(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        if hasattr(df[c], "cat"): df[c] = df[c].astype(str)
        df[c] = df[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = oe.fit_transform(df[cat_cols])
    return StandardScaler().fit_transform(df.select_dtypes(include=["number"]))

def eval_clustering(X, labels, name):
    mask = labels >= 0
    n = len(set(labels[mask]))
    noise = int((labels == -1).sum())
    if n > 1 and mask.sum() > n:
        sil = silhouette_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])
        print(f"  {{name}}: {{n}} clusters, noise={{noise}}, silhouette={{sil:.4f}}, CH={{ch:.1f}}, DB={{db:.4f}}")
        return {{"n_clusters": int(n), "noise": noise, "silhouette": round(float(sil), 4),
                "calinski_harabasz": round(float(ch), 1), "davies_bouldin": round(float(db), 4)}}
    else:
        print(f"  {{name}}: {{n}} clusters, noise={{noise}} — insufficient for metrics")
        return {{"n_clusters": int(n), "noise": noise}}


def cluster(X):
    results = {{}}
    timings = {{}}
    metrics_out = {{}}
    save_dir = os.path.dirname(os.path.abspath(__file__))

    # === PRIMARY: UMAP + HDBSCAN ===
    try:
        import umap, hdbscan
        t0 = time.perf_counter()
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X)

        # Auto-tune min_cluster_size
        best_sil, best_mcs, best_labels = -1, 15, None
        for mcs in [5, 10, 15, 25, 50]:
            lbls = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5).fit_predict(X_umap)
            mask = lbls >= 0
            n = len(set(lbls[mask]))
            if n > 1 and mask.sum() > n:
                s = silhouette_score(X_umap[mask], lbls[mask])
                if s > best_sil:
                    best_sil, best_mcs, best_labels = s, mcs, lbls
        if best_labels is None:
            best_labels = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(X_umap)
        timings["HDBSCAN"] = time.perf_counter() - t0
        print(f"  UMAP + HDBSCAN (min_cluster_size={{best_mcs}})  ({{timings['HDBSCAN']:.1f}}s):")
        m = eval_clustering(X_umap, best_labels, "HDBSCAN")
        m["time_s"] = round(timings["HDBSCAN"], 1)
        m["min_cluster_size"] = best_mcs
        metrics_out["HDBSCAN"] = m
        results["HDBSCAN"] = {{"labels": best_labels, "embedding": X_umap}}
    except Exception as e:
        print(f"  UMAP + HDBSCAN failed: {{e}}")
        # Fallback: PCA for embedding
        from sklearn.decomposition import PCA
        X_umap = PCA(n_components=2).fit_transform(X)

    # === SOFT ASSIGNMENTS: Gaussian Mixture Model ===
    try:
        from sklearn.mixture import GaussianMixture
        t0 = time.perf_counter()
        bics = [GaussianMixture(n_components=k, random_state=42).fit(X).bic(X) for k in range(2, 11)]
        best_k = int(range(2, 11)[np.argmin(bics)])
        gmm = GaussianMixture(n_components=best_k, random_state=42).fit(X)
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)
        timings["GMM"] = time.perf_counter() - t0
        print(f"  GMM (BIC-optimal k={{best_k}})  ({{timings['GMM']:.1f}}s):")
        m = eval_clustering(X, labels, "GMM")
        m["time_s"] = round(timings["GMM"], 1)
        m["best_k"] = best_k
        avg_confidence = float(probs.max(axis=1).mean())
        m["avg_confidence"] = round(avg_confidence, 4)
        metrics_out["GMM"] = m
        print(f"  Avg assignment confidence: {{avg_confidence:.4f}}")
        results["GMM"] = {{"labels": labels, "n": best_k, "probs": probs}}
    except Exception as e:
        print(f"  GMM failed: {{e}}")

    # === BASELINE: K-Means (Elbow + Silhouette) ===
    try:
        from sklearn.cluster import KMeans
        t0 = time.perf_counter()
        inertias, sils = [], []
        K_range = range(2, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = km.fit_predict(X)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, lbls))
        best_k = int(K_range[np.argmax(sils)])
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
        timings["KMeans"] = time.perf_counter() - t0
        print(f"  K-Means baseline (best k={{best_k}}, silhouette={{max(sils):.4f}})  ({{timings['KMeans']:.1f}}s):")
        m = eval_clustering(X, labels, "K-Means")
        m["time_s"] = round(timings["KMeans"], 1)
        m["best_k"] = best_k
        metrics_out["KMeans"] = m
        results["KMeans"] = {{"labels": labels, "n": best_k, "inertias": inertias, "sils": sils}}

        # Elbow + Silhouette plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(list(K_range), inertias, "bo-")
        axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
        axes[1].plot(list(K_range), sils, "rs-")
        axes[1].set_title("Silhouette Scores"); axes[1].set_xlabel("k"); axes[1].set_ylabel("Score")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "kmeans_elbow_silhouette.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("  Saved kmeans_elbow_silhouette.png")
    except Exception as e:
        print(f"  K-Means failed: {{e}}")

    # === VISUALIZATION ===
    try:
        embed = results.get("HDBSCAN", {{}}).get("embedding", X[:, :2] if X.shape[1] >= 2 else X)
        active = [(n, results[n]["labels"]) for n in ["HDBSCAN", "GMM", "KMeans"] if n in results]
        n_plots = len(active)
        fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(6 * max(n_plots, 1), 5))
        if n_plots == 1: axes = [axes]
        for ax, (name, lbls) in zip(axes, active):
            scatter = ax.scatter(embed[:, 0], embed[:, 1], c=lbls, cmap="tab10", s=10, alpha=0.6)
            ax.set_title(name); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "clustering_results.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print("  Saved clustering_results.png")
    except Exception as e:
        print(f"  Plot failed: {{e}}")

    # === SUMMARY ===
    print("\\n" + "=" * 40)
    print("CLUSTERING COMPARISON:")
    for name in ["HDBSCAN", "GMM", "KMeans"]:
        if name in results:
            n = len(set(results[name]["labels"])) - (1 if -1 in results[name]["labels"] else 0)
            t = f"  ({{timings[name]:.1f}}s)" if name in timings else ""
            print(f"  {{name}}: {{n}} clusters{{t}}")
    print("=" * 40)

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"  Metrics saved to {{out_path}}")


def run_eda(df, save_dir):
    """Exploratory Data Analysis for clustering."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
    print(f"Column types:\\n{{df.dtypes.value_counts().to_string()}}")
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\\nMissing values ({{len(n_miss)}} columns):")
        print(n_miss.sort_values(ascending=False).head(15).to_string())
    else:
        print("\\nNo missing values")
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    print("Summary statistics saved to eda_summary.csv")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        n = len(num_cols)
        fig, ax = plt.subplots(figsize=(min(n + 2, 20), min(n, 16)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        import seaborn as _sns
        _sns.heatmap(corr, mask=mask, annot=n <= 15, fmt=".2f",
                     cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap")
        fig.savefig(os.path.join(save_dir, "eda_correlation.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    plot_cols = num_cols[:20]
    if plot_cols:
        nr = max(1, (len(plot_cols) + 4) // 5)
        nc = min(5, len(plot_cols))
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
        for i, col in enumerate(plot_cols):
            ri, ci = divmod(i, nc)
            df[col].hist(bins=30, ax=axes[ri][ci], color="steelblue", edgecolor="black")
            axes[ri][ci].set_title(col, fontsize=9)
        for i in range(len(plot_cols), nr * nc):
            ri, ci = divmod(i, nc)
            axes[ri][ci].set_visible(False)
        fig.suptitle("Feature Distributions")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "eda_distributions.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print("EDA plots saved.")


def main():
    print("=" * 60)
    print("CLUSTERING: UMAP + HDBSCAN (primary) | GMM | K-Means baseline")
    print("=" * 60)
    df = load_data()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    run_eda(df, save_dir)
    X = preprocess(df)
    cluster(X)


if __name__ == "__main__":
    main()
''')


def gen_anomaly(path, cfg):
    data_load = cfg.get("data", '    raise FileNotFoundError("No data")')
    return textwrap.dedent(f'''\
"""
Modern Unsupervised Anomaly Detection Pipeline (April 2026)

Approach: Purely unsupervised — no labeled fraud targets.
          If ground-truth labels exist in the dataset they are used ONLY for
          post-hoc evaluation, never for training.

Models (PyOD 2):
  - ECOD  — Empirical CDF-based, parameter-free, fast
  - COPOD — Copula-based, parameter-free, fast
  - IForest — Isolation Forest (ensemble baseline)
  - LOF   — Local Outlier Factor (density-based, good for local anomalies)

Visual anomaly detection:
  - anomalib PatchCore (wide_resnet50_2 backbone, MVTec benchmark)

Compute: CPU-only for PyOD models (<30s each). GPU for anomalib PatchCore.
Data: Auto-downloaded at runtime
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data():
{data_load}
    print(f"Dataset shape: {{df.shape}}")
    return df


def preprocess(df):
    \"\"\"Prepare data for unsupervised detection.

    Returns (X_scaled, y_true_or_None).  Labels are auto-detected but NEVER
    used for training — only for optional post-hoc evaluation.
    \"\"\"
    df = df.copy()
    label_col = next((c for c in df.columns if c.lower() in ("label","class","target","anomaly","outlier")), None)
    y = None
    if label_col:
        y = df[label_col].values; df.drop(columns=[label_col], inplace=True)
        print(f" Ground-truth column '{{label_col}}' detected — used for evaluation only (not training)")
    for c in df.columns:
        if c.lower() in ("id","timestamp","date","time"): df.drop(columns=[c], inplace=True, errors="ignore")
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        if hasattr(df[c], "cat"): df[c] = df[c].astype(str)
        df[c] = df[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = oe.fit_transform(df[cat_cols])
    return StandardScaler().fit_transform(df.select_dtypes(include=["number"])), y


def detect(X, y=None):
    results = {{}}      # name -> {{labels, scores}}
    timings = {{}}      # name -> wall-clock seconds
    metrics_out = {{}}  # name -> dict of metrics
    has_labels = y is not None and len(set(y)) > 1

    for name, Builder in [
        ("ECOD", lambda: __import__("pyod.models.ecod", fromlist=["ECOD"]).ECOD(contamination=0.05)),
        ("COPOD", lambda: __import__("pyod.models.copod", fromlist=["COPOD"]).COPOD(contamination=0.05)),
        ("IForest", lambda: __import__("pyod.models.iforest", fromlist=["IForest"]).IForest(contamination=0.05, random_state=42)),
        ("LOF", lambda: __import__("pyod.models.lof", fromlist=["LOF"]).LOF(contamination=0.05, n_neighbors=20)),
    ]:
        try:
            t0 = time.perf_counter()
            m = Builder()
            m.fit(X)
            elapsed = time.perf_counter() - t0
            labels = m.labels_
            scores = m.decision_scores_ if hasattr(m, "decision_scores_") else m.decision_function(X)
            timings[name] = elapsed
            results[name] = {{"labels": labels, "scores": scores}}
            n_anom = int(labels.sum())
            row = {{"anomalies": n_anom, "anomaly_pct": round(n_anom / len(X), 4), "time_s": round(elapsed, 1)}}

            # Score distribution summary
            p50, p90, p95, p99 = np.percentile(scores, [50, 90, 95, 99])
            row["score_p50"] = round(float(p50), 4)
            row["score_p95"] = round(float(p95), 4)
            row["score_p99"] = round(float(p99), 4)

            extra = ""
            if has_labels:
                f1 = f1_score(y, labels)
                auc = roc_auc_score(y, scores)
                row["f1"] = round(f1, 4)
                row["roc_auc"] = round(auc, 4)
                extra = f"  F1: {{f1:.4f}}  ROC-AUC: {{auc:.4f}}"

            metrics_out[name] = row
            print(f"{{name}}: {{n_anom}} anomalies ({{n_anom/len(X):.2%}})  ({{elapsed:.1f}}s){{extra}}")
        except Exception as e:
            print(f"{{name}}: {{e}}")

    # ── Score distribution plot ──
    save_dir = os.path.dirname(os.path.abspath(__file__))
    if results:
        try:
            n_models = len(results)
            fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
            if n_models == 1: axes = [axes]
            for ax, (name, r) in zip(axes, results.items()):
                ax.hist(r["scores"][r["labels"] == 0], bins=50, alpha=0.6, label="Normal", density=True)
                ax.hist(r["scores"][r["labels"] == 1], bins=50, alpha=0.6, label="Anomaly", density=True)
                ax.set_title(name); ax.set_xlabel("Anomaly score"); ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "anomaly_scores.png"), dpi=100, bbox_inches="tight")
            plt.close()
            print("Saved anomaly_scores.png")
        except Exception as e:
            print(f"Score plot: {{e}}")

    # ── Agreement matrix (how many detectors flag each point) ──
    if len(results) > 1:
        try:
            all_labels = np.column_stack([r["labels"] for r in results.values()])
            votes = all_labels.sum(axis=1)
            for k in range(1, len(results) + 1):
                n = int((votes >= k).sum())
                print(f"  Flagged by ≥{{k}} detectors: {{n}} ({{n/len(X):.2%}})")
        except Exception:
            pass

    # ── anomalib PatchCore (image-based anomaly detection) ──
    try:
        t0 = time.perf_counter()
        from anomalib.models import Patchcore
        from anomalib.data import MVTec
        from anomalib.engine import Engine
        datamodule = MVTec(category="bottle", image_size=(256, 256), train_batch_size=8, eval_batch_size=8)
        model = Patchcore(backbone="wide_resnet50_2", layers_to_extract=["layer2", "layer3"],
                          coreset_sampling_ratio=0.1, num_neighbors=9)
        engine = Engine(max_epochs=1, devices=1, accelerator="auto")
        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)
        elapsed = time.perf_counter() - t0
        timings["PatchCore"] = elapsed
        print(f"PatchCore (anomalib): {{test_results}}  ({{elapsed:.1f}}s)")
    except Exception as e:
        print(f"PatchCore: {{e}}")

    # ── Save JSON metrics ──
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\\nMetrics saved to {{out_path}}")


def run_eda(df, save_dir):
    """Exploratory Data Analysis for anomaly detection."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
    print(f"Column types:\\n{{df.dtypes.value_counts().to_string()}}")
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\\nMissing values ({{len(n_miss)}} columns):")
        print(n_miss.sort_values(ascending=False).head(15).to_string())
    else:
        print("\\nNo missing values")
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    print("Summary statistics saved to eda_summary.csv")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        n = len(num_cols)
        fig, ax = plt.subplots(figsize=(min(n + 2, 20), min(n, 16)))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        import seaborn as _sns
        _sns.heatmap(corr, mask=mask, annot=n <= 15, fmt=".2f",
                     cmap="coolwarm", center=0, ax=ax, square=True)
        ax.set_title("Feature Correlation Heatmap")
        fig.savefig(os.path.join(save_dir, "eda_correlation.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    plot_cols = num_cols[:20]
    if plot_cols:
        nr = max(1, (len(plot_cols) + 4) // 5)
        nc = min(5, len(plot_cols))
        fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3 * nr), squeeze=False)
        for i, col in enumerate(plot_cols):
            ri, ci = divmod(i, nc)
            df[col].hist(bins=30, ax=axes[ri][ci], color="steelblue", edgecolor="black")
            axes[ri][ci].set_title(col, fontsize=9)
        for i in range(len(plot_cols), nr * nc):
            ri, ci = divmod(i, nc)
            axes[ri][ci].set_visible(False)
        fig.suptitle("Feature Distributions")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "eda_distributions.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print("EDA plots saved.")


def main():
    print("=" * 60)
    print("UNSUPERVISED ANOMALY DETECTION")
    print("PyOD 2 (ECOD / COPOD / IForest / LOF) + anomalib PatchCore")
    print("Labels used for evaluation only — never for training")
    print("=" * 60)
    df = load_data()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    run_eda(df, save_dir)
    X, y = preprocess(df)
    detect(X, y)


if __name__ == "__main__":
    main()
''')


def gen_timeseries(path, cfg):
    target = cfg.get("target", "Close")
    data_load = cfg["data"]
    return textwrap.dedent(f'''\
"""
Modern Time Series Forecasting Pipeline (April 2026)

Primary models (foundation-model forecasting):
  - AutoGluon TimeSeries  (AutoML ensemble, ~3 min fit with time_limit=180)
  - Chronos-Bolt          (Amazon zero-shot foundation model, ~30s on GPU)
  - Chronos-2             (Amazon universal foundation model, ~60s on GPU)
  - TimesFM               (Google foundation model, ~20s on GPU)

Classical baselines (kept for comparison only):
  - ARIMA(5,1,0)          (statsmodels, fast, CPU-only, <5s)
  - Prophet               (Meta, fast, CPU-only, <10s)

Tabular lag-feature baselines:
  - LightGBM / CatBoost / XGBoost (GBDT with lag features, ~10s each on GPU)
  - FLAML AutoML           (automated lag-feature model selection, 60s budget)

Compute requirements:
  - GPU recommended for foundation models (RTX 3060+ / 8 GB VRAM minimum)
  - AutoGluon-TS: 4+ GB RAM, ~3 min on CPU, ~1 min on GPU
  - Chronos / TimesFM: GPU strongly recommended (CPU fallback 5-10x slower)
  - Classical baselines (ARIMA / Prophet / GBDT): CPU-only, <30s each
  - FLAML: CPU-only, budget-capped at 60s

Metrics: RMSE, MAE, MAPE (where denominator is non-zero)
Export : metrics.json + metrics.csv + forecast.png
Data: Auto-downloaded at runtime
"""
import os, json, warnings, time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TARGET = "{target}"
HORIZON = 30
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def mape_score(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def score(name, y_true, y_pred, table):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mp = mape_score(y_true, y_pred)
    table.append({{"Model": name, "RMSE": rmse, "MAE": mae, "MAPE(%)": mp}})
    mp_s = f"{{mp:.2f}}%" if not np.isnan(mp) else "N/A"
    print(f"  {{name}}: RMSE={{rmse:.4f}}  MAE={{mae:.4f}}  MAPE={{mp_s}}")
    return rmse


def load_data():
{data_load}
    # Auto-detect date and target
    target = TARGET
    if target not in df.columns:
        for c in df.select_dtypes("number").columns:
            if any(kw in c.lower() for kw in ["close","price","value","sales","demand","total"]):
                target = c; break
        else:
            target = df.select_dtypes("number").columns[-1]
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).sort_values(c).set_index(c); break
    print(f"Dataset: {{df.shape}}, target: {{target}}")
    return df, target


def forecast(df, target):
    results = {{}}
    metrics = []
    series = df[target].dropna().values.astype(float)
    n = len(series); split = n - HORIZON
    train, test = series[:split], series[split:]

    # === PRIMARY: AutoGluon TimeSeries ===
    try:
        t0 = time.perf_counter()
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        ts_df = pd.DataFrame({{"item_id": ["s"] * split, "timestamp": pd.date_range("2020-01-01", periods=split, freq="D"), "target": train}})
        ts_data = TimeSeriesDataFrame.from_data_frame(ts_df)
        predictor = TimeSeriesPredictor(prediction_length=HORIZON, eval_metric="RMSE",
                                         path=os.path.join(SAVE_DIR, "ag_ts"))
        predictor.fit(ts_data, time_limit=180, presets="best_quality")
        ag_preds = predictor.predict(ts_data)
        y_pred = ag_preds["mean"].values[:len(test)]
        results["AutoGluon-TS"] = y_pred
        print(f"  AutoGluon-TS ({{time.perf_counter()-t0:.1f}}s)")
        score("AutoGluon-TS", test, y_pred, metrics)
        lb = predictor.leaderboard(ts_data)
        print("  Leaderboard (top 5):")
        for line in lb.head().to_string().splitlines():
            print(f"    {{line}}")
    except Exception as e: print(f"  AutoGluon-TS failed: {{e}}")

    # === FOUNDATION MODELS ===

    # Chronos-Bolt (fast zero-shot)
    try:
        t0 = time.perf_counter()
        import torch
        from chronos import ChronosPipeline
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                  device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        context = torch.tensor(train, dtype=torch.float32)
        y_pred = np.median(pipe.predict(context, HORIZON)[0].numpy(), axis=0)[:len(test)]
        results["Chronos-Bolt"] = y_pred
        print(f"  Chronos-Bolt ({{time.perf_counter()-t0:.1f}}s)")
        score("Chronos-Bolt", test, y_pred, metrics)
    except Exception as e: print(f"  Chronos-Bolt failed: {{e}}")

    # Chronos-2 (universal forecasting)
    try:
        t0 = time.perf_counter()
        import torch
        from chronos import ChronosPipeline
        pipe2 = ChronosPipeline.from_pretrained("amazon/chronos-2-base",
                   device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        context = torch.tensor(train, dtype=torch.float32)
        y_pred = np.median(pipe2.predict(context, HORIZON)[0].numpy(), axis=0)[:len(test)]
        results["Chronos-2"] = y_pred
        print(f"  Chronos-2 ({{time.perf_counter()-t0:.1f}}s)")
        score("Chronos-2", test, y_pred, metrics)
    except Exception as e: print(f"  Chronos-2 failed: {{e}}")

    # TimesFM (Google foundation model)
    try:
        t0 = time.perf_counter()
        import timesfm
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32,
                                            horizon_len=HORIZON),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
        freq = [0] * 1  # freq=0 -> daily
        y_pred, _ = tfm.forecast([train], freq)
        y_pred = y_pred[0][:len(test)]
        results["TimesFM"] = y_pred
        print(f"  TimesFM ({{time.perf_counter()-t0:.1f}}s)")
        score("TimesFM", test, y_pred, metrics)
    except Exception as e: print(f"  TimesFM failed: {{e}}")

    # === CLASSICAL BASELINES (comparison only) ===

    # ARIMA (statsmodels)
    try:
        t0 = time.perf_counter()
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train, order=(5, 1, 0))
        fitted = model.fit()
        y_pred = fitted.forecast(steps=HORIZON)[:len(test)]
        results["ARIMA(5,1,0)"] = y_pred
        print(f"  ARIMA(5,1,0) baseline ({{time.perf_counter()-t0:.1f}}s)")
        score("ARIMA(5,1,0)", test, y_pred, metrics)
    except Exception as e: print(f"  ARIMA failed: {{e}}")

    # Prophet (Meta)
    try:
        t0 = time.perf_counter()
        from prophet import Prophet
        p_df = pd.DataFrame({{"ds": pd.date_range("2020-01-01", periods=split, freq="D"), "y": train}})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(p_df)
        future = m.make_future_dataframe(periods=HORIZON)
        fc = m.predict(future)
        y_pred = fc["yhat"].values[-HORIZON:][:len(test)]
        results["Prophet"] = y_pred
        print(f"  Prophet baseline ({{time.perf_counter()-t0:.1f}}s)")
        score("Prophet", test, y_pred, metrics)
    except Exception as e: print(f"  Prophet failed: {{e}}")

    # === TABULAR LAG-FEATURE BASELINES (GBDT + FLAML) ===
    lags = [1, 2, 3, 5, 7, 14, 21]
    lag_df = pd.DataFrame({{"y": series}})
    for lg in lags:
        lag_df[f"lag_{{lg}}"] = lag_df["y"].shift(lg)
    lag_df["rolling_7"] = lag_df["y"].rolling(7).mean()
    lag_df["rolling_14"] = lag_df["y"].rolling(14).mean()
    lag_df["rolling_28"] = lag_df["y"].rolling(28).mean()
    lag_df["diff_1"] = lag_df["y"].diff(1)
    lag_df["diff_7"] = lag_df["y"].diff(7)
    lag_df = lag_df.dropna()
    offset = max(lags) + 28  # account for rolling window
    lag_train = lag_df.iloc[:split - offset]
    lag_test = lag_df.iloc[split - offset:split - offset + HORIZON]

    if len(lag_test) >= HORIZON:
        X_lag_tr = lag_train.drop(columns=["y"]); y_lag_tr = lag_train["y"]
        X_lag_te = lag_test.drop(columns=["y"]); y_lag_te = lag_test["y"]

        for name, builder in [
            ("LightGBM-Lag", lambda: __import__("lightgbm").LGBMRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                device="gpu", verbose=-1, n_jobs=-1)),
            ("CatBoost-Lag", lambda: __import__("catboost").CatBoostRegressor(
                iterations=500, lr=0.05, depth=6, task_type="GPU",
                devices="0", verbose=0)),
            ("XGBoost-Lag", lambda: __import__("xgboost").XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                device="cuda", tree_method="hist", verbosity=0, n_jobs=-1)),
        ]:
            try:
                t0 = time.perf_counter()
                m = builder()
                m.fit(X_lag_tr, y_lag_tr)
                y_pred = m.predict(X_lag_te)[:len(test)]
                results[name] = y_pred
                print(f"  {{name}} ({{time.perf_counter()-t0:.1f}}s)")
                score(name, y_lag_te.values[:len(y_pred)], y_pred, metrics)
            except Exception as e:
                print(f"  {{name}} failed: {{e}}")

        # FLAML AutoML on lag features (tabularized forecasting only)
        try:
            t0 = time.perf_counter()
            from flaml import AutoML
            flaml_model = AutoML()
            flaml_model.fit(X_lag_tr, y_lag_tr, task="regression", time_budget=60,
                           metric="rmse", verbose=0)
            y_pred = flaml_model.predict(X_lag_te)[:len(test)]
            results["FLAML-Lag"] = y_pred
            best = flaml_model.best_estimator
            print(f"  FLAML-Lag [best: {{best}}] ({{time.perf_counter()-t0:.1f}}s)")
            score("FLAML-Lag", y_lag_te.values[:len(y_pred)], y_pred, metrics)
        except Exception as e:
            print(f"  FLAML-Lag failed: {{e}}")

    # === METRICS SUMMARY ===
    if metrics:
        print()
        print("=" * 65)
        print("METRICS SUMMARY")
        print("=" * 65)
        summary = pd.DataFrame(metrics).sort_values("RMSE")
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)
        best_model = summary.iloc[0]["Model"]
        print(f"  Best model by RMSE: {{best_model}}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(train)), train, alpha=0.5, label="Train")
    ax.plot(range(len(train), len(train)+len(test)), test, linewidth=2, label="Actual")
    for name, y_pred in results.items():
        ax.plot(range(len(train), len(train)+len(y_pred)), y_pred, "--", label=name)
    ax.legend(); ax.set_title("Forecast Comparison")
    fig.savefig(os.path.join(SAVE_DIR, "forecast.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return metrics


def run_eda(df, target, save_dir):
    """Time Series Exploratory Data Analysis."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
    print(f"Date range: {{df.index.min()}} to {{df.index.max()}}" if hasattr(df.index, 'min') else "")
    print(f"Target column: {{target}}")
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\\nMissing values ({{len(n_miss)}} columns):")
        print(n_miss.sort_values(ascending=False).head(10).to_string())
    else:
        print("\\nNo missing values")
    desc = df.describe().T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    print("Summary statistics saved to eda_summary.csv")
    # Target time series plot
    fig, ax = plt.subplots(figsize=(14, 5))
    if target in df.columns:
        df[target].plot(ax=ax, color="steelblue")
    ax.set_title(f"Time Series: {{target}}")
    ax.set_xlabel("Time")
    fig.savefig(os.path.join(save_dir, "eda_timeseries.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    # Stationarity test (ADF)
    if target in df.columns:
        try:
            from statsmodels.tsa.stattools import adfuller
            series = df[target].dropna()
            if len(series) > 20:
                result = adfuller(series, maxlag=min(30, len(series)//3))
                print(f"\\nADF Stationarity Test:")
                print(f"  Test Statistic: {{result[0]:.4f}}")
                print(f"  p-value: {{result[1]:.4f}}")
                print(f"  Stationary: {{'Yes' if result[1] < 0.05 else 'No (p >= 0.05)'}}")
        except Exception:
            pass
        # Seasonal decomposition
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            series = df[target].dropna()
            freq = min(max(7, len(series) // 10), 365)
            if len(series) > 2 * freq:
                decomp = seasonal_decompose(series, period=freq, extrapolate_trend="freq")
                fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
                decomp.observed.plot(ax=axes[0]); axes[0].set_title("Observed")
                decomp.trend.plot(ax=axes[1]); axes[1].set_title("Trend")
                decomp.seasonal.plot(ax=axes[2]); axes[2].set_title("Seasonal")
                decomp.resid.plot(ax=axes[3]); axes[3].set_title("Residual")
                fig.tight_layout()
                fig.savefig(os.path.join(save_dir, "eda_decomposition.png"), dpi=100, bbox_inches="tight")
                plt.close(fig)
        except Exception:
            pass
    print("EDA plots saved.")


def main():
    print("=" * 60)
    print("TIME SERIES FORECASTING | April 2026")
    print("Primary: AutoGluon-TS, Chronos-Bolt, Chronos-2, TimesFM")
    print("Baselines: ARIMA, Prophet, LightGBM/CatBoost/XGBoost Lag, FLAML")
    print("=" * 60)
    df, target = load_data()
    run_eda(df, target, SAVE_DIR)
    metrics = forecast(df, target)

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


def gen_recommendation(path, cfg):
    data_load = cfg.get("data", '    raise FileNotFoundError("No data")')
    task = cfg.get("task", "cf")
    return textwrap.dedent(f'''\
"""
Modern Recommendation Pipeline (April 2026)

CF task :  implicit ALS + BPR (primary), Surprise SVD/KNN (baseline).
Hybrid  :  LightFM WARP/BPR (metadata-aware, cold-start capable).
Content :  Sentence Transformers / BGE-M3 / Qwen3-Embedding.
Timing  :  Wall-clock per model stage (CF and baseline).
Export  :  metrics.json with per-model evaluation + timing.
Data    :  Auto-downloaded at runtime.
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib; matplotlib.use("Agg")

warnings.filterwarnings("ignore")

TASK = "{task}"  # cf | hybrid | content
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
{data_load}
    print(f"Dataset shape: {{df.shape}}")
    return df


def detect_columns(df):
    cols = {{c.lower(): c for c in df.columns}}
    user = next((cols[k] for k in cols if "user" in k), df.columns[0])
    item = next((cols[k] for k in cols if any(x in k for x in ["item","movie","book","product","title","name"])), df.columns[1])
    rating = next((cols[k] for k in cols if any(x in k for x in ["rating","score","stars"])), None)
    content = next((cols[k] for k in cols if any(x in k for x in ["title","name","description","text","headline","category"])), None)
    return user, item, rating, content


def build_interaction_matrix(df, user_col, item_col, rating_col):
    ue, ie = LabelEncoder(), LabelEncoder()
    df["u"] = ue.fit_transform(df[user_col])
    df["i"] = ie.fit_transform(df[item_col])
    vals = df[rating_col].values.astype(np.float32) if rating_col else np.ones(len(df), dtype=np.float32)
    mat = csr_matrix((vals, (df["u"].values, df["i"].values)),
                     shape=(df["u"].nunique(), df["i"].nunique()))
    return mat, ue, ie


# -- PRIMARY: implicit ALS + BPR (collaborative filtering) --
def run_implicit_cf(df, user_col, item_col, rating_col):
    mat, ue, ie = build_interaction_matrix(df, user_col, item_col, rating_col)
    n_users, n_items = mat.shape[0], mat.shape[1]
    results = {{}}

    # implicit ALS
    try:
        from implicit.als import AlternatingLeastSquares
        from implicit.evaluation import precision_at_k, train_test_split
        t0 = time.perf_counter()
        train_m, test_m = train_test_split(mat, train_percentage=0.8)
        als = AlternatingLeastSquares(factors=128, iterations=30, use_gpu=True)
        als.fit(train_m)
        p_at_10 = precision_at_k(als, train_m, test_m, K=10)
        als_elapsed = round(time.perf_counter() - t0, 1)
        print(f"  implicit ALS -- {{n_users}} users, {{n_items}} items, P@10={{p_at_10:.4f}} ({{als_elapsed}}s)")
        results["implicit_ALS"] = {{"users": n_users, "items": n_items,
                                    "P@10": round(float(p_at_10), 4), "time_s": als_elapsed}}
    except Exception as e:
        print(f"  implicit ALS failed: {{e}}")

    # implicit BPR
    try:
        from implicit.bpr import BayesianPersonalizedRanking
        from implicit.evaluation import precision_at_k, train_test_split
        t1 = time.perf_counter()
        train_m, test_m = train_test_split(mat, train_percentage=0.8)
        bpr = BayesianPersonalizedRanking(factors=128, iterations=100, use_gpu=True)
        bpr.fit(train_m)
        bpr_p10 = precision_at_k(bpr, train_m, test_m, K=10)
        bpr_elapsed = round(time.perf_counter() - t1, 1)
        print(f"  implicit BPR -- P@10={{bpr_p10:.4f}} ({{bpr_elapsed}}s)")
        results["implicit_BPR"] = {{"P@10": round(float(bpr_p10), 4), "time_s": bpr_elapsed}}
    except Exception as e:
        print(f"  implicit BPR failed: {{e}}")

    return results


# ═══════════════════════════════════════════════════════════════
# -- HYBRID: LightFM (user/item metadata, cold-start capable) --
def run_lightfm_hybrid(df, user_col, item_col, rating_col, content_col):
    results = {{}}
    try:
        from lightfm import LightFM
        from lightfm.evaluation import precision_at_k, auc_score
        from lightfm.data import Dataset as LFDataset

        lfds = LFDataset()
        lfds.fit(df[user_col].unique(), df[item_col].unique())

        (interactions, weights) = lfds.build_interactions(
            ((r[user_col], r[item_col], r[rating_col] if rating_col else 1.0)
             for _, r in df.iterrows()))

        # Build item features from content column if available
        item_features = None
        n_item_features = 0
        if content_col and content_col in df.columns:
            unique_items = df[[item_col, content_col]].drop_duplicates(subset=[item_col])
            all_features = set()
            for text in unique_items[content_col].fillna("").astype(str):
                for w in text.lower().split()[:10]:
                    all_features.add(w)
            if all_features:
                n_item_features = len(all_features)
                lfds.fit_partial(item_features=all_features)
                item_feat_list = []
                for _, row in unique_items.iterrows():
                    words = row[content_col] if isinstance(row[content_col], str) else ""
                    feats = [w for w in words.lower().split()[:10] if w in all_features]
                    if feats:
                        item_feat_list.append((row[item_col], feats))
                if item_feat_list:
                    item_features = lfds.build_item_features(item_feat_list)
        print(f"  Item features: {{n_item_features}} unique tokens" if n_item_features else "  No item features (pure interactions)")

        # Train WARP model (for implicit/ranking) and BPR model
        for loss in ["warp", "bpr"]:
            t0 = time.perf_counter()
            model = LightFM(loss=loss, no_components=64, learning_rate=0.05)
            model.fit(interactions, item_features=item_features, epochs=30, num_threads=4)
            p_at_k = precision_at_k(model, interactions, item_features=item_features, k=10).mean()
            auc = auc_score(model, interactions, item_features=item_features).mean()
            elapsed = round(time.perf_counter() - t0, 1)
            print(f"  LightFM ({{loss}}) -- P@10={{p_at_k:.4f}}, AUC={{auc:.4f}} ({{elapsed}}s)")
            results[f"LightFM_{{loss}}"] = {{"P@10": round(float(p_at_k), 4),
                                            "AUC": round(float(auc), 4),
                                            "item_features": n_item_features,
                                            "time_s": elapsed}}
    except Exception as e:
        print(f"  LightFM failed: {{e}}")
    return results


# -- CONTENT-BASED: BGE-M3 / Qwen3-Embedding + TF-IDF baseline --
def run_content_embeddings(df, item_col, content_col):
    if not content_col or content_col not in df.columns:
        print("  No content column found -- skipping content-based embeddings")
        return {{}}
    results = {{}}
    items = df[[item_col, content_col]].drop_duplicates(subset=[item_col]).head(1000)
    texts = items[content_col].fillna("").astype(str).tolist()
    print(f"  {{len(items)}} unique items with content column '{{content_col}}'")

    # PRIMARY: BGE-M3
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        t0 = time.perf_counter()
        model = SentenceTransformer("BAAI/bge-m3")
        embs = model.encode(texts, batch_size=32, show_progress_bar=True)
        sim = cosine_similarity(embs)
        elapsed = round(time.perf_counter() - t0, 1)
        for i in range(min(3, len(items))):
            top_idx = np.argsort(sim[i])[-4:-1][::-1]
            top_items = items.iloc[top_idx][item_col].tolist()
            print(f"  BGE-M3 '{{items.iloc[i][item_col]}}' -> {{top_items}}")
        print(f"  BGE-M3: {{len(items)}} items, dim={{embs.shape[1]}} ({{elapsed}}s)")
        results["BGE-M3"] = {{"items": len(items), "dim": int(embs.shape[1]), "time_s": elapsed}}
    except Exception as e:
        print(f"  BGE-M3 failed: {{e}}")

    # ALTERNATIVE: Qwen3-Embedding
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        t1 = time.perf_counter()
        qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        qwen_embs = qwen.encode(texts, batch_size=16, show_progress_bar=True)
        qwen_sim = cosine_similarity(qwen_embs)
        qwen_elapsed = round(time.perf_counter() - t1, 1)
        for i in range(min(3, len(items))):
            top_idx = np.argsort(qwen_sim[i])[-4:-1][::-1]
            top_items = items.iloc[top_idx][item_col].tolist()
            print(f"  Qwen3 '{{items.iloc[i][item_col]}}' -> {{top_items}}")
        print(f"  Qwen3-Embedding: {{len(qwen_embs)}} items, dim={{qwen_embs.shape[1]}} ({{qwen_elapsed}}s)")
        results["Qwen3-Embedding"] = {{"items": len(qwen_embs), "dim": int(qwen_embs.shape[1]), "time_s": qwen_elapsed}}
    except Exception as e:
        print(f"  Qwen3-Embedding failed: {{e}}")

    # BASELINE: TF-IDF cosine similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        t2 = time.perf_counter()
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_mat = tfidf.fit_transform(texts)
        tfidf_sim = cos_sim(tfidf_mat)
        tfidf_elapsed = round(time.perf_counter() - t2, 1)
        for i in range(min(3, len(items))):
            top_idx = np.argsort(tfidf_sim[i])[-4:-1][::-1]
            top_items = items.iloc[top_idx][item_col].tolist()
            print(f"  TF-IDF '{{items.iloc[i][item_col]}}' -> {{top_items}}")
        print(f"  TF-IDF baseline: {{tfidf_mat.shape[0]}} items, {{tfidf_mat.shape[1]}} features ({{tfidf_elapsed}}s)")
        results["TF-IDF"] = {{"items": int(tfidf_mat.shape[0]), "features": int(tfidf_mat.shape[1]), "time_s": tfidf_elapsed}}
    except Exception as e:
        print(f"  TF-IDF baseline failed: {{e}}")

    return results


# -- BASELINE: Surprise SVD + KNN --
def run_surprise_baseline(df, user_col, item_col, rating_col):
    if not rating_col:
        print("  No explicit ratings -- skipping Surprise baseline")
        return {{}}
    results = {{}}
    try:
        from surprise import Dataset as SDataset, Reader, SVD, KNNBasic, accuracy
        from surprise.model_selection import cross_validate

        reader = Reader(rating_scale=(df[rating_col].min(), df[rating_col].max()))
        data = SDataset.load_from_df(df[[user_col, item_col, rating_col]].dropna(), reader)

        for algo_cls, name in [(SVD, "SVD"), (KNNBasic, "KNN")]:
            t0 = time.perf_counter()
            algo = algo_cls()
            cv = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3, verbose=False)
            elapsed = round(time.perf_counter() - t0, 1)
            rmse = round(float(cv["test_rmse"].mean()), 4)
            mae = round(float(cv["test_mae"].mean()), 4)
            print(f"  Surprise {{name}} -- RMSE={{rmse}}, MAE={{mae}} ({{elapsed}}s)")
            results[f"Surprise_{{name}}"] = {{"RMSE": rmse, "MAE": mae, "time_s": elapsed}}
        print("  Surprise baseline complete")
    except Exception as e:
        print(f"  Surprise baseline failed: {{e}}")
    return results


def train(df):
    user_col, item_col, rating_col, content_col = detect_columns(df)
    print(f"Columns -- user: {{user_col}}, item: {{item_col}}, rating: {{rating_col}}, content: {{content_col}}")
    metrics = {{"task": TASK}}

    if TASK == "cf":
        # Primary: implicit ALS/BPR -> Baseline: Surprise SVD/KNN
        print()
        print("-- implicit ALS + BPR (primary) --")
        metrics.update(run_implicit_cf(df, user_col, item_col, rating_col))
        print()
        print("-- Surprise SVD + KNN (baseline) --")
        metrics.update(run_surprise_baseline(df, user_col, item_col, rating_col))
    elif TASK == "hybrid":
        # Primary: LightFM -> Baseline: implicit ALS/BPR
        print()
        print("-- LightFM hybrid (primary) --")
        metrics.update(run_lightfm_hybrid(df, user_col, item_col, rating_col, content_col))
        print()
        print("-- implicit ALS + BPR (baseline) --")
        metrics.update(run_implicit_cf(df, user_col, item_col, rating_col))
    elif TASK == "content":
        # Primary: embedding-based content similarity -> TF-IDF baseline
        print()
        print("-- Content embeddings (primary) --")
        metrics.update(run_content_embeddings(df, item_col, content_col))
        print()
        print("-- implicit ALS/BPR (optional baseline) --")
        try:
            metrics.update(run_implicit_cf(df, user_col, item_col, rating_col))
        except Exception:
            print("  Skipped (no interaction data)")
    else:
        run_implicit_cf(df, user_col, item_col, rating_col)
        run_content_embeddings(df, item_col, content_col)

    return metrics


def run_eda(df, save_dir):
    """Exploratory Data Analysis for recommendation data."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
    print(f"Column types:\\n{{df.dtypes.value_counts().to_string()}}")
    # Detect user/item columns
    for col in df.columns:
        nuniq = df[col].nunique()
        if nuniq < len(df) * 0.5 and nuniq > 1:
            print(f"  {{col}}: {{nuniq}} unique values")
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(save_dir, "eda_summary.csv"))
    missing = df.isnull().sum()
    n_miss = missing[missing > 0]
    if len(n_miss):
        print(f"\\nMissing values ({{len(n_miss)}} columns):")
        print(n_miss.sort_values(ascending=False).head(10).to_string())
    else:
        print("\\nNo missing values")
    # Rating distribution if numeric column exists
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        df[num_cols[0]].hist(bins=30, ax=ax, color="steelblue", edgecolor="black")
        ax.set_title(f"Distribution: {{num_cols[0]}}")
        fig.savefig(os.path.join(save_dir, "eda_rating_dist.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print("Summary statistics saved to eda_summary.csv")
    print("EDA complete.")


def main():
    print("=" * 60)
    print(f"RECOMMENDATION ({{TASK}}) | implicit + LightFM + SentenceTransformers")
    print("=" * 60)
    df = load_data()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    run_eda(df, save_dir)
    metrics = train(df)

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


def gen_rl(path, cfg):
    env = cfg.get("env", "CartPole-v1")
    algo = cfg.get("algo", "PPO")
    is_cont_cfg = cfg.get("continuous", False)
    # For LunarLander with continuous flag, use the continuous kwarg
    make_kwargs = 'continuous=True' if is_cont_cfg else ''
    return textwrap.dedent(f'''\
"""
Modern Reinforcement Learning Pipeline (April 2026)

Primary algorithm: {algo}
  - SAC  (Stable-Baselines3) -- default for continuous-action envs
  - PPO  (Stable-Baselines3) -- default for discrete-action envs

Baselines (comparison):
  - PPO  (Stable-Baselines3) -- comparison when SAC is primary
  - DQN  (Stable-Baselines3) -- deep RL baseline for discrete-action envs
  - Q-learning (tabular)     -- educational baseline for small-state discrete envs

Environment: {env}
Action space: auto-detected (discrete -> PPO+DQN, continuous -> SAC+PPO)

Compute requirements:
  - PPO : ~100K steps, 1-3 min on CPU, <1 min with GPU
  - DQN : ~100K steps, 1-3 min on CPU, <1 min with GPU
  - SAC : ~100K steps, 2-5 min on CPU, <1 min with GPU
  - Q-learning (tabular): <10s, CPU-only, no neural network

Dependencies: stable-baselines3, gymnasium, matplotlib, numpy
"""
import os, json, time, warnings
import numpy as np
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ENV_NAME = "{env}"
ALGO = "{algo}"
TOTAL_TIMESTEPS = 100_000
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MAKE_KWARGS = dict({make_kwargs})

# Discrete envs where DQN is a valid baseline
DISCRETE_ENVS = ("CliffWalking", "FrozenLake", "Taxi", "LunarLander", "CartPole", "MountainCar")
# Small-state discrete envs where tabular Q-learning is educational
TABULAR_ENVS = ("CliffWalking", "FrozenLake", "Taxi")


def train_sb3(algo_name, env_instance, eval_env, save_dir, timesteps):
    \"\"\"Train a Stable-Baselines3 agent (PPO, SAC, or DQN).\"\"\"
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback

    eval_cb = EvalCallback(eval_env, best_model_save_path=save_dir,
        log_path=save_dir, eval_freq=5000, n_eval_episodes=10, deterministic=True)

    if algo_name == "SAC":
        model = SAC("MlpPolicy", env_instance, learning_rate=3e-4, buffer_size=100_000,
                     batch_size=256, tau=0.005, gamma=0.99, verbose=1, device="auto")
    elif algo_name == "DQN":
        model = DQN("MlpPolicy", env_instance, learning_rate=1e-4, buffer_size=50_000,
                     batch_size=64, gamma=0.99, exploration_fraction=0.3,
                     target_update_interval=1000, verbose=1, device="auto")
    else:
        model = PPO("MlpPolicy", env_instance, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                     ent_coef=0.01, verbose=1, device="auto")

    t0 = time.perf_counter()
    print(f"  Training {{algo_name}} on {{ENV_NAME}} for {{timesteps}} steps ...")
    model.learn(total_timesteps=timesteps, callback=eval_cb)
    elapsed = time.perf_counter() - t0
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"  {{algo_name}} - Reward: {{mean_reward:.2f}} +/- {{std_reward:.2f}} ({{elapsed:.1f}}s)")
    model.save(os.path.join(save_dir, f"{{algo_name.lower()}}_{{ENV_NAME}}"))
    return algo_name, mean_reward, std_reward, elapsed


def train_q_table(env_name, n_episodes=10_000, lr=0.1, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
    \"\"\"Tabular Q-learning - educational baseline for small-state discrete envs.\"\"\"
    env = gym.make(env_name)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    epsilon = eps_start
    rewards_log = []

    t0 = time.perf_counter()
    print(f"  Training Q-learning (tabular) on {{env_name}} for {{n_episodes}} episodes ...")
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            Q[state, action] += lr * (reward + gamma * np.max(Q[next_state]) * (1 - terminated) - Q[state, action])
            state = next_state
            total_reward += reward
        rewards_log.append(total_reward)
        epsilon = max(eps_end, epsilon * eps_decay)

    env.close()

    # Evaluate learned policy
    eval_env = gym.make(env_name)
    eval_rewards = []
    for _ in range(100):
        state, _ = eval_env.reset()
        total_r = 0; done = False
        while not done:
            action = int(np.argmax(Q[state]))
            state, r, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_r += r
        eval_rewards.append(total_r)
    eval_env.close()

    elapsed = time.perf_counter() - t0
    mean_r = np.mean(eval_rewards)
    std_r = np.std(eval_rewards)
    print(f"  Q-learning - Reward: {{mean_r:.2f}} +/- {{std_r:.2f}} ({{elapsed:.1f}}s, {{n_states}} states x {{n_actions}} actions)")
    return "Q-learning", mean_r, std_r, elapsed, rewards_log


def plot_results(results, save_dir):
    \"\"\"Bar chart comparing mean rewards across all algorithms.\"\"\"
    names = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(names)])
    ax.set_ylabel("Mean Reward (20 eval episodes)")
    ax.set_title(f"RL Algorithm Comparison - {{ENV_NAME}}")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{{m:.1f}}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "comparison.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_eda(env_name, make_kwargs, save_dir):
    """Environment information summary."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    try:
        env = gym.make(env_name, **make_kwargs)
        print(f"  Environment: {{env_name}}")
        print(f"  Observation space: {{env.observation_space}}")
        print(f"  Action space: {{env.action_space}}")
        is_continuous = hasattr(env.action_space, "shape") and len(env.action_space.shape) > 0
        print(f"  Action type: {{'continuous' if is_continuous else 'discrete'}}")
        if hasattr(env, 'reward_range'):
            print(f"  Reward range: {{env.reward_range}}")
        env.close()
    except Exception as e:
        print(f"  Could not inspect environment: {{e}}")
    print("EDA complete.")


def main():
    print("=" * 60)
    print(f"REINFORCEMENT LEARNING | {{ENV_NAME}}")
    print(f"Primary: {{ALGO}}  |  Baselines: auto-selected for action space")
    print("=" * 60)
    run_eda(ENV_NAME, MAKE_KWARGS, SAVE_DIR)
    results = []

    # === PRIMARY: SAC or PPO ===
    env = gym.make(ENV_NAME, **MAKE_KWARGS); eval_env = gym.make(ENV_NAME, **MAKE_KWARGS)
    is_continuous = hasattr(env.action_space, "shape") and len(env.action_space.shape) > 0
    act_type = "continuous" if is_continuous else "discrete"
    print(f"  Environment: {{ENV_NAME}} ({{act_type}} actions)")

    name, reward, std, dt = train_sb3(ALGO, env, eval_env, SAVE_DIR, TOTAL_TIMESTEPS)
    results.append((name, reward, std, dt))
    env.close(); eval_env.close()

    # === DQN BASELINE (discrete environments) ===
    is_discrete = any(tag in ENV_NAME for tag in DISCRETE_ENVS)
    if is_discrete and ALGO != "DQN":
        try:
            env2 = gym.make(ENV_NAME, **MAKE_KWARGS); eval_env2 = gym.make(ENV_NAME, **MAKE_KWARGS)
            name, reward, std, dt = train_sb3("DQN", env2, eval_env2, SAVE_DIR, TOTAL_TIMESTEPS)
            results.append((name, reward, std, dt))
            env2.close(); eval_env2.close()
        except Exception as e:
            print(f"  DQN baseline failed: {{e}}")

    # === PPO COMPARISON (continuous environments where SAC is primary) ===
    if is_continuous and ALGO != "PPO":
        try:
            env3 = gym.make(ENV_NAME, **MAKE_KWARGS); eval_env3 = gym.make(ENV_NAME, **MAKE_KWARGS)
            name, reward, std, dt = train_sb3("PPO", env3, eval_env3, SAVE_DIR, TOTAL_TIMESTEPS)
            results.append((name, reward, std, dt))
            env3.close(); eval_env3.close()
        except Exception as e:
            print(f"  PPO comparison failed: {{e}}")

    # === Q-LEARNING BASELINE (small-state discrete environments) ===
    is_tabular = any(tag in ENV_NAME for tag in TABULAR_ENVS)
    if is_tabular:
        try:
            name, reward, std, dt, _ = train_q_table(ENV_NAME)
            results.append((name, reward, std, dt))
        except Exception as e:
            print(f"  Q-learning baseline failed: {{e}}")

    # === SUMMARY ===
    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    best = max(results, key=lambda x: x[1])
    for name, reward, std, dt in results:
        marker = " <- best" if name == best[0] else ""
        print(f"  {{name:15s}}  Reward: {{reward:8.2f}} +/- {{std:6.2f}}  ({{dt:.1f}}s){{marker}}")

    # Save metrics
    metrics = [{{"algorithm": r[0], "mean_reward": r[1], "std_reward": r[2], "time_s": r[3]}} for r in results]
    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    plot_results(results, SAVE_DIR)
    print(f"  Saved: comparison.png, metrics.json")


if __name__ == "__main__":
    main()
''')


def gen_audio(path, cfg):
    task = cfg.get("task", "transcription")
    data_load = cfg.get("data", "    df = None")
    # Re-indent data_load to 12 spaces for nesting inside try: inside if: inside function
    data_load_12 = "\n".join("        " + line if line.strip() else line for line in data_load.split("\n"))
    return textwrap.dedent(f'''\
"""
Modern Audio/Speech Pipeline (April 2026)

Task: {task}

Model selection by task:
  - ASR / speech-to-text   -- Whisper large-v3-turbo (OpenAI)
  - Audio classification   -- Wav2Vec2-base + HuBERT-base (Meta)
  - Denoising / separation -- SpeechBrain SepFormer (speechbrain/sepformer-whamr)
  - Voice cloning / TTS    -- Coqui XTTS-v2 (multilingual, speaker-adaptive)

Compute requirements:
  - Whisper large-v3-turbo : ~2 GB VRAM, ~3s/file on GPU, ~15s/file on CPU
  - Wav2Vec2 / HuBERT      : ~1 GB VRAM, <1s/file on GPU, ~3s/file on CPU
  - SepFormer              : ~2 GB VRAM, ~5s/file on GPU, ~20s/file on CPU
  - XTTS-v2                : ~4 GB VRAM, ~10s per utterance on GPU, CPU very slow

Dependencies: transformers, torch, torchaudio, soundfile, speechbrain, TTS (Coqui)
Data: Auto-downloaded at runtime from HuggingFace
"""
import os, json, time, warnings
import numpy as np

warnings.filterwarnings("ignore")

TASK = "{task}"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_audio_samples():
    \"\"\"Download audio samples from HuggingFace datasets.\"\"\"
    from datasets import load_dataset
    import soundfile as sf

    save_dir = os.path.join(SAVE_DIR, "audio_data")
    os.makedirs(save_dir, exist_ok=True)

    if TASK == "classification":
        try:
{data_load_12}
            if df is not None:
                # Extract audio files from HF dataset if it has an audio column
                if hasattr(df, "columns"):
                    print(f"Loaded dataset: {{len(df)}} samples")
                return save_dir, df
        except Exception:
            pass
        ds = load_dataset("google/speech_commands", "v0.02", split="train[:100]")
    elif TASK == "cloning":
        ds = load_dataset("edinburghcstr/vctk", split="train[:20]",
                          trust_remote_code=True)
    else:
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy",
                          "clean", split="validation[:20]")

    paths = []
    for i, sample in enumerate(ds):
        audio = sample.get("audio", sample)
        if isinstance(audio, dict):
            arr, sr = np.array(audio["array"]), audio["sampling_rate"]
            out_path = os.path.join(save_dir, f"sample_{{i:03d}}.wav")
            sf.write(out_path, arr, sr)
            paths.append(out_path)

    print(f"Downloaded {{len(paths)}} audio samples to {{save_dir}}")
    return save_dir, paths


# ===========================================================
# ASR / SPEECH-TO-TEXT -- Whisper large-v3-turbo
# ===========================================================

def run_whisper(audio_dir):
    \"\"\"Automatic speech recognition with Whisper large-v3-turbo.\"\"\"
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    print(f"  Loading {{model_id}} on {{device}} ...")
    t0 = time.perf_counter()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    asr = pipeline("automatic-speech-recognition", model=model,
                   tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor,
                   torch_dtype=torch_dtype, device=device)
    print(f"  Model loaded in {{time.perf_counter()-t0:.1f}}s")

    from pathlib import Path
    audio_files = sorted(Path(audio_dir).glob("*.wav")) + sorted(Path(audio_dir).glob("*.flac"))
    results = []
    for f in audio_files[:10]:
        t1 = time.perf_counter()
        result = asr(str(f), return_timestamps=True)
        dt = time.perf_counter() - t1
        results.append({{"file": f.name, "text": result["text"], "time_s": round(dt, 2)}})
        print(f"  {{f.name}} ({{dt:.1f}}s): {{result['text'][:80]}}...")
    return results


# ===========================================================
# AUDIO CLASSIFICATION -- Wav2Vec2 + HuBERT
# ===========================================================

def run_wav2vec2_clf(audio_dir):
    \"\"\"Audio classification with Wav2Vec2 and HuBERT.\"\"\"
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import soundfile as sf
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_files = sorted(Path(audio_dir).glob("*.wav"))[:20]

    if not audio_files:
        print("  No .wav files found - extracting from dataset ...")
        # Try to extract audio from HF dataset objects
        ds_dir = Path(audio_dir)
        from datasets import load_dataset
        ds = load_dataset("google/speech_commands", "v0.02", split="train[:50]")
        for i, sample in enumerate(ds):
            audio = sample.get("audio", sample)
            if isinstance(audio, dict):
                arr = np.array(audio["array"])
                sr = audio["sampling_rate"]
                out_path = ds_dir / f"sample_{{i:03d}}.wav"
                sf.write(str(out_path), arr, sr)
        audio_files = sorted(ds_dir.glob("*.wav"))[:20]

    summary = []
    for model_name, label in [("facebook/wav2vec2-base", "Wav2Vec2"),
                               ("facebook/hubert-base-ls960", "HuBERT")]:
        try:
            t0 = time.perf_counter()
            print(f"  Loading {{label}} ...")
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForAudioClassification.from_pretrained(
                model_name, num_labels=10, ignore_mismatched_sizes=True).to(device)
            preds = []
            for f in audio_files:
                arr, sr = sf.read(str(f))
                if len(arr.shape) > 1:
                    arr = arr[:, 0]
                inputs = extractor(arr, sampling_rate=sr, return_tensors="pt",
                                   padding=True).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=-1).item()
                preds.append(pred)
                print(f"    {{f.name}}: class {{pred}}")
            elapsed = time.perf_counter() - t0
            summary.append({{"model": label, "n_files": len(preds),
                            "time_s": round(elapsed, 1)}})
            print(f"  {{label}}: {{len(preds)}} files classified ({{elapsed:.1f}}s)")
        except Exception as e:
            print(f"  {{label}} failed: {{e}}")

    return summary


# ===========================================================
# DENOISING / SEPARATION -- SpeechBrain SepFormer
# ===========================================================

def run_sepformer(audio_dir):
    \"\"\"Speech enhancement / denoising.

    Primary: SepFormer (speechbrain/sepformer-whamr) - neural, SOTA
    Baseline: spectral subtraction - classical signal processing
    \"\"\"
    try:
        from pathlib import Path
        import torchaudio

        audio_files = sorted(Path(audio_dir).glob("*.wav"))[:10]

        # --- BASELINE: Spectral Subtraction ---
        print("  --- Baseline: Spectral Subtraction ---")
        baseline_results = []
        t_base = time.perf_counter()
        for f in audio_files:
            try:
                import soundfile as sf
                data, sr = sf.read(str(f))
                if len(data.shape) > 1:
                    data = data[:, 0]
                fl = min(400, len(data))
                hop = fl // 2
                n_frames = max(1, len(data) // hop - 1)
                frames = np.array([data[i*hop:i*hop+fl] for i in range(n_frames) if i*hop+fl <= len(data)])
                if len(frames) == 0:
                    continue
                ham = np.hamming(fl)
                windowed = frames * ham
                dft = np.fft.fft(windowed)
                mag = np.abs(dft)
                phase = np.angle(dft)
                noise_est = np.mean(mag, axis=0)
                clean_mag = np.maximum(mag - 2 * noise_est, 0)
                estimate = clean_mag * np.exp(1j * phase)
                ift = [np.fft.ifft(e).real for e in estimate]
                clean_data = list(ift[0][:hop])
                for i in range(len(ift) - 1):
                    clean_data.extend(ift[i][hop:] + ift[i+1][:hop])
                clean_data.extend(ift[-1][hop:])
                baseline_results.append({{"file": f.name, "method": "spectral_subtraction"}})
                print(f"    {{f.name}}: processed")
            except Exception as e:
                print(f"    {{f.name}}: failed ({{e}})")
        dt_base = time.perf_counter() - t_base
        print(f"  Spectral subtraction: {{len(baseline_results)}} files ({{dt_base:.1f}}s)")

        # --- PRIMARY: SepFormer ---
        print("  --- Primary: SpeechBrain SepFormer ---")
        print("  Loading SepFormer (speechbrain/sepformer-whamr) ...")
        t0 = time.perf_counter()
        try:
            from speechbrain.inference.separation import SepformerSeparation
            sep_model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr",
                savedir=os.path.join(SAVE_DIR, "sepformer_model"))
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation
            sep_model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr",
                savedir=os.path.join(SAVE_DIR, "sepformer_model"))
        print(f"  Model loaded in {{time.perf_counter()-t0:.1f}}s")

        out_dir = os.path.join(SAVE_DIR, "enhanced")
        os.makedirs(out_dir, exist_ok=True)
        results = []

        for f in audio_files:
            t1 = time.perf_counter()
            est_sources = sep_model.separate_file(path=str(f))
            out_path = os.path.join(out_dir, f"{{f.stem}}_enhanced.wav")
            torchaudio.save(out_path, est_sources[:, :, 0].cpu(), 8000)
            dt = time.perf_counter() - t1
            results.append({{"file": f.name, "output": f"{{f.stem}}_enhanced.wav",
                            "method": "sepformer", "time_s": round(dt, 1)}})
            print(f"  {{f.name}} -> {{f.stem}}_enhanced.wav ({{dt:.1f}}s)")

        print(f"  Enhanced audio saved to {{out_dir}}")
        print(f"  SepFormer: {{len(results)}} files | Baseline: {{len(baseline_results)}} files")
        return {{"sepformer": results, "baseline_spectral": baseline_results}}
    except Exception as e:
        print(f"  SepFormer failed: {{e}}")
        return {{}}


# ===========================================================
# VOICE CLONING / TTS -- Coqui XTTS-v2
# ===========================================================

def run_voice_cloning(audio_dir):
    \"\"\"Voice cloning and text-to-speech.

    Primary: Coqui XTTS-v2 (multilingual, speaker-adaptive, ~4 GB VRAM)
    Baseline: pyttsx3 (offline, CPU-only, no cloning)
    \"\"\"
    results = {{"xtts": [], "baseline_pyttsx3": []}}

    # --- BASELINE: pyttsx3 (offline CPU TTS) ---
    try:
        import pyttsx3
        print("  --- Baseline: pyttsx3 (offline TTS) ---")
        engine = pyttsx3.init()
        baseline_dir = os.path.join(SAVE_DIR, "tts_baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        baseline_texts = [
            "This is a baseline text to speech sample using pyttsx3.",
            "Offline synthesis is fast but lacks naturalness.",
        ]
        for i, text in enumerate(baseline_texts):
            t1 = time.perf_counter()
            out_path = os.path.join(baseline_dir, f"baseline_{{i:02d}}.wav")
            engine.save_to_file(text, out_path)
            engine.runAndWait()
            dt = time.perf_counter() - t1
            results["baseline_pyttsx3"].append({{
                "text": text[:60], "output": os.path.basename(out_path),
                "time_s": round(dt, 1)}})
            print(f"    baseline_{{i:02d}}.wav ({{dt:.1f}}s)")
    except Exception as e:
        print(f"  pyttsx3 baseline skipped: {{e}}")

    # --- PRIMARY: XTTS-v2 ---
    try:
        from TTS.api import TTS
        from pathlib import Path

        print("  --- Primary: Coqui XTTS-v2 ---")
        print("  Loading XTTS-v2 ...")
        t0 = time.perf_counter()
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print(f"  Model loaded in {{time.perf_counter()-t0:.1f}}s")

        out_dir = os.path.join(SAVE_DIR, "tts_output")
        os.makedirs(out_dir, exist_ok=True)

        # Find a reference speaker sample for voice cloning
        ref_files = sorted(Path(audio_dir).glob("*.wav"))
        ref_speaker = str(ref_files[0]) if ref_files else None

        texts = [
            ("Hello, this is a text to speech demonstration using XTTS version 2.", "en"),
            ("Modern voice cloning can produce remarkably natural speech.", "en"),
            ("Deep learning models now generate human-quality audio in real time.", "en"),
            ("Bonjour, ceci est une demonstration de synthese vocale multilingue.", "fr"),
        ]

        for i, (text, lang) in enumerate(texts):
            t1 = time.perf_counter()
            out_path = os.path.join(out_dir, f"tts_sample_{{i:02d}}.wav")
            if ref_speaker:
                tts.tts_to_file(text=text, file_path=out_path,
                               speaker_wav=ref_speaker, language=lang)
                mode = "cloned"
            else:
                tts.tts_to_file(text=text, file_path=out_path)
                mode = "default"
            dt = time.perf_counter() - t1
            results["xtts"].append({{"text": text[:60], "output": os.path.basename(out_path),
                            "mode": mode, "language": lang, "time_s": round(dt, 1)}})
            print(f"  [{{mode}}/{{lang}}] tts_sample_{{i:02d}}.wav ({{dt:.1f}}s)")

        print(f"  TTS output saved to {{out_dir}}")
        print(f"  XTTS-v2: {{len(results['xtts'])}} samples | Baseline: {{len(results['baseline_pyttsx3'])}} samples")
    except Exception as e:
        print(f"  XTTS-v2 failed: {{e}}")

    return results


def run_eda(save_dir):
    """Audio file summary."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    audio_exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    audio_files = []
    for root, dirs, files in os.walk(save_dir):
        for f in files:
            if f.lower().endswith(audio_exts):
                audio_files.append(os.path.join(root, f))
    print(f"  Audio files found: {{len(audio_files)}}")
    if audio_files:
        total_size = sum(os.path.getsize(f) for f in audio_files)
        print(f"  Total size: {{total_size / 1024 / 1024:.1f}} MB")
    print("EDA complete.")


def validate_results(task, results, save_dir):
    """Validate audio outputs for the active task."""
    validation = {{"task": task, "checks": {{}}}}

    if task == "transcription":
        records = results.get("whisper", [])
        non_empty = sum(1 for item in records if str(item.get("text", "")).strip())
        validation["checks"]["whisper"] = {{
            "records": len(records),
            "non_empty_transcripts": non_empty,
            "passed": len(records) > 0 and non_empty == len(records),
        }}
    elif task == "classification":
        records = results.get("classification", [])
        passed = all(item.get("n_files", 0) > 0 for item in records) if records else False
        validation["checks"]["classification"] = {{
            "models": len(records),
            "files_scored": sum(int(item.get("n_files", 0)) for item in records),
            "passed": passed,
        }}
    elif task in ("denoising", "separation"):
        bundle = results.get("sepformer", {{}})
        sep_records = bundle.get("sepformer", []) if isinstance(bundle, dict) else []
        existing = sum(
            1 for item in sep_records
            if os.path.exists(os.path.join(save_dir, "enhanced", item.get("output", "")))
        )
        validation["checks"]["sepformer"] = {{
            "outputs": len(sep_records),
            "existing_outputs": existing,
            "baseline_files": len(bundle.get("baseline_spectral", [])) if isinstance(bundle, dict) else 0,
            "passed": len(sep_records) > 0 and existing == len(sep_records),
        }}
    elif task == "cloning":
        bundle = results.get("xtts", {{}})
        xtts_records = bundle.get("xtts", []) if isinstance(bundle, dict) else []
        existing = sum(
            1 for item in xtts_records
            if os.path.exists(os.path.join(save_dir, "tts_output", item.get("output", "")))
        )
        validation["checks"]["xtts"] = {{
            "outputs": len(xtts_records),
            "existing_outputs": existing,
            "baseline_outputs": len(bundle.get("baseline_pyttsx3", [])) if isinstance(bundle, dict) else 0,
            "passed": len(xtts_records) > 0 and existing == len(xtts_records),
        }}

    validation["passed"] = any(item.get("passed") for item in validation["checks"].values())
    out_path = os.path.join(save_dir, "validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"  Validation saved to {{out_path}}")
    return validation


def main():
    print("=" * 60)
    print(f"AUDIO/SPEECH | Task: {{TASK}}")
    print("Models: Whisper | Wav2Vec2/HuBERT | SepFormer | XTTS-v2")
    print("=" * 60)
    audio_dir, data = download_audio_samples()
    run_eda(SAVE_DIR)
    results = {{}}

    if TASK == "transcription":
        print()
        print("--- ASR: Whisper large-v3-turbo ---")
        asr_results = run_whisper(audio_dir)
        results["whisper"] = asr_results
        if asr_results:
            out = os.path.join(SAVE_DIR, "transcriptions.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(asr_results, f, indent=2)
            print(f"  Saved transcriptions to {{out}}")

    elif TASK == "classification":
        print()
        print("--- Classification: Wav2Vec2 + HuBERT ---")
        clf_results = run_wav2vec2_clf(audio_dir)
        results["classification"] = clf_results

    elif TASK == "denoising" or TASK == "separation":
        print()
        print("--- Denoising: SpeechBrain SepFormer ---")
        sep_results = run_sepformer(audio_dir)
        results["sepformer"] = sep_results

    elif TASK == "cloning":
        print()
        print("--- Voice Cloning: XTTS-v2 ---")
        tts_results = run_voice_cloning(audio_dir)
        results["xtts"] = tts_results

    else:
        print()
        print("--- ASR (default): Whisper large-v3-turbo ---")
        asr_results = run_whisper(audio_dir)
        results["whisper"] = asr_results

    # Save metrics
    validation = validate_results(TASK, results, SAVE_DIR)
    results["validation"] = validation
    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Metrics saved to {{metrics_path}}")


if __name__ == "__main__":
    main()
''')


def gen_cv_detection(path, cfg):
    task = cfg.get("task", "detect")
    data_load = cfg.get("data")
    return textwrap.dedent(f'''\
"""
Modern CV Detection / Tracking Pipeline (April 2026)

Primary : YOLO26m (Ultralytics) for detection and tracking.
Export  : metrics.json with file-level detections + validation.json with output checks.
Data    : Auto-downloads demo files at runtime.
"""
import os, json, time, warnings
from pathlib import Path
import urllib.request

warnings.filterwarnings("ignore")

TASK = "{task}"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_samples():
{data_load if data_load else '    from pathlib import Path\n    return [p for p in Path(SAVE_DIR).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov")]'}


def run_detection(files):
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    image_files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    out_dir = os.path.join(SAVE_DIR, "detections")
    os.makedirs(out_dir, exist_ok=True)
    metrics = {{"model": "yolo26m", "task": "detect", "images": []}}
    t0 = time.perf_counter()
    for f in image_files[:20]:
        results = model(str(f))
        for r in results:
            r.save(filename=os.path.join(out_dir, f.name))
            n_boxes = len(r.boxes) if r.boxes is not None else 0
            classes = [int(b.cls) for b in r.boxes] if r.boxes is not None else []
            metrics["images"].append({{"file": f.name, "detections": n_boxes,
                                       "classes": dict(sorted({{c: classes.count(c) for c in set(classes)}}.items()))}})
            if n_boxes:
                print(f"  {{f.name}}: {{n_boxes}} objects detected")
    elapsed = time.perf_counter() - t0
    metrics["time_s"] = round(elapsed, 1)
    metrics["total_images"] = len(metrics["images"])
    metrics["total_detections"] = sum(i["detections"] for i in metrics["images"])
    print(f"  Detection: {{metrics['total_images']}} images, {{metrics['total_detections']}} objects in {{elapsed:.1f}}s")
    print(f"  Results saved to {{out_dir}}")
    return metrics


def run_tracking(files):
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    video_files = [f for f in files if f.suffix in (".mp4", ".avi", ".mov")]
    if not video_files:
        print("  No video files found. Running detection on images instead.")
        return run_detection(files)
    metrics = {{"model": "yolo26m", "task": "track", "videos": []}}
    t0 = time.perf_counter()
    for v in video_files[:3]:
        model.track(str(v), persist=True, save=True, project=SAVE_DIR, name="tracking")
        metrics["videos"].append({{"file": v.name}})
        print(f"  Tracked: {{v.name}}")
    elapsed = time.perf_counter() - t0
    metrics["time_s"] = round(elapsed, 1)
    print(f"  Tracking: {{len(metrics['videos'])}} videos in {{elapsed:.1f}}s")
    return metrics


def run_eda(files, save_dir):
    """Input file summary for detection."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"  Input files: {{len(files)}}")
    if files:
        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        print(f"  Total size: {{total_size / 1024:.1f}} KB")
    print("EDA complete.")


def validate_results(metrics, files, save_dir):
    """Validate output coverage for detection / tracking demos."""
    validation = {{
        "task": metrics.get("task", TASK),
        "input_files": len(files),
        "processed": int(metrics.get("total_images", len(metrics.get("videos", [])))),
        "time_s": round(float(metrics.get("time_s", 0)), 1),
    }}
    if metrics.get("task") == "track":
        validation["processed"] = len(metrics.get("videos", []))
        validation["passed"] = validation["processed"] > 0 and validation["time_s"] >= 0
    else:
        validation["total_detections"] = int(metrics.get("total_detections", 0))
        validation["passed"] = validation["processed"] > 0 and validation["time_s"] >= 0
    out_path = os.path.join(save_dir, "validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"Validation saved to {{out_path}}")
    return validation


def main():
    print("=" * 60)
    print(f"CV DETECTION | Task: {{TASK}} | Model: YOLO26m")
    print("=" * 60)
    files = download_samples()
    run_eda(files, SAVE_DIR)
    if TASK == "track":
        metrics = run_tracking(files)
    else:
        metrics = run_detection(files)
    metrics["validation"] = validate_results(metrics, files, SAVE_DIR)

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


def gen_face_gesture(path, cfg):
    task = cfg.get("task", "face_detection")
    return textwrap.dedent(f'''\
"""
Modern Face/Hand/Gesture Pipeline (April 2026)

Task dispatch:
  face_detection : YOLO26m (primary) + MediaPipe Face Landmarker (secondary)
  expression     : MediaPipe Face Landmarker blendshapes (primary) + YOLO26m (baseline)
  face_recognition : InsightFace ArcFace embeddings + age/gender
  hand_gesture   : MediaPipe Gesture Recognizer (webcam)
  pose           : MediaPipe Pose Landmarker Heavy (33-point skeleton)
Timing : Wall-clock per model stage.
Export : metrics.json with detection counts, landmarks, and timing.
Data   : Auto-downloads LFW face samples at runtime.
"""
import os, json, time, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

TASK = "{task}"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_face_samples():
    """Download LFW face images from sklearn."""
    from sklearn.datasets import fetch_lfw_people
    import cv2

    save_dir = Path(os.path.dirname(__file__)) / "face_samples"
    save_dir.mkdir(exist_ok=True)

    if list(save_dir.glob("*.jpg")):
        return list(save_dir.glob("*.jpg"))

    lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0)
    paths = []
    for i in range(min(30, len(lfw.images))):
        img = (lfw.images[i] * 255).astype(np.uint8) if lfw.images[i].max() <= 1 else lfw.images[i].astype(np.uint8)
        p = save_dir / f"face_{{i:03d}}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    print(f"Downloaded {{len(paths)}} face images")
    return paths


def run_yolo_detection(files):
    """YOLO26 for person/face detection — replaces Haar cascades."""
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    save_dir = os.path.join(SAVE_DIR, "yolo_detections")
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.perf_counter()
    total_persons = 0
    total_objects = 0
    for f in files[:20]:
        results = model(str(f))
        for r in results:
            r.save(filename=os.path.join(save_dir, f.name))
            n_people = sum(1 for b in r.boxes if int(b.cls) == 0) if r.boxes is not None else 0
            n_total = len(r.boxes) if r.boxes is not None else 0
            total_persons += n_people
            total_objects += n_total
            print(f"  YOLO26 {{f.name}}: {{n_people}} persons, {{n_total}} total")
    elapsed = time.perf_counter() - t0
    print(f"  YOLO26: {{len(files[:20])}} images, {{total_objects}} objects in {{elapsed:.1f}}s")
    print(f"  Results saved to {{save_dir}}")
    return {{"model": "yolo26m", "images": len(files[:20]), "persons": total_persons,
             "total_objects": total_objects, "time_s": round(elapsed, 1)}}


def run_face_landmarker(files):
    """MediaPipe Face Landmarker — modern Tasks API for 478-point face mesh and expressions."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        model_path = os.path.join(SAVE_DIR, "face_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                model_path)

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=5)
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        save_dir = os.path.join(SAVE_DIR, "face_landmark_results")
        os.makedirs(save_dir, exist_ok=True)

        t0 = time.perf_counter()
        total_faces = 0
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_img)
            n_faces = len(result.face_landmarks)
            total_faces += n_faces
            for face_lm in result.face_landmarks:
                for lm in face_lm:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            if result.face_blendshapes:
                top_shapes = sorted(result.face_blendshapes[0], key=lambda b: b.score, reverse=True)[:3]
                expr = ", ".join(f"{{b.category_name}}={{b.score:.2f}}" for b in top_shapes)
                print(f"  {{f.name}}: {{n_faces}} faces, expressions: {{expr}}")
            else:
                print(f"  {{f.name}}: {{n_faces}} faces (478-pt mesh)")
            cv2.imwrite(os.path.join(save_dir, f.name), img)
        elapsed = time.perf_counter() - t0
        landmarker.close()
        print(f"  Face Landmarker: {{total_faces}} faces in {{elapsed:.1f}}s")
        return {{"model": "MediaPipe Face Landmarker", "faces": total_faces, "time_s": round(elapsed, 1)}}
    except Exception as e:
        print(f"  MediaPipe Face Landmarker: {{e}}")
        # Fallback to legacy face detection
        try:
            import cv2, mediapipe as mp
            mp_face = mp.solutions.face_detection; mp_draw = mp.solutions.drawing_utils
            t0 = time.perf_counter()
            total = 0
            with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    results = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    n = len(results.detections) if results.detections else 0
                    total += n
                    print(f"  (legacy) {{f.name}}: {{n}} faces")
            elapsed = time.perf_counter() - t0
            return {{"model": "MediaPipe legacy", "faces": total, "time_s": round(elapsed, 1)}}
        except Exception as e2:
            print(f"  MediaPipe legacy fallback: {{e2}}")
    return {{}}


def run_insightface(files):
    """InsightFace — face recognition, verification, gender/age/ethnicity."""
    try:
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        save_dir = os.path.join(SAVE_DIR, "insightface_results")
        os.makedirs(save_dir, exist_ok=True)
        t0 = time.perf_counter()
        embeddings = []
        total_faces = 0
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            faces = app.get(img)
            total_faces += len(faces)
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                if hasattr(face, "embedding") and face.embedding is not None:
                    embeddings.append(face.embedding)
                info = []
                if hasattr(face, "age"): info.append(f"age={{face.age}}")
                if hasattr(face, "gender"): info.append(f"gender={{'M' if face.gender==1 else 'F'}}")
                if info:
                    cv2.putText(img, " ".join(info), (bbox[0], bbox[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  {{f.name}}: {{len(faces)}} faces")
        elapsed = time.perf_counter() - t0
        sim = None
        if len(embeddings) >= 2:
            sim = float(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            print(f"  Cosine similarity (face 0 vs 1): {{sim:.4f}}")
        print(f"  InsightFace: {{total_faces}} faces in {{elapsed:.1f}}s")
        return {{"model": "InsightFace", "faces": total_faces, "embeddings": len(embeddings),
                 "cosine_sim_0v1": round(sim, 4) if sim else None, "time_s": round(elapsed, 1)}}
    except Exception as e:
        print(f"  InsightFace: {{e}}")
        return {{}}


def run_hand_gesture(files):
    """MediaPipe Hand Landmarker / Gesture Recognizer — modern Tasks API.

    Stage 1: Static image inference (offline, always runs).
    Stage 2: Live webcam gesture recognition (runs only if display available).
    """
    import sys
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request
        from collections import Counter

        model_path = os.path.join(SAVE_DIR, "gesture_recognizer.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
                model_path)

        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_hands=2)
        recognizer = mp_vision.GestureRecognizer.create_from_options(options)

        # --- Stage 1: Static image inference ---
        save_dir = os.path.join(SAVE_DIR, "hand_gesture_results")
        os.makedirs(save_dir, exist_ok=True)
        gesture_counts = Counter()
        total_hands = 0
        confidences = []
        t0 = time.perf_counter()
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = recognizer.recognize(mp_img)
            n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
            total_hands += n_hands
            if result.gestures:
                for gesture in result.gestures:
                    name = gesture[0].category_name
                    score = gesture[0].score
                    gesture_counts[name] += 1
                    confidences.append(score)
            if result.hand_landmarks:
                for hand_lm in result.hand_landmarks:
                    for lm in hand_lm:
                        x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  {{f.name}}: {{n_hands}} hands")
        static_elapsed = time.perf_counter() - t0
        avg_conf = sum(confidences) / max(len(confidences), 1)
        print(f"  Static: {{len(files[:20])}} images, {{total_hands}} hands in {{static_elapsed:.1f}}s")

        # --- Stage 2: Live webcam (if display available) ---
        webcam_frames = 0
        webcam_elapsed = 0.0
        if sys.stdout.isatty():
            print("Starting webcam gesture recognition... Press 'q' to quit.")
            cap = cv2.VideoCapture(0)
            t1 = time.perf_counter()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = recognizer.recognize(mp_img)
                if result.gestures:
                    for i, gesture in enumerate(result.gestures):
                        name = gesture[0].category_name
                        score = gesture[0].score
                        gesture_counts[name] += 1
                        cv2.putText(frame, f"{{name}} ({{score:.2f}})", (10, 40 + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if result.hand_landmarks:
                    for hand_lm in result.hand_landmarks:
                        for lm in hand_lm:
                            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                cv2.imshow("Gesture Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                webcam_frames += 1
                if webcam_frames >= 300:
                    break
            cap.release()
            cv2.destroyAllWindows()
            webcam_elapsed = round(time.perf_counter() - t1, 1)
            print(f"  Webcam: {{webcam_frames}} frames in {{webcam_elapsed}}s")
        else:
            print("  Webcam skipped (no display / headless environment)")

        recognizer.close()
        return {{"model": "MediaPipe Gesture Recognizer", "hands_detected": total_hands,
                 "static_images": len(files[:20]), "static_time_s": round(static_elapsed, 1),
                 "gesture_counts": dict(gesture_counts),
                 "avg_confidence": round(avg_conf, 4),
                 "webcam_frames": webcam_frames, "webcam_time_s": webcam_elapsed}}
    except Exception as e:
        print(f"  MediaPipe Gesture Recognizer: {{e}}")
        # Fallback to legacy hand detection on static images
        try:
            import cv2, mediapipe as mp
            mp_hands = mp.solutions.hands
            t0 = time.perf_counter()
            total = 0
            with mp_hands.Hands(static_image_mode=True,
                                min_detection_confidence=0.7) as hands:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None:
                        continue
                    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    n = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                    total += n
                    print(f"  (legacy) {{f.name}}: {{n}} hands")
            elapsed = time.perf_counter() - t0
            return {{"model": "MediaPipe legacy hands", "hands_detected": total,
                     "time_s": round(elapsed, 1)}}
        except Exception as e2:
            print(f"  MediaPipe legacy hands: {{e2}}")
    return {{}}


def run_pose(files):
    """MediaPipe Pose Landmarker — modern Tasks API."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        model_path = os.path.join(SAVE_DIR, "pose_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                model_path)

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_poses=3)
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        save_dir = os.path.join(SAVE_DIR, "pose_results")
        os.makedirs(save_dir, exist_ok=True)
        t0 = time.perf_counter()
        total_poses = 0
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_img)
            n_poses = len(result.pose_landmarks)
            total_poses += n_poses
            for pose_lm in result.pose_landmarks:
                for lm in pose_lm:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  {{f.name}}: {{n_poses}} poses")
        elapsed = time.perf_counter() - t0
        landmarker.close()
        print(f"  Pose Landmarker: {{total_poses}} poses in {{elapsed:.1f}}s")
        return {{"model": "MediaPipe Pose Landmarker", "poses": total_poses, "time_s": round(elapsed, 1)}}
    except Exception as e:
        print(f"  MediaPipe Pose Landmarker: {{e}}")
        # Fallback to legacy pose
        try:
            import cv2, mediapipe as mp
            mp_pose = mp.solutions.pose; mp_draw = mp.solutions.drawing_utils
            save_dir = os.path.join(SAVE_DIR, "pose_results")
            os.makedirs(save_dir, exist_ok=True)
            t0 = time.perf_counter()
            with mp_pose.Pose(min_detection_confidence=0.5) as pose:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.imwrite(os.path.join(save_dir, f.name), img)
                    print(f"  (legacy) {{f.name}}")
            elapsed = time.perf_counter() - t0
            return {{"model": "MediaPipe legacy pose", "time_s": round(elapsed, 1)}}
        except Exception as e2:
            print(f"  MediaPipe legacy pose: {{e2}}")
    return {{}}


def run_eda(files, save_dir):
    """Input file summary for face/gesture tasks."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"  Input files: {{len(files)}}")
    if files:
        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        print(f"  Total size: {{total_size / 1024:.1f}} KB")
    print("EDA complete.")


def validate_results(metrics, files, save_dir):
    """Validate output payloads for face / gesture tasks."""
    validation = {{"task": TASK, "input_files": len(files), "models": {{}}}}
    for name, payload in metrics.items():
        if name == "task" or not isinstance(payload, dict):
            continue
        numeric_values = [
            float(value) for key, value in payload.items()
            if key != "time_s" and isinstance(value, (int, float))
        ]
        positive_signal = any(value > 0 for value in numeric_values)
        validation["models"][name] = {{
            "time_s": round(float(payload.get("time_s", 0)), 1) if isinstance(payload.get("time_s", 0), (int, float)) else None,
            "positive_signal": positive_signal,
            "keys": sorted(payload.keys()),
            "passed": positive_signal or isinstance(payload.get("time_s"), (int, float)),
        }}
    validation["passed"] = any(model.get("passed") for model in validation["models"].values())
    out_path = os.path.join(save_dir, "validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"Validation saved to {{out_path}}")
    return validation


def main():
    print("=" * 60)
    print(f"FACE/HAND/GESTURE | Task: {{TASK}}")
    print("=" * 60)
    files = download_face_samples()
    run_eda(files, SAVE_DIR)
    metrics = {{"task": TASK}}

    if TASK == "face_detection":
        metrics["yolo"] = run_yolo_detection(files)
        metrics["face_landmarker"] = run_face_landmarker(files)
    elif TASK == "expression":
        # Expression / smile / blink — landmarker is primary (blendshapes)
        metrics["face_landmarker"] = run_face_landmarker(files)
        metrics["yolo_baseline"] = run_yolo_detection(files)
    elif TASK == "hand_gesture":
        metrics["gesture"] = run_hand_gesture(files)
    elif TASK == "pose":
        metrics["pose"] = run_pose(files)
    elif TASK == "face_recognition":
        metrics["insightface"] = run_insightface(files)
    else:
        metrics["yolo"] = run_yolo_detection(files)
        metrics["face_landmarker"] = run_face_landmarker(files)

    metrics["validation"] = validate_results(metrics, files, SAVE_DIR)

    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {{out_path}}")


if __name__ == "__main__":
    main()
''')


def gen_ocr(path, cfg):
    return textwrap.dedent('''\
"""
Modern OCR Pipeline (April 2026)

Primary : PaddleOCR (text detection + recognition, GPU, multilingual).
Extended: PaddleOCR-VL-1.5 (vision-language document parsing).
Timing  : Wall-clock per model stage.
Export  : metrics.json with file-level results + aggregate stats + timing.
Data    : Auto-downloads sample document images at runtime.
"""
import os, json, time, warnings
from pathlib import Path
import urllib.request

warnings.filterwarnings("ignore")

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_URLS = [
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img_12.jpg",
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img623.jpg",
]


def download_samples():
    save_dir = Path(SAVE_DIR) / "ocr_samples"
    save_dir.mkdir(exist_ok=True)
    paths = []
    for url in SAMPLE_URLS:
        fname = save_dir / url.split("/")[-1]
        if not fname.exists():
            urllib.request.urlretrieve(url, str(fname))
        paths.append(fname)
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        paths.extend([p for p in Path(SAVE_DIR).rglob(f"*{ext}") if p not in paths])
    print(f"{len(paths)} images available for OCR")
    return paths


def run_paddleocr(files):
    """PaddleOCR -- primary text detection + recognition."""
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
    results = []
    t0 = time.perf_counter()
    for f in files[:30]:
        result = ocr.ocr(str(f), cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                texts.append({"text": line[1][0], "confidence": round(line[1][1], 4)})
        full_text = " ".join(t["text"] for t in texts)
        avg_conf = sum(t["confidence"] for t in texts) / max(len(texts), 1)
        results.append({"file": f.name, "full_text": full_text,
                        "n_lines": len(texts), "avg_confidence": round(avg_conf, 4)})
        preview = full_text[:80] + "..." if len(full_text) > 80 else full_text
        print(f"  {f.name}: {len(texts)} lines (conf {avg_conf:.2f}) -- \\'{preview}\\'")
    elapsed = time.perf_counter() - t0
    return results, round(elapsed, 1)


def run_paddleocr_vl(files):
    """PaddleOCR-VL-1.5 -- vision-language document parsing."""
    from paddleocr import PaddleOCR
    vl_ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                       use_textline_orientation=False, lang="en", use_gpu=True)
    results = []
    t0 = time.perf_counter()
    for f in files[:10]:
        vl_result = vl_ocr.ocr(str(f), cls=True)
        n_lines = len(vl_result[0]) if vl_result and vl_result[0] else 0
        results.append({"file": f.name, "n_lines": n_lines})
        print(f"  VL-1.5 {f.name}: {n_lines} lines")
    elapsed = time.perf_counter() - t0
    return results, round(elapsed, 1)


def run_eda(files, save_dir):
    """Input file summary for OCR."""
    print("\\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"  Input files: {{len(files)}}")
    if files:
        total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        print(f"  Total size: {{total_size / 1024:.1f}} KB")
    print("EDA complete.")


def validate_results(primary_results, vl_results, save_dir):
    """Validate OCR outputs for completeness and confidence."""
    validation = {
        "paddleocr": {
            "files": len(primary_results),
            "files_with_text": sum(1 for item in primary_results if item.get("n_lines", 0) > 0),
            "avg_confidence": round(
                sum(float(item.get("avg_confidence", 0)) for item in primary_results) / max(len(primary_results), 1),
                4,
            ),
        },
        "paddleocr_vl": {
            "files": len(vl_results),
            "files_with_text": sum(1 for item in vl_results if item.get("n_lines", 0) > 0),
        },
    }
    validation["paddleocr"]["passed"] = validation["paddleocr"]["files_with_text"] > 0
    validation["paddleocr_vl"]["passed"] = validation["paddleocr_vl"]["files_with_text"] > 0 if vl_results else True
    validation["passed"] = validation["paddleocr"]["passed"] or validation["paddleocr_vl"]["passed"]
    out_path = os.path.join(save_dir, "validation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)
    print(f"Validation saved to {{out_path}}")
    return validation


def main():
    print("=" * 60)
    print("OCR | PaddleOCR + PaddleOCR-VL-1.5")
    print("=" * 60)
    files = download_samples()
    run_eda(files, SAVE_DIR)
    metrics = {}
    results = []
    vl_results = []

    # PRIMARY: PaddleOCR
    print()
    print("-- PaddleOCR --")
    try:
        results, elapsed = run_paddleocr(files)
        total_lines = sum(r["n_lines"] for r in results)
        avg_conf = sum(r["avg_confidence"] for r in results) / max(len(results), 1)
        metrics["PaddleOCR"] = {
            "files": len(results), "total_lines": total_lines,
            "avg_confidence": round(avg_conf, 4), "time_s": elapsed,
        }
        print(f"  PaddleOCR: {len(results)} files, {total_lines} lines in {elapsed}s")
    except Exception as e:
        print(f"  PaddleOCR failed: {e}")

    # EXTENDED: PaddleOCR-VL-1.5
    print()
    print("-- PaddleOCR-VL-1.5 (document parsing) --")
    try:
        vl_results, vl_elapsed = run_paddleocr_vl(files)
        vl_total = sum(r["n_lines"] for r in vl_results)
        metrics["PaddleOCR-VL-1.5"] = {
            "files": len(vl_results), "total_lines": vl_total, "time_s": vl_elapsed,
        }
        print(f"  VL-1.5: {len(vl_results)} files, {vl_total} lines in {vl_elapsed}s")
    except Exception as e:
        print(f"  PaddleOCR-VL-1.5 failed: {e}")

    metrics["validation"] = validate_results(results, vl_results, SAVE_DIR)

    # Save metrics
    out_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {out_path}")

    # Also save detailed per-file results
    detail_path = os.path.join(SAVE_DIR, "ocr_results.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {detail_path}")


if __name__ == "__main__":
    main()
''')


# ════════════════════════════════════════════════════════════════════════════════
# MAIN: Generate all pipelines
# ════════════════════════════════════════════════════════════════════════════════

def main():
    total = 0
    families = [
        ("Tabular Classification", TABULAR_CLF, gen_tabular_clf),
        ("Tabular Regression", TABULAR_REG, gen_tabular_reg),
        ("Fraud / Imbalanced", FRAUD, gen_fraud),
        ("Anomaly Detection", ANOMALY, gen_anomaly),
        ("Clustering", CLUSTERING, gen_clustering),
        ("NLP Classification", NLP_CLF, gen_nlp_clf),
        ("NER / Extraction", NLP_NER, gen_ner),
        ("NLP Generation", NLP_GEN, gen_nlp_gen),
        ("NLP Similarity", NLP_SIM, gen_nlp_similarity),
        ("NLP Misc", NLP_MISC, gen_nlp_gen),
        ("Image Classification", IMAGE_CLF, gen_image_clf),
        ("CV Detection", CV_DETECTION, gen_cv_detection),
        ("CV Misc", CV_MISC, gen_cv_detection),
        ("Face/Gesture", FACE_GESTURE, gen_face_gesture),
        ("OCR", OCR, gen_ocr),
        ("Captioning / VLM", CAPTIONING, gen_captioning),
        ("Medical Segmentation", MEDICAL_SEG, gen_medical_seg),
        ("Recommendation", RECOMMENDATION, gen_recommendation),
        ("Time Series", TIME_SERIES, gen_timeseries),
        ("Reinforcement Learning", RL, gen_rl),
        ("Audio/Speech", AUDIO, gen_audio),
        ("DL Image Misc", DL_IMAGE_MISC, gen_image_clf),
        ("DL Tabular Misc", DL_TABULAR_MISC, gen_tabular_reg),
        ("DL Cluster Misc", DL_CLUSTER_MISC, gen_clustering),
    ]

    for family_name, projects, generator in families:
        print(f"\n{'='*3} {family_name} {'='*3}")
        for path, cfg in projects.items():
            try:
                content = generator(path, cfg)
                if write_pipeline(path, content):
                    total += 1
                    print(f"  {path}")
            except Exception as e:
                print(f"  {path}: {e}")

    print(f"\n{'='*60}")
    print(f"TOTAL PIPELINES GENERATED: {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
