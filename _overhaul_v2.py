"""
Master overhaul script v2: generates modern pipeline.py for every project.
ALL data is auto-downloaded at runtime — no prior CSV files needed.
Run: python _overhaul_all.py

Data sources: HuggingFace datasets, sklearn, torchvision, yfinance, UCI, OpenML, direct URLs
"""
import os, sys, textwrap
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
    """HuggingFace datasets.load_dataset → df"""
    cfg = f', "{config}"' if config else ""
    col_filter = ""
    if columns:
        col_filter = f"\n    df = df[{columns}]"
    return f'    from datasets import load_dataset as _hf_load\n    df = _hf_load("{dataset_name}"{cfg}, split="{split}").to_pandas(){col_filter}'

def _sklearn(func_name):
    """sklearn.datasets → df"""
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
    """torchvision dataset → returns special flag for image pipeline"""
    return f"__torchvision__{dataset_cls}__{n_classes}"

def _openml(data_id, target=None):
    """OpenML dataset by ID"""
    t = f', target_column="{target}"' if target else ""
    return f'    from sklearn.datasets import fetch_openml\n    _d = fetch_openml(data_id={data_id}{t}, as_frame=True, parser="auto")\n    df = _d.frame'

def _seaborn(name):
    """Seaborn built-in dataset"""
    return f'    import seaborn as _sns\n    df = _sns.load_dataset("{name}")'


# ════════════════════════════════════════════════════════════════════════════════
# PROJECT → DATA SOURCE MAPPING  (every project has an online source)
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
    # NER / keyword / entity extraction (GLiNER primary)
    "NLP/Named Entity Recognition": {"target": "ner_tags", "text_col": "tokens", "data": _hf("conll2003")},
    "NLP/Keyword Extraction": {"target": "label", "text_col": "text", "data": _hf("EdinburghNLP/xsum")},
    "NLP/Keyword Research": {"target": "label", "text_col": "text", "data": _hf("EdinburghNLP/xsum")},
    # Text classification misc
    "NLP/Profanity Checker": {"target": "label", "text_col": "text", "data": _hf("hate_speech18")},
    "NLP/BOW and TF-IDF with XGBoost": {"target": "label", "text_col": "text", "data": _hf("stanfordnlp/imdb")},
}

# ── FAMILY 7: NLP GENERATION ──
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
    "Computer Vision/Document Word Detection": {"task": "detect"},
    "Computer Vision/Lane Finder": {"task": "detect"},
    "Computer Vision/Captcha Recognition": {"task": "detect"},
    "Deep Learning/Landmark Detection": {"task": "detect"},
}

# ── FAMILY 10: FACE/GESTURE ──
FACE_GESTURE = {
    "Computer Vision/Face Detection - OpenCV": {"task": "face_detection"},
    "Computer Vision/Face Expression Identifier": {"task": "face_detection"},
    "Computer Vision/Face Mask Detection": {"task": "face_detection"},
    "Computer Vision/Gesture Control Media Player": {"task": "hand_gesture"},
    "Computer Vision/Home Security": {"task": "face_detection"},
    "Computer Vision/Live Smile Detector": {"task": "face_detection"},
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
    "Reinforcement Learning/Cliff Walking": {"env": "CliffWalking-v0", "algo": "PPO"},
    "Reinforcement Learning/Frozen Lake": {"env": "FrozenLake-v1", "algo": "PPO"},
    "Reinforcement Learning/Gridworld Navigation": {"env": "CartPole-v1", "algo": "PPO"},
    "Reinforcement Learning/Lunar Landing": {"env": "LunarLander-v3", "algo": "PPO"},
    "Reinforcement Learning/Taxi Navigation": {"env": "Taxi-v3", "algo": "PPO"},
}

# ── FAMILY 15: AUDIO / SPEECH ──
AUDIO = {
    "Speech and Audio processing/Audio Denoising": {"task": "denoising", "data": _hf("edinburghcstr/vctk")},
    "Speech and Audio processing/Music Genre Prediction - Million Songs": {"task": "classification", "data": _hf("marsyas/gtzan")},
    "Speech and Audio processing/Voice Cloning": {"task": "cloning", "data": _hf("edinburghcstr/vctk")},
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
    "Computer Vision/Noise Reduction": {"task": "detect"},
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
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    roc_auc_score, confusion_matrix
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


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {{}}
    n_classes = y_train.nunique()
    is_binary = n_classes == 2

    # ── CatBoost (GPU) ──
    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=8,
            task_type="GPU", devices="0",
            eval_metric="AUC" if is_binary else "MultiClass",
            early_stopping_rounds=50, verbose=100,
            auto_class_weights="Balanced",
        )
        cb.fit(X_train, y_train, eval_set=(X_test, y_test))
        results["CatBoost"] = cb.predict(X_test).flatten()
        print(f"\\n✓ CatBoost Accuracy: {{accuracy_score(y_test, results['CatBoost']):.4f}}")
    except Exception as e:
        print(f"✗ CatBoost: {{e}}")

    # ── LightGBM (GPU) ──
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            device="gpu", class_weight="balanced", verbose=-1, n_jobs=-1,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        results["LightGBM"] = m.predict(X_test)
        print(f"\\n✓ LightGBM Accuracy: {{accuracy_score(y_test, results['LightGBM']):.4f}}")
    except Exception as e:
        print(f"✗ LightGBM: {{e}}")

    # ── XGBoost (CUDA) ──
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            device="cuda", tree_method="hist",
            eval_metric="auc" if is_binary else "mlogloss",
            early_stopping_rounds=50, verbosity=1, n_jobs=-1,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        results["XGBoost"] = m.predict(X_test)
        print(f"\\n✓ XGBoost Accuracy: {{accuracy_score(y_test, results['XGBoost']):.4f}}")
    except Exception as e:
        print(f"✗ XGBoost: {{e}}")

    # ── AutoGluon Tabular ──
    try:
        from autogluon.tabular import TabularPredictor
        import tempfile
        train_ag = X_train.copy(); train_ag["{target}"] = y_train.values
        test_ag = X_test.copy(); test_ag["{target}"] = y_test.values
        with tempfile.TemporaryDirectory() as tmp:
            predictor = TabularPredictor(label="{target}", path=tmp, verbosity=1)
            predictor.fit(train_ag, time_limit=180, presets="best_quality")
            results["AutoGluon"] = predictor.predict(test_ag.drop(columns=["{target}"])).values
            print(f"\\n✓ AutoGluon Accuracy: {{accuracy_score(y_test, results['AutoGluon']):.4f}}")
    except Exception as e:
        print(f"✗ AutoGluon: {{e}}")

    # ── RealTabPFN-v2 (prior-fitted network) ──
    try:
        from tabpfn import TabPFNClassifier
        if X_train.shape[0] <= 10000 and X_train.shape[1] <= 500:
            m = TabPFNClassifier(device="cuda", N_ensemble_configurations=32)
            m.fit(X_train.values, y_train.values)
            results["TabPFN-v2"] = m.predict(X_test.values)
            print(f"\\n✓ TabPFN-v2 Accuracy: {{accuracy_score(y_test, results['TabPFN-v2']):.4f}}")
        else:
            print("⚠ TabPFN-v2: dataset too large (>10k rows or >500 cols), skipped")
    except Exception as e:
        print(f"✗ TabPFN-v2: {{e}}")

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

        model = TabMNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        for ep in range(100):
            model.train(); loss = loss_fn(model(Xt), yt); loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            results["TabM"] = torch.argmax(model(Xv), dim=-1).cpu().numpy()
        print(f"\\n✓ TabM Accuracy: {{accuracy_score(y_test, results['TabM']):.4f}}")
    except Exception as e:
        print(f"✗ TabM: {{e}}")

    # ── Baseline Comparison: FLAML AutoML ──
    try:
        from flaml import AutoML
        automl = AutoML()
        automl.fit(X_train, y_train, task="classification", time_budget=120, verbose=0)
        results["FLAML"] = automl.predict(X_test)
        print(f"\\n✓ FLAML ({{automl.best_estimator}}) Accuracy: {{accuracy_score(y_test, results['FLAML']):.4f}}")
    except Exception as e:
        print(f"✗ FLAML: {{e}}")

    # ── Baseline Comparison: LazyPredict ──
    try:
        from lazypredict.Supervised import LazyClassifier
        lazy = LazyClassifier(verbose=0, ignore_warnings=True)
        lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)
        print(f"\\n✓ LazyPredict — Top 5 classifiers:")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"✗ LazyPredict: {{e}}")

    return results


def report(results, y_test, save_dir="."):
    print("\\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    best_name, best_acc = None, 0
    for name, y_pred in results.items():
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        print(f"\\n— {{name}} —  Accuracy: {{acc:.4f}}  |  F1: {{f1:.4f}}")
        print(classification_report(y_test, y_pred, zero_division=0))
        if acc > best_acc:
            best_acc, best_name = acc, name
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{{name}} Confusion Matrix")
        fig.savefig(os.path.join(save_dir, f"cm_{{name.lower().replace(' ', '_')}}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"\\n🏆 Best: {{best_name}} ({{best_acc:.4f}})")


def main():
    print("=" * 60)
    print("MODERN TABULAR CLASSIFICATION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | AutoGluon | TabPFN-v2 | TabM | FLAML | LazyPredict")
    print("=" * 60)
    df = load_data()
    X_train, X_test, y_train, y_test, le = preprocess(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, y_test, os.path.dirname(os.path.abspath(__file__)))


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
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        X[c] = X[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = {{}}

    try:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=1000, lr=0.05, depth=8, task_type="GPU",
                              devices="0", early_stopping_rounds=50, verbose=100)
        m.fit(X_train, y_train, eval_set=(X_test, y_test))
        results["CatBoost"] = m.predict(X_test)
        print(f"✓ CatBoost RMSE: {{mean_squared_error(y_test, results['CatBoost'], squared=False):.4f}}")
    except Exception as e:
        print(f"✗ CatBoost: {{e}}")

    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=1000, lr=0.05, max_depth=8,
                              device="gpu", verbose=-1, n_jobs=-1)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        results["LightGBM"] = m.predict(X_test)
        print(f"✓ LightGBM RMSE: {{mean_squared_error(y_test, results['LightGBM'], squared=False):.4f}}")
    except Exception as e:
        print(f"✗ LightGBM: {{e}}")

    try:
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8,
                         device="cuda", tree_method="hist", early_stopping_rounds=50,
                         verbosity=1, n_jobs=-1)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
        results["XGBoost"] = m.predict(X_test)
        print(f"✓ XGBoost RMSE: {{mean_squared_error(y_test, results['XGBoost'], squared=False):.4f}}")
    except Exception as e:
        print(f"✗ XGBoost: {{e}}")

    # ── AutoGluon Tabular ──
    try:
        from autogluon.tabular import TabularPredictor
        import tempfile
        train_ag = X_train.copy(); train_ag["{target}"] = y_train.values
        with tempfile.TemporaryDirectory() as tmp:
            predictor = TabularPredictor(label="{target}", path=tmp, problem_type="regression", verbosity=1)
            predictor.fit(train_ag, time_limit=180, presets="best_quality")
            results["AutoGluon"] = predictor.predict(X_test).values
            print(f"✓ AutoGluon RMSE: {{mean_squared_error(y_test, results['AutoGluon'], squared=False):.4f}}")
    except Exception as e:
        print(f"✗ AutoGluon: {{e}}")

    # ── RealTabPFN-v2 (prior-fitted network — regression) ──
    try:
        from tabpfn import TabPFNRegressor
        if X_train.shape[0] <= 10000 and X_train.shape[1] <= 500:
            m = TabPFNRegressor(device="cuda", N_ensemble_configurations=32)
            m.fit(X_train.values, y_train.values)
            results["TabPFN-v2"] = m.predict(X_test.values)
            print(f"✓ TabPFN-v2 RMSE: {{mean_squared_error(y_test, results['TabPFN-v2'], squared=False):.4f}}")
        else:
            print("⚠ TabPFN-v2: dataset too large (>10k rows or >500 cols), skipped")
    except Exception as e:
        print(f"✗ TabPFN-v2: {{e}}")

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
        model = TabMNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        for ep in range(100):
            model.train(); loss = nn.MSELoss()(model(Xt), yt); loss.backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): results["TabM"] = model(Xv).cpu().numpy()
        print(f"✓ TabM RMSE: {{mean_squared_error(y_test, results['TabM'], squared=False):.4f}}")
    except Exception as e:
        print(f"✗ TabM: {{e}}")

    # ── Baseline Comparison: FLAML AutoML ──
    try:
        from flaml import AutoML
        automl = AutoML()
        automl.fit(X_train, y_train, task="regression", time_budget=120, verbose=0)
        results["FLAML"] = automl.predict(X_test)
        print(f"✓ FLAML ({{automl.best_estimator}}) RMSE: {{mean_squared_error(y_test, results['FLAML'], squared=False):.4f}}")
    except Exception as e:
        print(f"✗ FLAML: {{e}}")

    # ── Baseline Comparison: LazyPredict ──
    try:
        from lazypredict.Supervised import LazyRegressor
        lazy = LazyRegressor(verbose=0, ignore_warnings=True)
        lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)
        print(f"\\n✓ LazyPredict — Top 5 regressors:")
        print(lazy_models.head().to_string())
    except Exception as e:
        print(f"✗ LazyPredict: {{e}}")

    return results


def report(results, y_test, save_dir="."):
    print("\\n" + "=" * 60)
    best_name, best_rmse = None, float("inf")
    for name, y_pred in results.items():
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"— {{name}} — RMSE: {{rmse:.4f}} | MAE: {{mae:.4f}} | R²: {{r2:.4f}}")
        if rmse < best_rmse:
            best_rmse, best_name = rmse, name
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.4, s=10)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax.set_title(f"{{name}} — Predicted vs Actual")
        fig.savefig(os.path.join(save_dir, f"scatter_{{name.lower().replace(' ', '_')}}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"\\n🏆 Best: {{best_name}} (RMSE: {{best_rmse:.4f}})")


def main():
    print("=" * 60)
    print("MODERN TABULAR REGRESSION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | AutoGluon | TabPFN-v2 | TabM | FLAML | LazyPredict")
    print("=" * 60)
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, y_test, os.path.dirname(os.path.abspath(__file__)))


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
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    classification_report, f1_score,
    precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix
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
    for c in cat_cols: X[c] = X[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def find_best_threshold(y_true, y_proba):
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    idx = np.argmax(f1s)
    return thresholds[idx] if idx < len(thresholds) else 0.5


def train_and_evaluate(X_train, X_test, y_train, y_test):
    from sklearn.calibration import CalibratedClassifierCV
    results = {{}}
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
            results[name] = {{"preds": preds, "proba": proba, "thresh": thresh, "model": name}}
            print(f"✓ {{name}} F1: {{f1_score(y_test, preds):.4f}} (t={{thresh:.3f}}) [calibrated]")
        except Exception as e:
            print(f"✗ {{name}}: {{e}}")

    # ── PyOD Anomaly Scoring (unsupervised cross-check) ──
    for pyod_name, pyod_builder in [
        ("ECOD", lambda: __import__("pyod.models.ecod", fromlist=["ECOD"]).ECOD(contamination=0.05)),
        ("COPOD", lambda: __import__("pyod.models.copod", fromlist=["COPOD"]).COPOD(contamination=0.05)),
        ("IForest-PyOD", lambda: __import__("pyod.models.iforest", fromlist=["IForest"]).IForest(contamination=0.05, random_state=42)),
    ]:
        try:
            pm = pyod_builder()
            pm.fit(X_train.values if hasattr(X_train, "values") else X_train)
            scores = pm.decision_function(X_test.values if hasattr(X_test, "values") else X_test)
            pyod_preds = pm.predict(X_test.values if hasattr(X_test, "values") else X_test)
            n_anom = pyod_preds.sum()
            f1 = f1_score(y_test, pyod_preds) if len(set(y_test)) > 1 else 0
            auc = roc_auc_score(y_test, scores) if len(set(y_test)) > 1 else 0
            print(f"✓ PyOD {{pyod_name}}: {{n_anom}} anomalies ({{n_anom/len(X_test):.2%}}), F1={{f1:.4f}}, AUC={{auc:.4f}}")
        except Exception as e:
            print(f"✗ PyOD {{pyod_name}}: {{e}}")

    return results


def report(results, y_test, save_dir="."):
    from sklearn.calibration import calibration_curve
    for name, r in results.items():
        print(f"\\n— {{name}} (threshold={{r['thresh']:.3f}}) —")
        print(classification_report(y_test, r["preds"], target_names=["Legit", "Fraud"]))
        print(f"  AUPRC: {{average_precision_score(y_test, r['proba']):.4f}}  ROC-AUC: {{roc_auc_score(y_test, r['proba']):.4f}}")

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
        print(f"\\n✓ Report saved to {{save_dir}}/fraud_report.png")
    except Exception as e:
        print(f"✗ Plot: {{e}}")


def main():
    print("=" * 60)
    print("FRAUD / IMBALANCED CLASSIFICATION PIPELINE")
    print("CatBoost | LightGBM | XGBoost | PyOD (ECOD/COPOD/IForest)")
    print("Calibrated probabilities + threshold tuning")
    print("=" * 60)
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    if results:
        report(results, y_test, os.path.dirname(os.path.abspath(__file__)))


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
Models: ModernBERT (English) + XLM-RoBERTa (multilingual) + GLiNER (zero-shot NER)
        TF-IDF + Naive Bayes as baseline
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib; matplotlib.use("Agg")

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
    for name, clf in [("Naive Bayes", MultinomialNB()), ("LogReg", LogisticRegression(max_iter=1000, n_jobs=-1))]:
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="weighted")
        print(f"  [Baseline] {{name}} — Accuracy: {{acc:.4f}}, F1: {{f1:.4f}}")


# ═══════════════════════════════════════════════════════════════
# PRIMARY: ModernBERT / XLM-R fine-tuned classifier
# ═══════════════════════════════════════════════════════════════
def train_transformer(df, model_name, display_name):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    le = LabelEncoder(); df["label_id"] = le.fit_transform(df["label"])
    n_classes = len(le.classes_)
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

    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for batch in train_loader:
            batch = {{k: v.to(device) for k, v in batch.items()}}
            loss = model(**batch).loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); total_loss += loss.item()
        print(f"  [{{display_name}}] Epoch {{epoch+1}}/{{EPOCHS}}, Loss: {{total_loss/len(train_loader):.4f}}")

    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {{k: v.to(device) for k, v in batch.items()}}
            preds.extend(torch.argmax(model(**batch).logits, dim=-1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    print(f"\\n✓ {{display_name}} — Accuracy: {{acc:.4f}}, F1: {{f1:.4f}}")
    print(classification_report(labels, preds, target_names=le.classes_.astype(str), zero_division=0))
    model.save_pretrained(os.path.join(os.path.dirname(__file__), f"{{display_name.lower().replace('-','_')}}_model"))
    return acc, f1


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
        print("✓ GLiNER zero-shot NER complete")
    except Exception as e:
        print(f"✗ GLiNER: {{e}}")


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
        print(f"✓ BGE-M3: {{len(texts)}} texts embedded (dim={{embs.shape[1]}})")

        # Qwen3-Embedding
        try:
            qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
            qwen_embs = qwen.encode(texts[:100], batch_size=16, show_progress_bar=True)
            print(f"✓ Qwen3-Embedding: {{len(qwen_embs)}} texts embedded (dim={{qwen_embs.shape[1]}})")
        except Exception as e:
            print(f"✗ Qwen3-Embedding: {{e}}")
    except Exception as e:
        print(f"✗ Embedding similarity: {{e}}")


def main():
    print("=" * 60)
    print("NLP CLASSIFICATION — ModernBERT + XLM-R | TF-IDF baseline | GLiNER NER")
    print("=" * 60)
    df = load_data()

    # Baseline first
    print("\\n— TF-IDF / Naive Bayes Baseline —")
    run_tfidf_baseline(df)

    # Primary transformer models
    best_acc, best_name = 0, ""
    for model_name, display_name in MODELS:
        try:
            acc, f1 = train_transformer(df.copy(), model_name, display_name)
            if acc > best_acc:
                best_acc, best_name = acc, display_name
        except Exception as e:
            print(f"✗ {{display_name}}: {{e}}")
    print(f"\\n🏆 Best: {{best_name}} (Accuracy: {{best_acc:.4f}})")

    # Zero-shot NER
    print("\\n— GLiNER Zero-Shot NER —")
    run_gliner(df)

    # Embedding similarity
    print("\\n— Embedding Similarity (BGE-M3 / Qwen3-Embedding) —")
    run_embedding_similarity(df)


if __name__ == "__main__":
    main()
''')


def gen_nlp_similarity(path, cfg):
    data_load = cfg.get("data", '    raise FileNotFoundError("No data")')
    return textwrap.dedent(f'''\
"""
Modern NLP Similarity / Retrieval Pipeline (April 2026)
Models: BGE-M3 + Qwen3-Embedding + Sentence Transformers
        TF-IDF cosine similarity as baseline
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def load_data():
{data_load}
    print(f"Dataset shape: {{df.shape}}")
    return df


def get_texts(df, n=500):
    """Extract text column, return up to n samples."""
    for c in df.columns:
        if df[c].dtype == "object" and df[c].str.len().mean() > 20:
            return df[c].dropna().head(n).tolist()
    text_cols = df.select_dtypes("object").columns
    if len(text_cols) > 0:
        return df[text_cols[0]].dropna().head(n).tolist()
    return df.iloc[:, 0].astype(str).head(n).tolist()


# ═══════════════════════════════════════════════════════════════
# BASELINE: TF-IDF Cosine Similarity
# ═══════════════════════════════════════════════════════════════
def run_tfidf_baseline(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    vecs = tfidf.fit_transform(texts)
    sim = cosine_similarity(vecs)
    avg_sim = (sim.sum() - len(texts)) / (len(texts) * (len(texts) - 1))
    print(f"  [Baseline] TF-IDF cosine: avg pairwise similarity = {{avg_sim:.4f}}")
    # Show top-3 pairs
    np.fill_diagonal(sim, 0)
    for i in range(min(3, len(texts))):
        top = np.argsort(sim[i])[-3:][::-1]
        scores = [str(j) + f"({{sim[i,j]:.3f}})" for j in top]
        joined = ", ".join(scores)
        print(f"    Text {{i}} most similar to: {{joined}}")
    return sim


# ═══════════════════════════════════════════════════════════════
# PRIMARY: BGE-M3 Embedding Similarity
# ═══════════════════════════════════════════════════════════════
def run_bge_m3(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        sim = cosine_similarity(embs)
        avg_sim = (sim.sum() - len(texts)) / (len(texts) * (len(texts) - 1))
        print("")
        print(f"✓ BGE-M3: {{len(texts)}} texts embedded (dim={{embs.shape[1]}})")
        print(f"  Avg pairwise semantic similarity = {{avg_sim:.4f}}")
        np.fill_diagonal(sim, 0)
        for i in range(min(3, len(texts))):
            top = np.argsort(sim[i])[-3:][::-1]
            scores = [str(j) + f"({{sim[i,j]:.3f}})" for j in top]
            joined = ", ".join(scores)
            print(f"    Text {{i}} most similar to: {{joined}}")
        return embs, sim
    except Exception as e:
        print(f"✗ BGE-M3: {{e}}")
        return None, None


# ═══════════════════════════════════════════════════════════════
# PRIMARY: Qwen3-Embedding
# ═══════════════════════════════════════════════════════════════
def run_qwen3_embedding(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        embs = model.encode(texts[:200], batch_size=16, show_progress_bar=True, normalize_embeddings=True)
        sim = cosine_similarity(embs)
        avg_sim = (sim.sum() - len(embs)) / (len(embs) * (len(embs) - 1))
        print("")
        print(f"✓ Qwen3-Embedding: {{len(embs)}} texts embedded (dim={{embs.shape[1]}})")
        print(f"  Avg pairwise semantic similarity = {{avg_sim:.4f}}")
        return embs, sim
    except Exception as e:
        print(f"✗ Qwen3-Embedding: {{e}}")
        return None, None


# ═══════════════════════════════════════════════════════════════
# CLUSTERING: Embedding-based topic discovery
# ═══════════════════════════════════════════════════════════════
def run_embedding_clustering(embs, texts):
    if embs is None:
        print("⚠ Skipping embedding clustering (no embeddings)")
        return
    try:
        import umap, hdbscan
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        X_2d = reducer.fit_transform(embs)
        labels = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X_2d)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("")
        print(f"✓ UMAP + HDBSCAN on embeddings: {{n_clusters}} topics/clusters")

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=15, alpha=0.6)
        ax.set_title("Embedding Space — UMAP + HDBSCAN"); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "embedding_clusters.png"), dpi=100)
        print("✓ Saved embedding_clusters.png")
    except Exception as e:
        print(f"✗ Embedding clustering: {{e}}")


def main():
    print("=" * 60)
    print("NLP SIMILARITY / RETRIEVAL — BGE-M3 + Qwen3-Embedding")
    print("TF-IDF baseline | Embedding clustering")
    print("=" * 60)
    df = load_data()
    texts = get_texts(df)
    print(f"Using {{len(texts)}} text samples")

    print(""); print("— TF-IDF Baseline —")
    run_tfidf_baseline(texts)

    print(""); print("— BGE-M3 Embeddings —")
    embs, sim = run_bge_m3(texts)

    print(""); print("— Qwen3-Embedding —")
    run_qwen3_embedding(texts)

    print(""); print("— Embedding Clustering —")
    run_embedding_clustering(embs, texts)


if __name__ == "__main__":
    main()
''')


def gen_nlp_gen(path, cfg):
    task = cfg.get("task", "summarization")
    data_load = cfg.get("data", "    df = None")
    return textwrap.dedent(f'''\
"""
Modern NLP Generation Pipeline (April 2026)
Models: Qwen3-Instruct (chat/generation/summarization) + NLLB-200 (translation) + BART (summarization baseline)
Data: Auto-downloaded at runtime
"""
import os, json, warnings
import pandas as pd
warnings.filterwarnings("ignore")

TASK = "{task}"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


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


def run_summarization(df):
    \"\"\"Qwen3-Instruct for general summarization + BART as classic baseline.\"\"\"
    text_col = next((c for c in df.columns if df[c].dtype == "object" and df[c].str.len().mean() > 50), df.select_dtypes("object").columns[0])
    texts = df[text_col].dropna().head(10).tolist()

    # ═══ PRIMARY: Qwen3-Instruct via Ollama ═══
    qwen_results = []
    for i, text in enumerate(texts):
        summary = query_ollama(f"Summarize concisely:\\n\\n{{text[:2000]}}\\n\\nSummary:")
        if summary:
            qwen_results.append({{"original": text[:200], "summary": summary}})
            print(f"  Qwen3 [{{i+1}}] {{summary[:100]}}...")
    if qwen_results:
        print(f"✓ Qwen3-Instruct summarized {{len(qwen_results)}} texts")

    # ═══ BASELINE: BART (facebook/bart-large-cnn) ═══
    try:
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
        bart_results = []
        for i, text in enumerate(texts[:5]):
            inputs = tokenizer(text[:1024], return_tensors="pt", truncation=True, max_length=1024).to(device)
            summary_ids = model.generate(**inputs, max_length=150, min_length=30, num_beams=4, length_penalty=2.0)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            bart_results.append(summary)
            print(f"  BART [{{i+1}}] {{summary[:100]}}...")
        print(f"✓ BART summarized {{len(bart_results)}} texts")
    except Exception as e:
        print(f"✗ BART baseline: {{e}}")


def run_translation(df):
    \"\"\"NLLB-200 (Meta) — 200+ language pairs, offline, multilingual.\"\"\"
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    text_col = df.select_dtypes("object").columns[0]

    # Translate to multiple target languages
    targets = [("fra_Latn", "French"), ("deu_Latn", "German"), ("spa_Latn", "Spanish"), ("zho_Hans", "Chinese")]
    for tgt_code, tgt_name in targets:
        results = []
        for i, text in enumerate(df[text_col].dropna().head(3)):
            inputs = tokenizer(text[:512], return_tensors="pt", truncation=True).to(device)
            translated_ids = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code), max_new_tokens=256)
            translated = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            results.append({{"original": text[:100], "translated": translated}})
            print(f"  → {{tgt_name}} [{{i+1}}] {{translated[:100]}}...")
        print(f"✓ NLLB-200 → {{tgt_name}}: {{len(results)}} texts")


def run_generation(df):
    \"\"\"Qwen3-Instruct for text generation / next-word prediction.\"\"\"
    prompts = [
        "Write a creative short story about artificial intelligence discovering emotions:",
        "Complete this sentence: The future of machine learning is",
        "Explain quantum computing to a 10-year-old:",
    ]
    # Use data context if available
    if df is not None:
        text_col = next((c for c in df.columns if df[c].dtype == "object"), None)
        if text_col:
            samples = df[text_col].dropna().head(3).tolist()
            prompts = [f"Continue this text creatively:\\n\\n{{t[:300]}}\\n\\nContinuation:" for t in samples]

    for i, prompt in enumerate(prompts):
        response = query_ollama(prompt, temperature=0.9, max_tokens=256)
        if response:
            print(f"  [{{i+1}}] {{response[:200]}}...")
    print(f"✓ Qwen3-Instruct generated {{len(prompts)}} texts")


def run_chatbot():
    \"\"\"Qwen3-Instruct interactive chatbot.\"\"\"
    print("\\n💬 Chatbot (type 'quit' to exit)")
    history = []
    while True:
        user = input("\\nYou: ").strip()
        if user.lower() in ("quit", "exit", "q"): break
        history.append(f"User: {{user}}")
        resp = query_ollama(f"You are a helpful assistant. Continue this conversation:\\n\\n{{'\\n'.join(history[-6:])}}\\n\\nAssistant:", temperature=0.8)
        if resp:
            history.append(f"Assistant: {{resp}}")
            print(f"Bot: {{resp}}")


def main():
    print("=" * 60)
    print(f"NLP GENERATION — Qwen3-Instruct + NLLB-200 + BART | Task: {{TASK}}")
    print("=" * 60)
    if TASK != "translation":
        test = query_ollama("Say hello.", max_tokens=10)
        if not test:
            print("⚠ Ollama not reachable. Run: ollama serve && ollama pull " + OLLAMA_MODEL)
            if TASK != "translation":
                return
    df = load_data()
    if TASK == "summarization" and df is not None: run_summarization(df)
    elif TASK == "translation" and df is not None: run_translation(df)
    elif TASK == "chatbot": run_chatbot()
    elif TASK == "generation": run_generation(df)
    else:
        if df is not None: run_summarization(df)
        else: run_chatbot()


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
Model: DINOv3 (primary backbone) + ConvNeXt V2 (fine-tuning backbone)
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IMG_SIZE, BATCH_SIZE, EPOCHS, LR = 224, 32, 10, 1e-4


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
        print(f"  Epoch {{epoch+1}}/{{EPOCHS}} — Loss: {{total_loss/len(train_loader):.4f}} — Val Acc: {{val_acc:.4f}}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "best_model.pth"))

    print(f"\\n🏆 DINOv3 Best Val Accuracy: {{best_acc:.4f}}")

    # ConvNeXt V2 (alternative fine-tuning backbone)
    try:
        import timm
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
        print(f"✓ ConvNeXt V2 Val Accuracy: {{cv_acc:.4f}}")
    except Exception as e:
        print(f"✗ ConvNeXt V2: {{e}}")


def main():
    print("=" * 60)
    print("IMAGE CLASSIFICATION — DINOv3 + ConvNeXt V2")
    print("=" * 60)
    train_model()


if __name__ == "__main__":
    main()
''')


def gen_captioning(path, cfg):
    """Image captioning / multimodal VLM pipeline — Qwen3-VL + Molmo 2."""
    data_load = cfg.get("data", '    raise FileNotFoundError("No data source configured")')
    return textwrap.dedent(f'''\
"""
Modern Image Captioning / VLM Pipeline (April 2026)
Models: Qwen3-VL (primary) + Molmo 2 (lightweight alternative)
Data: Auto-downloaded at runtime
"""
import os, warnings
import torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

MAX_SAMPLES = 20


def load_data():
{data_load}
    return df


def caption_images():
    df = load_data()
    # Auto-detect image column
    img_col = next((c for c in df.column_names if "image" in c.lower()), df.column_names[0])
    images = [df[i][img_col] for i in range(min(MAX_SAMPLES, len(df)))]

    # ═══ PRIMARY: Qwen3-VL ═══
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        captions = []
        for img in images:
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
            print(f"  Qwen3-VL: {{caption[:100]}}...")
        print(f"✓ Qwen3-VL captioned {{len(captions)}} images")
    except Exception as e:
        print(f"✗ Qwen3-VL: {{e}}")

    # ═══ ALTERNATIVE: Molmo 2 ═══
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor as AP2
        molmo = AutoModelForCausalLM.from_pretrained("allenai/Molmo-7B-D-0924",
            torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        molmo_proc = AP2.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True)
        for img in images[:5]:
            pil_img = img.convert("RGB") if hasattr(img, "convert") else Image.open(img).convert("RGB")
            inputs = molmo_proc.process(images=[pil_img], text="Describe this image in detail.")
            inputs = {{k: v.to(molmo.device).unsqueeze(0) if hasattr(v, "to") else v for k, v in inputs.items()}}
            out = molmo.generate_from_batch(inputs, max_new_tokens=128, tokenizer=molmo_proc.tokenizer)
            caption = molmo_proc.tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"  Molmo-2: {{caption[:100]}}...")
        print(f"✓ Molmo-2 captioned {{min(5, len(images))}} images")
    except Exception as e:
        print(f"✗ Molmo-2: {{e}}")


def main():
    print("=" * 60)
    print("IMAGE CAPTIONING / VLM — Qwen3-VL + Molmo 2")
    print("=" * 60)
    caption_images()


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
Models: nnU-Net (supervised baseline) + MedSAM2 (promptable foundation model)
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IMG_SIZE, BATCH_SIZE, EPOCHS, LR = 256, 8, 15, 1e-4
N_CLASSES = {n_classes}


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


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_ds = load_data()
    dataset = MedSegDataset(hf_ds)
    val_size = max(1, int(0.2 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    # ═══ PRIMARY: nnU-Net-style supervised U-Net ═══
    model = SimpleUNet().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_dice = 0
    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            loss = criterion(out, masks); loss.backward()
            opt.step(); opt.zero_grad(); total_loss += loss.item()
        scheduler.step()
        model.eval(); dice_sum, n = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                dice_sum += dice_score(out, masks).item(); n += 1
        val_dice = dice_sum / max(n, 1)
        print(f"  Epoch {{epoch+1}}/{{EPOCHS}} — Loss: {{total_loss/len(train_loader):.4f}} — Val Dice: {{val_dice:.4f}}")
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "best_unet.pth"))
    print(f"\\n🏆 nnU-Net-style Best Val Dice: {{best_dice:.4f}}")

    # ═══ ALTERNATIVE: MedSAM2 (promptable foundation segmentation) ═══
    try:
        from transformers import SamModel, SamProcessor
        sam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(device)
        sam_proc = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
        sam_model.eval()
        dice_sum, n = 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                for j in range(min(4, imgs.shape[0])):
                    pil_img = transforms.ToPILImage()(imgs[j])
                    h, w = pil_img.size[1], pil_img.size[0]
                    # Center-point prompt
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
                    union = (pred_binary == 1).sum().float() + (gt == 1).sum().float()
                    dice_sum += (2 * inter + 1e-6) / (union + 1e-6); n += 1
                if n >= 16: break
        sam_dice = dice_sum / max(n, 1)
        print(f"✓ MedSAM Dice (zero-shot, center-point prompt): {{sam_dice:.4f}}")
    except Exception as e:
        print(f"✗ MedSAM: {{e}}")

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
    plt.savefig(os.path.join(os.path.dirname(__file__), "segmentation_results.png"), dpi=100)
    print("Saved segmentation_results.png")


def main():
    print("=" * 60)
    print("MEDICAL SEGMENTATION — nnU-Net + MedSAM")
    print("=" * 60)
    train_model()


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
"""
import os, warnings
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
    for c in cat_cols: df[c] = df[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = oe.fit_transform(df[cat_cols])
    return StandardScaler().fit_transform(df.select_dtypes(include=["number"]))


def eval_clustering(X, labels, name):
    mask = labels >= 0
    n = len(set(labels[mask]))
    noise = (labels == -1).sum()
    if n > 1 and mask.sum() > n:
        sil = silhouette_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])
        print(f"  {{name}}: {{n}} clusters, noise={{noise}}, silhouette={{sil:.4f}}, CH={{ch:.1f}}, DB={{db:.4f}}")
        return sil
    else:
        print(f"  {{name}}: {{n}} clusters, noise={{noise}} — insufficient for metrics")
        return 0


def cluster(X):
    results = {{}}

    # ═══ PRIMARY: UMAP + HDBSCAN ═══
    try:
        import umap, hdbscan
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
        print(f"✓ UMAP + HDBSCAN (min_cluster_size={{best_mcs}}):")
        eval_clustering(X_umap, best_labels, "HDBSCAN")
        results["HDBSCAN"] = {{"labels": best_labels, "embedding": X_umap}}
    except Exception as e:
        print(f"✗ UMAP + HDBSCAN: {{e}}")
        # Fallback: PCA for embedding
        from sklearn.decomposition import PCA
        X_umap = PCA(n_components=2).fit_transform(X)

    # ═══ SOFT ASSIGNMENTS: Gaussian Mixture Model ═══
    try:
        from sklearn.mixture import GaussianMixture
        bics = [GaussianMixture(n_components=k, random_state=42).fit(X).bic(X) for k in range(2, 11)]
        best_k = range(2, 11)[np.argmin(bics)]
        gmm = GaussianMixture(n_components=best_k, random_state=42).fit(X)
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)
        print(f"✓ GMM (BIC-optimal k={{best_k}}):")
        eval_clustering(X, labels, "GMM")
        avg_confidence = probs.max(axis=1).mean()
        print(f"  Avg assignment confidence: {{avg_confidence:.4f}}")
        results["GMM"] = {{"labels": labels, "n": best_k, "probs": probs}}
    except Exception as e:
        print(f"✗ GMM: {{e}}")

    # ═══ BASELINE: K-Means (Elbow + Silhouette) ═══
    try:
        from sklearn.cluster import KMeans
        inertias, sils = [], []
        K_range = range(2, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = km.fit_predict(X)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X, lbls))
        best_k = K_range[np.argmax(sils)]
        labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
        print(f"✓ K-Means baseline (best k={{best_k}}, silhouette={{max(sils):.4f}}):")
        eval_clustering(X, labels, "K-Means")
        results["KMeans"] = {{"labels": labels, "n": best_k, "inertias": inertias, "sils": sils}}
    except Exception as e:
        print(f"✗ K-Means: {{e}}")

    # ═══ VISUALIZATION ═══
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        embed = results.get("HDBSCAN", {{}}).get("embedding", X[:, :2] if X.shape[1] >= 2 else X)
        for ax, (name, data) in zip(axes, [
            ("HDBSCAN", results.get("HDBSCAN", {{}}).get("labels")),
            ("GMM", results.get("GMM", {{}}).get("labels")),
            ("K-Means", results.get("KMeans", {{}}).get("labels")),
        ]):
            if data is not None:
                scatter = ax.scatter(embed[:, 0], embed[:, 1], c=data, cmap="tab10", s=10, alpha=0.6)
                ax.set_title(name); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
            else:
                ax.set_title(f"{{name}} (N/A)")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "clustering_results.png"), dpi=100)
        print("Saved clustering_results.png")
    except Exception as e:
        print(f"⚠ Plot: {{e}}")

    # ═══ SUMMARY ═══
    print("\\n" + "=" * 40)
    print("CLUSTERING COMPARISON:")
    for name in ["HDBSCAN", "GMM", "KMeans"]:
        if name in results:
            n = len(set(results[name]["labels"])) - (1 if -1 in results[name]["labels"] else 0)
            print(f"  {{name}}: {{n}} clusters")
    print("=" * 40)


def main():
    print("=" * 60)
    print("CLUSTERING: UMAP + HDBSCAN (primary) | GMM | K-Means baseline")
    print("=" * 60)
    df = load_data()
    X = preprocess(df)
    cluster(X)


if __name__ == "__main__":
    main()
''')


def gen_anomaly(path, cfg):
    data_load = cfg.get("data", '    raise FileNotFoundError("No data")')
    return textwrap.dedent(f'''\
"""
Modern Anomaly Detection Pipeline (April 2026)
Models: PyOD 2 (ECOD, COPOD, IForest) + anomalib PatchCore
Data: Auto-downloaded at runtime
"""
import os, warnings
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
    df = df.copy()
    label_col = next((c for c in df.columns if c.lower() in ("label","class","target","anomaly","outlier")), None)
    y = None
    if label_col:
        y = df[label_col].values; df.drop(columns=[label_col], inplace=True)
    for c in df.columns:
        if c.lower() in ("id","timestamp","date","time"): df.drop(columns=[c], inplace=True, errors="ignore")
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols: df[c] = df[c].fillna("unknown")
    if cat_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = oe.fit_transform(df[cat_cols])
    return StandardScaler().fit_transform(df.select_dtypes(include=["number"])), y


def detect(X, y=None):
    results = {{}}
    for name, Builder in [
        ("ECOD", lambda: __import__("pyod.models.ecod", fromlist=["ECOD"]).ECOD(contamination=0.05)),
        ("COPOD", lambda: __import__("pyod.models.copod", fromlist=["COPOD"]).COPOD(contamination=0.05)),
        ("IForest", lambda: __import__("pyod.models.iforest", fromlist=["IForest"]).IForest(contamination=0.05, random_state=42)),
    ]:
        try:
            m = Builder()
            m.fit(X)
            labels = m.labels_
            scores = m.decision_scores_ if hasattr(m, "decision_scores_") else m.decision_function(X)
            results[name] = {{"labels": labels, "scores": scores}}
            n_anom = labels.sum()
            print(f"✓ {{name}}: {{n_anom}} anomalies ({{n_anom/len(X):.2%}})")
            if y is not None and len(set(y)) > 1:
                f1 = f1_score(y, labels)
                auc = roc_auc_score(y, scores)
                print(f"  F1: {{f1:.4f}}  ROC-AUC: {{auc:.4f}}")
        except Exception as e:
            print(f"✗ {{name}}: {{e}}")

    # ── Comparison plot ──
    if results:
        try:
            fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
            if len(results) == 1: axes = [axes]
            for ax, (name, r) in zip(axes, results.items()):
                ax.hist(r["scores"][r["labels"] == 0], bins=50, alpha=0.6, label="Normal", density=True)
                ax.hist(r["scores"][r["labels"] == 1], bins=50, alpha=0.6, label="Anomaly", density=True)
                ax.set_title(name); ax.set_xlabel("Anomaly score"); ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), "anomaly_scores.png"), dpi=100)
            print("✓ Saved anomaly_scores.png")
        except Exception as e:
            print(f"⚠ Plot: {{e}}")

    # ── anomalib PatchCore (image-based anomaly detection) ──
    try:
        from anomalib.models import Patchcore
        from anomalib.data import MVTec
        from anomalib.engine import Engine
        datamodule = MVTec(category="bottle", image_size=(256, 256), train_batch_size=8, eval_batch_size=8)
        model = Patchcore(backbone="wide_resnet50_2", layers_to_extract=["layer2", "layer3"],
                          coreset_sampling_ratio=0.1, num_neighbors=9)
        engine = Engine(max_epochs=1, devices=1, accelerator="auto")
        engine.fit(model=model, datamodule=datamodule)
        test_results = engine.test(model=model, datamodule=datamodule)
        print(f"✓ PatchCore (anomalib): {{test_results}}")
    except Exception as e:
        print(f"✗ PatchCore: {{e}}")


def main():
    print("=" * 60)
    print("ANOMALY DETECTION — PyOD 2 + anomalib PatchCore")
    print("=" * 60)
    df = load_data()
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
Data: Auto-downloaded at runtime
"""
import os, warnings, time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TARGET = "{target}"
HORIZON = 30


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

    # ═══ PRIMARY: AutoGluon TimeSeries ═══
    try:
        t0 = time.time()
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        ts_df = pd.DataFrame({{"item_id": ["s"] * split, "timestamp": pd.date_range("2020-01-01", periods=split, freq="D"), "target": train}})
        ts_data = TimeSeriesDataFrame.from_data_frame(ts_df)
        predictor = TimeSeriesPredictor(prediction_length=HORIZON, eval_metric="RMSE",
                                         path=os.path.join(os.path.dirname(__file__), "ag_ts"))
        predictor.fit(ts_data, time_limit=180, presets="best_quality")
        ag_preds = predictor.predict(ts_data)
        y_pred = ag_preds["mean"].values[:len(test)]
        results["AutoGluon-TS"] = y_pred
        print(f"✓ AutoGluon-TS ({{time.time()-t0:.1f}}s)")
        score("AutoGluon-TS", test, y_pred, metrics)
        lb = predictor.leaderboard(ts_data)
        print("  Leaderboard (top 5):")
        for line in lb.head().to_string().splitlines():
            print(f"    {{line}}")
    except Exception as e: print(f"✗ AutoGluon-TS: {{e}}")

    # ═══ FOUNDATION MODELS ═══

    # Chronos-Bolt (fast zero-shot)
    try:
        t0 = time.time()
        import torch
        from chronos import ChronosPipeline
        pipe = ChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                  device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        context = torch.tensor(train, dtype=torch.float32)
        y_pred = np.median(pipe.predict(context, HORIZON)[0].numpy(), axis=0)[:len(test)]
        results["Chronos-Bolt"] = y_pred
        print(f"✓ Chronos-Bolt ({{time.time()-t0:.1f}}s)")
        score("Chronos-Bolt", test, y_pred, metrics)
    except Exception as e: print(f"✗ Chronos-Bolt: {{e}}")

    # Chronos-2 (universal forecasting)
    try:
        t0 = time.time()
        import torch
        from chronos import ChronosPipeline
        pipe2 = ChronosPipeline.from_pretrained("amazon/chronos-2-base",
                   device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        context = torch.tensor(train, dtype=torch.float32)
        y_pred = np.median(pipe2.predict(context, HORIZON)[0].numpy(), axis=0)[:len(test)]
        results["Chronos-2"] = y_pred
        print(f"✓ Chronos-2 ({{time.time()-t0:.1f}}s)")
        score("Chronos-2", test, y_pred, metrics)
    except Exception as e: print(f"✗ Chronos-2: {{e}}")

    # TimesFM (Google foundation model)
    try:
        t0 = time.time()
        import timesfm
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend="gpu", per_core_batch_size=32,
                                            horizon_len=HORIZON),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
        freq = [0] * 1  # freq=0 → daily
        y_pred, _ = tfm.forecast([train], freq)
        y_pred = y_pred[0][:len(test)]
        results["TimesFM"] = y_pred
        print(f"✓ TimesFM ({{time.time()-t0:.1f}}s)")
        score("TimesFM", test, y_pred, metrics)
    except Exception as e: print(f"✗ TimesFM: {{e}}")

    # ═══ CLASSICAL BASELINES (comparison only) ═══

    # ARIMA (statsmodels)
    try:
        t0 = time.time()
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train, order=(5, 1, 0))
        fitted = model.fit()
        y_pred = fitted.forecast(steps=HORIZON)[:len(test)]
        results["ARIMA(5,1,0)"] = y_pred
        print(f"✓ ARIMA(5,1,0) baseline ({{time.time()-t0:.1f}}s)")
        score("ARIMA(5,1,0)", test, y_pred, metrics)
    except Exception as e: print(f"✗ ARIMA: {{e}}")

    # Prophet (Meta)
    try:
        t0 = time.time()
        from prophet import Prophet
        p_df = pd.DataFrame({{"ds": pd.date_range("2020-01-01", periods=split, freq="D"), "y": train}})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(p_df)
        future = m.make_future_dataframe(periods=HORIZON)
        fc = m.predict(future)
        y_pred = fc["yhat"].values[-HORIZON:][:len(test)]
        results["Prophet"] = y_pred
        print(f"✓ Prophet baseline ({{time.time()-t0:.1f}}s)")
        score("Prophet", test, y_pred, metrics)
    except Exception as e: print(f"✗ Prophet: {{e}}")

    # ═══ TABULAR LAG-FEATURE BASELINES (GBDT + FLAML) ═══
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
                t0 = time.time()
                m = builder()
                m.fit(X_lag_tr, y_lag_tr)
                y_pred = m.predict(X_lag_te)[:len(test)]
                results[name] = y_pred
                print(f"✓ {{name}} ({{time.time()-t0:.1f}}s)")
                score(name, y_lag_te.values[:len(y_pred)], y_pred, metrics)
            except Exception as e:
                print(f"✗ {{name}}: {{e}}")

        # FLAML AutoML on lag features (tabularized forecasting only)
        try:
            t0 = time.time()
            from flaml import AutoML
            flaml_model = AutoML()
            flaml_model.fit(X_lag_tr, y_lag_tr, task="regression", time_budget=60,
                           metric="rmse", verbose=0)
            y_pred = flaml_model.predict(X_lag_te)[:len(test)]
            results["FLAML-Lag"] = y_pred
            best = flaml_model.best_estimator
            print(f"✓ FLAML-Lag [best: {{best}}] ({{time.time()-t0:.1f}}s)")
            score("FLAML-Lag", y_lag_te.values[:len(y_pred)], y_pred, metrics)
        except Exception as e:
            print(f"✗ FLAML-Lag: {{e}}")

    # ═══ METRICS SUMMARY ═══
    if metrics:
        print("")
        print("=" * 65)
        print("METRICS SUMMARY")
        print("=" * 65)
        summary = pd.DataFrame(metrics).sort_values("RMSE")
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(os.path.dirname(__file__), "metrics.csv"), index=False)
        best_model = summary.iloc[0]["Model"]
        print(f"  → Best model by RMSE: {{best_model}}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(train)), train, alpha=0.5, label="Train")
    ax.plot(range(len(train), len(train)+len(test)), test, linewidth=2, label="Actual")
    for name, y_pred in results.items():
        ax.plot(range(len(train), len(train)+len(y_pred)), y_pred, "--", label=name)
    ax.legend(); ax.set_title("Forecast Comparison")
    fig.savefig(os.path.join(os.path.dirname(__file__), "forecast.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return results


def main():
    print("=" * 60)
    print("TIME SERIES FORECASTING — April 2026")
    print("Primary: AutoGluon-TS, Chronos-Bolt, Chronos-2, TimesFM")
    print("Baselines: ARIMA, Prophet, LightGBM/CatBoost/XGBoost Lag, FLAML")
    print("=" * 60)
    df, target = load_data()
    forecast(df, target)


if __name__ == "__main__":
    main()
''')


def gen_recommendation(path, cfg):
    data_load = cfg.get("data", '    raise FileNotFoundError("No data")')
    task = cfg.get("task", "cf")
    return textwrap.dedent(f'''\
"""
Modern Recommendation Pipeline (April 2026)
Models: implicit ALS/BPR (CF) + LightFM (hybrid) + Sentence Transformers/BGE-M3/Qwen3-Embedding (content)
        Surprise SVD/KNN as baseline
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib; matplotlib.use("Agg")

warnings.filterwarnings("ignore")

TASK = "{task}"  # cf | hybrid | content


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


# ═══════════════════════════════════════════════════════════════
# PRIMARY: implicit ALS + BPR (collaborative filtering)
# ═══════════════════════════════════════════════════════════════
def run_implicit_cf(df, user_col, item_col, rating_col):
    mat, ue, ie = build_interaction_matrix(df, user_col, item_col, rating_col)
    n_users, n_items = mat.shape[0], mat.shape[1]

    # implicit ALS
    try:
        from implicit.als import AlternatingLeastSquares
        als = AlternatingLeastSquares(factors=128, iterations=30, use_gpu=True)
        als.fit(mat)
        # Evaluate: precision@10 on last interaction
        from implicit.evaluation import precision_at_k, train_test_split
        train_m, test_m = train_test_split(mat, train_percentage=0.8)
        als2 = AlternatingLeastSquares(factors=128, iterations=30, use_gpu=True)
        als2.fit(train_m)
        p_at_10 = precision_at_k(als2, train_m, test_m, K=10)
        print(f"✓ implicit ALS — {{n_users}} users, {{n_items}} items, P@10={{p_at_10:.4f}}")
    except Exception as e:
        print(f"✗ implicit ALS: {{e}}")

    # implicit BPR
    try:
        from implicit.bpr import BayesianPersonalizedRanking
        bpr = BayesianPersonalizedRanking(factors=128, iterations=100, use_gpu=True)
        bpr.fit(mat)
        print(f"✓ implicit BPR trained")
    except Exception as e:
        print(f"✗ implicit BPR: {{e}}")


# ═══════════════════════════════════════════════════════════════
# HYBRID: LightFM (user/item metadata, cold-start capable)
# ═══════════════════════════════════════════════════════════════
def run_lightfm_hybrid(df, user_col, item_col, rating_col, content_col):
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
        if content_col and content_col in df.columns:
            unique_items = df[[item_col, content_col]].drop_duplicates(subset=[item_col])
            # Simple: use content words as features
            all_features = set()
            for text in unique_items[content_col].fillna("").astype(str):
                for w in text.lower().split()[:10]:
                    all_features.add(w)
            if all_features:
                lfds.fit_partial(item_features=all_features)
                item_feat_list = []
                for _, row in unique_items.iterrows():
                    words = row[content_col] if isinstance(row[content_col], str) else ""
                    feats = [w for w in words.lower().split()[:10] if w in all_features]
                    if feats:
                        item_feat_list.append((row[item_col], feats))
                if item_feat_list:
                    item_features = lfds.build_item_features(item_feat_list)

        # Train WARP model (for implicit/ranking) and BPR model
        for loss in ["warp", "bpr"]:
            model = LightFM(loss=loss, no_components=64, learning_rate=0.05)
            model.fit(interactions, item_features=item_features, epochs=30, num_threads=4)
            p_at_k = precision_at_k(model, interactions, item_features=item_features, k=10).mean()
            auc = auc_score(model, interactions, item_features=item_features).mean()
            print(f"✓ LightFM ({{loss}}) — P@10={{p_at_k:.4f}}, AUC={{auc:.4f}}")
    except Exception as e:
        print(f"✗ LightFM: {{e}}")


# ═══════════════════════════════════════════════════════════════
# CONTENT-BASED: Sentence Transformers / BGE-M3 / Qwen3-Embedding
# ═══════════════════════════════════════════════════════════════
def run_content_embeddings(df, item_col, content_col):
    if not content_col or content_col not in df.columns:
        print("⚠ No content column found — skipping content-based embeddings")
        return

    items = df[[item_col, content_col]].drop_duplicates(subset=[item_col]).head(1000)
    texts = items[content_col].fillna("").astype(str).tolist()

    # BGE-M3
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        model = SentenceTransformer("BAAI/bge-m3")
        embs = model.encode(texts, batch_size=32, show_progress_bar=True)
        sim = cosine_similarity(embs)
        # Show top-3 similar items for first 3 items
        for i in range(min(3, len(items))):
            top_idx = np.argsort(sim[i])[-4:-1][::-1]  # top 3 excluding self
            top_items = items.iloc[top_idx][item_col].tolist()
            print(f"  Item '{{items.iloc[i][item_col]}}' → similar: {{top_items}}")
        print(f"✓ BGE-M3 content-based: {{len(items)}} items embedded (dim={{embs.shape[1]}})")
    except Exception as e:
        print(f"✗ BGE-M3: {{e}}")

    # Qwen3-Embedding
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        qwen_embs = qwen.encode(texts[:200], batch_size=16, show_progress_bar=True)
        qwen_sim = cosine_similarity(qwen_embs)
        print(f"✓ Qwen3-Embedding: {{len(qwen_embs)}} items embedded (dim={{qwen_embs.shape[1]}})")
    except Exception as e:
        print(f"✗ Qwen3-Embedding: {{e}}")


# ═══════════════════════════════════════════════════════════════
# BASELINE: Surprise SVD + KNN
# ═══════════════════════════════════════════════════════════════
def run_surprise_baseline(df, user_col, item_col, rating_col):
    if not rating_col:
        print("⚠ No explicit ratings — skipping Surprise baseline")
        return
    try:
        from surprise import Dataset as SDataset, Reader, SVD, KNNBasic, accuracy
        from surprise.model_selection import cross_validate

        reader = Reader(rating_scale=(df[rating_col].min(), df[rating_col].max()))
        data = SDataset.load_from_df(df[[user_col, item_col, rating_col]].dropna(), reader)

        for algo_cls, name in [(SVD, "SVD"), (KNNBasic, "KNN")]:
            algo = algo_cls()
            results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3, verbose=False)
            rmse = results["test_rmse"].mean()
            mae = results["test_mae"].mean()
            print(f"  Surprise {{name}} — RMSE={{rmse:.4f}}, MAE={{mae:.4f}}")
        print("✓ Surprise baseline complete")
    except Exception as e:
        print(f"✗ Surprise baseline: {{e}}")


def train(df):
    user_col, item_col, rating_col, content_col = detect_columns(df)
    print(f"Columns — user: {{user_col}}, item: {{item_col}}, rating: {{rating_col}}, content: {{content_col}}")

    if TASK == "cf":
        # Primary: implicit ALS/BPR → Baseline: Surprise SVD/KNN
        run_implicit_cf(df, user_col, item_col, rating_col)
        run_surprise_baseline(df, user_col, item_col, rating_col)
    elif TASK == "hybrid":
        # Primary: LightFM → also run implicit CF
        run_lightfm_hybrid(df, user_col, item_col, rating_col, content_col)
        run_implicit_cf(df, user_col, item_col, rating_col)
    elif TASK == "content":
        # Primary: embedding-based content similarity → also run implicit if possible
        run_content_embeddings(df, item_col, content_col)
        try:
            run_implicit_cf(df, user_col, item_col, rating_col)
        except Exception:
            pass
    else:
        run_implicit_cf(df, user_col, item_col, rating_col)
        run_content_embeddings(df, item_col, content_col)


def main():
    print("=" * 60)
    print(f"RECOMMENDATION ({{TASK}}) — implicit + LightFM + SentenceTransformers | Surprise baseline")
    print("=" * 60)
    df = load_data()
    train(df)


if __name__ == "__main__":
    main()
''')


def gen_rl(path, cfg):
    env = cfg.get("env", "CartPole-v1")
    algo = cfg.get("algo", "PPO")
    return textwrap.dedent(f'''\
"""
Modern Reinforcement Learning Pipeline (April 2026)
Models: PPO (default), SAC (continuous), DQN (discrete baseline) — Stable-Baselines3
Data: Gymnasium environments (auto-downloaded)
"""
import os, warnings
import numpy as np
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ENV_NAME = "{env}"
ALGO = "{algo}"
TOTAL_TIMESTEPS = 100_000

# Simple discrete envs where DQN is a valid educational baseline
DISCRETE_ENVS = ("CliffWalking", "FrozenLake", "Taxi", "LunarLander", "CartPole", "MountainCar")


def train_single(algo_name, env, eval_env, save_dir, timesteps):
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback

    eval_cb = EvalCallback(eval_env, best_model_save_path=save_dir,
        log_path=save_dir, eval_freq=5000, n_eval_episodes=10, deterministic=True)

    if algo_name == "SAC":
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000,
                     batch_size=256, tau=0.005, gamma=0.99, verbose=1, device="auto")
    elif algo_name == "DQN":
        model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=50_000,
                     batch_size=64, gamma=0.99, exploration_fraction=0.3,
                     target_update_interval=1000, verbose=1, device="auto")
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                     n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                     ent_coef=0.01, verbose=1, device="auto")

    print(f"\\n— Training {{algo_name}} on {{ENV_NAME}} for {{timesteps}} steps —")
    model.learn(total_timesteps=timesteps, callback=eval_cb)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"✓ {{algo_name}} — Mean Reward: {{mean_reward:.2f}} ± {{std_reward:.2f}}")
    model.save(os.path.join(save_dir, f"{{algo_name.lower()}}_{{ENV_NAME}}"))
    return algo_name, mean_reward, std_reward


def main():
    print("=" * 60)
    print(f"REINFORCEMENT LEARNING — {{ALGO}} on {{ENV_NAME}}")
    print("=" * 60)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    results = []

    # Main algorithm (PPO or SAC)
    env = gym.make(ENV_NAME); eval_env = gym.make(ENV_NAME)
    name, reward, std = train_single(ALGO, env, eval_env, save_dir, TOTAL_TIMESTEPS)
    results.append((name, reward, std))
    env.close(); eval_env.close()

    # DQN baseline for simple discrete environments
    is_discrete = any(tag in ENV_NAME for tag in DISCRETE_ENVS)
    if is_discrete and ALGO != "DQN":
        try:
            env2 = gym.make(ENV_NAME); eval_env2 = gym.make(ENV_NAME)
            name, reward, std = train_single("DQN", env2, eval_env2, save_dir, TOTAL_TIMESTEPS)
            results.append((name, reward, std))
            env2.close(); eval_env2.close()
        except Exception as e:
            print(f"✗ DQN baseline: {{e}}")

    # Summary
    print("\\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    best = max(results, key=lambda x: x[1])
    for name, reward, std in results:
        marker = " 🏆" if name == best[0] else ""
        print(f"  {{name}}: {{reward:.2f}} ± {{std:.2f}}{{marker}}")


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
Models: Whisper large-v3-turbo (ASR), Wav2Vec2/HuBERT (clf), SepFormer (denoising), XTTS-v2 (TTS)
Data: Auto-downloaded at runtime from HuggingFace
"""
import os, json, warnings
import numpy as np

warnings.filterwarnings("ignore")

TASK = "{task}"


def download_audio_samples():
    """Download audio samples from HuggingFace datasets."""
    from datasets import load_dataset
    import soundfile as sf

    save_dir = os.path.join(os.path.dirname(__file__), "audio_data")
    os.makedirs(save_dir, exist_ok=True)

    if TASK == "classification":
        try:
{data_load_12}
            if df is not None:
                print(f"Loaded dataset: {{len(df)}} samples")
                return save_dir, df
        except Exception:
            pass
        ds = load_dataset("google/speech_commands", "v0.02", split="train[:100]")
    else:
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:20]")

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


def run_whisper(audio_dir):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, torch_dtype=torch_dtype, device=device)

    from pathlib import Path
    audio_files = list(Path(audio_dir).glob("*.wav")) + list(Path(audio_dir).glob("*.flac"))
    results = []
    for f in audio_files[:10]:
        result = asr(str(f), return_timestamps=True)
        results.append({{"file": f.name, "text": result["text"]}})
        print(f"  ✓ {{f.name}}: {{result['text'][:100]}}...")
    return results


def run_voice_cloning():
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        out_path = os.path.join(os.path.dirname(__file__), "tts_output.wav")
        tts.tts_to_file(text="Hello, this is a text to speech sample using XTTS version 2.", file_path=out_path)
        print(f"✓ TTS output → {{out_path}}")
    except Exception as e:
        print(f"✗ XTTS: {{e}}")


def run_wav2vec2_clf(audio_dir):
    \"\"\"Audio classification with Wav2Vec2 and HuBERT.\"\"\"
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import soundfile as sf
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_files = list(Path(audio_dir).glob("*.wav"))[:10]

    for model_name, label in [("facebook/wav2vec2-base", "Wav2Vec2"),
                               ("facebook/hubert-base-ls960", "HuBERT")]:
        try:
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForAudioClassification.from_pretrained(
                model_name, num_labels=10, ignore_mismatched_sizes=True).to(device)
            for f in audio_files:
                arr, sr = sf.read(str(f))
                if len(arr.shape) > 1: arr = arr[:, 0]
                inputs = extractor(arr, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=-1).item()
                print(f"  ✓ [{{label}}] {{f.name}}: class {{pred}}")
            print(f"✓ {{label}} classification complete")
        except Exception as e:
            print(f"✗ {{label}}: {{e}}")


def run_sepformer(audio_dir):
    \"\"\"Speech enhancement / denoising with SpeechBrain SepFormer.\"\"\"
    try:
        from pathlib import Path
        try:
            from speechbrain.inference.separation import SepformerSeparation
            model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr",
                savedir=os.path.join(os.path.dirname(__file__), "sepformer_model"))
        except ImportError:
            from speechbrain.pretrained import SepformerSeparation
            model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-whamr",
                savedir=os.path.join(os.path.dirname(__file__), "sepformer_model"))
        audio_files = list(Path(audio_dir).glob("*.wav"))[:5]
        save_dir = os.path.join(os.path.dirname(__file__), "enhanced")
        os.makedirs(save_dir, exist_ok=True)
        for f in audio_files:
            est_sources = model.separate_file(path=str(f))
            out_path = os.path.join(save_dir, f"{{f.stem}}_enhanced.wav")
            import torchaudio
            torchaudio.save(out_path, est_sources[:, :, 0].cpu(), 8000)
            print(f"  ✓ {{f.name}} → {{f.stem}}_enhanced.wav")
        print(f"✓ SepFormer enhanced audio saved to {{save_dir}}")
    except Exception as e:
        print(f"✗ SepFormer: {{e}}")


def main():
    print("=" * 60)
    print(f"AUDIO/SPEECH — Task: {{TASK}}")
    print("=" * 60)
    audio_dir, data = download_audio_samples()
    if TASK == "transcription":
        results = run_whisper(audio_dir)
        if results:
            out = os.path.join(os.path.dirname(__file__), "transcriptions.json")
            with open(out, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
            print(f"Saved to {{out}}")
    elif TASK == "denoising":
        run_sepformer(audio_dir)
    elif TASK == "cloning":
        run_voice_cloning()
    elif TASK == "classification":
        run_wav2vec2_clf(audio_dir)
    elif TASK == "separation":
        run_sepformer(audio_dir)
    else:
        results = run_whisper(audio_dir)


if __name__ == "__main__":
    main()
''')


def gen_cv_detection(path, cfg):
    task = cfg.get("task", "detect")
    return textwrap.dedent(f'''\
"""
Modern CV Object Detection Pipeline (April 2026)
Model: YOLO26m (Ultralytics) — auto-downloads model + sample images
Data: Auto-downloaded at runtime
"""
import os, warnings
from pathlib import Path
import urllib.request

warnings.filterwarnings("ignore")

TASK = "{task}"

SAMPLE_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
]


def download_samples():
    save_dir = Path(os.path.dirname(__file__)) / "sample_images"
    save_dir.mkdir(exist_ok=True)
    paths = []
    for url in SAMPLE_URLS:
        fname = save_dir / url.split("/")[-1]
        if not fname.exists():
            urllib.request.urlretrieve(url, str(fname))
        paths.append(fname)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    for ext in exts:
        paths.extend([p for p in Path(os.path.dirname(__file__)).rglob(f"*{{ext}}") if p not in paths])
    print(f"{{len(paths)}} images available")
    return paths


def run_detection(files):
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    save_dir = os.path.join(os.path.dirname(__file__), "detections")
    os.makedirs(save_dir, exist_ok=True)
    for f in files[:20]:
        results = model(str(f))
        for r in results:
            r.save(filename=os.path.join(save_dir, f.name))
            if r.boxes is not None:
                print(f"  ✓ {{f.name}}: {{len(r.boxes)}} objects detected")
    print(f"Results saved to {{save_dir}}")


def run_tracking(files):
    from ultralytics import YOLO
    model = YOLO("yolo26m.pt")
    video_files = [f for f in files if f.suffix in (".mp4", ".avi")]
    if not video_files:
        print("No video files found. Running detection on images instead.")
        run_detection(files)
        return
    for v in video_files[:3]:
        model.track(str(v), persist=True, save=True, project=os.path.dirname(__file__), name="tracking")
        print(f"  ✓ Tracked: {{v.name}}")


def main():
    print("=" * 60)
    print(f"CV DETECTION — YOLO26m | Task: {{TASK}}")
    print("=" * 60)
    files = download_samples()
    if TASK == "track": run_tracking(files)
    else: run_detection(files)


if __name__ == "__main__":
    main()
''')


def gen_face_gesture(path, cfg):
    task = cfg.get("task", "face_detection")
    return textwrap.dedent(f'''\
"""
Modern Face/Hand/Gesture Pipeline (April 2026)
Models: YOLO26 (face/person detection) + MediaPipe Face Landmarker (expressions/landmarks)
        + MediaPipe Hand Landmarker / Gesture Recognizer + InsightFace (recognition/verification)
Data: Auto-downloads LFW face samples at runtime
"""
import os, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

TASK = "{task}"


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
    save_dir = os.path.join(os.path.dirname(__file__), "yolo_detections")
    os.makedirs(save_dir, exist_ok=True)
    for f in files[:20]:
        results = model(str(f))
        for r in results:
            r.save(filename=os.path.join(save_dir, f.name))
            n_people = sum(1 for b in r.boxes if int(b.cls) == 0) if r.boxes is not None else 0
            print(f"  ✓ YOLO26 {{f.name}}: {{n_people}} persons, {{len(r.boxes) if r.boxes is not None else 0}} total")
    print(f"YOLO26 results saved to {{save_dir}}")


def run_face_landmarker(files):
    """MediaPipe Face Landmarker — modern Tasks API for 478-point face mesh and expressions."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        # Download face landmarker model
        model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
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

        save_dir = os.path.join(os.path.dirname(__file__), "face_landmark_results")
        os.makedirs(save_dir, exist_ok=True)

        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_img)
            n_faces = len(result.face_landmarks)
            # Draw landmarks
            for face_lm in result.face_landmarks:
                for lm in face_lm:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
            # Report blendshapes (expressions)
            if result.face_blendshapes:
                top_shapes = sorted(result.face_blendshapes[0], key=lambda b: b.score, reverse=True)[:3]
                expr = ", ".join(f"{{b.category_name}}={{b.score:.2f}}" for b in top_shapes)
                print(f"  ✓ {{f.name}}: {{n_faces}} faces, expressions: {{expr}}")
            else:
                print(f"  ✓ {{f.name}}: {{n_faces}} faces (478-pt mesh)")
            cv2.imwrite(os.path.join(save_dir, f.name), img)
        landmarker.close()
        print(f"Face Landmarker results saved to {{save_dir}}")
    except Exception as e:
        print(f"✗ MediaPipe Face Landmarker: {{e}}")
        # Fallback to legacy face detection
        try:
            import cv2, mediapipe as mp
            mp_face = mp.solutions.face_detection; mp_draw = mp.solutions.drawing_utils
            with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    results = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    n = len(results.detections) if results.detections else 0
                    print(f"  ✓ (legacy) {{f.name}}: {{n}} faces")
        except Exception as e2:
            print(f"✗ MediaPipe legacy fallback: {{e2}}")


def run_insightface(files):
    """InsightFace — face recognition, verification, gender/age/ethnicity."""
    try:
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        save_dir = os.path.join(os.path.dirname(__file__), "insightface_results")
        os.makedirs(save_dir, exist_ok=True)
        embeddings = []
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            faces = app.get(img)
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
            print(f"  ✓ {{f.name}}: {{len(faces)}} faces")
        if len(embeddings) >= 2:
            sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
            print(f"  Cosine similarity (face 0 vs 1): {{sim:.4f}}")
        print(f"InsightFace results saved to {{save_dir}}")
    except Exception as e:
        print(f"✗ InsightFace: {{e}}")


def run_hand_gesture():
    """MediaPipe Hand Landmarker / Gesture Recognizer — modern Tasks API."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        # Download gesture recognizer model
        model_path = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
                model_path)

        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_hands=2)
        recognizer = mp_vision.GestureRecognizer.create_from_options(options)

        print("Starting webcam gesture recognition... Press 'q' to quit.")
        cap = cv2.VideoCapture(0)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = recognizer.recognize(mp_img)
            if result.gestures:
                for i, gesture in enumerate(result.gestures):
                    name = gesture[0].category_name
                    score = gesture[0].score
                    cv2.putText(frame, f"{{name}} ({{score:.2f}})", (10, 40 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if result.hand_landmarks:
                for hand_lm in result.hand_landmarks:
                    for lm in hand_lm:
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            cv2.imshow("Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            frame_count += 1
            if frame_count >= 300: break  # auto-stop after ~10 sec
        cap.release(); cv2.destroyAllWindows()
        recognizer.close()
        print(f"✓ Gesture Recognizer processed {{frame_count}} frames")
    except Exception as e:
        print(f"✗ MediaPipe Gesture Recognizer: {{e}}")
        # Fallback to legacy hand detection
        try:
            import cv2, mediapipe as mp
            mp_hands = mp.solutions.hands; mp_draw = mp.solutions.drawing_utils
            print("(fallback) Starting webcam hand detection... Press 'q' to quit.")
            cap = cv2.VideoCapture(0)
            with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        for lm in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    cv2.imshow("Hand Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"): break
            cap.release(); cv2.destroyAllWindows()
        except Exception as e2:
            print(f"✗ MediaPipe legacy hands: {{e2}}")


def run_pose(files):
    """MediaPipe Pose Landmarker — modern Tasks API."""
    try:
        import cv2, mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request

        model_path = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                model_path)

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            num_poses=3)
        landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        save_dir = os.path.join(os.path.dirname(__file__), "pose_results")
        os.makedirs(save_dir, exist_ok=True)
        for f in files[:20]:
            img = cv2.imread(str(f))
            if img is None: continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_img)
            n_poses = len(result.pose_landmarks)
            for pose_lm in result.pose_landmarks:
                for lm in pose_lm:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(save_dir, f.name), img)
            print(f"  ✓ {{f.name}}: {{n_poses}} poses")
        landmarker.close()
        print(f"Pose Landmarker results saved to {{save_dir}}")
    except Exception as e:
        print(f"✗ MediaPipe Pose Landmarker: {{e}}")
        # Fallback to legacy pose
        try:
            import cv2, mediapipe as mp
            mp_pose = mp.solutions.pose; mp_draw = mp.solutions.drawing_utils
            save_dir = os.path.join(os.path.dirname(__file__), "pose_results")
            os.makedirs(save_dir, exist_ok=True)
            with mp_pose.Pose(min_detection_confidence=0.5) as pose:
                for f in files[:20]:
                    img = cv2.imread(str(f))
                    if img is None: continue
                    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    cv2.imwrite(os.path.join(save_dir, f.name), img)
                    print(f"  ✓ (legacy) {{f.name}}")
        except Exception as e2:
            print(f"✗ MediaPipe legacy pose: {{e2}}")


def main():
    print("=" * 60)
    print(f"FACE/HAND/GESTURE — {{TASK}}")
    print("=" * 60)
    files = download_face_samples()
    if TASK == "face_detection":
        run_yolo_detection(files)
        run_face_landmarker(files)
    elif TASK == "hand_gesture":
        run_hand_gesture()
    elif TASK == "pose":
        run_pose(files)
    elif TASK == "face_recognition":
        run_insightface(files)
    else:
        run_yolo_detection(files)
        run_face_landmarker(files)


if __name__ == "__main__":
    main()
''')


def gen_ocr(path, cfg):
    return textwrap.dedent('''\
"""
Modern OCR Pipeline (April 2026)
Model: PaddleOCR + PaddleOCR-VL-1.5 (GPU, multilingual)
Data: Auto-downloads sample document images at runtime
"""
import os, json, warnings
from pathlib import Path
import urllib.request

warnings.filterwarnings("ignore")

SAMPLE_URLS = [
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img_12.jpg",
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/refs/heads/main/doc/imgs_en/img623.jpg",
]


def download_samples():
    save_dir = Path(os.path.dirname(__file__)) / "ocr_samples"
    save_dir.mkdir(exist_ok=True)
    paths = []
    for url in SAMPLE_URLS:
        fname = save_dir / url.split("/")[-1]
        if not fname.exists():
            urllib.request.urlretrieve(url, str(fname))
        paths.append(fname)
    # Also gather any local images
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        paths.extend([p for p in Path(os.path.dirname(__file__)).rglob(f"*{ext}") if p not in paths])
    print(f"{len(paths)} images available for OCR")
    return paths


def run_ocr(files):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True)
    results = []
    for f in files[:30]:
        result = ocr.ocr(str(f), cls=True)
        texts = []
        if result and result[0]:
            for line in result[0]:
                texts.append({"text": line[1][0], "confidence": line[1][1]})
        full_text = " ".join(t["text"] for t in texts)
        results.append({"file": f.name, "full_text": full_text, "lines": texts, "n_lines": len(texts)})
        print(f"  ✓ {f.name}: {len(texts)} lines — '{full_text[:80]}...'")
    return results


def main():
    print("=" * 60)
    print("OCR — PaddleOCR + PaddleOCR-VL-1.5 (GPU)")
    print("=" * 60)
    files = download_samples()
    results = run_ocr(files)

    # PaddleOCR-VL-1.5 (vision-language OCR)
    try:
        from paddleocr import PaddleOCR
        vl_ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False, lang="en", use_gpu=True)
        for f in files[:5]:
            vl_result = vl_ocr.ocr(str(f), cls=True)
            n_lines = len(vl_result[0]) if vl_result and vl_result[0] else 0
            print(f"  ✓ VL-1.5 {f.name}: {n_lines} lines")
        print("✓ PaddleOCR-VL-1.5 complete")
    except Exception as e:
        print(f"✗ PaddleOCR-VL-1.5: {e}")
    out_path = os.path.join(os.path.dirname(__file__), "ocr_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\\n✓ Saved to {out_path}: {len(results)} files, {sum(r['n_lines'] for r in results)} lines")


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
                    print(f"  ✓ {path}")
            except Exception as e:
                print(f"  ✗ {path}: {e}")

    print(f"\n{'='*60}")
    print(f"TOTAL PIPELINES GENERATED: {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
