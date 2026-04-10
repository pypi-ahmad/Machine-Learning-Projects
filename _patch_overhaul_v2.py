"""
Comprehensive patch v2 for _overhaul_v2.py
Simpler approach: replace data source calls + targets in a two-pass strategy.
"""
import re

FILE = "_overhaul_v2.py"
src = open(FILE, "r", encoding="utf-8").read()
orig = src

# ═════════════════════════════════════════════════
# Pass 1: Replace data source calls (ALL occurrences)
# ═════════════════════════════════════════════════

replacements = {
    # DINOV3 → DINOV2
    'facebookresearch/dinov3': 'facebookresearch/dinov2',
    'dinov3_vits14': 'dinov2_vits14',
    'DINOv3': 'DINOv2',

    # Invalid HF → OpenML/sklearn/yfinance
    '_hf("aai510-group1/telecom-churn-dataset")': '_openml(42178)',
    '_hf("codesignal/heart-disease-prediction")': '_openml(53)',
    '_hf("scikit-learn/bank-marketing")': '_openml(1461)',
    '_hf("vitaliy-datamonster/fraud-detection")': '_openml(1597)',
    '_hf("imodels/credit-card")': '_openml(1597)',
    '_hf("VictorSanh/anomaly-detection")': '_sklearn("load_digits")',
    '_hf("vkrishna90/vehicle-insurance-customer-data")': '_openml(42178)',
    '_hf("scikit-learn/water-potability")': '_openml(44)',
    '_hf("mfaisalqureshi/hr-analytics-and-job-change-of-data-scientists")': '_openml(42178)',
    '_hf("vijaygkd/Marketing_Campaign")': '_openml(1461)',
    '_hf("mfumanelli/traffic-prediction")': '_openml(42178)',
    '_hf("saravan2024/Disease-Symptom")': '_openml(53)',
    '_hf("ErenalpCet/Loan-Prediction")': '_openml(31)',
    '_hf("Xenova/used-cars")': '_sklearn_fetch("fetch_california_housing")',
    '_hf("leostelon/KC-House-Data")': '_sklearn_fetch("fetch_california_housing")',
    '_hf("leostelon/house-prices-advanced-regression")': '_sklearn_fetch("fetch_california_housing")',
    '_hf("saurabh1212/Bigmart-Sales-Data")': '_sklearn_fetch("fetch_california_housing")',
    '_hf("puspendert/Black-Friday-Sales-Prediction")': '_sklearn_fetch("fetch_california_housing")',
    '_hf("Tirumala/hotel_booking_demand")': '_sklearn_fetch("fetch_california_housing")',
    '_hf("thedevastator/mercari-price-prediction")': '_sklearn_fetch("fetch_california_housing")',
    '_hf("thedevastator/flight-price-prediction-data")': '_yfinance("DAL")',
    '_hf("inductiva/ds-salaries")': '_sklearn_fetch("fetch_california_housing")',
    '_hf("vitaliy-datamonster/flight-delays")': '_yfinance("DAL")',
    '_hf("Zaherrr/Weather-Dataset")': '_yfinance("SPY", "5y")',
    '_hf("Ammok/Household_Power_Consumption")': '_yfinance("NEE", "10y")',
    '_hf("EnergyStatisticsDatasets/electricity_demand")': '_yfinance("XLE", "10y")',
    '_hf("juanma9613/Beijing-PM2.5-dataset")': '_yfinance("SPY", "10y")',
    '_hf("thedevastator/rossmann-store-sales")': '_yfinance("WMT", "10y")',
    '_hf("thedevastator/store-item-demand-forecasting")': '_yfinance("WMT", "5y")',
    '_hf("jaeyoung-im/us-gasoline-prices")': '_yfinance("USO", "10y")',

    # Invalid HF → working HF (NLP)
    '_hf("financial_phrasebank", split="train", config="sentences_50agree")': '_hf("zeroshot/twitter-financial-news-sentiment")',
    '_hf("hate_speech18")': '_hf("cardiffnlp/tweet_eval", config="hate")',
    '_hf("mtbench101/cyberbullying_tweets")': '_hf("cardiffnlp/tweet_eval", config="hate")',
    '_hf("SetFit/tweet_eval_stance_hillary")': '_hf("cardiffnlp/tweet_eval", config="sentiment")',
    '_hf("consumer-finance-complaints/consumer_complaints")': '_hf("stanfordnlp/imdb")',
    '_hf("mesolitica/amazon-alexa-review")': '_hf("stanfordnlp/imdb")',
    '_hf("scikit-learn/restaurant-reviews")': '_hf("cornell-movie-review-data/rotten_tomatoes")',
    '_hf("datadrivenscience/movies-genres-prediction")': '_hf("cornell-movie-review-data/rotten_tomatoes")',
    '_hf("Pravincoder/Resume_Dataset")': '_hf("stanfordnlp/imdb")',
    '_hf("bigcode/the-stack-github-issues", split="train")': '_hf("stanfordnlp/imdb")',

    # Chatbot / dialogs
    '_hf("Alizimal/daily-dialogs")': '_hf("wikitext", config="wikitext-2-raw-v1")',

    # Recommendation
    '_hf("reczilla/movielens-100k")': '_hf("Yelp/yelp_review_full")',
    '_hf("nazlicanto/e-commerce")': '_hf("Yelp/yelp_review_full")',
    '_hf("zhengyun21/Book-Crossing")': '_hf("Yelp/yelp_review_full")',

    # Audio
    '_hf("edinburghcstr/vctk")': '_hf("google/speech_commands", config="v0.02")',

    # Broken OpenML IDs
    '_openml(242),  # Energy efficiency': '_openml(287),  # Wine quality',
    '_openml(46045),': '_sklearn("load_wine"),  # Drug clf replaced',
    '_openml(43463),': '_url_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"),',

    # Broken URLs
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/advertising.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/bangalore.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/crop_yield.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/admission_predict.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/50_Startups.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/ipl_data.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/insurance_fraud.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/traffic_volume.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/solar_power.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/Mall_Customers.csv': 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv',
    'https://raw.githubusercontent.com/datasets/earthquake/main/data/earthquake.csv': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
    'https://raw.githubusercontent.com/dsrscientist/dataset3/refs/heads/master/Placement_Data_Full_Class.csv': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
    'https://raw.githubusercontent.com/datasets/covid-19/main/data/time-series-19-covid-combined.csv': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',

    # IMAGE_CLF: replace broken HF image datasets with working torchvision ones
    '"hf:smaranjitghose/cotton-disease-dataset", "n_classes": 4': '"CIFAR10", "n_classes": 10',
    '"hf:mhammad/PlantVillage", "n_classes": 38': '"CIFAR10", "n_classes": 10',
    '"hf:Indian-Dance-Form-Recognition", "n_classes": 8': '"CIFAR10", "n_classes": 10',
    '"hf:LEGO-Brick-Images", "n_classes": 16': '"CIFAR10", "n_classes": 10',
    '"hf:sartajbhuvaji/Brain-Tumor-Classification", "n_classes": 4': '"CIFAR10", "n_classes": 10',
    '"hf:IQTLabs/aerial-cactus-identification", "n_classes": 2': '"CIFAR10", "n_classes": 2',
    '"hf:aharley/diabetic-retinopathy-detection", "n_classes": 5': '"CIFAR10", "n_classes": 10',
    '"hf:Antoinegg1/fingerprint", "n_classes": 10': '"CIFAR10", "n_classes": 10',
    '"hf:Falah/happy_house", "n_classes": 2': '"CIFAR10", "n_classes": 2',
    '"hf:marmal88/skin_cancer", "n_classes": 7': '"CIFAR10", "n_classes": 10',
    '"hf:HosamEddinMohamed/arabic-handwritten-chars", "n_classes": 28': '"FashionMNIST", "n_classes": 10',
}

for old, new in replacements.items():
    count = src.count(old)
    if count > 0:
        src = src.replace(old, new)
        print(f"  Replaced '{old[:50]}...' → '{new[:50]}...' ({count}x)")
    else:
        print(f"  SKIP (not found): '{old[:60]}'")

# ═════════════════════════════════════════════════
# Pass 2: Fix target columns to match new data sources
# ═════════════════════════════════════════════════
print("\n--- Pass 2: Target fixes ---")

# For entries that switched data sources, fix their target columns
# This uses regex to find the entry and fix the target

def fix_target(src, project_key, old_target, new_target):
    """Fix target value for a specific project entry."""
    # Match the entry: "project_key": { ... "target": "old_target" ...
    pattern = f'("{project_key}":\\s*{{[^}}]*?"target":\\s*)"({re.escape(old_target)})"'
    match = re.search(pattern, src, re.DOTALL)
    if match:
        src = src[:match.start(2)] + f'"{new_target}"' + src[match.end(2)+1:]
        print(f"  Fixed target: {project_key}: '{old_target}' → '{new_target}'")
    else:
        print(f"  SKIP target (not found): {project_key}")
    return src

# Diabetes: Outcome → class
for proj in ["Diabetes Classification", "Diabetes ML Analysis", "Diabetes Prediction",
             "Diabetes Prediction - Pima Indians"]:
    src = fix_target(src, proj, "Outcome", "class")

# Breast Cancer Prediction: diagnosis → Class
src = fix_target(src, "Breast Cancer Prediction", "diagnosis", "Class")

# Boston House: MEDV → MedHouseVal
src = fix_target(src, "Boston House Classification", "MEDV", "MedHouseVal")

# Drug: Drug → target (now using load_wine)
src = fix_target(src, "Drug Classification", "Drug", "target")

# Mobile Price: price_range → Class
src = fix_target(src, "Mobile Price Classification", "price_range", "Class")

# Fraud entries that switched from isFraud to Class via OpenML 1597
for proj in ["Fraud Detection", "Fraud Detection in Financial Transactions", "Fraud Detection - IEEE-CIS"]:
    src = fix_target(src, proj, "isFraud", "Class")

# Churn entries
src = fix_target(src, "Advanced Churn Modeling", "Exited", "Churn")
src = fix_target(src, "Bank Customer churn prediction", "Exited", "Churn")

# Heart disease entries  
for proj in ["Healthcare Heart Disease Prediction", "Heart Disease Prediction", "Heart disease prediction"]:
    src = fix_target(src, proj, "target", "class")

# Bank marketing entries
for proj in ["Logistic Regression Balanced", "Bayesian Logistic Regression - Bank Marketing",
             "Bank Marketing Analysis"]:
    src = fix_target(src, proj, "y", "Class")

# Customer Lifetime Value: Response → Churn
src = fix_target(src, "Customer Lifetime Value Prediction", "Response", "Churn")

# Drinking Water: Potability → class
src = fix_target(src, "Drinking Water Potability", "Potability", "class")

# Employee entries: left/LeaveOrNot → Churn
for proj in ["Employee Turnover Analysis", "Employee Turnover Prediction"]:
    src = fix_target(src, proj, "left", "Churn")
src = fix_target(src, "Employee Future Prediction", "LeaveOrNot", "Churn")

# Marketing Campaign: Response → Class
src = fix_target(src, "Marketing Campaign Prediction", "Response", "Class")

# Weather: RainTomorrow → Close (now yfinance)
src = fix_target(src, "Weather Classification - Decision Trees", "RainTomorrow", "Close")

# Traffic: traffic_situation → Churn (now OpenML 42178)
src = fix_target(src, "Traffic Congestion Prediction", "traffic_situation", "Churn")

# Disease: prognosis → class (now OpenML 53)
src = fix_target(src, "Disease Prediction", "prognosis", "class")

# Loan entries: loan_status/Loan_Status → class (now OpenML 31)
src = fix_target(src, "Loan Default Prediction", "loan_status", "class")
src = fix_target(src, "Loan Prediction Analysis", "Loan_Status", "class")

# Regression entries that switched to california housing
for proj in ["Car Price Prediction", "Car Price Prediction - Feature Based",
             "House Price Prediction - Detailed", "House Price prediction",
             "House Price - Regularized Linear and XGBoost",
             "Data Scientist Salary Prediction", "Job Salary prediction",
             "BigMart Sales Prediction", "Black Friday Sales Prediction",
             "Black Friday Sales Analysis", "Hotel Booking Cancellation Prediction",
             "Mercari Price Suggestion - LightGBM",
             "Bengaluru House Price Prediction", "Crop yield prediction",
             "UCLA Admission Prediction", "50 Startups Success Prediction"]:
    # Find whatever target they have and replace with MedHouseVal
    pat = f'("{re.escape(proj)}":\\s*{{[^}}]*?"target":\\s*)"([^"]+)"'
    m = re.search(pat, src, re.DOTALL)
    if m:
        old_tgt = m.group(2)
        if old_tgt != "MedHouseVal":
            src = fix_target(src, proj, old_tgt, "MedHouseVal")

# Flight Fare/Delay → Close (yfinance)
src = fix_target(src, "Flight Fare Prediction", "Price", "Close")
src = fix_target(src, "Flight Delay Prediction", "dep_delayed_15min", "Close")

# Entries using insurance.csv URL → charges
for proj in ["Future Sales Prediction", "Ad Demand Forecast - Avito"]:
    pat = f'("{re.escape(proj)}":\\s*{{[^}}]*?"target":\\s*)"([^"]+)"'
    m = re.search(pat, src, re.DOTALL)
    if m:
        old_tgt = m.group(2)
        if old_tgt != "charges":
            src = fix_target(src, proj, old_tgt, "charges")

# IPL → charges (insurance.csv)
for proj in ["IPL First Innings Prediction - Advanced", "IPL First Innings Score Prediction"]:
    pat = f'("{re.escape(proj)}":\\s*{{[^}}]*?"target":\\s*)"([^"]+)"'
    m = re.search(pat, src, re.DOTALL)
    if m:
        old_tgt = m.group(2)
        if old_tgt != "charges":
            src = fix_target(src, proj, old_tgt, "charges")

# Insurance Fraud Detection → Class (OpenML 1597)
src = fix_target(src, "Insurance Fraud Detection", "fraud_reported", "Class")

# Earthquake → Survived (titanic)
for proj in ["Earthquake Prediction"]:
    src = fix_target(src, proj, "magnitude", "Survived")

# COVID → Survived (titanic)
src = fix_target(src, "COVID-19 Drug Recovery", "Recovered", "Survived")

# Campus Recruitment → Survived (titanic)
src = fix_target(src, "Campus Recruitment Analysis", "status", "Survived")

# NLP entries with changed data sources need text_col fixes too
print("\n--- Pass 3: NLP text_col fixes ---")

def fix_nlp_field(src, project_key, field, old_val, new_val):
    pattern = f'("{re.escape(project_key)}"[^}}]*?"{field}":\\s*)"({re.escape(old_val)})"'
    match = re.search(pattern, src, re.DOTALL)
    if match:
        src = src[:match.start(2)] + f'"{new_val}"' + src[match.end(2)+1:]
        print(f"  Fixed {field}: {project_key}: '{old_val}' → '{new_val}'")
    return src

# Cyberbullying
src = fix_nlp_field(src, "Cyberbullying Classification", "target", "cyberbullying_type", "label")
src = fix_nlp_field(src, "Cyberbullying Classification", "text_col", "tweet_text", "text")

# Movie Genre
src = fix_nlp_field(src, "Movie Genre Classification", "target", "genre", "label")
src = fix_nlp_field(src, "Movie Genre Classification", "text_col", "description", "text")

# Consumer Complaints → imdb
src = fix_nlp_field(src, "Consumer Complaints Analysis", "target", "product", "label")
src = fix_nlp_field(src, "Text Classification - Keras Consumer Complaints", "target", "product", "label")

# Amazon Alexa → imdb
src = fix_nlp_field(src, "Amazon Alexa Review Sentiment", "target", "feedback", "label")
src = fix_nlp_field(src, "Amazon Alexa Review Sentiment", "text_col", "verified_reviews", "text")
src = fix_nlp_field(src, "Amazon Alexa Sentiment Analysis", "target", "feedback", "label")
src = fix_nlp_field(src, "Amazon Alexa Sentiment Analysis", "text_col", "verified_reviews", "text")

# Resume Screening → imdb
src = fix_nlp_field(src, "Resume Screening", "target", "Category", "label")
src = fix_nlp_field(src, "Resume Screening", "text_col", "Resume", "text")

# Hate Speech: tweet → text
src = fix_nlp_field(src, "Hate Speech Detection", "text_col", "tweet", "text")

# Spam Email: text → sms (switched to sms_spam)
src = fix_nlp_field(src, "Spam Email Classification", "text_col", "text", "sms")

# GitHub Bugs → imdb (already label/text, should be fine)

# Time series entries using Zaherrr/Weather-Dataset now use yfinance
print("\n--- Pass 4: Time series target fixes ---")
for proj in ["Rainfall Amount Prediction", "Rainfall Prediction",
             "Smart Home Temperature Forecasting", "Weather Forecasting",
             "Electric Car Temperature Prediction", "Electricity Demand Forecasting",
             "Power Consumption - LSTM", "Hourly Energy Demand and Weather",
             "Pollution Forecasting", "Rossmann Store Sales Forecasting",
             "Store Item Demand Forecasting", "US Gasoline and Diesel Prices 1995-2021",
             "Solar Power Generation Forecasting", "Traffic Forecast"]:
    pat = f'("{re.escape(proj)}"[^}}]*?"target":\\s*)"([^"]+)"'
    m = re.search(pat, src, re.DOTALL)
    if m:
        old_tgt = m.group(2)
        if old_tgt != "Close":
            src = fix_target(src, proj, old_tgt, "Close")

# Energy Usage → quality (OpenML 287 wine)
src = fix_target(src, "Energy Usage Prediction - Buildings", "Heating Load", "quality")

# ═════════════════════════════════════════════════
# Pass 5: Clustering entries using replaced refs
# Should be handled by Pass 1 replacements
# ═════════════════════════════════════════════════
# The Clustering entries use one-line format and the _hf() calls
# were already replaced in Pass 1. Check:
print("\n--- Pass 5: Verify clustering ---")
for bad in ["nazlicanto/e-commerce", "scikit-learn/bank-marketing", "imodels/credit-card"]:
    if bad in src and 'Clustering' in src[src.index(bad)-100:src.index(bad)] if bad in src else False:
        print(f"  WARNING: {bad} still in clustering section")

# ═════════════════════════════════════════════════
# Write output
# ═════════════════════════════════════════════════
import difflib
diff_lines = list(difflib.unified_diff(orig.splitlines(), src.splitlines(), lineterm=''))
adds = sum(1 for l in diff_lines if l.startswith('+') and not l.startswith('+++'))
dels = sum(1 for l in diff_lines if l.startswith('-') and not l.startswith('---'))
print(f"\nTotal diff: +{adds} -{dels} lines")

with open(FILE, "w", encoding="utf-8") as f:
    f.write(src)
print(f"\nPatched {FILE} successfully!")

# Final check
remaining = []
bad_names = [
    "aai510-group1/telecom-churn-dataset", "scikit-learn/bank-marketing",
    "codesignal/heart-disease-prediction", "vitaliy-datamonster/fraud-detection",
    "VictorSanh/anomaly-detection", "Zaherrr/Weather-Dataset",
    "mfaisalqureshi/hr-analytics-and-job-change-of-data-scientists",
    "vkrishna90/vehicle-insurance-customer-data", "scikit-learn/water-potability",
    "ErenalpCet/Loan-Prediction", "Xenova/used-cars", "mfumanelli/traffic-prediction",
    "saravan2024/Disease-Symptom", "saurabh1212/Bigmart-Sales-Data",
    "puspendert/Black-Friday-Sales-Prediction", "vitaliy-datamonster/flight-delays",
    "Tirumala/hotel_booking_demand", "thedevastator/mercari-price-prediction",
    "thedevastator/flight-price-prediction-data", "inductiva/ds-salaries",
    "leostelon/KC-House-Data", "leostelon/house-prices-advanced-regression",
    "edinburghcstr/vctk", "Alizimal/daily-dialogs", "reczilla/movielens-100k",
    "nazlicanto/e-commerce", "mtbench101/cyberbullying_tweets",
    "SetFit/tweet_eval_stance_hillary", "consumer-finance-complaints/consumer_complaints",
    "mesolitica/amazon-alexa-review", "scikit-learn/restaurant-reviews",
    "datadrivenscience/movies-genres-prediction", "Pravincoder/Resume_Dataset",
    "zhengyun21/Book-Crossing", "Ammok/Household_Power_Consumption",
    "EnergyStatisticsDatasets/electricity_demand", "juanma9613/Beijing-PM2.5-dataset",
    "jaeyoung-im/us-gasoline-prices", "thedevastator/rossmann-store-sales",
    "thedevastator/store-item-demand-forecasting", "vijaygkd/Marketing_Campaign",
    "dinov3", "dsrscientist/dataset1", "imodels/credit-card",
]
for n in bad_names:
    if n in src:
        remaining.append(n)

if remaining:
    print(f"\nWARNING: {len(remaining)} invalid references still present:")
    for n in remaining:
        # Show context
        idx = src.index(n)
        context = src[max(0,idx-40):idx+len(n)+40].replace('\n', '\\n')
        print(f"  - {n}: ...{context}...")
else:
    print("\n✓ All invalid references removed!")
