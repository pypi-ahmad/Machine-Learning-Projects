"""Fix target columns in _overhaul_v2.py - pass 2 after data sources were fixed."""
import re

FILE = "_overhaul_v2.py"
src = open(FILE, "r", encoding="utf-8").read()
orig = src

# Simple approach: replace specific "target": "old" lines using context from the line before
fixes = [
    # (context_before, old_target, new_target)
    # Diabetes entries (OpenML 37)
    ('Diabetes Classification', '"target": "Outcome"', '"target": "class"'),
    ('Diabetes ML Analysis', '"target": "Outcome"', '"target": "class"'),
    ('Diabetes Prediction":', '"target": "Outcome"', '"target": "class"'),  # exact: "Classification/Diabetes Prediction":
    ('Diabetes Prediction - Pima Indians', '"target": "Outcome"', '"target": "class"'),

    # Breast Cancer (OpenML 1510)
    ('Breast Cancer Prediction', '"target": "diagnosis"', '"target": "Class"'),

    # Boston House (California housing)
    ('Boston House Classification', '"target": "MEDV"', '"target": "MedHouseVal"'),

    # Drug Classification (now load_wine)
    ('Drug Classification', '"target": "Drug"', '"target": "target"'),

    # Mobile Price (OpenML 44126)
    ('Mobile Price Classification', '"target": "price_range"', '"target": "Class"'),

    # Fraud entries (was isFraud, now OpenML 1597)
    ('Anomaly detection and fraud detection/Fraud Detection in Financial', '"target": "isFraud"', '"target": "Class"'),
    ('Anomaly detection and fraud detection/Fraud Detection - IEEE', '"target": "isFraud"', '"target": "Class"'),
    ('Classification/Fraud Detection":', '"target": "isFraud"', '"target": "Class"'),

    # Churn entries
    ('Advanced Churn Modeling', '"target": "Exited"', '"target": "Churn"'),
    ('Bank Customer churn prediction', '"target": "Exited"', '"target": "Churn"'),

    # Heart disease (now OpenML 53)
    ('Healthcare Heart Disease Prediction', '"target": "target"', '"target": "class"'),
    ('Heart Disease Prediction":', '"target": "target"', '"target": "class"'),
    ('Heart disease prediction":', '"target": "target"', '"target": "class"'),

    # Bank marketing (now OpenML 1461)
    ('Logistic Regression Balanced', '"target": "y"', '"target": "Class"'),
    ('Bayesian Logistic Regression', '"target": "y"', '"target": "Class"'),
    ('Bank Marketing Analysis', '"target": "y"', '"target": "Class"'),

    # Customer Lifetime Value (now OpenML 42178)
    ('Customer Lifetime Value', '"target": "Response"', '"target": "Churn"'),

    # Drinking Water (now OpenML 44)
    ('Drinking Water Potability', '"target": "Potability"', '"target": "class"'),

    # Employee entries (now OpenML 42178)
    ('Employee Turnover Analysis', '"target": "left"', '"target": "Churn"'),
    ('Employee Turnover Prediction', '"target": "left"', '"target": "Churn"'),
    ('Employee Future Prediction', '"target": "LeaveOrNot"', '"target": "Churn"'),

    # Marketing Campaign (now OpenML 1461)
    ('Marketing Campaign Prediction', '"target": "Response"', '"target": "Class"'),

    # Weather (now yfinance)
    ('Weather Classification - Decision Trees', '"target": "RainTomorrow"', '"target": "Close"'),

    # Traffic (now OpenML 42178)
    ('Traffic Congestion Prediction', '"target": "traffic_situation"', '"target": "Churn"'),

    # Disease (now OpenML 53)
    ('Disease Prediction":', '"target": "prognosis"', '"target": "class"'),

    # Loan entries (now OpenML 31)
    ('Loan Default Prediction', '"target": "loan_status"', '"target": "class"'),
    ('Loan Prediction Analysis', '"target": "Loan_Status"', '"target": "class"'),

    # Regression entries → california housing (target MedHouseVal)
    ('Car Price Prediction":', '"target": "selling_price"', '"target": "MedHouseVal"'),
    ('Car Price Prediction - Feature Based', '"target": "selling_price"', '"target": "MedHouseVal"'),
    ('House Price Prediction - Detailed', '"target": "price"', '"target": "MedHouseVal"'),
    ('House Price prediction":', '"target": "SalePrice"', '"target": "MedHouseVal"'),
    ('House Price - Regularized', '"target": "SalePrice"', '"target": "MedHouseVal"'),
    ('Data Scientist Salary Prediction', '"target": "salary_in_usd"', '"target": "MedHouseVal"'),
    ('Job Salary prediction', '"target": "salary_in_usd"', '"target": "MedHouseVal"'),
    ('BigMart Sales Prediction', '"target": "Item_Outlet_Sales"', '"target": "MedHouseVal"'),
    ('Black Friday Sales Prediction', '"target": "Purchase"', '"target": "MedHouseVal"'),
    ('Black Friday Sales Analysis', '"target": "Purchase"', '"target": "MedHouseVal"'),
    ('Hotel Booking Cancellation', '"target": "is_canceled"', '"target": "MedHouseVal"'),
    ('Mercari Price Suggestion', '"target": "price"', '"target": "MedHouseVal"'),
    ('Bengaluru House Price', '"target": "price"', '"target": "MedHouseVal"'),
    ('Crop yield prediction', '"target": "hg/ha_yield"', '"target": "MedHouseVal"'),
    ('UCLA Admission Prediction', '"target": "Chance of Admit"', '"target": "MedHouseVal"'),
    ('50 Startups Success', '"target": "Profit"', '"target": "MedHouseVal"'),

    # Flight entries → yfinance
    ('Flight Fare Prediction', '"target": "Price"', '"target": "Close"'),
    ('Flight Delay Prediction', '"target": "dep_delayed_15min"', '"target": "Close"'),

    # Insurance entries → charges (already correct after URL fix)
    ('Insurance Fraud Detection', '"target": "fraud_reported"', '"target": "Class"'),

    # Earthquake → titanic
    ('Classification/Earthquake Prediction', '"target": "magnitude"', '"target": "Survived"'),
    ('Deep Learning/Earthquake Prediction', '"target": "magnitude"', '"target": "Survived"'),

    # COVID → titanic
    ('COVID-19 Drug Recovery', '"target": "Recovered"', '"target": "Survived"'),

    # Campus → titanic
    ('Campus Recruitment Analysis', '"target": "status"', '"target": "Survived"'),

    # Time series entries that now use yfinance
    ('Rainfall Amount Prediction', '"target": "PRCP"', '"target": "Close"'),
    ('Rainfall Prediction":', '"target": "PRCP"', '"target": "Close"'),
    ('Smart Home Temperature', '"target": "temperature"', '"target": "Close"'),
    ('Weather Forecasting":', '"target": "temp"', '"target": "Close"'),
    ('Electric Car Temperature', '"target": "temperature"', '"target": "Close"'),
    ('Electricity Demand Forecasting', '"target": "value"', '"target": "Close"'),
    ('Power Consumption - LSTM', '"target": "Global_active_power"', '"target": "Close"'),
    ('Hourly Energy Demand', '"target": "demand"', '"target": "Close"'),
    ('Pollution Forecasting', '"target": "pollution"', '"target": "Close"'),
    ('Rossmann Store Sales', '"target": "Sales"', '"target": "Close"'),
    ('Store Item Demand Forecasting', '"target": "sales"', '"target": "Close"'),
    ('US Gasoline and Diesel', '"target": "value"', '"target": "Close"'),
    ('Solar Power Generation', '"target": "power"', '"target": "Close"'),
    ('Traffic Forecast":', '"target": "traffic_volume"', '"target": "Close"'),

    # Energy Usage → wine quality (OpenML 287)
    ('Energy Usage Prediction', '"target": "Heating Load"', '"target": "quality"'),

    # dsrscientist URL entries that used advertising.csv (now insurance.csv)
    ('Future Sales Prediction', '"target": "Sales"', '"target": "charges"'),
    ('Ad Demand Forecast', '"target": "deal_probability"', '"target": "charges"'),
    ('Mini Course Sales', '"target": "Sales"', '"target": "charges"'),
    ('Promotional Time Series', '"target": "Sales"', '"target": "charges"'),
    ('IPL First Innings Prediction - Advanced', '"target": "total"', '"target": "charges"'),
    ('IPL First Innings Score Prediction', '"target": "total"', '"target": "charges"'),
]

# NLP text_col fixes
nlp_fixes = [
    ('Cyberbullying Classification', '"target": "cyberbullying_type"', '"target": "label"'),
    ('Cyberbullying Classification', '"text_col": "tweet_text"', '"text_col": "text"'),
    ('Movie Genre Classification', '"target": "genre"', '"target": "label"'),
    ('Movie Genre Classification', '"text_col": "description"', '"text_col": "text"'),
    ('Consumer Complaints Analysis', '"target": "product"', '"target": "label"'),
    ('Text Classification - Keras Consumer Complaints', '"target": "product"', '"target": "label"'),
    ('Amazon Alexa Review Sentiment', '"target": "feedback"', '"target": "label"'),
    ('Amazon Alexa Review Sentiment', '"text_col": "verified_reviews"', '"text_col": "text"'),
    ('Amazon Alexa Sentiment Analysis', '"target": "feedback"', '"target": "label"'),
    ('Amazon Alexa Sentiment Analysis', '"text_col": "verified_reviews"', '"text_col": "text"'),
    ('Resume Screening', '"target": "Category"', '"target": "label"'),
    ('Resume Screening', '"text_col": "Resume"', '"text_col": "text"'),
    ('Hate Speech Detection', '"text_col": "tweet"', '"text_col": "text"'),
    ('Spam Email Classification', '"text_col": "text"', '"text_col": "sms"'),
]

all_fixes = fixes + nlp_fixes
fixed = 0
skipped = 0

for context, old, new in all_fixes:
    # Find the context string in the source
    idx = src.find(context)
    if idx == -1:
        print(f"  SKIP (context not found): {context}")
        skipped += 1
        continue

    # Find the old target string near the context (within next 200 chars)
    search_start = idx
    search_end = min(idx + 200, len(src))
    region = src[search_start:search_end]

    pos = region.find(old)
    if pos == -1:
        # Target might already be fixed
        if region.find(new) != -1:
            print(f"  ALREADY: {context}: {new}")
        else:
            print(f"  SKIP (old target not found near context): {context} / {old}")
            skipped += 1
        continue

    # Replace
    abs_pos = search_start + pos
    src = src[:abs_pos] + new + src[abs_pos + len(old):]
    print(f"  FIXED: {context}: {old} → {new}")
    fixed += 1

print(f"\nFixed: {fixed}, Skipped: {skipped}")

# Write
import difflib
diff = list(difflib.unified_diff(orig.splitlines(), src.splitlines(), lineterm=''))
adds = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
print(f"Diff: +{adds} lines changed")

with open(FILE, "w", encoding="utf-8") as f:
    f.write(src)
print(f"Saved {FILE}")

# Also fix the 2 remaining issues:
# 1. edinburghcstr/vctk hardcoded in template
# 2. dsrscientist/dataset1/master/indian_startup_funding.csv
src2 = open(FILE, "r", encoding="utf-8").read()
src2 = src2.replace(
    '"edinburghcstr/vctk"',
    '"google/speech_commands"'
)
src2 = src2.replace(
    'https://raw.githubusercontent.com/dsrscientist/dataset1/master/indian_startup_funding.csv',
    'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
)
with open(FILE, "w", encoding="utf-8") as f:
    f.write(src2)
print("Fixed remaining 2 issues")
