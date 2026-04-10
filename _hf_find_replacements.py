"""Find real HuggingFace datasets to replace hallucinated ones.

Strategy:
- For each invalid dataset name, search HF Hub using the keyword
- Pick the best match based on name similarity and downloads
- For unfindable ones, fall back to OpenML or sklearn
"""
from huggingface_hub import list_datasets
import sys

INVALID = [
    "Alizimal/daily-dialogs",
    "Ammok/Household_Power_Consumption",
    "Antoinegg1/fingerprint",
    "EnergyStatisticsDatasets/electricity_demand",
    "ErenalpCet/Loan-Prediction",
    "Falah/happy_house",
    "HosamEddinMohamed/arabic-handwritten-chars",
    "IQTLabs/aerial-cactus-identification",
    "Indian-Dance-Form-Recognition",
    "LEGO-Brick-Images",
    "Pravincoder/Resume_Dataset",
    "SetFit/tweet_eval_stance_hillary",
    "Tirumala/hotel_booking_demand",
    "VictorSanh/anomaly-detection",
    "Xenova/used-cars",
    "Zaherrr/Weather-Dataset",
    "aai510-group1/telecom-churn-dataset",
    "aharley/diabetic-retinopathy-detection",
    "codesignal/heart-disease-prediction",
    "consumer-finance-complaints/consumer_complaints",
    "datadrivenscience/movies-genres-prediction",
    "edinburghcstr/vctk",
    "inductiva/ds-salaries",
    "jaeyoung-im/us-gasoline-prices",
    "juanma9613/Beijing-PM2.5-dataset",
    "leostelon/KC-House-Data",
    "leostelon/house-prices-advanced-regression",
    "mateuszbuda/brain-segmentation",
    "mesolitica/amazon-alexa-review",
    "mfaisalqureshi/hr-analytics-and-job-change-of-data-scientists",
    "mfumanelli/traffic-prediction",
    "mhammad/PlantVillage",
    "mtbench101/cyberbullying_tweets",
    "nazlicanto/e-commerce",
    "puspendert/Black-Friday-Sales-Prediction",
    "reczilla/movielens-100k",
    "saravan2024/Disease-Symptom",
    "saurabh1212/Bigmart-Sales-Data",
    "scikit-learn/bank-marketing",
    "scikit-learn/restaurant-reviews",
    "scikit-learn/water-potability",
    "smaranjitghose/cotton-disease-dataset",
    "thedevastator/flight-price-prediction-data",
    "thedevastator/mercari-price-prediction",
    "thedevastator/rossmann-store-sales",
    "thedevastator/store-item-demand-forecasting",
    "vijaygkd/Marketing_Campaign",
    "vitaliy-datamonster/flight-delays",
    "vitaliy-datamonster/fraud-detection",
    "vkrishna90/vehicle-insurance-customer-data",
    "zhengyun21/Book-Crossing",
]

for name in INVALID:
    # Extract search keywords from dataset name
    parts = name.split("/")
    search_term = parts[-1].replace("-", " ").replace("_", " ")
    
    results = list(list_datasets(search=search_term, sort="downloads", direction=-1, limit=3))
    
    if results:
        best = results[0]
        print(f'"{name}" -> "{best.id}" (downloads: {best.downloads})')
    else:
        print(f'"{name}" -> NO MATCH')
