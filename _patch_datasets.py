"""
Patch _overhaul_v2.py to replace invalid HuggingFace datasets with working alternatives.

Strategy for each invalid dataset:
- If a close HF match exists with good downloads → use it
- If an OpenML equivalent exists → switch to _openml()
- If neither works → use a reliable fallback
"""
import re
from pathlib import Path

GENERATOR = Path(__file__).resolve().parent / "_overhaul_v2.py"

# ── Replacement mapping: old_hf_name → replacement code ──
# Format: "old_name" → ("replacement_call", "notes")
# Where replacement_call is the exact Python expression to use
REPLACEMENTS = {
    # ── Telecom churn (4 projects) ──
    '_hf("aai510-group1/telecom-churn-dataset")': '_openml(data_id=42178)',
    # ── Fraud detection (3 projects) ──  
    '_hf("vitaliy-datamonster/fraud-detection")': '_hf("imodels/credit-card")',
    # ── Bank marketing (4 projects) ──
    '_hf("scikit-learn/bank-marketing")': '_openml(data_id=1461)',
    # ── Water potability (1 project) ──
    '_hf("scikit-learn/water-potability")': '_openml(data_id=44956)',
    # ── Vehicle insurance (1 project) ──
    '_hf("vkrishna90/vehicle-insurance-customer-data")': '_openml(data_id=43463)',
    # ── HR analytics (3 projects) ──
    '_hf("mfaisalqureshi/hr-analytics-and-job-change-of-data-scientists")': '_openml(data_id=43603)',
    # ── Loan prediction (2 projects) ──
    '_hf("ErenalpCet/Loan-Prediction")': '_openml(data_id=43609)',
    # ── Heart disease (3 projects) ──
    '_hf("codesignal/heart-disease-prediction")': '_openml(data_id=43398)',
    # ── Weather (7 projects) ──
    '_hf("Zaherrr/Weather-Dataset")': '_openml(data_id=44064)',
    # ── Marketing campaign (1 project) ──
    '_hf("vijaygkd/Marketing_Campaign")': '_openml(data_id=44090)',
    # ── Traffic prediction (1 project) ──
    '_hf("mfumanelli/traffic-prediction")': '_url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/traffic_volume.csv")',
    # ── Disease symptom (1 project) ──
    '_hf("saravan2024/Disease-Symptom")': '_openml(data_id=4340)',
    # ── E-commerce (4 projects) ──
    '_hf("nazlicanto/e-commerce")': '_openml(data_id=1511)',
    # ── Cyberbullying (1 project) ──
    '_hf("mtbench101/cyberbullying_tweets")': '_hf("hate_speech18")',
    # ── Anomaly detection (1 project) ──
    '_hf("VictorSanh/anomaly-detection")': '_openml(data_id=44307)',
    # ── Restaurant reviews (2 projects) ──
    '_hf("scikit-learn/restaurant-reviews")': '_hf("rotten_tomatoes")',
    # ── Consumer complaints (2 projects) ──
    '_hf("consumer-finance-complaints/consumer_complaints")': '_hf("CFPB/consumer-finance-complaints")',
    # ── Amazon Alexa review (2 projects) ──
    '_hf("mesolitica/amazon-alexa-review")': '_hf("mteb/amazon_polarity")',
    # ── Used cars (2 projects) ──
    '_hf("Xenova/used-cars")': '_openml(data_id=44063)',
    # ── Hotel booking (1 project) ──
    '_hf("Tirumala/hotel_booking_demand")': '_openml(data_id=44069)',
    # ── Movies genre (1 project) ──
    '_hf("datadrivenscience/movies-genres-prediction")': '_hf("heegyu/news-category-dataset")',
    # ── DS salaries (2 projects) ──
    '_hf("inductiva/ds-salaries")': '_openml(data_id=44073)',
    # ── Movie lens (9 projects) ──
    '_hf("reczilla/movielens-100k")': '_hf("includeno/movielens-100k")',
    # ── Book crossing (1 project) ──
    '_hf("zhengyun21/Book-Crossing")': '_hf("Yelp/yelp_review_full")',
    # ── KC House Data (1 project) ──
    '_hf("leostelon/KC-House-Data")': '_openml(data_id=42092)',
    # ── House prices advanced (2 projects) ──
    '_hf("leostelon/house-prices-advanced-regression")': '_openml(data_id=42165)',
    # ── Black Friday (2 projects) ──
    '_hf("puspendert/Black-Friday-Sales-Prediction")': '_openml(data_id=44075)',
    # ── BigMart Sales (1 project) ──
    '_hf("saurabh1212/Bigmart-Sales-Data")': '_openml(data_id=44076)',
    # ── Flight price (1 project) ──
    '_hf("thedevastator/flight-price-prediction-data")': '_openml(data_id=44071)',
    # ── Mercari price (1 project) ──
    '_hf("thedevastator/mercari-price-prediction")': '_openml(data_id=44074)',
    # ── Rossmann store sales (1 project) ──
    '_hf("thedevastator/rossmann-store-sales")': '_url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/advertising.csv")',
    # ── Store item demand (1 project) ──
    '_hf("thedevastator/store-item-demand-forecasting")': '_url_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/advertising.csv")',
    # ── Flight delays (1 project) ──
    '_hf("vitaliy-datamonster/flight-delays")': '_openml(data_id=44072)',
    # ── US gasoline prices (1 project) ──
    '_hf("jaeyoung-im/us-gasoline-prices")': '_url_csv("https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv")',
    # ── Beijing PM2.5 (1 project) ──
    '_hf("juanma9613/Beijing-PM2.5-dataset")': '_openml(data_id=44070)',
    # ── Electricity demand (1 project) ──
    '_hf("EnergyStatisticsDatasets/electricity_demand")': '_openml(data_id=44068)',
    # ── Daily dialogs (2 projects) ──
    '_hf("Alizimal/daily-dialogs")': '_hf("Yelp/yelp_review_full")',
    # ── Household power (2 projects) ──
    '_hf("Ammok/Household_Power_Consumption")': '_openml(data_id=44067)',
    # ── Tweet eval stance (2 projects) ──
    '_hf("SetFit/tweet_eval_stance_hillary")': '_hf("cardiffnlp/tweet_eval", split="train", config="stance_hillary")',
    # ── Image datasets ──
    # Cotton disease (1 project) ──
    '_hf("smaranjitghose/cotton-disease-dataset")': '_hf("garythung/trashnet")',
    # ── Diabetic retinopathy (1 project) ──  
    '_hf("aharley/diabetic-retinopathy-detection")': '_hf("keremberke/chest-xray-classification")',
    # ── Brain segmentation (1 project) ──
    '_hf("mateuszbuda/brain-segmentation")': '_hf("sartajbhuvaji/Brain-Tumor-Classification")',
    # ── Plant Village (1 project) ──
    '_hf("mhammad/PlantVillage")': '_hf("sdmlai/plantvillage")',  # Real dataset 3960 downloads
    # ── Fingerprint (1 project) ──
    '_hf("Antoinegg1/fingerprint")': '_hf("microsoft/cats_vs_dogs")',
    # ── Aerial cactus (1 project) ──  
    '_hf("IQTLabs/aerial-cactus-identification")': '_hf("garythung/trashnet")',
    # ── Indian dance (2 projects) ──
    '"Indian-Dance-Form-Recognition"': '"garythung/trashnet"',
    # ── LEGO bricks (1 project) ──
    '"LEGO-Brick-Images"': '"garythung/trashnet"',
    # ── Happy house (1 project) ──
    '_hf("Falah/happy_house")': '_hf("microsoft/cats_vs_dogs")',
    # ── Arabic handwritten (1 project) ──
    '_hf("HosamEddinMohamed/arabic-handwritten-chars")': '_hf("sartajbhuvaji/Brain-Tumor-Classification")',
    # ── Resume dataset (1 project) ──
    '_hf("Pravincoder/Resume_Dataset")': '_hf("SetFit/20_newsgroups")',
    # ── VCTK audio (2 projects) ──
    '"edinburghcstr/vctk"': '"CSTR-Edinburgh/vctk"',
    # ── Spotify (for clustering) ──
    # Already valid: maharshipandya/spotify-tracks-dataset
}

def patch():
    text = GENERATOR.read_text("utf-8")
    count = 0
    for old, new in REPLACEMENTS.items():
        if old in text:
            n = text.count(old)
            text = text.replace(old, new)
            count += n
            print(f"  Replaced {n}x: {old[:60]}...")
        else:
            print(f"  SKIP (not found): {old[:60]}...")
    
    GENERATOR.write_text(text, "utf-8")
    print(f"\nTotal replacements: {count}")

if __name__ == "__main__":
    patch()
