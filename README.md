<div align="center">

# Machine Learning Projects

**315 production-grade ML pipelines across 23 problem families, built on April 2026 foundation models.**

Every pipeline is generated from a single orchestrator, runs GPU-first, auto-downloads its data, and produces structured JSON metrics — ready for experimentation or deployment.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Table of Contents

- [Highlights](#highlights)
- [Getting Started](#getting-started)
- [Model Stack](#model-stack)
- [Projects](#projects)
  - [Classification](#classification)
  - [Regression](#regression)
  - [NLP](#nlp)
  - [Computer Vision](#computer-vision)
  - [Deep Learning](#deep-learning)
  - [Time Series Analysis](#time-series-analysis)
  - [Clustering](#clustering)
  - [Recommendation Systems](#recommendation-systems)
  - [Anomaly Detection and Fraud Detection](#anomaly-detection-and-fraud-detection)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Speech and Audio Processing](#speech-and-audio-processing)
  - [Data Analysis](#data-analysis)
  - [Associate Rule Learning](#associate-rule-learning)
  - [Conceptual](#conceptual)
  - [Python Scripts](#python-scripts)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [License](#license)

---

## Highlights

| Metric | Value |
|--------|-------|
| **Total projects** | 315 |
| **Problem families** | 23 |
| **Generator functions** | 19 |
| **Generated pipeline code** | 68,600+ lines |
| **Generator source** | ~4,200 lines |
| **Code amplification** | 16× |
| **Stack vintage** | April 2026 |
| **GPU acceleration** | CUDA-first defaults across all families |

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- NVIDIA GPU with CUDA drivers (recommended; CPU fallback available)
- 8 GB+ RAM (16 GB recommended for transformer models)

### Installation

```bash
git clone https://github.com/pypi-ahmad/Machine-Learning-Projects.git
cd Machine-Learning-Projects

python -m venv .venv
```

**Activate the virtual environment:**

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

**Install PyTorch first** (match your CUDA version):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**Then install all other dependencies:**

```bash
pip install -r requirements.txt
```

### Run Any Pipeline

```bash
# Tabular classification
python "Classification/Adult Salary Prediction/pipeline.py"

# Time series forecasting
python "Time Series Analysis/Stock Price Forecasting/pipeline.py"

# Reinforcement learning
python "Reinforcement Learning/Lunar Landing/pipeline.py"

# NLP classification
python "NLP/Sentiment Analysis IMDB/pipeline.py"

# Computer vision
python "Computer Vision/Face Recognition Door Lock - AWS Rekognition/pipeline.py"
```

Each pipeline auto-downloads its dataset, trains models, evaluates performance, and writes results to `metrics.json` or `results.json` in the project directory.

---

## Model Stack

All models target **April 2026** best-in-class architectures. Every family pairs primary foundation models with lightweight baselines for comparison.

### Tabular ML

| Family | Count | Primary Models | Baselines |
|--------|-------|----------------|-----------|
| **Classification** | 46 | CatBoost (GPU), LightGBM (GPU), XGBoost (CUDA), AutoGluon, TabPFN-v2, TabM | FLAML AutoML, LazyPredict |
| **Regression** | 44 | CatBoost (GPU), LightGBM (GPU), XGBoost (CUDA), AutoGluon, TabPFN-v2, TabM | FLAML AutoML, LazyPredict |
| **Fraud Detection** | 7 | Calibrated GBDT trio (isotonic), PyOD ECOD / COPOD / IForest | — |
| **Anomaly Detection** | 7 | PyOD ensemble, anomalib PatchCore (`wide_resnet50_2`) | — |

### Natural Language Processing

| Family | Count | Primary Models | Baselines |
|--------|-------|----------------|-----------|
| **Text Classification** | 44 | ModernBERT (`answerdotai/ModernBERT-base`), XLM-RoBERTa, GLiNER, BGE-M3 | TF-IDF + MultinomialNB |
| **Generation & Translation** | 16 | Qwen3-Instruct (8B, Ollama), BART (`bart-large-cnn`), NLLB-200 | — |
| **Semantic Similarity** | 4 | BGE-M3 (`BAAI/bge-m3`), Qwen3-Embedding (0.6B) | TF-IDF cosine |
| **NLP Misc** | 6 | Qwen3-Instruct, BART, NLLB-200 | — |

### Computer Vision

| Family | Count | Primary Models | Baselines |
|--------|-------|----------------|-----------|
| **Image Classification** | 31 | DINOv3 (ViT-S/14), ConvNeXt V2 (`convnextv2_tiny`) | — |
| **Detection & Processing** | 11 | YOLO26m (Ultralytics) | — |
| **Face & Gesture** | 12 | YOLO26m, MediaPipe Face/Hand/Pose Landmarkers, InsightFace | — |
| **Captioning / VLM** | 3 | Qwen3-VL (`Qwen3-VL-2B-Instruct`), Molmo 2 (7B) | — |
| **OCR** | 3 | PaddleOCR (GPU), PaddleOCR-VL-1.5 | — |
| **Medical Segmentation** | 2 | nnU-Net (SimpleUNet), MedSAM (`medsam-vit-base`) | — |

### Time Series & Forecasting

| Family | Count | Primary Models | Baselines |
|--------|-------|----------------|-----------|
| **Forecasting** | 25 | AutoGluon-TS, Chronos-Bolt, Chronos-2, TimesFM 2.0 | ARIMA(5,1,0), Prophet, GBDT lag-features, FLAML |

### Other Domains

| Family | Count | Primary Models | Baselines |
|--------|-------|----------------|-----------|
| **Clustering** | 22 | UMAP + HDBSCAN, Gaussian Mixture | K-Means |
| **Recommendation** | 19 | implicit ALS / BPR, LightFM (WARP), BGE-M3, Qwen3-Embedding | Surprise SVD / KNN |
| **Reinforcement Learning** | 5 | PPO, SAC (Stable-Baselines3) | DQN, tabular Q-learning |
| **Audio & Speech** | 4 | Whisper large-v3-turbo, Wav2Vec2, HuBERT, SepFormer, XTTS-v2 | — |

---

## Projects

### Classification

<details>
<summary><strong>57 projects — CatBoost · LightGBM · XGBoost · AutoGluon · TabPFN-v2 · TabM</strong></summary>

- [Adult Salary Prediction](Classification/Adult%20Salary%20Prediction)
- [Advanced Credit Card Fraud Detection](Classification/Advanced%20Credit%20Card%20Fraud%20Detection)
- [Autoencoder Fashion MNIST](Classification/Autoencoder%20Fashion%20MNIST)
- [Autoencoder for Customer Churn](Classification/Autoencoder%20for%20Customer%20Churn)
- [Bayesian Logistic Regression - Bank Marketing](Classification/Bayesian%20Logistic%20Regression%20-%20Bank%20Marketing)
- [Boston House Classification](Classification/Boston%20House%20Classification)
- [Breast Cancer Detection](Classification/Breast%20Cancer%20Detection)
- [Breast Cancer Prediction](Classification/Breast%20Cancer%20Prediction)
- [CIFAR-10 Classification](Classification/CIFAR-10%20Classification)
- [Cotton Disease Prediction](Classification/Cotton%20Disease%20Prediction)
- [Credit Card Fraud - Imbalanced Dataset](Classification/Credit%20Card%20Fraud%20-%20Imbalanced%20Dataset)
- [Credit Risk Modeling - German Credit](Classification/Credit%20Risk%20Modeling%20-%20German%20Credit)
- [Customer Churn Prediction - Telecom](Classification/Customer%20Churn%20Prediction%20-%20Telecom)
- [Customer Lifetime Value Prediction](Classification/Customer%20Lifetime%20Value%20Prediction)
- [Customer Segmentation - E-Commerce](Classification/Customer%20Segmentation%20-%20E-Commerce)
- [Cyberbullying Classification](Classification/Cyberbullying%20Classification)
- [Diabetes Classification](Classification/Diabetes%20Classification)
- [Diabetes ML Analysis](Classification/Diabetes%20ML%20Analysis)
- [Diabetes Prediction](Classification/Diabetes%20Prediction)
- [Digit Recognition - MNIST Sequence](Classification/Digit%20Recognition%20-%20MNIST%20Sequence)
- [Dog vs Cat Classification](Classification/Dog%20vs%20Cat%20Classification)
- [Drinking Water Potability](Classification/Drinking%20Water%20Potability)
- [Drug Classification](Classification/Drug%20Classification)
- [Earthquake Prediction](Classification/Earthquake%20Prediction)
- [Employee Turnover Analysis](Classification/Employee%20Turnover%20Analysis)
- [Employee Turnover Prediction](Classification/Employee%20Turnover%20Prediction)
- [Fashion MNIST Analysis](Classification/Fashion%20MNIST%20Analysis)
- [Flower Species Classification](Classification/Flower%20Species%20Classification)
- [Fraud Detection](Classification/Fraud%20Detection)
- [Garbage Classification](Classification/Garbage%20Classification)
- [Glass Classification](Classification/Glass%20Classification)
- [Groundhog Day Predictions](Classification/Groundhog%20Day%20Predictions)
- [H2O Higgs Boson](Classification/H2O%20Higgs%20Boson)
- [Hand Digit Recognition](Classification/Hand%20Digit%20Recognition)
- [Healthcare Heart Disease Prediction](Classification/Healthcare%20Heart%20Disease%20Prediction)
- [Heart Disease Prediction](Classification/Heart%20Disease%20Prediction)
- [Income Classification](Classification/Income%20Classification)
- [Iris Dataset Analysis](Classification/Iris%20Dataset%20Analysis)
- [Loan Default Prediction](Classification/Loan%20Default%20Prediction)
- [Loan Prediction Analysis](Classification/Loan%20Prediction%20Analysis)
- [Logistic Regression Balanced](Classification/Logistic%20Regression%20Balanced)
- [Marketing Campaign Prediction](Classification/Marketing%20Campaign%20Prediction)
- [Mobile Price Classification](Classification/Mobile%20Price%20Classification)
- [Movie Genre Classification](Classification/Movie%20Genre%20Classification)
- [Plant Disease Recognition](Classification/Plant%20Disease%20Recognition)
- [Pneumonia Classification](Classification/Pneumonia%20Classification)
- [Simple Classification Problem](Classification/Simple%20Classification%20Problem)
- [Social Network Ads Analysis](Classification/Social%20Network%20Ads%20Analysis)
- [SONAR Rock vs Mine Prediction](Classification/SONAR%20Rock%20vs%20Mine%20Prediction)
- [Spam Email Classification](Classification/Spam%20Email%20Classification)
- [Student Performance Prediction](Classification/Student%20Performance%20Prediction)
- [Titanic - Handling Missing Values](Classification/Titanic%20-%20Handling%20Missing%20Values)
- [Titanic Survival Prediction](Classification/Titanic%20Survival%20Prediction)
- [Traffic Congestion Prediction](Classification/Traffic%20Congestion%20Prediction)
- [Weather Classification - Decision Trees](Classification/Weather%20Classification%20-%20Decision%20Trees)
- [Wine Quality Analysis](Classification/Wine%20Quality%20Analysis)
- [Wine Quality Prediction](Classification/Wine%20Quality%20Prediction)

</details>

---

### Regression

<details>
<summary><strong>43 projects — CatBoost · LightGBM · XGBoost · AutoGluon · TabPFN-v2 · TabM</strong></summary>

- [50 Startups Success Prediction](Regression/50%20Startups%20Success%20Prediction)
- [Ad Demand Forecast - Avito](Regression/Ad%20Demand%20Forecast%20-%20Avito)
- [Bank Customer churn prediction](Regression/Bank%20Customer%20churn%20prediction)
- [Bengaluru House Price Prediction](Regression/Bengaluru%20House%20Price%20Prediction)
- [BigMart Sales Prediction](Regression/BigMart%20Sales%20Prediction)
- [Bike Sharing Demand Analysis](Regression/Bike%20Sharing%20Demand%20Analysis)
- [Bitcoin Price Prediction](Regression/Bitcoin%20Price%20Prediction)
- [Bitcoin Price Prediction - Advanced](Regression/Bitcoin%20Price%20Prediction%20-%20Advanced)
- [Black Friday Sales Analysis](Regression/Black%20Friday%20Sales%20Analysis)
- [Black Friday Sales Prediction](Regression/Black%20Friday%20Sales%20Prediction)
- [Boston House Classification](Regression/Boston%20House%20Classification)
- [Boston Housing Analysis](Regression/Boston%20Housing%20Analysis)
- [Boston Housing Prediction Analysis](Regression/Boston%20Housing%20Prediction%20Analysis)
- [California Housing Prediction](Regression/California%20Housing%20Prediction)
- [Car Price Prediction](Regression/Car%20Price%20Prediction)
- [Car Price Prediction - Feature Based](Regression/Car%20Price%20Prediction%20-%20Feature%20Based)
- [China GDP Estimation](Regression/China%20GDP%20Estimation)
- [Crop yield prediction](Regression/Crop%20yield%20prediction)
- [Data Scientist Salary Prediction](Regression/Data%20Scientist%20Salary%20Prediction)
- [Diabetes Prediction - Pima Indians](Regression/Diabetes%20Prediction%20-%20Pima%20Indians)
- [Employee Future Prediction](Regression/Employee%20Future%20Prediction)
- [Energy Usage Prediction - Buildings](Regression/Energy%20Usage%20Prediction%20-%20Buildings)
- [Flight Delay Prediction](Regression/Flight%20Delay%20Prediction)
- [Flight Fare Prediction](Regression/Flight%20Fare%20Prediction)
- [Future Sales Prediction](Regression/Future%20Sales%20Prediction)
- [Gold Price Prediction](Regression/Gold%20Price%20Prediction)
- [Heart disease prediction](Regression/Heart%20disease%20prediction)
- [Hotel Booking Cancellation Prediction](Regression/Hotel%20Booking%20Cancellation%20Prediction)
- [House Price - Regularized Linear and XGBoost](Regression/House%20Price%20-%20Regularized%20Linear%20and%20XGBoost)
- [House Price prediction](Regression/House%20Price%20prediction)
- [House Price Prediction - Detailed](Regression/House%20Price%20Prediction%20-%20Detailed)
- [Insurance premium prediction](Regression/Insurance%20premium%20prediction)
- [IPL First Innings Prediction - Advanced](Regression/IPL%20First%20Innings%20Prediction%20-%20Advanced)
- [IPL First Innings Score Prediction](Regression/IPL%20First%20Innings%20Score%20Prediction)
- [Job Salary prediction](Regression/Job%20Salary%20prediction)
- [Medical Cost Personal](Regression/Medical%20Cost%20Personal)
- [Mercari Price Suggestion - LightGBM](Regression/Mercari%20Price%20Suggestion%20-%20LightGBM)
- [Rainfall Amount Prediction](Regression/Rainfall%20Amount%20Prediction)
- [Rainfall Prediction](Regression/Rainfall%20Prediction)
- [Stock price prediction](Regression/Stock%20price%20prediction)
- [Tesla Car Price Prediction](Regression/Tesla%20Car%20Price%20Prediction)
- [TPOT Mercedes Prediction](Regression/TPOT%20Mercedes%20Prediction)
- [UCLA Admission Prediction](Regression/UCLA%20Admission%20Prediction)

</details>

---

### NLP

<details>
<summary><strong>60 projects — ModernBERT · XLM-RoBERTa · Qwen3-Instruct · BART · NLLB-200 · BGE-M3</strong></summary>

- [Amazon Alexa Review Sentiment](NLP/Amazon%20Alexa%20Review%20Sentiment)
- [Amazon Sentiment Analysis](NLP/Amazon%20Sentiment%20Analysis)
- [Autocorrect](NLP/Autocorrect)
- [BOW and TF-IDF with XGBoost](NLP/BOW%20and%20TF-IDF%20with%20XGBoost)
- [Clinton vs Trump Tweets Analysis](NLP/Clinton%20vs%20Trump%20Tweets%20Analysis)
- [Consumer Complaints Analysis](NLP/Consumer%20Complaints%20Analysis)
- [Cross Language Information Retrieval](NLP/Cross%20Language%20Information%20Retrieval)
- [Disaster or Not Disaster](NLP/Disaster%20or%20Not%20Disaster)
- [DJIA Sentiment Analysis - News Headlines](NLP/DJIA%20Sentiment%20Analysis%20-%20News%20Headlines)
- [DJIA Sentiment Analysis - Stock Prediction](NLP/DJIA%20Sentiment%20Analysis%20-%20Stock%20Prediction)
- [Document Summary Creator](NLP/Document%20Summary%20Creator)
- [Fake News Detection](NLP/Fake%20News%20Detection)
- [GitHub Bugs Prediction](NLP/GitHub%20Bugs%20Prediction)
- [Hate Speech Detection](NLP/Hate%20Speech%20Detection)
- [IMDB Sentiment Analysis - Deep Learning](NLP/IMDB%20Sentiment%20Analysis%20-%20Deep%20Learning)
- [IMDB Sentiment Review Analysis](NLP/IMDB%20Sentiment%20Review%20Analysis)
- [Keyword Extraction](NLP/Keyword%20Extraction)
- [Keyword Research](NLP/Keyword%20Research)
- [Language Translation Model](NLP/Language%20Translation%20Model)
- [Language Translator](NLP/Language%20Translator)
- [Message Spam Detection](NLP/Message%20Spam%20Detection)
- [Movie Review Sentiments](NLP/Movie%20Review%20Sentiments)
- [Named Entity Recognition](NLP/Named%20Entity%20Recognition)
- [Next Word Prediction](NLP/Next%20Word%20Prediction)
- [NLP for Other Languages](NLP/NLP%20for%20Other%20Languages)
- [Plagiarism Checker](NLP/Plagiarism%20Checker)
- [Profanity Checker](NLP/Profanity%20Checker)
- [Restaurant Review Sentiment Analysis](NLP/Restaurant%20Review%20Sentiment%20Analysis)
- [Resume Screening](NLP/Resume%20Screening)
- [Sentiment Analysis](NLP/Sentiment%20Analysis)
- [Sentiment Analysis - Flask Web App](NLP/Sentiment%20Analysis%20-%20Flask%20Web%20App)
- [Sentiment Analysis - Restaurant Reviews](NLP/Sentiment%20Analysis%20-%20Restaurant%20Reviews)
- [SMS Spam Detection](NLP/SMS%20Spam%20Detection)
- [SMS Spam Detection - Detailed](NLP/SMS%20Spam%20Detection%20-%20Detailed)
- [SMS Spam Detection Analysis](NLP/SMS%20Spam%20Detection%20Analysis)
- [Spam Classifier](NLP/Spam%20Classifier)
- [Spam SMS Classification](NLP/Spam%20SMS%20Classification)
- [Spell Checker](NLP/Spell%20Checker)
- [Spelling Correction](NLP/Spelling%20Correction)
- [Stop Words in 28 Languages](NLP/Stop%20Words%20in%2028%20Languages)
- [Text Classification](NLP/Text%20Classification)
- [Text Classification - Keras Consumer Complaints](NLP/Text%20Classification%20-%20Keras%20Consumer%20Complaints)
- [Text Classification with NLP](NLP/Text%20Classification%20with%20NLP)
- [Text Clustering and Topic Modelling](NLP/Text%20Clustering%20and%20Topic%20Modelling)
- [Text File Analysis](NLP/Text%20File%20Analysis)
- [Text Generation](NLP/Text%20Generation)
- [Text Processing and Analysis](NLP/Text%20Processing%20and%20Analysis)
- [Text Similarity](NLP/Text%20Similarity)
- [Text Summarization](NLP/Text%20Summarization)
- [Text Summarization - Medium](NLP/Text%20Summarization%20-%20Medium)
- [Text Summarization - Word Frequency](NLP/Text%20Summarization%20-%20Word%20Frequency)
- [Text Summarization - Word Frequency Method](NLP/Text%20Summarization%20-%20Word%20Frequency%20Method)
- [Three-Way Sentiment Analysis - Tweets](NLP/Three-Way%20Sentiment%20Analysis%20-%20Tweets)
- [Twitter Sentiment Analysis](NLP/Twitter%20Sentiment%20Analysis)
- [Twitter Sentiment Analysis - ML](NLP/Twitter%20Sentiment%20Analysis%20-%20ML)
- [Twitter US Airline Sentiment](NLP/Twitter%20US%20Airline%20Sentiment)
- [US Election Prediction](NLP/US%20Election%20Prediction)
- [WhatsApp Chat Analysis](NLP/WhatsApp%20Chat%20Analysis)
- [WhatsApp Group Chat Analysis](NLP/WhatsApp%20Group%20Chat%20Analysis)
- [Wikipedia Search Word Cloud](NLP/Wikipedia%20Search%20Word%20Cloud)

</details>

---

### Computer Vision

<details>
<summary><strong>Pipeline Projects — YOLO · DINOv2 · MediaPipe · PaddleOCR · InsightFace · Qwen3-VL</strong></summary>

- [Aerial Cactus Identification](Computer%20Vision/Aerial%20Cactus%20Identification)
- [Aerial Imagery Segmentation](Computer%20Vision/Aerial%20Imagery%20Segmentation)
- [Age Gender Recognition](Computer%20Vision/Age%20Gender%20Recognition)
- [Blink Headpose Analyzer](Computer%20Vision/Blink%20Headpose%20Analyzer)
- [Brain Tumour Detection](Computer%20Vision/Brain%20Tumour%20Detection)
- [Building Footprint Change Detector](Computer%20Vision/Building%20Footprint%20Change%20Detector)
- [Building Footprint Segmentation](Computer%20Vision/Building%20Footprint%20Segmentation)
- [Business Card Reader](Computer%20Vision/Business%20Card%20Reader)
- [Captcha Recognition](Computer%20Vision/Captcha%20Recognition)
- [Car and Pedestrian Tracker](Computer%20Vision/Car%20and%20Pedestrian%20Tracker)
- [Cartoonize The Image](Computer%20Vision/Cartoonize%20The%20Image)
- [Celebrity Face Recognition](Computer%20Vision/Celebrity%20Face%20Recognition)
- [Cell Counting Instance Segmentation](Computer%20Vision/Cell%20Counting%20Instance%20Segmentation)
- [Cell Nuclei Segmentation](Computer%20Vision/Cell%20Nuclei%20Segmentation)
- [Conveyor Part Defect Detector](Computer%20Vision/Conveyor%20Part%20Defect%20Detector)
- [Crop Row Weed Segmentation](Computer%20Vision/Crop%20Row%20Weed%20Segmentation)
- [Crowd Zone Counter](Computer%20Vision/Crowd%20Zone%20Counter)
- [Document Layout Block Detector](Computer%20Vision/Document%20Layout%20Block%20Detector)
- [Document Type Classifier Router](Computer%20Vision/Document%20Type%20Classifier%20Router)
- [Document Word Detection](Computer%20Vision/Document%20Word%20Detection)
- [Dominant Color Analysis](Computer%20Vision/Dominant%20Color%20Analysis)
- [Dominant Color Extraction](Computer%20Vision/Dominant%20Color%20Extraction)
- [Driver Drowsiness Monitor](Computer%20Vision/Driver%20Drowsiness%20Monitor)
- [Drone Ship OBB Detector](Computer%20Vision/Drone%20Ship%20OBB%20Detector)
- [Ecommerce Item Attribute Tagger](Computer%20Vision/Ecommerce%20Item%20Attribute%20Tagger)
- [Emotion Recognition from Facial Expression](Computer%20Vision/Emotion%20Recognition%20from%20Facial%20Expression)
- [Exam Sheet Parser](Computer%20Vision/Exam%20Sheet%20Parser)
- [Exercise Rep Counter](Computer%20Vision/Exercise%20Rep%20Counter)
- [Face Anti Spoofing Detection](Computer%20Vision/Face%20Anti%20Spoofing%20Detection)
- [Face Clustering Photo Organizer](Computer%20Vision/Face%20Clustering%20Photo%20Organizer)
- [Face Detection - OpenCV](Computer%20Vision/Face%20Detection%20-%20OpenCV)
- [Face Expression Identifier](Computer%20Vision/Face%20Expression%20Identifier)
- [Face Landmark Detection](Computer%20Vision/Face%20Landmark%20Detection)
- [Face Mask Detection](Computer%20Vision/Face%20Mask%20Detection)
- [Face Recognition Door Lock - AWS Rekognition](Computer%20Vision/Face%20Recognition%20Door%20Lock%20-%20AWS%20Rekognition)
- [Face Verification Attendance System](Computer%20Vision/Face%20Verification%20Attendance%20System)
- [Finger Counter Pro](Computer%20Vision/Finger%20Counter%20Pro)
- [Fire and Smoke Detection](Computer%20Vision/Fire%20and%20Smoke%20Detection)
- [Fire Area Segmentation](Computer%20Vision/Fire%20Area%20Segmentation)
- [Food Freshness Grader](Computer%20Vision/Food%20Freshness%20Grader)
- [Food Image Recognition & Calories Estimation](Computer%20Vision/Food%20Image%20Recognition%20%26%20Calories%20Estimation)
- [Food Object Detection](Computer%20Vision/Food%20Object%20Detection)
- [Form OCR Checkbox Extractor](Computer%20Vision/Form%20OCR%20Checkbox%20Extractor)
- [Gaze Direction Estimator](Computer%20Vision/Gaze%20Direction%20Estimator)
- [Gesture Control Media Player](Computer%20Vision/Gesture%20Control%20Media%20Player)
- [Gesture Controlled Slideshow](Computer%20Vision/Gesture%20Controlled%20Slideshow)
- [Handwriting Recognition](Computer%20Vision/Handwriting%20Recognition)
- [Handwritten Note to Markdown](Computer%20Vision/Handwritten%20Note%20to%20Markdown)
- [Home Security](Computer%20Vision/Home%20Security)
- [ID Card KYC Parser](Computer%20Vision/ID%20Card%20KYC%20Parser)
- [Image Captioning](Computer%20Vision/Image%20Captioning)
- [Image Cartoonify](Computer%20Vision/Image%20Cartoonify)
- [Image Text Extraction - OCR](Computer%20Vision/Image%20Text%20Extraction%20-%20OCR)
- [Image to Sketch](Computer%20Vision/Image%20to%20Sketch)
- [Image to Text Conversion - OCR](Computer%20Vision/Image%20to%20Text%20Conversion%20-%20OCR)
- [Image Watermark](Computer%20Vision/Image%20Watermark)
- [Indian Classical Dance Classification](Computer%20Vision/Indian%20Classical%20Dance%20Classification)
- [Industrial Scratch Crack Segmentation](Computer%20Vision/Industrial%20Scratch%20Crack%20Segmentation)
- [Interactive Video Object Cutout Studio](Computer%20Vision/Interactive%20Video%20Object%20Cutout%20Studio)
- [Invoice Field Extractor](Computer%20Vision/Invoice%20Field%20Extractor)
- [Lane Finder](Computer%20Vision/Lane%20Finder)
- [Licence Plate Detector](Computer%20Vision/Licence%20Plate%20Detector)
- [Live Smile Detector](Computer%20Vision/Live%20Smile%20Detector)
- [Logo Detection and Brand Recognition](Computer%20Vision/Logo%20Detection%20and%20Brand%20Recognition)
- [Logo Retrieval Brand Match](Computer%20Vision/Logo%20Retrieval%20Brand%20Match)
- [Lung Segmentation from Chest X-Ray](Computer%20Vision/Lung%20Segmentation%20from%20Chest%20X-Ray)
- [Medical Image Segmentation for Tumour Detection](Computer%20Vision/Medical%20Image%20Segmentation%20for%20Tumour%20Detection)
- [Noise Reduction](Computer%20Vision/Noise%20Reduction)
- [Number Plate Reader Pro](Computer%20Vision/Number%20Plate%20Reader%20Pro)
- [Parking Occupancy Monitor](Computer%20Vision/Parking%20Occupancy%20Monitor)
- [Pedestrian Detection](Computer%20Vision/Pedestrian%20Detection)
- [Plant Disease Prediction](Computer%20Vision/Plant%20Disease%20Prediction)
- [Plant Disease Severity Estimator](Computer%20Vision/Plant%20Disease%20Severity%20Estimator)
- [Polyp Lesion Segmentation](Computer%20Vision/Polyp%20Lesion%20Segmentation)
- [PPE Compliance Monitor](Computer%20Vision/PPE%20Compliance%20Monitor)
- [Prescription OCR Parser](Computer%20Vision/Prescription%20OCR%20Parser)
- [Product Counterfeit Visual Checker](Computer%20Vision/Product%20Counterfeit%20Visual%20Checker)
- [QR Code Readability](Computer%20Vision/QR%20Code%20Readability)
- [Real-time Object Tracking](Computer%20Vision/Real-time%20Object%20Tracking)
- [Receipt Digitizer](Computer%20Vision/Receipt%20Digitizer)
- [Retail Shelf Stockout Detector](Computer%20Vision/Retail%20Shelf%20Stockout%20Detector)
- [Road Lane Detection](Computer%20Vision/Road%20Lane%20Detection)
- [Road Pothole Segmentation](Computer%20Vision/Road%20Pothole%20Segmentation)
- [Road Segmentation for Autonomous Vehicles](Computer%20Vision/Road%20Segmentation%20for%20Autonomous%20Vehicles)
- [Room Security - Webcam](Computer%20Vision/Room%20Security%20-%20Webcam)
- [Scene Text Reader Translator](Computer%20Vision/Scene%20Text%20Reader%20Translator)
- [Sign Language Alphabet Recognizer](Computer%20Vision/Sign%20Language%20Alphabet%20Recognizer)
- [Sign Language Recognition](Computer%20Vision/Sign%20Language%20Recognition)
- [Similar Image Finder](Computer%20Vision/Similar%20Image%20Finder)
- [Skin Cancer Detection](Computer%20Vision/Skin%20Cancer%20Detection)
- [Sports Ball Possession Tracker](Computer%20Vision/Sports%20Ball%20Possession%20Tracker)
- [Traffic Sign Recognition](Computer%20Vision/Traffic%20Sign%20Recognition)
- [Traffic Sign Recognizer](Computer%20Vision/Traffic%20Sign%20Recognizer)
- [Traffic Violation Analyzer](Computer%20Vision/Traffic%20Violation%20Analyzer)
- [Video Event Search](Computer%20Vision/Video%20Event%20Search)
- [Visual Anomaly Detector](Computer%20Vision/Visual%20Anomaly%20Detector)
- [Waste Sorting Detector](Computer%20Vision/Waste%20Sorting%20Detector)
- [Waterbody Flood Extent Segmentation](Computer%20Vision/Waterbody%20Flood%20Extent%20Segmentation)
- [Wildlife Image Classification](Computer%20Vision/Wildlife%20Image%20Classification)
- [Wildlife Species Retrieval](Computer%20Vision/Wildlife%20Species%20Retrieval)
- [Wound Area Measurement](Computer%20Vision/Wound%20Area%20Measurement)
- [Yoga Pose Correction Coach](Computer%20Vision/Yoga%20Pose%20Correction%20Coach)

</details>

<details>
<summary><strong>Mini Projects (Project 1–50 series)</strong></summary>

- [Project 1 - Real Time Angle Detector](Computer%20Vision/Project%201%20-%20Real%20Time%20Angle%20Detector)
- [Project 2 - Real Time Document Scanner](Computer%20Vision/Project%202%20-%20Real%20Time%20Document%20Scanner)
- [Project 3 - Real Time Face Detector](Computer%20Vision/Project%203%20-%20Real%20Time%20Face%20Detector)
- [Project 4 - Facial Landmarking](Computer%20Vision/Project%204%20-%20Facial%20Landmarking)
- [Project 5 - Finger Counter](Computer%20Vision/Project%205%20-%20Finger%20Counter)
- [Project 6 - Live Hand Tracking Module](Computer%20Vision/Project%206%20-%20Live%20Hand%20Tracking%20Module)
- [Project 7 - Real Time Object Size Detector](Computer%20Vision/Project%207%20-%20Real%20Time%20Object%20Size%20Detector)
- [Project 8 - OMR Evaluator](Computer%20Vision/Project%208%20-%20OMR%20Evaluator)
- [Project 9 - Real Time Camera Painter](Computer%20Vision/Project%209%20-%20Real%20Time%20Camera%20Painter)
- [Project 10 - Live Pose Detector](Computer%20Vision/Project%2010%20-%20Live%20Pose%20Detector)
- [Project 11 - Live QR Reader](Computer%20Vision/Project%2011%20-%20Live%20QR%20Reader)
- [Project 12 - Real Time Object Detection](Computer%20Vision/Project%2012%20-%20Real%20Time%20Object%20Detection)
- [Project 13 - Real Time Sudoku Solver](Computer%20Vision/Project%2013%20-%20Real%20Time%20Sudoku%20Solver)
- [Project 14 - Click Detect on Image](Computer%20Vision/Project%2014%20-%20Click%20Detect%20on%20Image)
- [Project 15 - Live Image Cartoonifier](Computer%20Vision/Project%2015%20-%20Live%20Image%20Cartoonifier)
- [Project 16 - Live Car Detection](Computer%20Vision/Project%2016%20-%20Live%20Car%20Detection)
- [Project 17 - Blink Detection](Computer%20Vision/Project%2017%20-%20Blink%20Detection)
- [Project 18 - Live Ball Tracking](Computer%20Vision/Project%2018%20-%20Live%20Ball%20Tracking)
- [Project 19 - Grayscale Converter](Computer%20Vision/Project%2019%20-%20Grayscale%20Converter)
- [Project 20 - Image Finder](Computer%20Vision/Project%2020%20-%20Image%20Finder)
- [Project 21 - Volume Controller](Computer%20Vision/Project%2021%20-%20Volume%20Controller)
- [Project 22 - Live Color Picker](Computer%20Vision/Project%2022%20-%20Live%20Color%20Picker)
- [Project 23 - Crop and Resize Image](Computer%20Vision/Project%2023%20-%20Crop%20and%20Resize%20Image)
- [Project 24 - Custom Object Detection](Computer%20Vision/Project%2024%20-%20Custom%20Object%20Detection)
- [Project 25 - Real Time Object Measurement](Computer%20Vision/Project%2025%20-%20Real%20Time%20Object%20Measurement)
- [Project 26 - Real Time Color Detection](Computer%20Vision/Project%2026%20-%20Real%20Time%20Color%20Detection)
- [Project 27 - Real Time Shape Detection](Computer%20Vision/Project%2027%20-%20Real%20Time%20Shape%20Detection)
- [Project 28 - Watermarking on Image](Computer%20Vision/Project%2028%20-%20Watermarking%20on%20Image)
- [Project 29 - Live Virtual Pen](Computer%20Vision/Project%2029%20-%20Live%20Virtual%20Pen)
- [Project 30 - Contrast Enhancement of Color Images](Computer%20Vision/Project%2030%20-%20Contrast%20Enhancement%20of%20Color%20Images)
- [Project 31 - Contrast Enhancement of Grayscale Image](Computer%20Vision/Project%2031%20-%20Contrast%20Enhancement%20of%20Grayscale%20Image)
- [Project 32 - Draw Vertical Lines of Coin](Computer%20Vision/Project%2032%20-%20Draw%20Vertical%20Lines%20of%20Coin)
- [Project 33 - Image Blurring](Computer%20Vision/Project%2033%20-%20Image%20Blurring)
- [Project 34 - Live Motion Blurring](Computer%20Vision/Project%2034%20-%20Live%20Motion%20Blurring)
- [Project 35 - Image Sharpening](Computer%20Vision/Project%2035%20-%20Image%20Sharpening)
- [Project 36 - Thresholding Techniques](Computer%20Vision/Project%2036%20-%20Thresholding%20Techniques)
- [Project 38 - Pencil Sketch Effect](Computer%20Vision/Project%2038%20-%20Pencil%20Sketch%20Effect)
- [Project 39 - Noise Removal](Computer%20Vision/Project%2039%20-%20Noise%20Removal)
- [Project 40 - Non-Photorealistic Rendering](Computer%20Vision/Project%2040%20-%20Non-Photorealistic%20Rendering)
- [Project 41 - Image Segmentation](Computer%20Vision/Project%2041%20-%20Image%20Segmentation)
- [Project 42 - Image Resizing](Computer%20Vision/Project%2042%20-%20Image%20Resizing)
- [Project 43 - Funny Cartoonizing Images](Computer%20Vision/Project%2043%20-%20Funny%20Cartoonizing%20Images)
- [Project 44 - Joining Multiple Images to Display](Computer%20Vision/Project%2044%20-%20Joining%20Multiple%20Images%20to%20Display)
- [Project 45 - Detecting Clicks on Images](Computer%20Vision/Project%2045%20-%20Detecting%20Clicks%20on%20Images)
- [Project 48 - Face Gender and Ethnicity Recognizer](Computer%20Vision/Project%2048%20-%20Face%20Gender%20and%20Ethnicity%20Recognizer)
- [Project 49 - Real Time Text Detection](Computer%20Vision/Project%2049%20-%20Real%20Time%20Text%20Detection)
- [Project 50 - Reversing Video](Computer%20Vision/Project%2050%20-%20Reversing%20Video)

</details>

---

### Deep Learning

<details>
<summary><strong>51 projects — PyTorch · TensorFlow · Keras · CNNs · GANs · LSTMs</strong></summary>

- [Advanced Churn Modeling](Deep%20Learning/Advanced%20Churn%20Modeling)
- [Advanced ResNet-50](Deep%20Learning/Advanced%20ResNet-50)
- [All Space Missions Analysis](Deep%20Learning/All%20Space%20Missions%20Analysis)
- [Amazon Alexa Sentiment Analysis](Deep%20Learning/Amazon%20Alexa%20Sentiment%20Analysis)
- [Amazon Stock Price Analysis](Deep%20Learning/Amazon%20Stock%20Price%20Analysis)
- [Arabic Character Recognition](Deep%20Learning/Arabic%20Character%20Recognition)
- [Bank Marketing Analysis](Deep%20Learning/Bank%20Marketing%20Analysis)
- [Bottle vs Can Classification](Deep%20Learning/Bottle%20vs%20Can%20Classification)
- [Brain MRI Segmentation](Deep%20Learning/Brain%20MRI%20Segmentation)
- [Brain Tumor Recognition](Deep%20Learning/Brain%20Tumor%20Recognition)
- [Cactus Aerial Image Recognition](Deep%20Learning/Cactus%20Aerial%20Image%20Recognition)
- [Caffe Face Detector - OpenCV](Deep%20Learning/Caffe%20Face%20Detector%20-%20OpenCV)
- [Campus Recruitment Analysis](Deep%20Learning/Campus%20Recruitment%20Analysis)
- [Cat and Dog Voice Recognition](Deep%20Learning/Cat%20and%20Dog%20Voice%20Recognition)
- [Cat vs Dog Classification](Deep%20Learning/Cat%20vs%20Dog%20Classification)
- [Chatbot](Deep%20Learning/Chatbot)
- [ChatBot - Neural Network](Deep%20Learning/ChatBot%20-%20Neural%20Network)
- [Clothing Prediction - Flask App](Deep%20Learning/Clothing%20Prediction%20-%20Flask%20App)
- [Concrete Strength Prediction](Deep%20Learning/Concrete%20Strength%20Prediction)
- [COVID-19 Drug Recovery](Deep%20Learning/COVID-19%20Drug%20Recovery)
- [COVID-19 Lung CT Scan Analysis](Deep%20Learning/COVID-19%20Lung%20CT%20Scan%20Analysis)
- [Dance Form Identification](Deep%20Learning/Dance%20Form%20Identification)
- [Diabetic Retinopathy](Deep%20Learning/Diabetic%20Retinopathy)
- [Disease Prediction](Deep%20Learning/Disease%20Prediction)
- [Earthquake Prediction](Deep%20Learning/Earthquake%20Prediction)
- [Electric Car Temperature Prediction](Deep%20Learning/Electric%20Car%20Temperature%20Prediction)
- [Face Gender and Ethnicity Recognizer](Deep%20Learning/Face%20Gender%20and%20Ethnicity%20Recognizer)
- [Face Mask Detection](Deep%20Learning/Face%20Mask%20Detection)
- [Fingerprint Recognition](Deep%20Learning/Fingerprint%20Recognition)
- [GANs](Deep%20Learning/GANs)
- [Glass Detection](Deep%20Learning/Glass%20Detection)
- [Happy House Predictor](Deep%20Learning/Happy%20House%20Predictor)
- [Hourly Energy Demand and Weather](Deep%20Learning/Hourly%20Energy%20Demand%20and%20Weather)
- [Image Colorization](Deep%20Learning/Image%20Colorization)
- [IMDB Sentiment Analysis](Deep%20Learning/IMDB%20Sentiment%20Analysis)
- [Indian Startup Data Analysis](Deep%20Learning/Indian%20Startup%20Data%20Analysis)
- [Keep Babies Safe](Deep%20Learning/Keep%20Babies%20Safe)
- [Landmark Detection](Deep%20Learning/Landmark%20Detection)
- [Lego Brick Classification](Deep%20Learning/Lego%20Brick%20Classification)
- [Movie Title Prediction](Deep%20Learning/Movie%20Title%20Prediction)
- [News Category Prediction](Deep%20Learning/News%20Category%20Prediction)
- [Parkinson Pose Estimation](Deep%20Learning/Parkinson%20Pose%20Estimation)
- [Pneumonia Detection](Deep%20Learning/Pneumonia%20Detection)
- [Pokemon Generation Clustering](Deep%20Learning/Pokemon%20Generation%20Clustering)
- [Sentiment Analysis - Flask App](Deep%20Learning/Sentiment%20Analysis%20-%20Flask%20App)
- [Sheep Breed Classification - CNN](Deep%20Learning/Sheep%20Breed%20Classification%20-%20CNN)
- [Skin Cancer Recognition](Deep%20Learning/Skin%20Cancer%20Recognition)
- [Stock Market Prediction](Deep%20Learning/Stock%20Market%20Prediction)
- [Sudoku Solver - Neural Network](Deep%20Learning/Sudoku%20Solver%20-%20Neural%20Network)
- [Walking or Running Classification](Deep%20Learning/Walking%20or%20Running%20Classification)
- [World Currency Coin Detection](Deep%20Learning/World%20Currency%20Coin%20Detection)

</details>

---

### Time Series Analysis

<details>
<summary><strong>21 projects — AutoGluon-TS · Chronos-Bolt · TimesFM · ARIMA · Prophet · LSTM</strong></summary>

- [Cryptocurrency Price Forecasting](Time%20Series%20Analysis/Cryptocurrency%20Price%20Forecasting)
- [Electricity Demand Forecasting](Time%20Series%20Analysis/Electricity%20Demand%20Forecasting)
- [Forecasting with ARIMA](Time%20Series%20Analysis/Forecasting%20with%20ARIMA)
- [Gold Price Forecasting](Time%20Series%20Analysis/Gold%20Price%20Forecasting)
- [Granger Causality Test](Time%20Series%20Analysis/Granger%20Causality%20Test)
- [Mini Course Sales Forecasting](Time%20Series%20Analysis/Mini%20Course%20Sales%20Forecasting)
- [Pollution Forecasting](Time%20Series%20Analysis/Pollution%20Forecasting)
- [Power Consumption - LSTM](Time%20Series%20Analysis/Power%20Consumption%20-%20LSTM)
- [Promotional Time Series](Time%20Series%20Analysis/Promotional%20Time%20Series)
- [Rossmann Store Sales Forecasting](Time%20Series%20Analysis/Rossmann%20Store%20Sales%20Forecasting)
- [Smart Home Temperature Forecasting](Time%20Series%20Analysis/Smart%20Home%20Temperature%20Forecasting)
- [Solar Power Generation Forecasting](Time%20Series%20Analysis/Solar%20Power%20Generation%20Forecasting)
- [Stock Market Analysis - Tech Stocks](Time%20Series%20Analysis/Stock%20Market%20Analysis%20-%20Tech%20Stocks)
- [Stock Price Forecasting](Time%20Series%20Analysis/Stock%20Price%20Forecasting)
- [Store Item Demand Forecasting](Time%20Series%20Analysis/Store%20Item%20Demand%20Forecasting)
- [Time Series Forecasting](Time%20Series%20Analysis/Time%20Series%20Forecasting)
- [Time Series Forecasting - Introduction](Time%20Series%20Analysis/Time%20Series%20Forecasting%20-%20Introduction)
- [Time Series with LSTM](Time%20Series%20Analysis/Time%20Series%20with%20LSTM)
- [Traffic Forecast](Time%20Series%20Analysis/Traffic%20Forecast)
- [US Gasoline and Diesel Prices 1995-2021](Time%20Series%20Analysis/US%20Gasoline%20and%20Diesel%20Prices%201995-2021)
- [Weather Forecasting](Time%20Series%20Analysis/Weather%20Forecasting)

</details>

---

### Clustering

<details>
<summary><strong>20 projects — UMAP · HDBSCAN · K-Means · Gaussian Mixture</strong></summary>

- [Credit Card Customer Segmentation](Clustering/Credit%20Card%20Customer%20Segmentation)
- [Customer Segmentation](Clustering/Customer%20Segmentation)
- [Customer Segmentation - Bank](Clustering/Customer%20Segmentation%20-%20Bank)
- [Financial Time Series Clustering](Clustering/Financial%20Time%20Series%20Clustering)
- [Housing Price Segmentation](Clustering/Housing%20Price%20Segmentation)
- [KMeans Clustering - Imagery Analysis](Clustering/KMeans%20Clustering%20-%20Imagery%20Analysis)
- [Mall Customer Segmentation](Clustering/Mall%20Customer%20Segmentation)
- [Mall Customer Segmentation - Advanced](Clustering/Mall%20Customer%20Segmentation%20-%20Advanced)
- [Mall Customer Segmentation - Detailed](Clustering/Mall%20Customer%20Segmentation%20-%20Detailed)
- [Mall Customer Segmentation Data](Clustering/Mall%20Customer%20Segmentation%20Data)
- [Online Retail Customer Segmentation](Clustering/Online%20Retail%20Customer%20Segmentation)
- [Online Retail Segmentation Analysis](Clustering/Online%20Retail%20Segmentation%20Analysis)
- [Spotify Song Cluster Analysis](Clustering/Spotify%20Song%20Cluster%20Analysis)
- [Turkiye Student Evaluation - Advanced](Clustering/Turkiye%20Student%20Evaluation%20-%20Advanced)
- [Turkiye Student Evaluation Analysis](Clustering/Turkiye%20Student%20Evaluation%20Analysis)
- [Vehicle Crash Data Clustering](Clustering/Vehicle%20Crash%20Data%20Clustering)
- [Weather Data Clustering - KMeans](Clustering/Weather%20Data%20Clustering%20-%20KMeans)
- [Wholesale Customer Segmentation](Clustering/Wholesale%20Customer%20Segmentation)
- [Wholesale Segmentation Analysis](Clustering/Wholesale%20Segmentation%20Analysis)
- [Wine Segmentation](Clustering/Wine%20Segmentation)

</details>

---

### Recommendation Systems

<details>
<summary><strong>19 projects — implicit ALS/BPR · LightFM · Surprise SVD · BGE-M3 · Qwen3-Embedding</strong></summary>

- [Article Recommendation System](Recommendation%20Systems/Article%20Recommendation%20System)
- [Articles Recommender](Recommendation%20Systems/Articles%20Recommender)
- [Book Recommendation System](Recommendation%20Systems/Book%20Recommendation%20System)
- [Building Recommender in an Hour](Recommendation%20Systems/Building%20Recommender%20in%20an%20Hour)
- [Collaborative Filtering - TensorFlow](Recommendation%20Systems/Collaborative%20Filtering%20-%20TensorFlow)
- [E-Commerce Recommendation System](Recommendation%20Systems/E-Commerce%20Recommendation%20System)
- [Event Recommendation System](Recommendation%20Systems/Event%20Recommendation%20System)
- [Hotel Recommendation System](Recommendation%20Systems/Hotel%20Recommendation%20System)
- [Million Songs Recommendation Engine](Recommendation%20Systems/Million%20Songs%20Recommendation%20Engine)
- [Movie Recommendation Engine](Recommendation%20Systems/Movie%20Recommendation%20Engine)
- [Movie Recommendation System](Recommendation%20Systems/Movie%20Recommendation%20System)
- [Movies Recommender](Recommendation%20Systems/Movies%20Recommender)
- [Music Recommendation System](Recommendation%20Systems/Music%20Recommendation%20System)
- [Recipe Recommendation System](Recommendation%20Systems/Recipe%20Recommendation%20System)
- [Recommender Systems Fundamentals](Recommendation%20Systems/Recommender%20Systems%20Fundamentals)
- [Recommender with Surprise Library](Recommendation%20Systems/Recommender%20with%20Surprise%20Library)
- [Restaurant Recommendation System](Recommendation%20Systems/Restaurant%20Recommendation%20System)
- [Seattle Hotels Recommender](Recommendation%20Systems/Seattle%20Hotels%20Recommender)
- [TV Show Recommendation System](Recommendation%20Systems/TV%20Show%20Recommendation%20System)

</details>

---

### Anomaly Detection and Fraud Detection

<details>
<summary><strong>11 projects — PyOD · anomalib PatchCore · CatBoost · LightGBM · IForest</strong></summary>

- [Anomaly Detection - Numenta Benchmark](Anomaly%20detection%20and%20fraud%20detection/Anomaly%20Detection%20-%20Numenta%20Benchmark)
- [Anomaly Detection - Social Networks Twitter Bot](Anomaly%20detection%20and%20fraud%20detection/Anomaly%20Detection%20-%20Social%20Networks%20Twitter%20Bot)
- [Anomaly Detection in Images - CIFAR-10](Anomaly%20detection%20and%20fraud%20detection/Anomaly%20Detection%20in%20Images%20-%20CIFAR-10)
- [Banknote Authentication](Anomaly%20detection%20and%20fraud%20detection/Banknote%20Authentication)
- [Breast Cancer Detection - Wisconsin Dataset](Anomaly%20detection%20and%20fraud%20detection/Breast%20Cancer%20Detection%20-%20Wisconsin%20Dataset)
- [Fraud Detection - IEEE-CIS](Anomaly%20detection%20and%20fraud%20detection/Fraud%20Detection%20-%20IEEE-CIS)
- [Fraud Detection in Financial Transactions](Anomaly%20detection%20and%20fraud%20detection/Fraud%20Detection%20in%20Financial%20Transactions)
- [Fraudulent Credit Card Transaction Detection](Anomaly%20detection%20and%20fraud%20detection/Fraudulent%20Credit%20Card%20Transaction%20Detection)
- [Insurance Fraud Detection](Anomaly%20detection%20and%20fraud%20detection/Insurance%20Fraud%20Detection)
- [Intrusion Detection](Anomaly%20detection%20and%20fraud%20detection/Intrusion%20Detection)
- [Traffic Flow Prediction - METR-LA](Anomaly%20detection%20and%20fraud%20detection/Traffic%20Flow%20Prediction%20-%20METR-LA)

</details>

---

### Reinforcement Learning

<details>
<summary><strong>9 projects — PPO · SAC · DQN · Stable-Baselines3 · Gymnasium</strong></summary>

- [Bipedal Walker](Reinforcement%20Learning/Bipedal%20Walker)
- [Cliff Walking](Reinforcement%20Learning/Cliff%20Walking)
- [Frozen Lake](Reinforcement%20Learning/Frozen%20Lake)
- [Gridworld Navigation](Reinforcement%20Learning/Gridworld%20Navigation)
- [Lunar Lander Continuous](Reinforcement%20Learning/Lunar%20Lander%20Continuous)
- [Lunar Landing](Reinforcement%20Learning/Lunar%20Landing)
- [Mountain Car Continuous](Reinforcement%20Learning/Mountain%20Car%20Continuous)
- [Pendulum Control](Reinforcement%20Learning/Pendulum%20Control)
- [Taxi Navigation](Reinforcement%20Learning/Taxi%20Navigation)

</details>

---

### Speech and Audio Processing

<details>
<summary><strong>5 projects — Whisper · Wav2Vec2 · HuBERT · SepFormer · XTTS-v2</strong></summary>

- [Audio Denoising](Speech%20and%20Audio%20processing/Audio%20Denoising)
- [Image Captioning](Speech%20and%20Audio%20processing/Image%20Captioning)
- [Music Genre Prediction - Million Songs](Speech%20and%20Audio%20processing/Music%20Genre%20Prediction%20-%20Million%20Songs)
- [Speech to Text](Speech%20and%20Audio%20processing/Speech%20to%20Text)
- [Voice Cloning](Speech%20and%20Audio%20processing/Voice%20Cloning)

</details>

---

### Data Analysis

<details>
<summary><strong>39 projects — EDA · Pandas · Matplotlib · Seaborn · PySpark</strong></summary>

- [2016 General Election Poll Analysis](Data%20Analysis/2016%20General%20Election%20Poll%20Analysis)
- [911 Calls Exploratory Analysis](Data%20Analysis/911%20Calls%20Exploratory%20Analysis)
- [Airbnb Data Analysis](Data%20Analysis/Airbnb%20Data%20Analysis)
- [Bank Payment Fraud Detection](Data%20Analysis/Bank%20Payment%20Fraud%20Detection)
- [CLV Non-Contractual](Data%20Analysis/CLV%20Non-Contractual)
- [CLV Online Retail](Data%20Analysis/CLV%20Online%20Retail)
- [Coffee Quality Analysis](Data%20Analysis/Coffee%20Quality%20Analysis)
- [COVID-19 Global Data Analysis](Data%20Analysis/COVID-19%20Global%20Data%20Analysis)
- [COVID-19 Tracking](Data%20Analysis/COVID-19%20Tracking)
- [Customer Lifetime Value Prediction](Data%20Analysis/Customer%20Lifetime%20Value%20Prediction)
- [Cybersecurity Anomaly Detection](Data%20Analysis/Cybersecurity%20Anomaly%20Detection)
- [Data Science Salaries Analysis](Data%20Analysis/Data%20Science%20Salaries%20Analysis)
- [Drive Data Analysis](Data%20Analysis/Drive%20Data%20Analysis)
- [FIFA 21 Data Cleaning](Data%20Analysis/FIFA%2021%20Data%20Cleaning)
- [FIFA Data Analysis](Data%20Analysis/FIFA%20Data%20Analysis)
- [Food Delivery Analysis](Data%20Analysis/Food%20Delivery%20Analysis)
- [Heart Failure Prediction](Data%20Analysis/Heart%20Failure%20Prediction)
- [Indians Diabetes Prediction](Data%20Analysis/Indians%20Diabetes%20Prediction)
- [Medical Insurance Cost Analysis](Data%20Analysis/Medical%20Insurance%20Cost%20Analysis)
- [Melbourne Housing Price Analysis](Data%20Analysis/Melbourne%20Housing%20Price%20Analysis)
- [Mobile Price Prediction](Data%20Analysis/Mobile%20Price%20Prediction)
- [Outliers Detection](Data%20Analysis/Outliers%20Detection)
- [Pokemon Data Analysis](Data%20Analysis/Pokemon%20Data%20Analysis)
- [Price Elasticity of Demand](Data%20Analysis/Price%20Elasticity%20of%20Demand)
- [Principal Component Analysis - Detailed](Data%20Analysis/Principal%20Component%20Analysis%20-%20Detailed)
- [PySpark Basic DataFrame Operations](Data%20Analysis/PySpark%20Basic%20DataFrame%20Operations)
- [Red Wine Quality Analysis](Data%20Analysis/Red%20Wine%20Quality%20Analysis)
- [Sleep Health Analysis](Data%20Analysis/Sleep%20Health%20Analysis)
- [Spark DataFrames Exercise](Data%20Analysis/Spark%20DataFrames%20Exercise)
- [Spotify Health Analysis](Data%20Analysis/Spotify%20Health%20Analysis)
- [Student Alcohol Consumption Analysis](Data%20Analysis/Student%20Alcohol%20Consumption%20Analysis)
- [Student Performance Analysis](Data%20Analysis/Student%20Performance%20Analysis)
- [Titanic Data Analysis](Data%20Analysis/Titanic%20Data%20Analysis)
- [Titanic Exploratory Analysis](Data%20Analysis/Titanic%20Exploratory%20Analysis)
- [Tokyo 2020 Tweets Frequency](Data%20Analysis/Tokyo%202020%20Tweets%20Frequency)
- [Top Billionaires List Analysis](Data%20Analysis/Top%20Billionaires%20List%20Analysis)
- [Video Game Sales Analysis](Data%20Analysis/Video%20Game%20Sales%20Analysis)
- [Wine Production Analysis](Data%20Analysis/Wine%20Production%20Analysis)
- [YouTube Trending Videos Analysis](Data%20Analysis/YouTube%20Trending%20Videos%20Analysis)

</details>

---

### Associate Rule Learning

<details>
<summary><strong>4 projects — Apriori · FP-Growth</strong></summary>

- [Grocery Store](Associate%20Rule%20Learning/Grocery%20Store)
- [Online News](Associate%20Rule%20Learning/Online%20News)
- [Online Retail](Associate%20Rule%20Learning/Online%20Retail)
- [Online video game store](Associate%20Rule%20Learning/Online%20video%20game%20store)

</details>

---

### Conceptual

<details>
<summary><strong>15 projects — Statistics · XGBoost · ROC Curves · SHAP · Databases</strong></summary>

- [Advanced Hyperparameter Tuning](Conceptual/Advanced%20Hyperparameter%20Tuning)
- [Advanced SHAP Values](Conceptual/Advanced%20SHAP%20Values)
- [Bayesian Statistics - PyMC3 and ArviZ](Conceptual/Bayesian%20Statistics%20-%20PyMC3%20and%20ArviZ)
- [Machine Learning with spaCy](Conceptual/Machine%20Learning%20with%20spaCy)
- [NHANES Confidence Intervals](Conceptual/NHANES%20Confidence%20Intervals)
- [NHANES Hypothesis Testing](Conceptual/NHANES%20Hypothesis%20Testing)
- [Practical Statistics in Python](Conceptual/Practical%20Statistics%20in%20Python)
- [Regression Diagnostics](Conceptual/Regression%20Diagnostics)
- [Supervised Learning - Part I](Conceptual/Supervised%20Learning%20-%20Part%20I)
- [Supervised Learning - Part II](Conceptual/Supervised%20Learning%20-%20Part%20II)
- [Text Classification - Keras](Conceptual/Text%20Classification%20-%20Keras)
- [Threshold Selection - ROC Curve](Conceptual/Threshold%20Selection%20-%20ROC%20Curve)
- [Working with Databases](Conceptual/Working%20with%20Databases)
- [XGBoost Algorithm](Conceptual/XGBoost%20Algorithm)
- [XGBoost with BOW and TF-IDF](Conceptual/XGBoost%20with%20BOW%20and%20TF-IDF)

</details>

---

### Python Scripts

<details>
<summary><strong>130+ utility and automation scripts</strong></summary>

- [Age Calculator GUI](Python%20Scripts/Age%20Calculator%20GUI)
- [Alarm Clock](Python%20Scripts/Alarm%20Clock)
- [Alphabetical File Organizer](Python%20Scripts/Alphabetical%20File%20Organizer)
- [Attachment Downloader](Python%20Scripts/Attachment%20Downloader)
- [AudioBook Creator](Python%20Scripts/AudioBook%20Creator)
- [Auto Backup](Python%20Scripts/Auto%20Backup)
- [Auto Birthday Wisher](Python%20Scripts/Auto%20Birthday%20Wisher)
- [Auto Draw](Python%20Scripts/Auto%20Draw)
- [Auto Fill Google Forms](Python%20Scripts/Auto%20Fill%20Google%20Forms)
- [Automate Facebook bot](Python%20Scripts/Automate%20Facebook%20bot)
- [Automatic Certificate Generator](Python%20Scripts/Automatic%20Certificate%20Generator)
- [Automatic FB login](Python%20Scripts/Automatic%20FB%20login)
- [AWS Endorsement Management](Python%20Scripts/AWS%20Endorsement%20Management)
- [Brick Breaker game](Python%20Scripts/Brick%20Breaker%20game)
- [Bubble Shooter Game](Python%20Scripts/Bubble%20Shooter%20Game)
- [Calculate Age](Python%20Scripts/Calculate%20Age)
- [Calculator App](Python%20Scripts/Calculator%20App)
- [Calculator GUI](Python%20Scripts/Calculator%20GUI)
- [Calendar GUI](Python%20Scripts/Calendar%20GUI)
- [Chat bot](Python%20Scripts/Chat%20bot)
- [Chrome Automation](Python%20Scripts/Chrome%20Automation)
- [CLI Proxy Tester](Python%20Scripts/CLI%20Proxy%20Tester)
- [CLI Todo App](Python%20Scripts/CLI%20Todo%20App)
- [COVID-19 Real-Time Notification](Python%20Scripts/COVID-19%20Real-Time%20Notification)
- [COVID-19 Update Bot](Python%20Scripts/COVID-19%20Update%20Bot)
- [CSV File Merger](Python%20Scripts/CSV%20File%20Merger)
- [Currency Converter](Python%20Scripts/Currency%20Converter)
- [Currency Exchange Rates](Python%20Scripts/Currency%20Exchange%20Rates)
- [Current Weather Fetcher](Python%20Scripts/Current%20Weather%20Fetcher)
- [Decimal Binary Converter](Python%20Scripts/Decimal%20Binary%20Converter)
- [Desktop Notification Application](Python%20Scripts/Desktop%20Notification%20Application)
- [Device Shutdown and Restart](Python%20Scripts/Device%20Shutdown%20and%20Restart)
- [Dictionary GUI](Python%20Scripts/Dictionary%20GUI)
- [Dictionary to Python Object](Python%20Scripts/Dictionary%20to%20Python%20Object)
- [Digital Clock](Python%20Scripts/Digital%20Clock)
- [Digital Clock With GUI](Python%20Scripts/Digital%20Clock%20With%20GUI)
- [Discord Bot](Python%20Scripts/Discord%20Bot)
- [DNS Record Lookup](Python%20Scripts/DNS%20Record%20Lookup)
- [Download Folder Organizer](Python%20Scripts/Download%20Folder%20Organizer)
- [Duplicate File Remover](Python%20Scripts/Duplicate%20File%20Remover)
- [Easy Video Player](Python%20Scripts/Easy%20Video%20Player)
- [Email GUI](Python%20Scripts/Email%20GUI)
- [Email Sender](Python%20Scripts/Email%20Sender)
- [Email Sender from CSV](Python%20Scripts/Email%20Sender%20from%20CSV)
- [Email to CSV Store](Python%20Scripts/Email%20to%20CSV%20Store)
- [Email Validator](Python%20Scripts/Email%20Validator)
- [Facebook Auto Login](Python%20Scripts/Facebook%20Auto%20Login)
- [Facebook DP Downloader](Python%20Scripts/Facebook%20DP%20Downloader)
- [Facebook Video Downloader](Python%20Scripts/Facebook%20Video%20Downloader)
- [Fidget Spinner Game](Python%20Scripts/Fidget%20Spinner%20Game)
- [File and Folder Compressor](Python%20Scripts/File%20and%20Folder%20Compressor)
- [File and Folder Encryption](Python%20Scripts/File%20and%20Folder%20Encryption)
- [File Compare](Python%20Scripts/File%20Compare)
- [File Encryptor](Python%20Scripts/File%20Encryptor)
- [File Renamer](Python%20Scripts/File%20Renamer)
- [File Splitter](Python%20Scripts/File%20Splitter)
- [File Unzipper](Python%20Scripts/File%20Unzipper)
- [Flames Game](Python%20Scripts/Flames%20Game)
- [Folder Splitter](Python%20Scripts/Folder%20Splitter)
- [GeeksForGeeks Article Downloader](Python%20Scripts/GeeksForGeeks%20Article%20Downloader)
- [Geocoding](Python%20Scripts/Geocoding)
- [GitHub Automation](Python%20Scripts/GitHub%20Automation)
- [GitHub Repo Automation](Python%20Scripts/GitHub%20Repo%20Automation)
- [GOOGLE API](Python%20Scripts/GOOGLE%20API)
- [Google Classroom Bot](Python%20Scripts/Google%20Classroom%20Bot)
- [Google News Scraper](Python%20Scripts/Google%20News%20Scraper)
- [Hacker News Scraper](Python%20Scripts/Hacker%20News%20Scraper)
- [Hangman Game](Python%20Scripts/Hangman%20Game)
- [High Quality YouTube Video Downloader](Python%20Scripts/High%20Quality%20YouTube%20Video%20Downloader)
- [Hostname and IP Finder](Python%20Scripts/Hostname%20and%20IP%20Finder)
- [HTTP Status Code Checker](Python%20Scripts/HTTP%20Status%20Code%20Checker)
- [Image Converter](Python%20Scripts/Image%20Converter)
- [Image File Size Reducer](Python%20Scripts/Image%20File%20Size%20Reducer)
- [Image Meta Info Extractor](Python%20Scripts/Image%20Meta%20Info%20Extractor)
- [Image to PDF Converter](Python%20Scripts/Image%20to%20PDF%20Converter)
- [Image to Speech](Python%20Scripts/Image%20to%20Speech)
- [IMDB Rating Finder](Python%20Scripts/IMDB%20Rating%20Finder)
- [Instagram Follow and Message Bot](Python%20Scripts/Instagram%20Follow%20and%20Message%20Bot)
- [Instagram Follow Tracker](Python%20Scripts/Instagram%20Follow%20Tracker)
- [Instagram Image Downloader](Python%20Scripts/Instagram%20Image%20Downloader)
- [Instagram Liker Bot](Python%20Scripts/Instagram%20Liker%20Bot)
- [Instagram Profile Viewer](Python%20Scripts/Instagram%20Profile%20Viewer)
- [Instagram Scraper](Python%20Scripts/Instagram%20Scraper)
- [Instagram Unfollower Tracker](Python%20Scripts/Instagram%20Unfollower%20Tracker)
- [Internet Connection Checker](Python%20Scripts/Internet%20Connection%20Checker)
- [Internet Speed Test](Python%20Scripts/Internet%20Speed%20Test)
- [IPL Statistics GUI](Python%20Scripts/IPL%20Statistics%20GUI)
- [JPEG to PNG Converter](Python%20Scripts/JPEG%20to%20PNG%20Converter)
- [JSON to CSV Converter](Python%20Scripts/JSON%20to%20CSV%20Converter)
- [Leap Year Checker](Python%20Scripts/Leap%20Year%20Checker)
- [LinkedIn Connections Scraper](Python%20Scripts/LinkedIn%20Connections%20Scraper)
- [LinkedIn Email Scraper](Python%20Scripts/LinkedIn%20Email%20Scraper)
- [Live Cricket Score](Python%20Scripts/Live%20Cricket%20Score)
- [Lyrics Genius API](Python%20Scripts/Lyrics%20Genius%20API)
- [Medium Article Scraper](Python%20Scripts/Medium%20Article%20Scraper)
- [Movie Info Telegram Bot](Python%20Scripts/Movie%20Info%20Telegram%20Bot)
- [Movie Information Scraper](Python%20Scripts/Movie%20Information%20Scraper)
- [Multi-File String Search](Python%20Scripts/Multi-File%20String%20Search)
- [Music Player](Python%20Scripts/Music%20Player)
- [News Updater With Voice](Python%20Scripts/News%20Updater%20With%20Voice)
- [News Website Scraper](Python%20Scripts/News%20Website%20Scraper)
- [NSE Stocks GUI](Python%20Scripts/NSE%20Stocks%20GUI)
- [Number Guessing Game](Python%20Scripts/Number%20Guessing%20Game)
- [Numbers to Words Converter](Python%20Scripts/Numbers%20to%20Words%20Converter)
- [Open Port Scanner](Python%20Scripts/Open%20Port%20Scanner)
- [PageSpeed API](Python%20Scripts/PageSpeed%20API)
- [Paint App](Python%20Scripts/Paint%20App)
- [Password Hashing](Python%20Scripts/Password%20Hashing)
- [Password Manager CLI](Python%20Scripts/Password%20Manager%20CLI)
- [Password Manager GUI](Python%20Scripts/Password%20Manager%20GUI)
- [PDF Merger](Python%20Scripts/PDF%20Merger)
- [PDF Reader with Voice](Python%20Scripts/PDF%20Reader%20with%20Voice)
- [PDF Text Extractor](Python%20Scripts/PDF%20Text%20Extractor)
- [PDF to CSV Converter](Python%20Scripts/PDF%20to%20CSV%20Converter)
- [PDF to Text Converter](Python%20Scripts/PDF%20to%20Text%20Converter)
- [PNG to ICO Converter](Python%20Scripts/PNG%20to%20ICO%20Converter)
- [Pomodoro Timer GUI](Python%20Scripts/Pomodoro%20Timer%20GUI)
- [PyDoku](Python%20Scripts/PyDoku)
- [PyQt5 Password Generator](Python%20Scripts/PyQt5%20Password%20Generator)
- [pyWeather](Python%20Scripts/pyWeather)
- [QR Code Generator](Python%20Scripts/QR%20Code%20Generator)
- [Quotes Scraper](Python%20Scripts/Quotes%20Scraper)
- [Racing Bar Chart Animation](Python%20Scripts/Racing%20Bar%20Chart%20Animation)
- [Random Email Generator](Python%20Scripts/Random%20Email%20Generator)
- [Random Password Generator](Python%20Scripts/Random%20Password%20Generator)
- [Random Wikipedia Article](Python%20Scripts/Random%20Wikipedia%20Article)
- [Random Word from List](Python%20Scripts/Random%20Word%20from%20List)
- [Raspberry Pi Sonoff](Python%20Scripts/Raspberry%20Pi%20Sonoff)
- [Recursive Password Generator](Python%20Scripts/Recursive%20Password%20Generator)
- [Reddit Scraper](Python%20Scripts/Reddit%20Scraper)
- [Rock Paper Scissors Game](Python%20Scripts/Rock%20Paper%20Scissors%20Game)
- [Screen Recorder](Python%20Scripts/Screen%20Recorder)
- [Screenshot Capture](Python%20Scripts/Screenshot%20Capture)
- [Simple Stopwatch](Python%20Scripts/Simple%20Stopwatch)
- [Sine vs Cosine](Python%20Scripts/Sine%20vs%20Cosine)
- [Site Blocker](Python%20Scripts/Site%20Blocker)
- [SMS Automation](Python%20Scripts/SMS%20Automation)
- [Snake Game GUI](Python%20Scripts/Snake%20Game%20GUI)
- [Spaceship Game](Python%20Scripts/Spaceship%20Game)
- [Speech to Text](Python%20Scripts/Speech%20to%20Text)
- [Speech to Text Converter](Python%20Scripts/Speech%20to%20Text%20Converter)
- [Speed Test](Python%20Scripts/Speed%20Test)
- [Spreadsheet Automation](Python%20Scripts/Spreadsheet%20Automation)
- [Take a Break Reminder](Python%20Scripts/Take%20a%20Break%20Reminder)
- [Terminal Hangman Game](Python%20Scripts/Terminal%20Hangman%20Game)
- [Terminal Progress Bar - Image Resize](Python%20Scripts/Terminal%20Progress%20Bar%20-%20Image%20Resize)
- [Text Editor](Python%20Scripts/Text%20Editor)
- [Text Encryption and Decryption](Python%20Scripts/Text%20Encryption%20and%20Decryption)
- [Text Message Sender](Python%20Scripts/Text%20Message%20Sender)
- [Text to Speech](Python%20Scripts/Text%20to%20Speech)
- [Tic Tac Toe](Python%20Scripts/Tic%20Tac%20Toe)
- [Tic Tac Toe with AI](Python%20Scripts/Tic%20Tac%20Toe%20with%20AI)
- [Todo App](Python%20Scripts/Todo%20App)
- [Tweet Fetcher and Store](Python%20Scripts/Tweet%20Fetcher%20and%20Store)
- [Twitter Scraper](Python%20Scripts/Twitter%20Scraper)
- [Typing Speed Test](Python%20Scripts/Typing%20Speed%20Test)
- [Unique Words in File](Python%20Scripts/Unique%20Words%20in%20File)
- [Unsplash Wallpaper Downloader](Python%20Scripts/Unsplash%20Wallpaper%20Downloader)
- [URL Shortener](Python%20Scripts/URL%20Shortener)
- [USSD Service Data](Python%20Scripts/USSD%20Service%20Data)
- [Video Frame Capture](Python%20Scripts/Video%20Frame%20Capture)
- [Video Splitter by Time](Python%20Scripts/Video%20Splitter%20by%20Time)
- [Video to Audio Converter](Python%20Scripts/Video%20to%20Audio%20Converter)
- [Voice Translator](Python%20Scripts/Voice%20Translator)
- [Wallpaper Changer](Python%20Scripts/Wallpaper%20Changer)
- [Weather App](Python%20Scripts/Weather%20App)
- [Web Crawler Link Finder](Python%20Scripts/Web%20Crawler%20Link%20Finder)
- [Web Page Summarizer](Python%20Scripts/Web%20Page%20Summarizer)
- [Website Blocker](Python%20Scripts/Website%20Blocker)
- [Website Connectivity Checker](Python%20Scripts/Website%20Connectivity%20Checker)
- [Website Image Downloader](Python%20Scripts/Website%20Image%20Downloader)
- [Website Load Time Checker](Python%20Scripts/Website%20Load%20Time%20Checker)
- [Website Snapshot](Python%20Scripts/Website%20Snapshot)
- [WhatsApp Auto Messenger](Python%20Scripts/WhatsApp%20Auto%20Messenger)
- [WhatsApp Automation](Python%20Scripts/WhatsApp%20Automation)
- [WhatsApp Bot](Python%20Scripts/WhatsApp%20Bot)
- [WiFi Password Retriever](Python%20Scripts/WiFi%20Password%20Retriever)
- [Wikipedia Infobox Scraper](Python%20Scripts/Wikipedia%20Infobox%20Scraper)
- [Wikipedia Scraper](Python%20Scripts/Wikipedia%20Scraper)
- [Wikipedia Summary GUI](Python%20Scripts/Wikipedia%20Summary%20GUI)
- [Word Games](Python%20Scripts/Word%20Games)
- [Work Setup Automation](Python%20Scripts/Work%20Setup%20Automation)
- [XML to JSON Converter](Python%20Scripts/XML%20to%20JSON%20Converter)
- [YouTube Audio Downloader](Python%20Scripts/YouTube%20Audio%20Downloader)
- [YouTube Comment Scraper](Python%20Scripts/YouTube%20Comment%20Scraper)
- [YouTube Trending Feed Scraper](Python%20Scripts/YouTube%20Trending%20Feed%20Scraper)
- [YouTube Video Downloader](Python%20Scripts/YouTube%20Video%20Downloader)
- [ZIP File Extractor](Python%20Scripts/ZIP%20File%20Extractor)

</details>

---

## Repository Structure

```
Machine-Learning-Projects/
│
├── Classification/                        # 46 tabular classification projects
├── Regression/                            # 44 tabular regression projects
├── NLP/                                   # 70 NLP projects (classification, generation, similarity, misc)
├── Computer Vision/                       # 23 CV detection, face/gesture, and misc projects
├── Deep Learning/                         # 37 image classification + DL-specific projects
├── Time Series Analysis/                  # 25 time series forecasting projects
├── Clustering/                            # 22 clustering and segmentation projects
├── Recommendation Systems/                # 19 recommender system projects
├── Anomaly detection and fraud detection/ # 14 fraud + anomaly detection projects
├── Reinforcement Learning/                # 5 RL projects
├── Speech and Audio processing/           # 4 audio/speech projects
├── Data Analysis/                         # EDA-only notebooks (pre-existing, not generated)
├── Conceptual/                            # Conceptual tutorials (PCA, ROC curves, etc.)
├── Associate Rule Learning/               # Apriori / FP-Growth projects
├── Python Scripts/                        # Standalone utility scripts
│
├── _overhaul_v2.py                        # Master pipeline generator (19 functions, 4,200 LOC)
├── requirements.txt                       # Dependency manifest (grouped by category)
├── conftest.py                            # Shared pytest fixtures
├── pytest.ini                             # Test configuration and markers
├── generate_tests.py                      # Auto-generates test suites from pipeline metadata
├── dataset_registry.json                  # Project metadata registry (source, target, models)
├── standardize_ml.py                      # ML standardization utilities
├── MIGRATION_PLAN.md                      # Detailed family-by-family migration audit
└── WORKSPACE_OVERVIEW.md                  # Architecture overview document
```

---

## Architecture

### Generator System

All 315 pipelines are produced by a single Python orchestrator — [`_overhaul_v2.py`](_overhaul_v2.py). The generator implements a **dictionary-dispatch architecture**:

1. **23 dictionaries** map each project name to its metadata (dataset path, target column, task type).
2. **19 generator functions** (`gen_tabular_clf`, `gen_nlp_clf`, `gen_image_clf`, etc.) emit complete, self-contained pipeline code for a given family.
3. A **dispatch table** routes each dictionary to the correct generator.
4. Running `python _overhaul_v2.py` regenerates all 315 `pipeline.py` files in one pass.

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  23 Project     │────▶│  Dispatch    │────▶│  19 Generator   │
│  Dictionaries   │     │  Table       │     │  Functions      │
│  (315 entries)  │     │              │     │                 │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │  315 pipeline  │
                                              │  .py files     │
                                              │  (68,600+ LOC) │
                                              └───────────────┘
```

### Pipeline Design

Each generated `pipeline.py` follows a consistent structure:

- **Auto-detection** of GPU availability (CUDA → CPU fallback)
- **Auto-download** of datasets at runtime — no manual data management
- **Foundation model primary** with classical baselines for comparison
- **Structured output** — metrics written to JSON for programmatic consumption
- **Wall-clock timing** for every training step
- **Compute documentation** embedded as comments (VRAM, training time estimates)

### Regeneration

```bash
python _overhaul_v2.py           # regenerate all 315 pipelines
```

Edit the dictionaries to add new projects, or modify a `gen_*` function to update the model stack for an entire family at once.

---

## Testing

The repository includes a pytest-based test infrastructure:

```bash
pytest                                    # run all tests
pytest -m "sklearn"                       # run only scikit-learn tests
pytest -m "not slow"                      # skip long-running deep learning tests
pytest -k "classification"                # filter by name pattern
```

**Registered markers:** `data`, `slow`, `sklearn`, `keras`, `pytorch`, `nlp`, `cv`, `clustering`, `timeseries`, `eda`, `conceptual`

**Shared fixtures** (defined in `conftest.py`):
- `workspace_root` — absolute path to the repository root
- `data_root` — absolute path to the `data/` directory

---

## Dependencies

Dependencies are organized by category in [`requirements.txt`](requirements.txt):

| Category | Key Packages |
|----------|-------------|
| **Core DL** | PyTorch, torchmetrics |
| **Hugging Face** | transformers, datasets, accelerate, evaluate, sentence-transformers |
| **Gradient Boosting** | catboost, xgboost, lightgbm (all GPU-enabled) |
| **AutoML** | FLAML, LazyPredict, AutoGluon (tabular + time series), TabPFN |
| **Computer Vision** | timm, opencv-python, albumentations, ultralytics |
| **Anomaly Detection** | PyOD, anomalib |
| **Clustering** | umap-learn, hdbscan |
| **Time Series** | statsforecast, chronos-forecasting, timesfm |
| **NLP** | nltk, GLiNER, qwen-vl-utils |
| **RL** | gymnasium, stable-baselines3 |
| **Face / Pose** | insightface |
| **Audio** | librosa, soundfile |
| **Visualization** | matplotlib, seaborn |
| **Data** | numpy, pandas, scipy |

> **Note:** PyTorch must be installed separately with the correct CUDA index URL before running `pip install -r requirements.txt`. Some packages (mediapipe, paddleocr, TTS, implicit, lightfm) are commented out in `requirements.txt` due to platform-specific build requirements — uncomment as needed.

---

## License

This project is licensed under the MIT License.
