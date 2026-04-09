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

## Project Catalog

<details>
<summary><strong>Tabular Classification</strong> — 46 projects</summary>

Projects include: Adult Salary Prediction, Titanic Survival, Credit Card Fraud, Heart Disease, Diabetes Prediction, Iris Classification, Wine Quality, Mushroom Classification, Customer Churn, Bank Marketing, and 36 more — each with CatBoost / LightGBM / XGBoost / AutoGluon / TabPFN-v2 / TabM ensembles plus FLAML and LazyPredict baselines.

</details>

<details>
<summary><strong>Tabular Regression</strong> — 44 projects</summary>

Projects include: Boston Housing, Bitcoin Price Prediction, Gold Price Prediction, Flight Fare Prediction, Rain Fall Prediction, Medical Cost Prediction, Used Car Price Prediction, Diamond Price Prediction, and 36 more — same model stack as classification but with regressor variants and MSE / MAE / R² metrics.

</details>

<details>
<summary><strong>NLP Classification</strong> — 44 projects</summary>

Projects include: Sentiment Analysis (IMDB, Twitter, Amazon), Spam Detection, Hate Speech Detection, Fake News Classification, Disaster Tweet Classification, Emotion Detection, Sarcasm Detection, and 35 more — all fine-tuned with ModernBERT + XLM-RoBERTa and compared against TF-IDF baselines.

</details>

<details>
<summary><strong>Image Classification</strong> — 31 projects</summary>

Projects include: Dog vs Cat, Fashion MNIST, Flower Species, CIFAR-10, Plant Disease Recognition, Cotton Disease Prediction, Pneumonia Classification, Face Mask Detection, and 23 more — all using DINOv3 (ViT-S/14) and ConvNeXt V2 with mixed-precision training.

</details>

<details>
<summary><strong>Time Series Forecasting</strong> — 25 projects</summary>

Projects include: Stock Price Forecasting (multiple tickers), Bitcoin Forecasting, Weather Prediction, Energy Demand, Sales Forecasting, COVID-19 Trends, Traffic Prediction, and 18 more — using AutoGluon-TS, Chronos-Bolt / Chronos-2, TimesFM, with ARIMA / Prophet / FLAML baselines.

</details>

<details>
<summary><strong>Computer Vision</strong> — 23 projects (Detection + Face/Gesture + Misc)</summary>

Projects include: Object Detection, Face Recognition, Gesture Control, Pose Estimation, Sign Language Detection, Face Expression Identification, Live Smile Detector, Age & Gender Detection, and 15 more — using YOLO26m, MediaPipe Tasks, and InsightFace.

</details>

<details>
<summary><strong>NLP Generation & Translation</strong> — 22 projects</summary>

Projects include: Text Summarization, Machine Translation, Chatbot Development, Story Generation, Code Generation, and 17 more — using Qwen3-Instruct via Ollama, BART for summarization, and NLLB-200 for multilingual translation.

</details>

<details>
<summary><strong>Clustering</strong> — 22 projects</summary>

Projects include: Customer Segmentation, Market Basket Analysis, Image Segmentation, Document Clustering, Anomaly Clustering, and 17 more — using UMAP dimensionality reduction + HDBSCAN with silhouette / Calinski-Harabasz / Davies-Bouldin evaluation.

</details>

<details>
<summary><strong>Recommendation Systems</strong> — 19 projects</summary>

Projects include: Movie Recommendation, Book Recommendation, Music Playlist, E-Commerce Product Suggestion, Restaurant Recommendation, and 14 more — combining collaborative filtering (implicit ALS/BPR, Surprise SVD) with content-based embeddings (BGE-M3, Qwen3-Embedding) and LightFM hybrid models.

</details>

<details>
<summary><strong>Anomaly & Fraud Detection</strong> — 14 projects</summary>

Projects include: Credit Card Fraud Detection, Network Intrusion Detection, Insurance Fraud, Transaction Anomaly Detection, and 10 more — using calibrated GBDT ensembles, PyOD (ECOD / COPOD / IForest), and anomalib PatchCore for visual anomaly detection.

</details>

<details>
<summary><strong>Reinforcement Learning</strong> — 5 projects</summary>

Projects include: Lunar Landing, CartPole, Mountain Car, and more — using PPO / SAC / DQN from Stable-Baselines3 with tabular Q-learning as an educational baseline for small-state environments. Auto-selects SAC for continuous and DQN for discrete action spaces.

</details>

<details>
<summary><strong>Audio & Speech</strong> — 4 projects</summary>

Projects include: Speech-to-Text Transcription, Audio Classification, Audio Denoising, Voice Cloning — using Whisper large-v3-turbo, Wav2Vec2 / HuBERT for classification, SepFormer for source separation, and XTTS-v2 for voice synthesis.

</details>

<details>
<summary><strong>Other Families</strong> — NLP Similarity (4), Captioning/VLM (3), OCR (3), Medical Segmentation (2)</summary>

- **NLP Similarity:** Semantic search and document retrieval with BGE-M3 / Qwen3-Embedding
- **Captioning / VLM:** Image captioning and visual question answering with Qwen3-VL and Molmo 2
- **OCR:** Text extraction from documents with PaddleOCR and PaddleOCR-VL-1.5
- **Medical Segmentation:** Organ / lesion segmentation with nnU-Net and MedSAM

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
