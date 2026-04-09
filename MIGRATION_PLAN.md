# Model Modernization Plan — April 2026

> **Repository:** Machine-Learning-Projects  
> **Total pipelines:** 315 (generated) + 26 pre-existing Data Analysis (PyCaret, not managed by generator)  
> **Generator:** `_overhaul_v2.py` — 18 generator functions, 22 family→generator mappings  
> **Status:** ✅ **ALL MIGRATIONS COMPLETE** — every family upgraded, verified, committed  

---

## Migration Timeline

| Commit | Phase | Family | Summary |
|--------|-------|--------|---------|
| `d587254` | 0 | All | v2 overhaul: 315 pipelines auto-download data, Python 3.13 venv |
| `a83dda5` | 1 | Multi | AutoGluon+TabPFN+TabM, XLM-R+GLiNER, DINOv3+Qwen3-VL, YOLO26m, anomalib, Chronos-2+TimesFM |
| `cd45b34` | 2 | Tabular | FLAML + LazyPredict as baseline comparison for tabular clf/reg |
| `f0284e0` | 3 | Multi | TabPFN-v2 reg, fraud GBDT+PyOD, NLLB-200, ConvNeXtV2+Molmo2, DQN baseline, PaddleOCR-VL-1.5 |
| `03e9dc5` | 4 | Audio | SepFormer denoising, Wav2Vec2+HuBERT classification, Whisper ASR, XTTS-v2 cloning |
| `215ddfd` | 5 | RL | PPO primary, DQN baseline on discrete envs |
| `724a2a5` | 6 | Time Series | AutoGluon-TS + Chronos-Bolt/2 + TimesFM + GBDT lag baselines |
| `34bea87` | 7 | Vision | DINOv3+ConvNeXtV2 clf, Qwen3-VL+Molmo2 captioning, nnU-Net+MedSAM medical seg |
| `2f9f322` | 8 | Face/Gesture | YOLO26 + MediaPipe Tasks API + InsightFace |
| `38fa5fe` | 9 | Recommenders | implicit ALS/BPR + LightFM + BGE-M3/Qwen3-Embedding + Surprise baseline |
| `fec6901` | 10 | NLP Gen | Qwen3-Instruct + NLLB-200 + BART baseline |
| `23a9e02` | 11 | NLP Clf | ModernBERT + XLM-R + TF-IDF/NB baseline + GLiNER NER |
| `4434638` | 12 | Clustering | UMAP+HDBSCAN + GMM + K-Means baseline |
| `a1c74d1` | 13 | Fraud/Anomaly | Calibrated GBDT trio + PyOD ECOD/COPOD/IForest + anomalib PatchCore |

---

## Family-by-Family Audit

### 1. Tabular Classification — 46 projects → `gen_tabular_clf`

**Status:** ✅ Complete

| Model | Role | GPU |
|-------|------|-----|
| CatBoost | Primary | ✅ GPU, `auto_class_weights="Balanced"` |
| LightGBM | Primary | ✅ GPU, `class_weight="balanced"` |
| XGBoost | Primary | ✅ CUDA, `tree_method="hist"` |
| AutoGluon Tabular | AutoML ensemble | CPU (time_limit=180s, `best_quality`) |
| TabPFN-v2 | Prior-fitted network | ✅ CUDA (≤10K rows, ≤500 cols guard) |
| TabM | Deep tabular | ✅ CUDA (3-block multi-head architecture) |
| FLAML | Baseline AutoML | CPU (time_budget=120s) |
| LazyPredict | Baseline comparison | CPU (top-5 classifiers) |

**Legacy models removed:** KNN, Decision Tree, AdaBoost, SVC, Naive Bayes, plain MLP, Random Forest  
**Projects include:** Adult Salary, Breast Cancer, Credit Risk, Customer Churn, Diabetes, Heart Disease, Wine Quality, Titanic, Iris, and 37 more

---

### 2. Tabular Regression — 44 projects → `gen_tabular_reg`

**Status:** ✅ Complete

| Model | Role | GPU |
|-------|------|-----|
| CatBoostRegressor | Primary | ✅ GPU |
| LGBMRegressor | Primary | ✅ GPU |
| XGBRegressor | Primary | ✅ CUDA |
| AutoGluon Tabular | AutoML (`regression`, `best_quality`) | CPU |
| TabPFNRegressor | Prior-fitted network | ✅ CUDA (size guard) |
| TabM | Deep tabular (MSE loss) | ✅ CUDA |
| FLAML | Baseline | CPU |
| LazyRegressor | Baseline | CPU |

**Projects include:** House Price, Insurance Premium, Gold Price, Flight Fare, Boston Housing, California Housing, and 38 more

---

### 3. Fraud / Imbalanced Classification — 7 projects → `gen_fraud`

**Status:** ✅ Complete

| Model | Role | GPU |
|-------|------|-----|
| CatBoost | Primary, `scale_pos_weight` | ✅ GPU |
| LightGBM | Primary, `scale_pos_weight` | ✅ GPU |
| XGBoost | Primary, `scale_pos_weight` | ✅ CUDA |
| CalibratedClassifierCV | Isotonic calibration on 15% held-out split | — |
| PyOD ECOD | Unsupervised cross-check | CPU |
| PyOD COPOD | Unsupervised cross-check | CPU |
| PyOD IForest | Unsupervised cross-check | CPU |

**Special features:** PR-curve threshold tuning, reliability diagram, confusion matrix plot  
**FLAML/LazyPredict:** ❌ Not included (not appropriate for imbalanced/fraud-specific stack)

---

### 4. Anomaly Detection — 7 projects → `gen_anomaly`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| PyOD ECOD | Tabular anomaly | Unsupervised, contamination=0.05 |
| PyOD COPOD | Tabular anomaly | Unsupervised, contamination=0.05 |
| PyOD IForest | Tabular anomaly | `pyod.models.iforest` (not sklearn) |
| anomalib PatchCore | Image anomaly | WideResNet-50, MVTec benchmark |

**Special features:** Score distribution comparison plot, F1/AUC when ground-truth labels exist  
**FLAML/LazyPredict:** ❌ Not included (unsupervised detection, not tabular supervised)

---

### 5. Clustering — 21 projects → `gen_clustering`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| UMAP | Dimensionality reduction | `n_components=2`, `min_dist=0.1` |
| HDBSCAN | Primary clustering | Auto-tuned `min_cluster_size` |
| Gaussian Mixture (GMM) | Soft assignments | BIC model selection |
| K-Means | Baseline | Elbow + Silhouette analysis |

**Metrics:** Silhouette, Calinski-Harabasz, Davies-Bouldin + comparison scatter plot

---

### 6. NLP Classification — 40 projects → `gen_nlp_clf`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| ModernBERT | Primary English encoder | `answerdotai/ModernBERT-base`, fine-tuned |
| XLM-RoBERTa | Multilingual fallback | `xlm-roberta-base` |
| TF-IDF + Naive Bayes | Baseline | MultinomialNB |
| TF-IDF + Logistic Regression | Baseline | Regularized LR |
| GLiNER | Zero-shot NER | NER/keyword extraction projects |
| BGE-M3 / Qwen3-Embedding | Semantic similarity | Embedding-based retrieval |

**Routing note:** NER, keyword extraction, profanity, and BOW projects moved from NLP_MISC → NLP_CLF

---

### 7. NLP Generation — 16 projects → `gen_nlp_gen`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| Qwen3-Instruct | Chat/generation/summarization | Via Ollama local inference |
| NLLB-200 | Translation | 200+ languages, 4 target languages |
| BART | Summarization baseline | `facebook/bart-large-cnn` |

**Note:** Translation works independently of Ollama via HuggingFace NLLB pipeline

---

### 8. NLP Miscellaneous — 10 projects → `gen_nlp_gen`

**Status:** ✅ Complete (reuses `gen_nlp_gen`)

Covers: Plagiarism checker, stop words analysis, text clustering/topic modeling, WhatsApp chat analysis, cross-language IR

---

### 9. Image Classification — 31 projects → `gen_image_clf`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| DINOv3 | Primary backbone | ViT-S/14, frozen features + linear probe |
| ConvNeXt V2 | Fine-tunable alternative | Full fine-tuning head |

**Legacy removed:** VGG, ResNet, plain CNN

---

### 10. CV Object Detection — 5 projects → `gen_cv_detection`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| YOLO26m | Detection + tracking | Ultralytics API, supports detect/segment/pose/OBB |

---

### 11. CV Miscellaneous — 6 projects → `gen_cv_detection`

**Status:** ✅ Complete (reuses `gen_cv_detection`)

Covers: Dominant color, cartoonify, sketch, watermark, noise reduction

---

### 12. Face / Hand / Gesture — 12 projects → `gen_face_gesture`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| YOLO26 | Person/face detection | Replaces Haar Cascades |
| MediaPipe Face Landmarker | Face mesh (478 points) | Modern Tasks API |
| MediaPipe Hand Landmarker | Hand gesture recognition | Modern Tasks API |
| MediaPipe Pose Landmarker | Skeleton/keypoints | Modern Tasks API |
| InsightFace | Face recognition/verification | Age, gender, ethnicity |

**Legacy removed:** Haar Cascades, LBPHFaceRecognizer, old `mp.solutions` API

---

### 13. OCR / Document Understanding — 3 projects → `gen_ocr`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| PaddleOCR | Primary OCR | GPU-accelerated |
| PaddleOCR-VL-1.5 | Document parsing | Vision-language OCR |

**Legacy removed:** Tesseract

---

### 14. Captioning / Vision-Language — 3 projects → `gen_captioning`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| Qwen3-VL | Primary VLM | Multimodal understanding |
| Molmo 2 | Lightweight alternative | Efficient captioning |

---

### 15. Medical Segmentation — 2 projects → `gen_medical_seg`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| nnU-Net | Supervised baseline | U-Net architecture |
| MedSAM | Foundation model | Promptable zero-shot segmentation |

---

### 16. Recommendation Systems — 18 projects → `gen_recommendation`

**Status:** ✅ Complete

| Model | Role | Projects |
|-------|------|----------|
| implicit ALS | Collaborative filtering | 9 CF projects |
| implicit BPR | Bayesian personalized ranking | 9 CF projects |
| LightFM | Hybrid (metadata-aware) | 5 hybrid projects |
| BGE-M3 / Qwen3-Embedding | Content-based semantic | 5 content projects |
| Surprise SVD | Baseline | All projects |
| Surprise KNN | Baseline | All projects |

**Task-routed:** CF, hybrid, and content projects get different primary models

---

### 17. Time Series — 25 projects → `gen_timeseries`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| AutoGluon-TS | Primary ensemble | Foundation model ensemble |
| Chronos-Bolt | Foundation forecaster | Fast zero-shot |
| Chronos-2 | Foundation forecaster | Universal forecasting |
| TimesFM | Foundation forecaster | Google pretrained |
| LightGBM | Lag-feature baseline | GPU |
| CatBoost | Lag-feature baseline | GPU |
| XGBoost | Lag-feature baseline | CUDA |

**Legacy removed:** ARIMA, Prophet, plain LSTM, StatsForecast  
**FLAML/LazyPredict:** ❌ Not included (replaced by AutoGluon-TS)

---

### 18. Reinforcement Learning — 5 projects → `gen_rl`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| PPO | Primary (all envs) | Stable-Baselines3 |
| SAC | Continuous control | When action space is continuous |
| DQN | Discrete baseline | Comparison only |

**Legacy removed:** Plain Q-learning, manual Q-tables

---

### 19. Audio / Speech — 4 projects → `gen_audio`

**Status:** ✅ Complete

| Model | Role | Notes |
|-------|------|-------|
| Whisper large-v3-turbo | ASR / transcription | OpenAI |
| Wav2Vec2 | Audio classification | HuggingFace |
| HuBERT | Audio representation | HuggingFace |
| SepFormer | Denoising / enhancement | SpeechBrain |
| XTTS-v2 | Voice cloning / TTS | Coqui |

---

### 20. DL Image Misc — 2 projects → `gen_image_clf`

GANs, Sudoku Solver — uses DINOv3/ConvNeXt V2 backbone

### 21. DL Tabular Misc — 2 projects → `gen_tabular_reg`

Space Missions, Indian Startup — uses full tabular regression stack

### 22. DL Cluster Misc — 1 project → `gen_clustering`

Pokemon Generation Clustering — uses UMAP+HDBSCAN stack

---

## FLAML / LazyPredict Scope

| Family | FLAML | LazyPredict | Rationale |
|--------|-------|-------------|-----------|
| Tabular Classification | ✅ | ✅ | Classical supervised tabular — ideal use case |
| Tabular Regression | ✅ | ✅ | Classical supervised tabular — ideal use case |
| Fraud / Imbalanced | ❌ | ❌ | Specialized GBDT+calibration+PyOD stack |
| Anomaly Detection | ❌ | ❌ | Unsupervised — not applicable |
| Clustering | ❌ | ❌ | Unsupervised — not applicable |
| NLP Classification | ❌ | ❌ | Transformer-based — not applicable |
| NLP Generation | ❌ | ❌ | LLM-based — not applicable |
| Image Classification | ❌ | ❌ | Vision foundation models — not applicable |
| CV Detection | ❌ | ❌ | YOLO detection — not applicable |
| Face/Gesture | ❌ | ❌ | MediaPipe/InsightFace — not applicable |
| OCR | ❌ | ❌ | PaddleOCR — not applicable |
| Captioning/VLM | ❌ | ❌ | Multimodal — not applicable |
| Medical Segmentation | ❌ | ❌ | Specialized segmentation — not applicable |
| Recommendation | ❌ | ❌ | CF/hybrid/embedding — not applicable |
| Time Series | ❌ | ❌ | Foundation forecasters — not applicable |
| RL | ❌ | ❌ | Policy optimization — not applicable |
| Audio/Speech | ❌ | ❌ | Speech transformers — not applicable |

---

## Pre-existing Issues (Not Managed by Generator)

17 `Data Analysis/*/pipeline.py` files use PyCaret `import *` which causes `SyntaxError` on Python 3.13. These are **not generated** by `_overhaul_v2.py` and are outside migration scope.

---

## Verification

- **315/315 generated pipelines compile clean** (`py_compile`)
- **17/17 Data Analysis pipelines** have pre-existing PyCaret errors (out of scope)
- **All model stacks verified:** no legacy KNN/SVM/RandomForest/DecisionTree/AdaBoost/MLP as primary models in any generated pipeline
- **Naive Bayes** appears only as TF-IDF baseline in NLP classification pipelines (40 projects) — intentional, not primary
- **FLAML/LazyPredict** confirmed present only in `gen_tabular_clf` and `gen_tabular_reg`
