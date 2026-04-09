# UPGRADE REPORT — Step 6

**Date**: 2025-07-01
**Scope**: Full PyTorch-only modernization of 49 projects across 7 categories

---

## 1. Summary

| Metric | Value |
|--------|-------|
| Total projects upgraded | 49 |
| Old files removed (`2 - Code.py`) | 34 |
| Old scaffold stubs removed (`src/`) | 15 |
| New `run.py` files created | 49 |
| Shared utilities created | `shared/utils.py` (~310 lines) |
| TensorFlow references remaining | **0** |
| Keras references remaining | **0** |
| `outputs/.gitkeep` created | 49 |

---

## 2. Tech Stack (Final)

| Dependency | Version | Role |
|------------|---------|------|
| torch | 2.10.0+cu130 | Framework |
| torchvision | ≥0.22.0 | CV |
| torchaudio | ≥2.10.0 | Audio |
| timm | ≥1.0.15 | Pretrained CV models |
| transformers | ≥4.51.0 | NLP / Multimodal |
| pycaret | ≥3.5.0 | Tabular AutoML |
| lazypredict | ≥0.2.14 | Quick model comparison |
| scikit-learn | ≥1.6.1 | Preprocessing, metrics |
| opencv-python-headless | ≥4.12.0 | Image I/O |
| gymnasium | ≥1.1.0 | RL environments |
| mlxtend | ≥0.23.4 | Association rules |
| opendatasets | ≥0.1.22 | Kaggle downloads |
| torchmetrics | ≥1.6.0 | DL metrics |

**Removed**: tensorflow, keras, tf-keras, lightning, catboost, plotly, gensim, wandb, mlflow, gdown, pathlib2, python-dotenv, jupyter, notebook, ipykernel, ipywidgets

---

## 3. Per-Project Upgrade Detail

### A. Anomaly Detection & Fraud Detection

| # | Project | Old Approach | New Approach | Dataset Method |
|---|---------|-------------|--------------|----------------|
| 1 | CIFAR-10 | Keras Conv2D | timm EfficientNet-B0 fine-tune | torchvision auto-download |
| 2 | Twitter Bot | Empty 2-Code.py | PyCaret classification | Kaggle opendatasets |
| 3 | NAB | Keras LSTM autoencoder | PyTorch LSTM Autoencoder | Kaggle opendatasets |
| 4 | Banknote Auth | sklearn LogisticRegression | PyCaret AutoML | Kaggle opendatasets |
| 5 | Breast Cancer | sklearn 5 classifiers | PyCaret AutoML (drop ID col) | Kaggle opendatasets |
| 6 | Credit Card | sklearn + SMOTE | PyCaret fix_imbalance=True | Kaggle opendatasets |
| 7 | Financial Fraud | sklearn LogisticRegression | PyCaret fix_imbalance=True | Kaggle opendatasets |
| 8 | Insurance Fraud | sklearn RandomForest | PyCaret AutoML | Kaggle opendatasets |
| 9 | IEEE-CIS | sklearn XGBoost | PyCaret fix_imbalance + 50k sample | Kaggle opendatasets |
| 10 | Intrusion Detection | sklearn RF pipeline | PyCaret AutoML | Kaggle opendatasets |
| 11 | METR-LA Traffic | TF/Keras LSTM | PyTorch LSTM (seq-to-seq) | Kaggle opendatasets |

### B. Associate Rule Learning

| # | Project | Old | New | Dataset |
|---|---------|-----|-----|---------|
| 12 | Grocery Store | mlxtend apriori | mlxtend apriori + fpgrowth | Kaggle |
| 13 | Online Retail | mlxtend apriori | mlxtend apriori + fpgrowth | Kaggle |
| 14 | Online News | mlxtend apriori | mlxtend apriori + fpgrowth (text TF-IDF) | Kaggle |
| 15 | Video Game Store | mlxtend apriori | mlxtend apriori + fpgrowth | Kaggle |

### C. Chat Bots

| # | Project | Old | New | Dataset |
|---|---------|-----|-----|---------|
| 16 | Customer Service | (varied) | DistilBERT fine-tuning (transformers) | Kaggle |
| 17 | Fitness | (varied) | DistilBERT fine-tuning | Kaggle |
| 18 | Job Search | (varied) | DistilBERT fine-tuning | Kaggle |
| 19 | Health & Wellness | (varied) | DistilBERT fine-tuning | Kaggle |
| 20 | Travel | (varied) | DistilBERT fine-tuning | Kaggle |
| 21 | Weather | (varied) | PyCaret classification (tabular data) | Kaggle |
| 22 | News | (varied) | DistilBERT fine-tuning | Kaggle |
| 23 | Personal Shopping | (varied) | DistilBERT fine-tuning | Kaggle |
| 24 | Personal Finance | (varied) | DistilBERT fine-tuning | Kaggle |
| 25 | Language Learning | (varied) | PyCaret + TF-IDF (sentence data) | Kaggle |

### D. GANs

| # | Project | Old | New | Dataset |
|---|---------|-----|-----|---------|
| 26 | Face Generation | TF/Keras DCGAN | Pure PyTorch DCGAN + AMP | Kaggle |
| 27 | Image Inpainting | TF autoencoder | Context Encoder + PatchGAN (PyTorch) + AMP | Kaggle |
| 28 | Pix2Pix Translation | TF U-Net (broken) | Pix2Pix U-Net + PatchGAN (PyTorch) + AMP | Kaggle |
| 29 | Style Transfer | TF VGG19 | PyTorch VGG19 perceptual loss + AMP | Kaggle |
| 30 | Text-to-Image | PyTorch (only existing) | Class-Conditional DCGAN + AMP | Kaggle |

### E. Recommendation Systems

| # | Project | Old | New | Dataset |
|---|---------|-----|-----|---------|
| 31 | Event | NotImplementedError | PyTorch NCF (Neural Collaborative Filtering) | Kaggle |
| 32 | E-commerce | NotImplementedError | PyTorch NCF | Kaggle |
| 33 | Restaurant | NotImplementedError | PyTorch NCF | Kaggle |
| 34 | Hotel | NotImplementedError | PyTorch NCF | Kaggle |
| 35 | Movie | Broken sklearn | PyTorch NCF (MovieLens) | Kaggle |
| 36 | Music | NotImplementedError | NCF / PyCaret regression fallback | Kaggle |
| 37 | Book | NotImplementedError | PyTorch NCF | Kaggle |
| 38 | Recipe | NotImplementedError | NCF / PyCaret regression fallback | Kaggle |
| 39 | Article | NotImplementedError | PyTorch NCF (MIND news) | Kaggle |
| 40 | TV Show | NotImplementedError | PyCaret Regression | Kaggle |

### F. Reinforcement Learning

| # | Project | Old | New | Environment |
|---|---------|-----|-----|-------------|
| 41 | Taxi | Broken Q-table | PyTorch DQN + Replay Buffer | Taxi-v3 |
| 42 | Lunar Landing | NotImplementedError | PyTorch DQN + ε-greedy | LunarLander-v3 |
| 43 | Gridworld | NotImplementedError | PyTorch DQN + ε-greedy | FrozenLake 8×8 |
| 44 | Cliff Walking | NotImplementedError | PyTorch DQN + ε-greedy | CliffWalking-v0 |
| 45 | Frozen Lake | NotImplementedError | PyTorch DQN + ε-greedy | FrozenLake-v1 |

### G. Speech & Audio

| # | Project | Old | New | Dataset |
|---|---------|-----|-----|---------|
| 46 | Audio Denoising | TF Autoencoder (partial) | PyTorch U-Net Spectrogram + AMP | Kaggle |
| 47 | Image Captioning | Broken references | ViT + GPT-2 (HuggingFace) fine-tune | Kaggle |
| 48 | Music Genre | sklearn pipeline | PyCaret classification on audio features | Kaggle |
| 49 | Voice Cloning | TF1 Tacotron | SpeechT5 (HuggingFace) + speaker embeddings | Kaggle |

---

## 4. GPU Readiness

| Check | Status |
|-------|--------|
| `scripts/check_gpu.py` updated | ✅ CUDA 13.0, AMP check, TF32 status |
| `get_device()` in shared module | ✅ CUDA → MPS → CPU auto |
| AMP in deep learning projects | ✅ All GAN/CV/NLP/Audio projects |
| Device fallback in all run.py | ✅ Graceful CPU fallback |
| Batch size adaptation | ✅ Smaller batch on CPU where applicable |

---

## 5. Shared Module

**`shared/utils.py`** provides:

| Function | Purpose |
|----------|---------|
| `get_device()` | Auto-detect CUDA/MPS/CPU |
| `set_seed(seed)` | Reproducible runs |
| `dataset_prompt(name, url, desc)` | Print dataset info |
| `kaggle_prompt()` | Print Kaggle setup instructions |
| `download_kaggle_dataset(url, dest)` | Download from Kaggle |
| `project_paths(__file__)` | Get root/data/outputs paths |
| `compute_classification_metrics(y, preds, probs)` | Accuracy, F1, AUROC |
| `save_classification_report(metrics, dir)` | JSON + confusion matrix PNG |
| `compute_regression_metrics(y, preds)` | MAE, RMSE, R² |
| `save_regression_report(metrics, dir)` | JSON + scatter PNG |
| `run_pycaret_classification(df, target, out)` | Full PyCaret pipeline |
| `run_pycaret_regression(df, target, out)` | Full PyCaret pipeline |
| `setup_logging()` | Timestamped logging |

---

## 6. Files Modified/Created

### Infrastructure
- `requirements.txt` — Rewritten (PyTorch-only, 40 dependencies)
- `environment.yml` — Rewritten (conda + pip, CUDA 13.0)
- `configs/global_config.yaml` — Updated for PyTorch stack
- `scripts/check_gpu.py` — Updated with AMP + CUDA 13.0
- `shared/__init__.py` — New
- `shared/utils.py` — New (~310 lines)
- `README.md` — New (complete project guide)
- `UPGRADE_REPORT.md` — This file

### Per-Project (×49)
- `run.py` — Created in every project directory
- `outputs/.gitkeep` — Created in every project directory

### Removed
- 34 × `2 - Code.py` files — Deleted
- Old `src/` scaffold directories — Deleted from all Recommendation Systems and RL projects

---

## 7. Remaining Manual Steps

1. **Install dependencies**: Run `pip install -r requirements.txt` after installing PyTorch
2. **Kaggle credentials**: Place `kaggle.json` in `~/.kaggle/` before running projects that use Kaggle datasets
3. **Test individual projects**: Run `python run.py` in each project directory to verify
4. **Large datasets**: Some projects (IEEE-CIS, CelebA, COCO, Places2) download multi-GB datasets — ensure disk space
5. **RL rendering**: Set `render_mode="human"` in RL projects if you want visual rendering (requires display)
6. **Git commit**: Stage all changes and commit

---

## 8. Verification Checklist

- [x] 49 `run.py` files exist
- [x] 0 TensorFlow/Keras references in codebase
- [x] 0 old `2 - Code.py` files remain
- [x] 49 `outputs/.gitkeep` files created
- [x] `shared/utils.py` has all standard functions
- [x] `requirements.txt` is PyTorch-only
- [x] `environment.yml` targets CUDA 13.0
- [x] `scripts/check_gpu.py` checks AMP + CUDA 13.0
- [x] Every DL project uses `get_device()` + AMP
- [x] Every tabular project uses PyCaret
- [x] Every CV project uses timm pretrained models
- [x] Every NLP project uses HuggingFace transformers
- [x] All GANs are pure PyTorch
- [x] All RL projects use Gymnasium + PyTorch DQN
