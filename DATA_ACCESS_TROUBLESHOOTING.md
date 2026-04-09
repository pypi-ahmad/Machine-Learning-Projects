# Data Access Troubleshooting

This document explains how to resolve dataset download failures for projects in
this workspace.  Every project's `run.py` will **still produce
`outputs/metrics.json`** even when data cannot be downloaded — the status field
will explain what went wrong and how to fix it.

---

## 1. Kaggle API Token

All Kaggle-sourced projects need a valid API token.  Set **one** of:

| Method | Details |
|--------|---------|
| **Env var (recommended)** | `export KAGGLE_API_TOKEN=KGAT_xxxx…` (or `$env:KAGGLE_API_TOKEN = "…"` on Windows) |
| **kaggle.json** | Save to `~/.kaggle/kaggle.json` (`{"username":"…","key":"…"}`) |
| **Env vars** | `KAGGLE_USERNAME` + `KAGGLE_KEY` |

Get a token at <https://www.kaggle.com> → Account → API → *Create New Token*.

---

## 2. Competition Rules — 401 Unauthorized (5 projects)

These datasets are hosted as **Kaggle Competitions** and require you to accept
the competition rules before the API can download them.

| # | Project | Competition | Action |
|---|---------|-------------|--------|
| P10 | Advanced ResNet50 (Plant Pathology) | [plant-pathology-2021-fgvc8](https://www.kaggle.com/c/plant-pathology-2021-fgvc8) | Visit [Rules](https://www.kaggle.com/c/plant-pathology-2021-fgvc8/rules) and click **"I Understand and Accept"** |
| P11 | Cat Vs Dog | [dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats) | Visit [Rules](https://www.kaggle.com/c/dogs-vs-cats/rules) and accept |
| P12 | Keep Babies Safe | [state-farm-distracted-driver-detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) | Visit [Rules](https://www.kaggle.com/c/state-farm-distracted-driver-detection/rules) and accept |
| P18 | Diabetic Retinopathy | [diabetic-retinopathy-detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) | Visit [Rules](https://www.kaggle.com/c/diabetic-retinopathy-detection/rules) and accept |
| P47 | Cactus or Not Cactus | [aerial-cactus-identification](https://www.kaggle.com/c/aerial-cactus-identification) | Visit [Rules](https://www.kaggle.com/c/aerial-cactus-identification/rules) and accept |

**After accepting**, re-run:

```bash
python "Deep Learning Projects 10 - Advanced rsnet50/run.py" --smoke-test
```

---

## 3. Restricted / Deprecated Datasets — 403 Forbidden (9 projects)

These datasets have been restricted, deprecated, or delisted on Kaggle.
The Kaggle API returns **403 Forbidden**. To fix, either:

* **Find an alternative dataset** (see suggestions below), or
* **Download manually** from an archive / mirror and place files in the
  project's `data/` directory.

| # | Project | Kaggle Slug | Alternative |
|---|---------|-------------|-------------|
| P15 | Happy House Predictor | `uciml/boston-housing-dataset` | Use `sklearn.datasets.fetch_california_housing()` or [UCI ML](https://archive.ics.uci.edu/dataset/228/housing) |
| P27 | Concrete Strength | `uciml/concrete-compressive-strength-data-set` | [UCI ML Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) |
| P29 | Indian Startup Analysis | `ruchi798/startup-investments-crunchbase` | Search for "startup funding" on Kaggle or data.world |
| P30 | Amazon Stock Price | `rohanrao/amazon-stock-price` | Download from Yahoo Finance (`AMZN` ticker) or try `yfinance` Python package |
| P31 | Dance Form Identification | `arjunbhasin2013/indian-classical-dance` | Search Kaggle for "indian dance form" or use `google-images-download` |
| P32 | Glass or No Glass Detector | `jehanbhathena/eyeglasses-dataset` | Search Kaggle for "eyeglasses" or "glasses detection" |
| P37 | Sheep Breed Classification | `warcoder/sheep-face-images` | Search Kaggle for "sheep" or "animal faces" |
| P42 | Bottle or Cans Classifier | `trolukovich/bottles-and-cans` | Search Kaggle for "bottles cans" or similar |
| P44 | Image Colorization | `landrykezebou/vizwiz-colorization` | Use any colour-image dataset (e.g., CIFAR-10 via torchvision, or ImageNet subset) |

---

## 4. GCS-Only Dataset — P4 Landmark Detection

Project 4 uses the **Google Landmarks Dataset**, which only provides CSV
metadata on Kaggle. The actual images must be downloaded separately from
Google Cloud Storage.

**Steps:**

1. The Kaggle download (`google/google-landmarks-dataset`) gives you CSV files
   with image URLs.
2. Follow the official guide at
   <https://github.com/cvdfoundation/google-landmark> to download the images
   from GCS.
3. Place the downloaded images under
   `Deep Learning Projects 4 - Landmark Detection Model/data/`.

This project is marked as `dataset_missing` in smoke tests until images are
manually downloaded.

---

## 5. Smoke Test Status Codes

When a project cannot access its data, `run.py` writes `outputs/metrics.json`
with one of these statuses:

| Status | Meaning |
|--------|---------|
| `ok` | Training completed successfully |
| `dataset_missing` | Dataset cannot be downloaded (403 / 404 / GCS-only) |
| `manual_action_required` | Competition rules need accepting (401) |
| `auth_error` | Kaggle credentials missing or invalid |
| `missing_dependency` | Required Python package not installed |
| `download_error` | Network / transient download failure |
| `error` | Unhandled exception (code bug) |

The smoke-test runner (`scripts/smoke_test_all.py`) classifies:

* `ok` → **PASS**
* `dataset_missing`, `manual_action_required`, etc. → **EXPECTED**
* No `metrics.json` or unhandled crash → **FAIL**

A fully green CI run means **0 FAIL** — all failures are accounted for.
