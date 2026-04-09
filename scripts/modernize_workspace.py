#!/usr/bin/env python3
"""
Modernize Workspace
===================
Rewrites global config files and generates run.py + README.md for all 50 projects.

* PyTorch 2.10.0 (cu130) — NO TensorFlow / Keras anywhere
* timm for CV, HuggingFace for NLP, PyCaret for tabular
* Mixed-precision (AMP), cosine schedule, AdamW

Usage:
    python scripts/modernize_workspace.py
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def w(rel_path: str, content: str) -> None:
    """Write *content* to ROOT / rel_path, creating parents."""
    p = ROOT / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(f"  + {rel_path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. GLOBAL FILES                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

REQUIREMENTS_TXT = """\
# ==============================================================================
# Deep Learning Projects Monorepo - Requirements (PyTorch-only)
# ==============================================================================
# Step 1 — install PyTorch with CUDA 13.0 support:
#   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
#
# Step 2 — install everything else:
#   pip install -r requirements.txt
# ==============================================================================

# -- CV / Transfer Learning ---------------------------------------------------
timm>=1.0.12
opencv-python>=4.12.0
Pillow>=11.0.0
albumentations>=2.0.0

# -- NLP / HuggingFace --------------------------------------------------------
transformers>=4.47.0
datasets>=3.2.0
accelerate>=1.2.0
evaluate>=0.4.3
tokenizers>=0.21.0

# -- Metrics -------------------------------------------------------------------
torchmetrics>=1.6.0
scikit-learn>=1.6.0

# -- AutoML (tabular) ---------------------------------------------------------
pycaret>=3.4.0
lazypredict>=0.2.12

# -- Data / Numerical ---------------------------------------------------------
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.14.0

# -- Visualization -------------------------------------------------------------
matplotlib>=3.9.0
seaborn>=0.13.0

# -- Utilities -----------------------------------------------------------------
tqdm>=4.67.0
pyyaml>=6.0.2
requests>=2.32.0
kaggle>=1.6.0
gdown>=5.2.0

# -- Audio ---------------------------------------------------------------------
librosa>=0.10.2

# -- Notebook ------------------------------------------------------------------
jupyter>=1.1.0
ipykernel>=6.29.0
openpyxl>=3.1.5
"""

ENVIRONMENT_YML = """\
# ==============================================================================
# Conda Environment — PyTorch-only stack
# ==============================================================================
# conda env create -f environment.yml && conda activate dl-projects
# ==============================================================================
name: dl-projects
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pip>=24.0
  - pytorch>=2.10.0
  - torchvision>=0.20.0
  - torchaudio>=2.5.0
  - pytorch-cuda=13.0
  - numpy>=1.26.0
  - pandas>=2.2.0
  - scipy>=1.14.0
  - scikit-learn>=1.6.0
  - matplotlib>=3.9.0
  - seaborn>=0.13.0
  - opencv>=4.12.0
  - pillow>=11.0.0
  - tqdm>=4.67.0
  - pyyaml>=6.0.2
  - requests>=2.32.0
  - jupyter>=1.1.0
  - ipykernel>=6.29.0
  - pip:
      - timm>=1.0.12
      - transformers>=4.47.0
      - datasets>=3.2.0
      - accelerate>=1.2.0
      - evaluate>=0.4.3
      - torchmetrics>=1.6.0
      - pycaret>=3.4.0
      - lazypredict>=0.2.12
      - kaggle>=1.6.0
      - gdown>=5.2.0
      - albumentations>=2.0.0
      - librosa>=0.10.2
      - openpyxl>=3.1.5
"""

GLOBAL_CONFIG_YAML = """\
# ==============================================================================
# Global Configuration — PyTorch-only stack
# ==============================================================================
device: "cuda"
precision: "amp"              # automatic mixed precision
seed: 42
deterministic: true

batch_size: 32
learning_rate: 0.0001
num_epochs: 15
optimizer: "adamw"
weight_decay: 0.0001
scheduler: "cosine"

num_workers: 4
pin_memory: true

train_split: 0.70
val_split: 0.15
test_split: 0.15

log_level: "INFO"
save_top_k: 1
monitor_metric: "val_acc"
monitor_mode: "max"

image:
  size: 224
  channels: 3
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]

paths:
  data: "data"
  outputs: "outputs"
"""

CHECK_GPU_PY = '''\
#!/usr/bin/env python3
"""GPU / CUDA Validation — PyTorch-only stack.

Usage:  python scripts/check_gpu.py
"""
import sys


def check_gpu() -> bool:
    print("=" * 60)
    print("  GPU / CUDA Validation Report")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("\\n[FAIL] PyTorch not installed.")
        print("       pip3 install torch torchvision torchaudio "
              "--index-url https://download.pytorch.org/whl/cu130")
        return False

    print(f"\\nPyTorch version : {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available  : {cuda}")
    if not cuda:
        print("\\n[WARN] CUDA NOT available. Training will use CPU only.")
        return False

    print(f"CUDA version    : {torch.version.cuda}")
    print(f"cuDNN version   : {torch.backends.cudnn.version()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"\\n  GPU {i}: {p.name}  |  {p.total_mem / 1024**3:.1f} GB  "
              f"|  CC {p.major}.{p.minor}")

    # functional test
    try:
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        print(f"\\n  MatMul test: OK (sum={c.sum().item():.2f})")
    except Exception as e:
        print(f"\\n  [FAIL] {e}")
        return False

    print("\\n" + "=" * 60)
    print("  [PASS] GPU is fully functional for PyTorch.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    sys.exit(0 if check_gpu() else 1)
'''


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. RUN.PY TEMPLATES                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# --- CV Classification -------------------------------------------------------
CV_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Model   : {model} (timm)
Task    : Image Classification
Usage   : python run.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.utils import (seed_everything, get_device, dataset_prompt,
                           kaggle_download, ensure_dir)
from shared.cv import (create_dataloaders, build_timm_model,
                        train_model, evaluate_model)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}
MODEL   = {model_repr}
CLASSES = {num_classes}
EPOCHS  = {epochs}


def get_data():
    """Download and prepare the dataset."""
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    seed_everything(42)
    device = get_device()
    ensure_dir(OUTPUT_DIR)

    data_root = get_data()
    train_dl, val_dl, test_dl, class_names = create_dataloaders(
        data_root, img_size=224, batch_size=32,
    )
    model = build_timm_model(MODEL, num_classes=CLASSES or len(class_names))
    model = train_model(
        model, train_dl, val_dl,
        epochs=EPOCHS, lr=1e-4, device=device, output_dir=OUTPUT_DIR,
    )
    evaluate_model(model, test_dl, class_names, device=device,
                   output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

# --- Tabular Classification --------------------------------------------------
TAB_CLS_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Task    : Tabular Classification (PyCaret AutoML)
Usage   : python run.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (seed_everything, dataset_prompt,
                           kaggle_download, ensure_dir)
from shared.tabular import run_pycaret_classification

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}
TARGET  = {target_repr}


def get_data() -> pd.DataFrame:
    """Download and load the dataset."""
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    seed_everything(42)
    ensure_dir(OUTPUT_DIR)
    df = get_data()
    print(f"  Shape: {{df.shape}}  |  Target: {{TARGET}}")
    print(f"  Classes: {{df[TARGET].nunique()}}")
    run_pycaret_classification(df, target=TARGET, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

# --- Tabular Regression ------------------------------------------------------
TAB_REG_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Task    : Tabular Regression (PyCaret AutoML)
Usage   : python run.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (seed_everything, dataset_prompt,
                           kaggle_download, ensure_dir)
from shared.tabular import run_pycaret_regression

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}
TARGET  = {target_repr}


def get_data() -> pd.DataFrame:
    """Download and load the dataset."""
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    seed_everything(42)
    ensure_dir(OUTPUT_DIR)
    df = get_data()
    print(f"  Shape: {{df.shape}}  |  Target: {{TARGET}}")
    run_pycaret_regression(df, target=TARGET, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

# --- Tabular Clustering ------------------------------------------------------
TAB_CLUST_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Task    : Tabular Clustering (PyCaret)
Usage   : python run.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (seed_everything, dataset_prompt,
                           kaggle_download, ensure_dir)
from shared.tabular import run_pycaret_clustering

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}


def get_data() -> pd.DataFrame:
    """Download and load the dataset."""
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    seed_everything(42)
    ensure_dir(OUTPUT_DIR)
    df = get_data()
    print(f"  Shape: {{df.shape}}")
    run_pycaret_clustering(df, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

# --- NLP Classification ------------------------------------------------------
NLP_TPL = '''\
#!/usr/bin/env python3
"""
Project {num} -- {title}

Dataset : {dataset}
Model   : {hf_model} (HuggingFace)
Task    : Text Classification
Usage   : python run.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from shared.utils import (seed_everything, get_device, dataset_prompt,
                           kaggle_download, ensure_dir)
from shared.nlp import (build_hf_classifier, tokenize_texts,
                         train_hf_classifier, evaluate_hf_classifier)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "outputs"

DATASET = {dataset_repr}
LINKS   = {links_repr}
KAGGLE  = {kaggle_repr}
HF_MODEL = {hf_model_repr}
NUM_LABELS = {num_labels}


def get_data():
    """Download, load and split text data.

    Returns (train_texts, train_labels, test_texts, test_labels, class_names).
    """
    dataset_prompt(DATASET, LINKS)
{get_data_body}


def main():
    seed_everything(42)
    device = get_device()
    ensure_dir(OUTPUT_DIR)

    train_texts, train_labels, test_texts, test_labels, class_names = get_data()
    print(f"  Train: {{len(train_texts)}}  |  Test: {{len(test_texts)}}  "
          f"|  Classes: {{class_names}}")

    model, tokenizer = build_hf_classifier(HF_MODEL, num_labels=len(class_names))
    train_ds = tokenize_texts(train_texts, train_labels, tokenizer)
    test_ds  = tokenize_texts(test_texts, test_labels, tokenizer)

    trainer = train_hf_classifier(model, train_ds, test_ds, OUTPUT_DIR)
    evaluate_hf_classifier(trainer, test_ds, class_names, OUTPUT_DIR)


if __name__ == "__main__":
    main()
'''

# --- README template ----------------------------------------------------------
README_TPL = '''\
# Project {num} -- {title}

## Dataset
{dataset}

{links_md}

## Stack
| Component | Choice |
|-----------|--------|
| Framework | PyTorch 2.10.0 (cu130) |
| Model     | {model_desc} |
| Task      | {task_desc} |
| AutoML    | {automl_desc} |

## Usage

```bash
# 1. Install PyTorch (once)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 2. Install dependencies (once)
pip install -r requirements.txt

# 3. Run this project
python run.py
```

## Outputs
After training, check `outputs/` for:
- `best_model.pth` (or `.pkl` for tabular) -- saved model weights
- `metrics.json` -- accuracy, F1, etc.
- `training_curves.png` -- loss / accuracy over epochs
- `confusion_matrix.png` -- per-class performance
- `classification_report.txt` -- detailed metrics
'''


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. HELPER FACTORIES for get_data bodies                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _cv_kaggle(subdir: str | None = None, competition: bool = False) -> str:
    """Generate get_data body for Kaggle-sourced ImageFolder CV datasets."""
    comp = ", competition=True" if competition else ""
    if subdir:
        return (
            f'    target = DATA_DIR / "{subdir}"\n'
            f'    if not target.exists():\n'
            f'        kaggle_download(KAGGLE, DATA_DIR{comp})\n'
            f'    return target'
        )
    return (
        f'    if not list(DATA_DIR.rglob("*.jpg")) and not list(DATA_DIR.rglob("*.png")):\n'
        f'        kaggle_download(KAGGLE, DATA_DIR{comp})\n'
        f'    # auto-detect data root\n'
        f'    for child in sorted(DATA_DIR.iterdir()):\n'
        f'        if child.is_dir() and child.name != "__MACOSX":\n'
        f'            return child\n'
        f'    return DATA_DIR'
    )


def _tab_kaggle(csv_hint: str | None = None) -> str:
    """Generate get_data body for Kaggle CSV tabular datasets."""
    if csv_hint:
        return (
            f'    csv = DATA_DIR / "{csv_hint}"\n'
            f'    if not csv.exists():\n'
            f'        kaggle_download(KAGGLE, DATA_DIR)\n'
            f'        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            f'        csv = csvs[0] if csvs else csv\n'
            f'    return pd.read_csv(csv)'
        )
    return (
        '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
        '    if not csvs:\n'
        '        kaggle_download(KAGGLE, DATA_DIR)\n'
        '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
        '    return pd.read_csv(csvs[0])'
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. PROJECT DEFINITIONS  (50 entries)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

PROJECTS: list[dict] = [
    # ── P1  Pneumonia Detection ──────────────────────────────────────────────
    dict(
        num=1,
        folder="Deep Learning Projects 1 - Pnemonia Detection",
        title="Pneumonia Detection",
        type="cv",
        dataset="Chest X-Ray Pneumonia",
        links=["https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"],
        kaggle="paultimothymooney/chest-xray-pneumonia",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=2,
        epochs=15,
        get_data_body=_cv_kaggle("chest_xray"),
    ),

    # ── P2  Face Mask Detection ──────────────────────────────────────────────
    dict(
        num=2,
        folder="Deep Learning Projects 2 - Face Mask Detection",
        title="Face Mask Detection",
        type="cv",
        dataset="Face Mask 12K",
        links=["https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset"],
        kaggle="ashishjangra27/face-mask-12k-images-dataset",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=2,
        epochs=10,
        get_data_body=_cv_kaggle("Face Mask Dataset"),
    ),

    # ── P3  Earthquake Prediction ────────────────────────────────────────────
    dict(
        num=3,
        folder="Deep Learning Projects 3 - Earthquack Prediction model",
        title="Earthquake Prediction",
        type="tabular_reg",
        dataset="Earthquake Prediction",
        links=["https://www.kaggle.com/datasets/henryshan/earthquake-prediction"],
        kaggle="henryshan/earthquake-prediction",
        target="magnitude",
        get_data_body=_tab_kaggle(),
    ),

    # ── P4  Landmark Detection ───────────────────────────────────────────────
    dict(
        num=4,
        folder="Deep Learning Projects 4 - Landmark Detection Model",
        title="Landmark Detection",
        type="cv",
        dataset="Google Landmarks Dataset",
        links=["https://www.kaggle.com/datasets/google/google-landmarks-dataset"],
        kaggle="google/google-landmarks-dataset",
        model="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        num_classes=0,  # auto from dataset
        epochs=15,
        get_data_body=_cv_kaggle(),
    ),

    # ── P5  Chatbot / Intent Classification ──────────────────────────────────
    dict(
        num=5,
        folder="Deep Learning Projects 5 - Chatbot With Deep Learning",
        title="Chatbot Intent Classification",
        type="nlp",
        dataset="Chatbot Intent Recognition",
        links=["https://www.kaggle.com/datasets/elvinagammed/chatbots-intent-recognition-dataset"],
        kaggle="elvinagammed/chatbots-intent-recognition-dataset",
        hf_model="distilbert-base-uncased",
        num_labels=0,
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    # Expect columns like "text"/"query" and "intent"/"label"\n'
            '    text_col = [c for c in df.columns if c.lower() in ("text","query","utterance","sentence")][0]\n'
            '    label_col = [c for c in df.columns if c.lower() in ("intent","label","category","tag")][0]\n'
            '    labels_unique = sorted(df[label_col].unique())\n'
            '    lab2id = {l: i for i, l in enumerate(labels_unique)}\n'
            '    df["_label"] = df[label_col].map(lab2id)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr[text_col].tolist(), tr["_label"].tolist(),\n'
            '            te[text_col].tolist(), te["_label"].tolist(), labels_unique)'
        ),
    ),

    # ── P6  Movie Title / Genre Prediction ───────────────────────────────────
    dict(
        num=6,
        folder="Deep Learning Projects 6 - Movies Title Prediction",
        title="Movie Genre Prediction",
        type="nlp",
        dataset="Wikipedia Movie Plots",
        links=["https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots"],
        kaggle="jrobischon/wikipedia-movie-plots",
        hf_model="distilbert-base-uncased",
        num_labels=0,
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    # Use top-10 genres for tractability\n'
            '    top = df["Genre"].value_counts().head(10).index.tolist()\n'
            '    df = df[df["Genre"].isin(top)].copy()\n'
            '    labels = sorted(top)\n'
            '    lab2id = {l: i for i, l in enumerate(labels)}\n'
            '    df["_label"] = df["Genre"].map(lab2id)\n'
            '    # Truncate plot to first 512 chars for speed\n'
            '    df["_text"] = df["Plot"].str[:512]\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["_text"].tolist(), tr["_label"].tolist(),\n'
            '            te["_text"].tolist(), te["_label"].tolist(), labels)'
        ),
    ),

    # ── P7  Churn Modeling ───────────────────────────────────────────────────
    dict(
        num=7,
        folder="Deep Learning Projects 7 - Advanced Churn Modeling",
        title="Customer Churn Prediction",
        type="tabular_cls",
        dataset="Churn Modelling",
        links=["https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling"],
        kaggle="shrutimechlearn/churn-modelling",
        target="Exited",
        get_data_body=_tab_kaggle("Churn_Modelling.csv"),
    ),

    # ── P8  Disease Prediction ───────────────────────────────────────────────
    dict(
        num=8,
        folder="Deep Learning Projects 8 - Disease Prediction Model",
        title="Disease Prediction",
        type="tabular_cls",
        dataset="Disease Prediction Using ML",
        links=["https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning"],
        kaggle="kaushil268/disease-prediction-using-machine-learning",
        target="prognosis",
        get_data_body=_tab_kaggle(),
    ),

    # ── P9  IMDB Sentiment ───────────────────────────────────────────────────
    dict(
        num=9,
        folder="Deep Learning Projects 9 - IMDB Sentiment Analysis using Deep Learning",
        title="IMDB Sentiment Analysis",
        type="nlp",
        dataset="IMDB 50K Movie Reviews",
        links=["https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"],
        kaggle="lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        hf_model="distilbert-base-uncased",
        num_labels=2,
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    lab_map = {"positive": 1, "negative": 0}\n'
            '    df["_label"] = df["sentiment"].map(lab_map)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["review"].tolist(), tr["_label"].tolist(),\n'
            '            te["review"].tolist(), te["_label"].tolist(),\n'
            '            ["negative", "positive"])'
        ),
    ),

    # ── P10  Plant Pathology / ResNet50 ──────────────────────────────────────
    dict(
        num=10,
        folder="Deep Learning Projects 10 - Advanced rsnet50",
        title="Plant Pathology (ResNet50 -> ConvNeXt)",
        type="cv",
        dataset="Plant Pathology 2021",
        links=["https://www.kaggle.com/c/plant-pathology-2021-fgvc8"],
        kaggle="plant-pathology-2021-fgvc8",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=0,
        epochs=15,
        get_data_body=_cv_kaggle(competition=True),
    ),

    # ── P11  Cat Vs Dog ──────────────────────────────────────────────────────
    dict(
        num=11,
        folder="Deep Learning Projects 11 - Cat Vs Dog",
        title="Cat vs Dog Classification",
        type="cv",
        dataset="Dogs vs Cats",
        links=["https://www.kaggle.com/c/dogs-vs-cats"],
        kaggle="dogs-vs-cats",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=2,
        epochs=10,
        get_data_body=(
            '    import shutil\n'
            '    target = DATA_DIR / "organized"\n'
            '    if not target.exists():\n'
            '        kaggle_download(KAGGLE, DATA_DIR, competition=True)\n'
            '        # Reorganize flat cat.*.jpg / dog.*.jpg into class dirs\n'
            '        for split, ratio in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:\n'
            '            (target / split / "cat").mkdir(parents=True, exist_ok=True)\n'
            '            (target / split / "dog").mkdir(parents=True, exist_ok=True)\n'
            '        src = DATA_DIR / "train"\n'
            '        if not src.is_dir():\n'
            '            src = DATA_DIR\n'
            '        cats = sorted(src.glob("cat*.jpg"))\n'
            '        dogs = sorted(src.glob("dog*.jpg"))\n'
            '        import random; random.seed(42)\n'
            '        random.shuffle(cats); random.shuffle(dogs)\n'
            '        for imgs, cls in [(cats,"cat"), (dogs,"dog")]:\n'
            '            n = len(imgs)\n'
            '            splits = {"train": imgs[:int(.8*n)], "val": imgs[int(.8*n):int(.9*n)], "test": imgs[int(.9*n):]}\n'
            '            for sp, files in splits.items():\n'
            '                for f in files:\n'
            '                    shutil.copy2(f, target / sp / cls / f.name)\n'
            '    return target'
        ),
    ),

    # ── P12  Keep Babies Safe / Distracted Driver ────────────────────────────
    dict(
        num=12,
        folder="Deep Learning Projects 12 - Keep Babies Safe",
        title="Distracted Driver Detection",
        type="cv",
        dataset="State Farm Distracted Driver Detection",
        links=["https://www.kaggle.com/c/state-farm-distracted-driver-detection"],
        kaggle="state-farm-distracted-driver-detection",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=10,
        epochs=15,
        get_data_body=_cv_kaggle(competition=True),
    ),

    # ── P13  COVID Drug Recovery ─────────────────────────────────────────────
    dict(
        num=13,
        folder="Deep Learning Projects 13 - Covid 19 Drug Recovery using Deep Learning",
        title="COVID-19 Drug Recovery Analysis",
        type="tabular_cls",
        dataset="UNCOVER COVID-19 Challenge",
        links=["https://www.kaggle.com/datasets/roche-data-science-coalition/uncover"],
        kaggle="roche-data-science-coalition/uncover",
        target="outcome",
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    # Use the first CSV with an outcome column\n'
            '    for csv in csvs:\n'
            '        df = pd.read_csv(csv, low_memory=False)\n'
            '        if TARGET in df.columns:\n'
            '            return df\n'
            '    # Fallback: use largest CSV, find best target\n'
            '    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)\n'
            '    df = pd.read_csv(csvs[0], low_memory=False)\n'
            '    return df'
        ),
    ),

    # ── P14  Face, Gender & Ethnicity ────────────────────────────────────────
    dict(
        num=14,
        folder="Deep Learning Projects 14 - Face, Gender & Ethincity recognizer model",
        title="Face / Gender / Ethnicity Recognizer",
        type="cv",
        dataset="UTKFace",
        links=[
            "https://www.kaggle.com/datasets/jangedoo/utkface-new",
            "https://www.kaggle.com/datasets/jessicali9530/fairface-dataset",
        ],
        kaggle="jangedoo/utkface-new",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=5,  # ethnicity: 5 classes
        epochs=15,
        get_data_body=(
            '    import shutil\n'
            '    target = DATA_DIR / "ethnicity"\n'
            '    if not target.exists():\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        # UTKFace filenames: age_gender_race_date.jpg  (race 0-4)\n'
            '        races = {0: "white", 1: "black", 2: "asian", 3: "indian", 4: "others"}\n'
            '        src = None\n'
            '        for d in [DATA_DIR / "UTKFace", DATA_DIR / "utkface_aligned_cropped", DATA_DIR]:\n'
            '            if d.is_dir() and list(d.glob("*.jpg")):\n'
            '                src = d; break\n'
            '        if src is None:\n'
            '            raise FileNotFoundError("No images found after Kaggle download")\n'
            '        imgs = list(src.glob("*.jpg")) + list(src.glob("*.png"))\n'
            '        import random; random.seed(42); random.shuffle(imgs)\n'
            '        n = len(imgs)\n'
            '        for i, f in enumerate(imgs):\n'
            '            parts = f.stem.split("_")\n'
            '            if len(parts) < 3: continue\n'
            '            race_id = int(parts[2]) if parts[2].isdigit() else 4\n'
            '            race = races.get(race_id, "others")\n'
            '            split = "train" if i < .8*n else ("val" if i < .9*n else "test")\n'
            '            dest = target / split / race\n'
            '            dest.mkdir(parents=True, exist_ok=True)\n'
            '            shutil.copy2(f, dest / f.name)\n'
            '    return target'
        ),
    ),

    # ── P15  Boston Housing / Happy House ────────────────────────────────────
    dict(
        num=15,
        folder="Deep Learning Projects 15 - Happy house Predictor model",
        title="Boston Housing Price Prediction",
        type="tabular_reg",
        dataset="Boston Housing Dataset",
        links=["https://www.kaggle.com/datasets/uciml/boston-housing-dataset"],
        kaggle="uciml/boston-housing-dataset",
        target="MEDV",
        get_data_body=_tab_kaggle(),
    ),

    # ── P16  Brain MRI Segmentation (CUSTOM) ─────────────────────────────────
    dict(
        num=16,
        folder="Deep Learning Projects 16 - Brain MRI Segmentation modling",
        title="Brain MRI Segmentation",
        type="custom",
        dataset="LGG MRI Segmentation",
        links=["https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"],
        kaggle="mateuszbuda/lgg-mri-segmentation",
        model="deeplabv3_resnet50",
        custom_run=(
            '#!/usr/bin/env python3\n'
            '"""Project 16 -- Brain MRI Segmentation\n'
            '\n'
            'Dataset : LGG MRI Segmentation\n'
            'Model   : DeepLabV3-ResNet50 (torchvision)\n'
            'Task    : Binary Segmentation\n'
            'Usage   : python run.py\n'
            '"""\n'
            'import sys\n'
            'from pathlib import Path\n'
            '\n'
            'ROOT = Path(__file__).resolve().parents[1]\n'
            'sys.path.insert(0, str(ROOT))\n'
            '\n'
            'import numpy as np\n'
            'import torch, torch.nn as nn\n'
            'from torch.utils.data import DataLoader, Dataset, random_split\n'
            'from torchvision import transforms\n'
            'from torchvision.models.segmentation import deeplabv3_resnet50\n'
            'from PIL import Image\n'
            'from tqdm import tqdm\n'
            'from shared.utils import seed_everything, get_device, dataset_prompt, kaggle_download, ensure_dir, save_metrics\n'
            '\n'
            'PROJECT_DIR = Path(__file__).resolve().parent\n'
            'DATA_DIR    = PROJECT_DIR / "data"\n'
            'OUTPUT_DIR  = PROJECT_DIR / "outputs"\n'
            'KAGGLE = "mateuszbuda/lgg-mri-segmentation"\n'
            'EPOCHS, LR, BATCH = 20, 1e-4, 8\n'
            '\n'
            '\n'
            'class MRIDataset(Dataset):\n'
            '    def __init__(self, images, masks, size=256):\n'
            '        self.images, self.masks, self.size = images, masks, size\n'
            '        self.tf = transforms.Compose([transforms.Resize((size,size)),\n'
            '                                      transforms.ToTensor()])\n'
            '    def __len__(self): return len(self.images)\n'
            '    def __getitem__(self, i):\n'
            '        img = self.tf(Image.open(self.images[i]).convert("RGB"))\n'
            '        msk = self.tf(Image.open(self.masks[i]).convert("L"))\n'
            '        return img, (msk > 0.5).float()\n'
            '\n'
            '\n'
            'def get_data():\n'
            '    dataset_prompt("LGG MRI Segmentation",\n'
            '                   ["https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"])\n'
            '    if not list(DATA_DIR.rglob("*_mask.tif")):\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '    imgs = sorted(DATA_DIR.rglob("*[!_mask].tif"))\n'
            '    msks = sorted(DATA_DIR.rglob("*_mask.tif"))\n'
            '    # fallback: also check png\n'
            '    if not imgs:\n'
            '        imgs = sorted(p for p in DATA_DIR.rglob("*.png") if "_mask" not in p.stem)\n'
            '        msks = sorted(DATA_DIR.rglob("*_mask.png"))\n'
            '    return imgs, msks\n'
            '\n'
            '\n'
            'def dice_score(pred, target, eps=1e-7):\n'
            '    pred = (pred > 0.5).float()\n'
            '    inter = (pred * target).sum()\n'
            '    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)\n'
            '\n'
            '\n'
            'def main():\n'
            '    seed_everything(42)\n'
            '    device = get_device()\n'
            '    ensure_dir(OUTPUT_DIR)\n'
            '    imgs, msks = get_data()\n'
            '    ds = MRIDataset(imgs, msks)\n'
            '    n_val = int(0.15 * len(ds))\n'
            '    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])\n'
            '    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)\n'
            '    val_dl   = DataLoader(val_ds, batch_size=BATCH, num_workers=2)\n'
            '\n'
            '    model = deeplabv3_resnet50(weights="DEFAULT")\n'
            '    model.classifier[4] = nn.Conv2d(256, 1, 1)\n'
            '    model = model.to(device)\n'
            '    opt = torch.optim.AdamW(model.parameters(), lr=LR)\n'
            '    bce = nn.BCEWithLogitsLoss()\n'
            '    best_dice = 0.0\n'
            '\n'
            '    for ep in range(EPOCHS):\n'
            '        model.train(); loss_sum = 0\n'
            '        for X, y in tqdm(train_dl, desc=f"Epoch {ep+1}/{EPOCHS}"):\n'
            '            X, y = X.to(device), y.to(device)\n'
            '            out = model(X)["out"]\n'
            '            out = nn.functional.interpolate(out, size=y.shape[-2:], mode="bilinear")\n'
            '            loss = bce(out, y)\n'
            '            opt.zero_grad(); loss.backward(); opt.step()\n'
            '            loss_sum += loss.item()\n'
            '        # val\n'
            '        model.eval(); dsc = []\n'
            '        with torch.no_grad():\n'
            '            for X, y in val_dl:\n'
            '                X, y = X.to(device), y.to(device)\n'
            '                out = torch.sigmoid(model(X)["out"])\n'
            '                out = nn.functional.interpolate(out, size=y.shape[-2:], mode="bilinear")\n'
            '                dsc.append(dice_score(out, y).item())\n'
            '        mean_dice = np.mean(dsc)\n'
            '        print(f"  loss={loss_sum/len(train_dl):.4f}  dice={mean_dice:.4f}")\n'
            '        if mean_dice > best_dice:\n'
            '            best_dice = mean_dice\n'
            '            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")\n'
            '\n'
            '    save_metrics({"best_dice": best_dice}, OUTPUT_DIR)\n'
            '\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        ),
    ),

    # ── P17  Parkinson's Disease ─────────────────────────────────────────────
    dict(
        num=17,
        folder="Deep Learning Projects 17 - Parkension Post Estimation using deep learning",
        title="Parkinson's Disease Detection",
        type="tabular_cls",
        dataset="Parkinson's Disease Data Set",
        links=["https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set"],
        kaggle="vikasukani/parkinsons-disease-data-set",
        target="status",
        get_data_body=_tab_kaggle(),
    ),

    # ── P18  Diabetic Retinopathy ────────────────────────────────────────────
    dict(
        num=18,
        folder="Deep Learning Projects 18 - Diabetic Retinopathy project",
        title="Diabetic Retinopathy Detection",
        type="cv",
        dataset="Diabetic Retinopathy Detection",
        links=["https://www.kaggle.com/c/diabetic-retinopathy-detection"],
        kaggle="diabetic-retinopathy-detection",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=5,
        epochs=15,
        get_data_body=_cv_kaggle(competition=True),
    ),

    # ── P19  Arabic Character Recognition ────────────────────────────────────
    dict(
        num=19,
        folder="Deep Learning Projects 19 - Arabic character recognization using deep learning",
        title="Arabic Handwritten Character Recognition",
        type="cv",
        dataset="AHCD (Arabic Handwritten Characters)",
        links=["https://www.kaggle.com/datasets/mloey1/ahcd1"],
        kaggle="mloey1/ahcd1",
        model="efficientnet_b0.ra_in1k",
        num_classes=28,
        epochs=15,
        get_data_body=_cv_kaggle(),
    ),

    # ── P20  Brain Tumor Recognition ─────────────────────────────────────────
    dict(
        num=20,
        folder="Deep Learning Projects 20 - Brain Tumor Recognization using Deep Learning",
        title="Brain Tumor Classification",
        type="cv",
        dataset="Brain MRI Images for Tumor Detection",
        links=["https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection"],
        kaggle="navoneel/brain-mri-images-for-brain-tumor-detection",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=0,
        epochs=20,
        get_data_body=_cv_kaggle(),
    ),

    # ── P21  Walking or Running ──────────────────────────────────────────────
    dict(
        num=21,
        folder="Deep Learning Projects 21 - Image Walking or Running",
        title="Human Action Recognition (Walk / Run)",
        type="cv",
        dataset="HAR Dataset",
        links=["https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset"],
        kaggle="meetnagadia/human-action-recognition-har-dataset",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=0,
        epochs=15,
        get_data_body=_cv_kaggle(),
    ),

    # ── P22  Space Missions ──────────────────────────────────────────────────
    dict(
        num=22,
        folder="Deep Learning Projects 22- 1957 All Space Missions",
        title="Space Missions Success Prediction",
        type="tabular_cls",
        dataset="All Space Missions from 1957",
        links=["https://www.kaggle.com/datasets/agirlcoding/all-space-missions-from-1957"],
        kaggle="agirlcoding/all-space-missions-from-1957",
        target="Status Mission",
        get_data_body=_tab_kaggle(),
    ),

    # ── P23  Sudoku Solver (CUSTOM) ──────────────────────────────────────────
    dict(
        num=23,
        folder="Deep Learning Projects 23 - 1 Million Suduku Solver using neural nets",
        title="Sudoku Solver with Neural Network",
        type="custom",
        dataset="1M Sudoku Games",
        links=["https://www.kaggle.com/datasets/bryanpark/sudoku"],
        kaggle="bryanpark/sudoku",
        model="custom CNN",
        custom_run=(
            '#!/usr/bin/env python3\n'
            '"""Project 23 -- Sudoku Solver with Neural Network\n'
            '\n'
            'Dataset : 1M Sudoku Games\n'
            'Model   : Custom CNN (PyTorch)\n'
            'Usage   : python run.py\n'
            '"""\n'
            'import sys\n'
            'from pathlib import Path\n'
            '\n'
            'ROOT = Path(__file__).resolve().parents[1]\n'
            'sys.path.insert(0, str(ROOT))\n'
            '\n'
            'import numpy as np\n'
            'import torch, torch.nn as nn\n'
            'from torch.utils.data import DataLoader, TensorDataset\n'
            'from tqdm import tqdm\n'
            'import pandas as pd\n'
            'from shared.utils import seed_everything, get_device, dataset_prompt, kaggle_download, ensure_dir, save_metrics\n'
            '\n'
            'PROJECT_DIR = Path(__file__).resolve().parent\n'
            'DATA_DIR    = PROJECT_DIR / "data"\n'
            'OUTPUT_DIR  = PROJECT_DIR / "outputs"\n'
            'KAGGLE = "bryanpark/sudoku"\n'
            'EPOCHS, LR, BATCH = 10, 1e-3, 256\n'
            '\n'
            '\n'
            'class SudokuNet(nn.Module):\n'
            '    def __init__(self):\n'
            '        super().__init__()\n'
            '        self.conv = nn.Sequential(\n'
            '            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),\n'
            '            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),\n'
            '            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),\n'
            '            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),\n'
            '            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),\n'
            '        )\n'
            '        self.head = nn.Conv2d(128, 9, 1)  # 9 classes per cell\n'
            '\n'
            '    def forward(self, x):  # x: (B,1,9,9)\n'
            '        return self.head(self.conv(x))  # (B,9,9,9)\n'
            '\n'
            '\n'
            'def parse_sudoku(s):\n'
            '    return np.array([int(c) for c in s]).reshape(9, 9).astype(np.float32)\n'
            '\n'
            '\n'
            'def get_data():\n'
            '    dataset_prompt("1M Sudoku Games", ["https://www.kaggle.com/datasets/bryanpark/sudoku"])\n'
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0], nrows=100_000)  # cap for speed\n'
            '    quizzes = np.stack([parse_sudoku(q) for q in df.iloc[:, 0]])\n'
            '    solutions = np.stack([parse_sudoku(s) for s in df.iloc[:, 1]])\n'
            '    X = torch.from_numpy(quizzes / 9.0).unsqueeze(1)  # (N,1,9,9)\n'
            '    y = torch.from_numpy(solutions).long() - 1        # classes 0-8\n'
            '    n = len(X); split = int(0.9 * n)\n'
            '    return (TensorDataset(X[:split], y[:split]),\n'
            '            TensorDataset(X[split:], y[split:]))\n'
            '\n'
            '\n'
            'def main():\n'
            '    seed_everything(42)\n'
            '    device = get_device()\n'
            '    ensure_dir(OUTPUT_DIR)\n'
            '    train_ds, val_ds = get_data()\n'
            '    train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=2)\n'
            '    val_dl   = DataLoader(val_ds, BATCH, num_workers=2)\n'
            '\n'
            '    model = SudokuNet().to(device)\n'
            '    opt = torch.optim.Adam(model.parameters(), lr=LR)\n'
            '    crit = nn.CrossEntropyLoss()\n'
            '    best_acc = 0.0\n'
            '\n'
            '    for ep in range(EPOCHS):\n'
            '        model.train(); total, correct = 0, 0\n'
            '        for X, y in tqdm(train_dl, desc=f"Epoch {ep+1}/{EPOCHS}"):\n'
            '            X, y = X.to(device), y.to(device)\n'
            '            out = model(X)  # (B,9,9,9)\n'
            '            loss = crit(out, y)\n'
            '            opt.zero_grad(); loss.backward(); opt.step()\n'
            '            correct += (out.argmax(1) == y).sum().item()\n'
            '            total += y.numel()\n'
            '        # val\n'
            '        model.eval(); vc, vt = 0, 0\n'
            '        with torch.no_grad():\n'
            '            for X, y in val_dl:\n'
            '                X, y = X.to(device), y.to(device)\n'
            '                pred = model(X).argmax(1)\n'
            '                vc += (pred == y).sum().item(); vt += y.numel()\n'
            '        vacc = vc / vt\n'
            '        print(f"  train_acc={correct/total:.4f}  val_acc={vacc:.4f}")\n'
            '        if vacc > best_acc:\n'
            '            best_acc = vacc\n'
            '            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")\n'
            '\n'
            '    save_metrics({"best_cell_accuracy": best_acc}, OUTPUT_DIR)\n'
            '\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        ),
    ),

    # ── P24  Electric Motor Temperature ──────────────────────────────────────
    dict(
        num=24,
        folder="Deep Learning Projects 24 -Electric Car Temperature Predictor using Deep Learning",
        title="Electric Motor Temperature Prediction",
        type="tabular_reg",
        dataset="Electric Motor Temperature",
        links=["https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature"],
        kaggle="wkirgsn/electric-motor-temperature",
        target="pm",
        get_data_body=_tab_kaggle(),
    ),

    # ── P25  Hourly Energy Demand ────────────────────────────────────────────
    dict(
        num=25,
        folder="Deep Learning Projects 25-Hourly energy demand generation and weather",
        title="Energy Demand Prediction",
        type="tabular_reg",
        dataset="Hourly Energy Consumption",
        links=["https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption"],
        kaggle="robikscube/hourly-energy-consumption",
        target="PJME_MW",
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    # Use the largest CSV (PJME_hourly)\n'
            '    csvs.sort(key=lambda p: p.stat().st_size, reverse=True)\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    # Add time features\n'
            '    if "Datetime" in df.columns:\n'
            '        df["Datetime"] = pd.to_datetime(df["Datetime"])\n'
            '        df["hour"] = df["Datetime"].dt.hour\n'
            '        df["dayofweek"] = df["Datetime"].dt.dayofweek\n'
            '        df["month"] = df["Datetime"].dt.month\n'
            '        df.drop(columns=["Datetime"], inplace=True)\n'
            '    return df'
        ),
    ),

    # ── P26  Face Detection (CUSTOM - inference only) ────────────────────────
    dict(
        num=26,
        folder="Deep Learning Projects 26 - Caffe Face Detector (OpenCV Pre-trained Model)",
        title="Face Detection (OpenCV DNN)",
        type="custom",
        dataset="OpenCV DNN Model Zoo",
        links=["https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"],
        kaggle="",
        model="OpenCV DNN",
        custom_run=(
            '#!/usr/bin/env python3\n'
            '"""Project 26 -- Face Detection (OpenCV DNN)\n'
            '\n'
            'Model  : OpenCV SSD face detector (pre-trained)\n'
            'Task   : Inference-only face detection\n'
            'Usage  : python run.py [--image path/to/image.jpg]\n'
            '"""\n'
            'import sys, argparse\n'
            'from pathlib import Path\n'
            '\n'
            'ROOT = Path(__file__).resolve().parents[1]\n'
            'sys.path.insert(0, str(ROOT))\n'
            '\n'
            'import cv2\n'
            'import numpy as np\n'
            'from shared.utils import ensure_dir, url_download\n'
            '\n'
            'PROJECT_DIR = Path(__file__).resolve().parent\n'
            'DATA_DIR    = PROJECT_DIR / "data"\n'
            'OUTPUT_DIR  = PROJECT_DIR / "outputs"\n'
            '\n'
            'PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"\n'
            'MODEL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"\n'
            '\n'
            '\n'
            'def get_model():\n'
            '    ensure_dir(DATA_DIR)\n'
            '    proto = url_download(PROTO, DATA_DIR, "deploy.prototxt")\n'
            '    model = url_download(MODEL, DATA_DIR, "face_model.caffemodel")\n'
            '    return cv2.dnn.readNetFromCaffe(str(proto), str(model))\n'
            '\n'
            '\n'
            'def detect_faces(net, image_path, conf=0.5):\n'
            '    img = cv2.imread(str(image_path))\n'
            '    h, w = img.shape[:2]\n'
            '    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,\n'
            '                                 (300, 300), (104, 177, 123))\n'
            '    net.setInput(blob)\n'
            '    dets = net.forward()\n'
            '    faces = []\n'
            '    for i in range(dets.shape[2]):\n'
            '        c = dets[0, 0, i, 2]\n'
            '        if c > conf:\n'
            '            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])\n'
            '            x1, y1, x2, y2 = box.astype(int)\n'
            '            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)\n'
            '            faces.append((x1, y1, x2, y2, float(c)))\n'
            '    return img, faces\n'
            '\n'
            '\n'
            'def main():\n'
            '    ap = argparse.ArgumentParser()\n'
            '    ap.add_argument("--image", default=None)\n'
            '    args = ap.parse_args()\n'
            '    ensure_dir(OUTPUT_DIR)\n'
            '    net = get_model()\n'
            '    if args.image:\n'
            '        result, faces = detect_faces(net, args.image)\n'
            '        out_path = OUTPUT_DIR / "detected.jpg"\n'
            '        cv2.imwrite(str(out_path), result)\n'
            '        print(f"  Found {len(faces)} face(s) -> {out_path}")\n'
            '    else:\n'
            '        # Demo with webcam or sample\n'
            '        print("  Pass --image <path> to run. Example:")\n'
            '        print("    python run.py --image data/sample.jpg")\n'
            '\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        ),
    ),

    # ── P27  Concrete Strength ───────────────────────────────────────────────
    dict(
        num=27,
        folder="Deep Learning Projects 27- Calculate Concrete Strength",
        title="Concrete Compressive Strength Prediction",
        type="tabular_reg",
        dataset="Concrete Compressive Strength",
        links=["https://www.kaggle.com/datasets/uciml/concrete-compressive-strength-data-set"],
        kaggle="uciml/concrete-compressive-strength-data-set",
        target="csMPa",
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    # Rename target if needed\n'
            '    target_candidates = [c for c in df.columns if "strength" in c.lower() or "csm" in c.lower()]\n'
            '    if target_candidates and TARGET not in df.columns:\n'
            '        df = df.rename(columns={target_candidates[0]: TARGET})\n'
            '    return df'
        ),
    ),

    # ── P28  Stock Market / News Sentiment ───────────────────────────────────
    dict(
        num=28,
        folder="Deep Learning Projects 28 - Stock Market Prediction",
        title="Stock Market Prediction from News",
        type="nlp",
        dataset="Stock News Sentiment",
        links=["https://www.kaggle.com/datasets/aaron7sun/stocknews"],
        kaggle="aaron7sun/stocknews",
        hf_model="distilbert-base-uncased",
        num_labels=2,
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    # Combine headline columns into one text\n'
            '    text_cols = [c for c in df.columns if c.startswith("Top")]\n'
            '    if text_cols:\n'
            '        df["_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)\n'
            '    else:\n'
            '        df["_text"] = df.iloc[:, 2:].fillna("").agg(" ".join, axis=1)\n'
            '    label_col = "Label" if "Label" in df.columns else df.columns[1]\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42)\n'
            '    return (tr["_text"].tolist(), tr[label_col].astype(int).tolist(),\n'
            '            te["_text"].tolist(), te[label_col].astype(int).tolist(),\n'
            '            ["down", "up"])'
        ),
    ),

    # ── P29  Indian Startup Analysis ─────────────────────────────────────────
    dict(
        num=29,
        folder="Deep Learning Projects 29 - Indian Startup data Analysis",
        title="Indian Startup Funding Prediction",
        type="tabular_reg",
        dataset="Startup Investments (CrunchBase)",
        links=["https://www.kaggle.com/datasets/ruchi798/startup-investments-crunchbase"],
        kaggle="ruchi798/startup-investments-crunchbase",
        target="funding_total_usd",
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0], low_memory=False)\n'
            '    # Clean funding column\n'
            '    if TARGET in df.columns:\n'
            '        df[TARGET] = pd.to_numeric(df[TARGET].astype(str).str.replace(",","").str.strip(), errors="coerce")\n'
            '        df = df.dropna(subset=[TARGET])\n'
            '        df = df[df[TARGET] > 0]\n'
            '    return df'
        ),
    ),

    # ── P30  Amazon Stock ────────────────────────────────────────────────────
    dict(
        num=30,
        folder="Deep Learning Projects 30 - Amazon Stock Price Deep Analysis",
        title="Amazon Stock Price Prediction",
        type="tabular_reg",
        dataset="Amazon Stock Price",
        links=["https://www.kaggle.com/datasets/rohanrao/amazon-stock-price"],
        kaggle="rohanrao/amazon-stock-price",
        target="Close",
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    # Add lag features for time-series\n'
            '    if "Date" in df.columns:\n'
            '        df["Date"] = pd.to_datetime(df["Date"])\n'
            '        df = df.sort_values("Date").reset_index(drop=True)\n'
            '        for lag in [1, 3, 7]:\n'
            '            df[f"close_lag{lag}"] = df["Close"].shift(lag)\n'
            '        df.dropna(inplace=True)\n'
            '        df.drop(columns=["Date"], inplace=True)\n'
            '    return df'
        ),
    ),

    # ── P31  Dance Form ──────────────────────────────────────────────────────
    dict(
        num=31,
        folder="Deep Learning Projects 31 - Indentifying Dance Form Using Deep Learning-20210724T041140Z-001",
        title="Indian Classical Dance Form Classification",
        type="cv",
        dataset="Indian Classical Dance",
        links=["https://www.kaggle.com/datasets/arjunbhasin2013/indian-classical-dance"],
        kaggle="arjunbhasin2013/indian-classical-dance",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=0,
        epochs=20,
        get_data_body=_cv_kaggle(),
    ),

    # ── P32  Glasses Detection ───────────────────────────────────────────────
    dict(
        num=32,
        folder="Deep Learning Projects 32 - Glass or No Glass Detector Model using DL",
        title="Eyeglasses Detection",
        type="cv",
        dataset="Eyeglasses Dataset",
        links=["https://www.kaggle.com/datasets/jehanbhathena/eyeglasses-dataset"],
        kaggle="jehanbhathena/eyeglasses-dataset",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=2,
        epochs=10,
        get_data_body=_cv_kaggle(),
    ),

    # ── P33  Fingerprint Recognition ─────────────────────────────────────────
    dict(
        num=33,
        folder="Deep Learning Projects 33 - Fingerprint Recognizer Model using DL",
        title="Fingerprint Recognition",
        type="cv",
        dataset="SOCOFing Fingerprint Dataset",
        links=["https://www.kaggle.com/datasets/ruizgara/socofing"],
        kaggle="ruizgara/socofing",
        model="efficientnet_b0.ra_in1k",
        num_classes=0,
        epochs=15,
        get_data_body=_cv_kaggle(),
    ),

    # ── P34  Coin Detection ──────────────────────────────────────────────────
    dict(
        num=34,
        folder="Deep Learning Projects 34 - World Currency Coin Detector Model using DL",
        title="World Currency Coin Classification",
        type="cv",
        dataset="Coin Images",
        links=["https://www.kaggle.com/datasets/wanderdust/coin-images"],
        kaggle="wanderdust/coin-images",
        model="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        num_classes=0,
        epochs=20,
        get_data_body=_cv_kaggle(),
    ),

    # ── P35  News Category ───────────────────────────────────────────────────
    dict(
        num=35,
        folder="Deep Learning Projects 35 - News Category Prediction using DL",
        title="News Category Prediction",
        type="nlp",
        dataset="News Category Dataset",
        links=["https://www.kaggle.com/datasets/rmisra/news-category-dataset"],
        kaggle="rmisra/news-category-dataset",
        hf_model="distilbert-base-uncased",
        num_labels=0,
        get_data_body=(
            '    import json as _json\n'
            '    files = list(DATA_DIR.rglob("*.json"))\n'
            '    if not files:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        files = list(DATA_DIR.rglob("*.json"))\n'
            '    # Parse newline-delimited JSON\n'
            '    records = []\n'
            '    for f in files:\n'
            '        for line in f.read_text(encoding="utf-8").splitlines():\n'
            '            if line.strip():\n'
            '                records.append(_json.loads(line))\n'
            '    df = pd.DataFrame(records)\n'
            '    # Top-10 categories\n'
            '    top = df["category"].value_counts().head(10).index.tolist()\n'
            '    df = df[df["category"].isin(top)].copy()\n'
            '    labels = sorted(top)\n'
            '    lab2id = {l: i for i, l in enumerate(labels)}\n'
            '    df["_label"] = df["category"].map(lab2id)\n'
            '    df["_text"] = df["headline"].fillna("") + " " + df["short_description"].fillna("")\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["_text"].tolist(), tr["_label"].tolist(),\n'
            '            te["_text"].tolist(), te["_label"].tolist(), labels)'
        ),
    ),

    # ── P36  Lego Brick ──────────────────────────────────────────────────────
    dict(
        num=36,
        folder="Deep Learning Projects 36 - Lego Brick Code Problem",
        title="Lego Brick Classification",
        type="cv",
        dataset="Lego Brick Images",
        links=["https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images"],
        kaggle="joosthazelzet/lego-brick-images",
        model="efficientnet_b0.ra_in1k",
        num_classes=0,
        epochs=15,
        get_data_body=_cv_kaggle(),
    ),

    # ── P37  Sheep Breed ─────────────────────────────────────────────────────
    dict(
        num=37,
        folder="Deep Learning Projects 37 - Sheep Breed Classification using CNN DL",
        title="Sheep Breed Classification",
        type="cv",
        dataset="Sheep Face Images",
        links=["https://www.kaggle.com/datasets/warcoder/sheep-face-images"],
        kaggle="warcoder/sheep-face-images",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=4,
        epochs=15,
        get_data_body=_cv_kaggle(),
    ),

    # ── P38  Campus Recruitment ──────────────────────────────────────────────
    dict(
        num=38,
        folder="Deep Learning Projects 38 - Campus Recruitment Success rate analysis",
        title="Campus Recruitment Prediction",
        type="tabular_cls",
        dataset="Campus Recruitment",
        links=["https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement"],
        kaggle="benroshan/factors-affecting-campus-placement",
        target="status",
        get_data_body=_tab_kaggle(),
    ),

    # ── P39  Bank Marketing ──────────────────────────────────────────────────
    dict(
        num=39,
        folder="Deep Learning Projects 39 - Bank Marketing",
        title="Bank Marketing Prediction",
        type="tabular_cls",
        dataset="Bank Marketing",
        links=["https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing"],
        kaggle="henriqueyamahata/bank-marketing",
        target="y",
        get_data_body=_tab_kaggle(),
    ),

    # ── P40  Pokemon Clustering ──────────────────────────────────────────────
    dict(
        num=40,
        folder="Deep Learning Projects 40 - Pokemon Generation Clustering",
        title="Pokemon Generation Clustering",
        type="tabular_cluster",
        dataset="Pokemon Dataset",
        links=["https://www.kaggle.com/datasets/abcsds/pokemon"],
        kaggle="abcsds/pokemon",
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    # Keep numeric columns for clustering\n'
            '    df = df.select_dtypes(include="number").dropna()\n'
            '    return df'
        ),
    ),

    # ── P41  Cat/Dog Voice (CUSTOM - audio) ──────────────────────────────────
    dict(
        num=41,
        folder="Deep Learning Projects 41 - Cat _ Dog Voice Recognizer Model",
        title="Cat vs Dog Audio Classification",
        type="custom",
        dataset="Audio Cats and Dogs",
        links=["https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs"],
        kaggle="mmoreaux/audio-cats-and-dogs",
        model="CNN on mel-spectrogram",
        custom_run=(
            '#!/usr/bin/env python3\n'
            '"""Project 41 -- Cat vs Dog Audio Classification\n'
            '\n'
            'Dataset: Audio Cats and Dogs\n'
            'Model  : Simple CNN on mel-spectrograms (PyTorch + torchaudio)\n'
            'Usage  : python run.py\n'
            '"""\n'
            'import sys\n'
            'from pathlib import Path\n'
            '\n'
            'ROOT = Path(__file__).resolve().parents[1]\n'
            'sys.path.insert(0, str(ROOT))\n'
            '\n'
            'import numpy as np\n'
            'import torch, torch.nn as nn\n'
            'from torch.utils.data import DataLoader, Dataset, random_split\n'
            'import torchaudio\n'
            'from torchaudio.transforms import MelSpectrogram, Resample\n'
            'from tqdm import tqdm\n'
            'from shared.utils import (seed_everything, get_device, dataset_prompt,\n'
            '                          kaggle_download, ensure_dir, save_metrics)\n'
            '\n'
            'PROJECT_DIR = Path(__file__).resolve().parent\n'
            'DATA_DIR    = PROJECT_DIR / "data"\n'
            'OUTPUT_DIR  = PROJECT_DIR / "outputs"\n'
            'KAGGLE = "mmoreaux/audio-cats-and-dogs"\n'
            'EPOCHS, LR, BATCH = 15, 1e-3, 32\n'
            'SR = 16000\n'
            '\n'
            '\n'
            'class AudioDS(Dataset):\n'
            '    def __init__(self, files, labels):\n'
            '        self.files, self.labels = files, labels\n'
            '        self.mel = MelSpectrogram(sample_rate=SR, n_mels=64, n_fft=1024)\n'
            '\n'
            '    def __len__(self): return len(self.files)\n'
            '\n'
            '    def __getitem__(self, i):\n'
            '        wav, sr = torchaudio.load(str(self.files[i]))\n'
            '        if sr != SR:\n'
            '            wav = Resample(sr, SR)(wav)\n'
            '        wav = wav.mean(0, keepdim=True)  # mono\n'
            '        # Pad/trim to 3 seconds\n'
            '        target_len = SR * 3\n'
            '        if wav.shape[1] < target_len:\n'
            '            wav = nn.functional.pad(wav, (0, target_len - wav.shape[1]))\n'
            '        else:\n'
            '            wav = wav[:, :target_len]\n'
            '        spec = self.mel(wav)  # (1, 64, T)\n'
            '        return spec, self.labels[i]\n'
            '\n'
            '\n'
            'class AudioCNN(nn.Module):\n'
            '    def __init__(self):\n'
            '        super().__init__()\n'
            '        self.features = nn.Sequential(\n'
            '            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n'
            '            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n'
            '            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),\n'
            '        )\n'
            '        self.classifier = nn.Sequential(\n'
            '            nn.Flatten(), nn.Linear(128*4*4, 128), nn.ReLU(), nn.Dropout(0.3),\n'
            '            nn.Linear(128, 2),\n'
            '        )\n'
            '\n'
            '    def forward(self, x):\n'
            '        return self.classifier(self.features(x))\n'
            '\n'
            '\n'
            'def get_data():\n'
            '    dataset_prompt("Audio Cats and Dogs",\n'
            '                   ["https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs"])\n'
            '    wavs = list(DATA_DIR.rglob("*.wav"))\n'
            '    if not wavs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        wavs = list(DATA_DIR.rglob("*.wav"))\n'
            '    files, labels = [], []\n'
            '    for w in wavs:\n'
            '        low = w.stem.lower()\n'
            '        if "cat" in low or "cat" in str(w.parent).lower():\n'
            '            files.append(w); labels.append(0)\n'
            '        elif "dog" in low or "dog" in str(w.parent).lower():\n'
            '            files.append(w); labels.append(1)\n'
            '    return AudioDS(files, labels)\n'
            '\n'
            '\n'
            'def main():\n'
            '    seed_everything(42)\n'
            '    device = get_device()\n'
            '    ensure_dir(OUTPUT_DIR)\n'
            '\n'
            '    ds = get_data()\n'
            '    n_val = int(0.2 * len(ds))\n'
            '    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])\n'
            '    train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=2)\n'
            '    val_dl   = DataLoader(val_ds, BATCH, num_workers=2)\n'
            '\n'
            '    model = AudioCNN().to(device)\n'
            '    opt = torch.optim.Adam(model.parameters(), lr=LR)\n'
            '    crit = nn.CrossEntropyLoss()\n'
            '    best = 0.0\n'
            '\n'
            '    for ep in range(EPOCHS):\n'
            '        model.train()\n'
            '        for X, y in tqdm(train_dl, desc=f"Epoch {ep+1}/{EPOCHS}"):\n'
            '            X, y = X.to(device), y.to(device)\n'
            '            loss = crit(model(X), y)\n'
            '            opt.zero_grad(); loss.backward(); opt.step()\n'
            '        model.eval(); c, t = 0, 0\n'
            '        with torch.no_grad():\n'
            '            for X, y in val_dl:\n'
            '                X, y = X.to(device), y.to(device)\n'
            '                c += (model(X).argmax(1) == y).sum().item(); t += len(y)\n'
            '        acc = c / t\n'
            '        print(f"  val_acc={acc:.4f}")\n'
            '        if acc > best:\n'
            '            best = acc\n'
            '            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")\n'
            '\n'
            '    save_metrics({"best_val_accuracy": best}, OUTPUT_DIR)\n'
            '\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        ),
    ),

    # ── P42  Bottle or Cans ──────────────────────────────────────────────────
    dict(
        num=42,
        folder="Deep Learning Projects 42 - Bottle or Cans Classifier using DL",
        title="Bottle vs Can Classification",
        type="cv",
        dataset="Bottles and Cans",
        links=["https://www.kaggle.com/datasets/trolukovich/bottles-and-cans"],
        kaggle="trolukovich/bottles-and-cans",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=0,
        epochs=15,
        get_data_body=_cv_kaggle(),
    ),

    # ── P43  Skin Cancer ─────────────────────────────────────────────────────
    dict(
        num=43,
        folder="Deep Learning Projects 43 - Skin Cancer Recognizer using DL",
        title="Skin Cancer (HAM10000) Classification",
        type="cv",
        dataset="Skin Cancer MNIST: HAM10000",
        links=["https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"],
        kaggle="kmader/skin-cancer-mnist-ham10000",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=7,
        epochs=20,
        get_data_body=(
            '    import shutil\n'
            '    target = DATA_DIR / "organized"\n'
            '    if not target.exists():\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        # HAM10000 has images + CSV metadata; organise into class dirs\n'
            '        csv_files = list(DATA_DIR.rglob("*metadata*")) + list(DATA_DIR.rglob("*HAM*.csv"))\n'
            '        if csv_files:\n'
            '            import pandas as _pd\n'
            '            meta = _pd.read_csv(csv_files[0])\n'
            '            imgs = list(DATA_DIR.rglob("*.jpg"))\n'
            '            img_map = {p.stem: p for p in imgs}\n'
            '            import random; random.seed(42)\n'
            '            meta = meta.sample(frac=1, random_state=42)\n'
            '            n = len(meta)\n'
            '            for i, row in enumerate(meta.itertuples()):\n'
            '                split = "train" if i < .8*n else ("val" if i < .9*n else "test")\n'
            '                cls = row.dx\n'
            '                src_img = img_map.get(row.image_id)\n'
            '                if src_img:\n'
            '                    dest = target / split / cls\n'
            '                    dest.mkdir(parents=True, exist_ok=True)\n'
            '                    shutil.copy2(src_img, dest / src_img.name)\n'
            '    return target'
        ),
    ),

    # ── P44  Image Colorization (CUSTOM) ─────────────────────────────────────
    dict(
        num=44,
        folder="Deep Learning Projects 44 - Image Colorization using Deep Learning",
        title="Image Colorization",
        type="custom",
        dataset="VizWiz Colorization",
        links=["https://www.kaggle.com/datasets/landrykezebou/vizwiz-colorization"],
        kaggle="landrykezebou/vizwiz-colorization",
        model="UNet autoencoder",
        custom_run=(
            '#!/usr/bin/env python3\n'
            '"""Project 44 -- Image Colorization\n'
            '\n'
            'Dataset : VizWiz Colorization\n'
            'Model   : Simple UNet autoencoder (grayscale -> color)\n'
            'Usage   : python run.py\n'
            '"""\n'
            'import sys\n'
            'from pathlib import Path\n'
            '\n'
            'ROOT = Path(__file__).resolve().parents[1]\n'
            'sys.path.insert(0, str(ROOT))\n'
            '\n'
            'import numpy as np\n'
            'import torch, torch.nn as nn\n'
            'from torch.utils.data import DataLoader, Dataset, random_split\n'
            'from torchvision import transforms\n'
            'from PIL import Image\n'
            'from tqdm import tqdm\n'
            'from shared.utils import (seed_everything, get_device, dataset_prompt,\n'
            '                          kaggle_download, ensure_dir, save_metrics)\n'
            '\n'
            'PROJECT_DIR = Path(__file__).resolve().parent\n'
            'DATA_DIR    = PROJECT_DIR / "data"\n'
            'OUTPUT_DIR  = PROJECT_DIR / "outputs"\n'
            'KAGGLE = "landrykezebou/vizwiz-colorization"\n'
            'EPOCHS, LR, BATCH = 20, 1e-3, 16\n'
            'IMG_SIZE = 128\n'
            '\n'
            '\n'
            'class ColorDS(Dataset):\n'
            '    def __init__(self, paths, size=128):\n'
            '        self.paths, self.size = paths, size\n'
            '        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])\n'
            '    def __len__(self): return len(self.paths)\n'
            '    def __getitem__(self, i):\n'
            '        color = self.tf(Image.open(self.paths[i]).convert("RGB"))\n'
            '        gray  = color.mean(0, keepdim=True)  # grayscale\n'
            '        return gray, color\n'
            '\n'
            '\n'
            'class MiniUNet(nn.Module):\n'
            '    def __init__(self):\n'
            '        super().__init__()\n'
            '        def block(ci, co): return nn.Sequential(nn.Conv2d(ci,co,3,1,1), nn.BatchNorm2d(co), nn.ReLU())\n'
            '        self.enc1 = block(1, 64)\n'
            '        self.enc2 = nn.Sequential(nn.MaxPool2d(2), block(64, 128))\n'
            '        self.enc3 = nn.Sequential(nn.MaxPool2d(2), block(128, 256))\n'
            '        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)\n'
            '        self.dec2 = block(256, 128)\n'
            '        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)\n'
            '        self.dec1 = block(128, 64)\n'
            '        self.head = nn.Conv2d(64, 3, 1)\n'
            '    def forward(self, x):\n'
            '        e1 = self.enc1(x)\n'
            '        e2 = self.enc2(e1)\n'
            '        e3 = self.enc3(e2)\n'
            '        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))\n'
            '        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))\n'
            '        return torch.sigmoid(self.head(d1))\n'
            '\n'
            '\n'
            'def get_data():\n'
            '    dataset_prompt("VizWiz Colorization",\n'
            '                   ["https://www.kaggle.com/datasets/landrykezebou/vizwiz-colorization"])\n'
            '    imgs = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))\n'
            '    if not imgs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        imgs = list(DATA_DIR.rglob("*.jpg")) + list(DATA_DIR.rglob("*.png"))\n'
            '    return ColorDS(imgs[:5000])  # cap for speed\n'
            '\n'
            '\n'
            'def main():\n'
            '    seed_everything(42)\n'
            '    device = get_device()\n'
            '    ensure_dir(OUTPUT_DIR)\n'
            '    ds = get_data()\n'
            '    n_val = int(0.15 * len(ds))\n'
            '    train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])\n'
            '    train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=2)\n'
            '    val_dl   = DataLoader(val_ds, BATCH, num_workers=2)\n'
            '\n'
            '    model = MiniUNet().to(device)\n'
            '    opt = torch.optim.Adam(model.parameters(), lr=LR)\n'
            '    crit = nn.MSELoss()\n'
            '    best_loss = float("inf")\n'
            '\n'
            '    for ep in range(EPOCHS):\n'
            '        model.train(); total_loss = 0\n'
            '        for gray, color in tqdm(train_dl, desc=f"Epoch {ep+1}/{EPOCHS}"):\n'
            '            gray, color = gray.to(device), color.to(device)\n'
            '            pred = model(gray)\n'
            '            loss = crit(pred, color)\n'
            '            opt.zero_grad(); loss.backward(); opt.step()\n'
            '            total_loss += loss.item()\n'
            '        # val\n'
            '        model.eval(); vloss = 0\n'
            '        with torch.no_grad():\n'
            '            for gray, color in val_dl:\n'
            '                gray, color = gray.to(device), color.to(device)\n'
            '                vloss += crit(model(gray), color).item()\n'
            '        vloss /= max(len(val_dl), 1)\n'
            '        print(f"  train_loss={total_loss/len(train_dl):.4f}  val_loss={vloss:.4f}")\n'
            '        if vloss < best_loss:\n'
            '            best_loss = vloss\n'
            '            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")\n'
            '\n'
            '    save_metrics({"best_val_mse": best_loss}, OUTPUT_DIR)\n'
            '\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        ),
    ),

    # ── P45  Amazon Alexa Sentiment ──────────────────────────────────────────
    dict(
        num=45,
        folder="Deep Learning Projects 45 - Amazon Alexa Review Sentiment Analysis",
        title="Amazon Alexa Review Sentiment",
        type="nlp",
        dataset="Amazon Alexa Reviews",
        links=["https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews"],
        kaggle="sid321axn/amazon-alexa-reviews",
        hf_model="distilbert-base-uncased",
        num_labels=2,
        get_data_body=(
            '    files = list(DATA_DIR.rglob("*.tsv")) + list(DATA_DIR.rglob("*.csv"))\n'
            '    if not files:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        files = list(DATA_DIR.rglob("*.tsv")) + list(DATA_DIR.rglob("*.csv"))\n'
            '    f = files[0]\n'
            '    df = pd.read_csv(f, sep="\\t" if f.suffix == ".tsv" else ",", encoding="latin-1")\n'
            '    text_col = [c for c in df.columns if "review" in c.lower() or "verified" in c.lower() or "text" in c.lower()]\n'
            '    text_col = text_col[0] if text_col else df.columns[-2]\n'
            '    label_col = "feedback" if "feedback" in df.columns else "rating"\n'
            '    if df[label_col].nunique() > 2:\n'
            '        df["_label"] = (df[label_col] >= 4).astype(int)  # 4-5 positive\n'
            '    else:\n'
            '        df["_label"] = df[label_col].astype(int)\n'
            '    df["_text"] = df[text_col].astype(str)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["_text"].tolist(), tr["_label"].tolist(),\n'
            '            te["_text"].tolist(), te["_label"].tolist(),\n'
            '            ["negative", "positive"])'
        ),
    ),

    # ── P46  Chatbot (Neural Network) ────────────────────────────────────────
    dict(
        num=46,
        folder="Deep Learning Projects 46 - Build_ChatBot_using_Neural_Network",
        title="Chatbot Intent Classification",
        type="nlp",
        dataset="Chatbot Intent Recognition",
        links=["https://www.kaggle.com/datasets/elvinagammed/chatbots-intent-recognition-dataset"],
        kaggle="elvinagammed/chatbots-intent-recognition-dataset",
        hf_model="distilbert-base-uncased",
        num_labels=0,
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    text_col = [c for c in df.columns if c.lower() in ("text","query","utterance","sentence")][0]\n'
            '    label_col = [c for c in df.columns if c.lower() in ("intent","label","category","tag")][0]\n'
            '    labels_unique = sorted(df[label_col].unique())\n'
            '    lab2id = {l: i for i, l in enumerate(labels_unique)}\n'
            '    df["_label"] = df[label_col].map(lab2id)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr[text_col].tolist(), tr["_label"].tolist(),\n'
            '            te[text_col].tolist(), te["_label"].tolist(), labels_unique)'
        ),
    ),

    # ── P47  Cactus Identification ───────────────────────────────────────────
    dict(
        num=47,
        folder="Deep Learning Projects 47 - Cactus or Not Cactus Ariel Image Recognizer",
        title="Aerial Cactus Identification",
        type="cv",
        dataset="Aerial Cactus Identification",
        links=["https://www.kaggle.com/c/aerial-cactus-identification"],
        kaggle="aerial-cactus-identification",
        model="efficientnet_b0.ra_in1k",
        num_classes=2,
        epochs=10,
        get_data_body=(
            '    import shutil\n'
            '    target = DATA_DIR / "organized"\n'
            '    if not target.exists():\n'
            '        kaggle_download(KAGGLE, DATA_DIR, competition=True)\n'
            '        csv_files = list(DATA_DIR.rglob("*.csv"))\n'
            '        import pandas as _pd\n'
            '        df = _pd.read_csv(csv_files[0]) if csv_files else _pd.DataFrame()\n'
            '        img_dir = DATA_DIR / "train"\n'
            '        if not img_dir.is_dir():\n'
            '            img_dir = DATA_DIR\n'
            '        if "has_cactus" in df.columns:\n'
            '            import random; random.seed(42)\n'
            '            df = df.sample(frac=1, random_state=42)\n'
            '            n = len(df)\n'
            '            for i, row in enumerate(df.itertuples()):\n'
            '                split = "train" if i < .8*n else ("val" if i < .9*n else "test")\n'
            '                cls = "cactus" if row.has_cactus else "no_cactus"\n'
            '                dest = target / split / cls\n'
            '                dest.mkdir(parents=True, exist_ok=True)\n'
            '                src = img_dir / row.id\n'
            '                if src.exists():\n'
            '                    shutil.copy2(src, dest / row.id)\n'
            '        else:\n'
            '            return DATA_DIR  # fallback\n'
            '    return target'
        ),
    ),

    # ── P48  Fashion MNIST / Clothing ────────────────────────────────────────
    dict(
        num=48,
        folder="Deep Learning Projects 48 -  Build_Clothing_Prediction_Flask_Web_App",
        title="Fashion MNIST Clothing Classification",
        type="custom",
        dataset="Fashion-MNIST (torchvision)",
        links=["https://github.com/zalandoresearch/fashion-mnist"],
        kaggle="",
        model="efficientnet_b0.ra_in1k",
        custom_run=(
            '#!/usr/bin/env python3\n'
            '"""Project 48 -- Fashion MNIST Clothing Classification\n'
            '\n'
            'Dataset : Fashion-MNIST (auto-downloaded via torchvision)\n'
            'Model   : efficientnet_b0 (timm, pretrained)\n'
            'Usage   : python run.py\n'
            '"""\n'
            'import sys\n'
            'from pathlib import Path\n'
            '\n'
            'ROOT = Path(__file__).resolve().parents[1]\n'
            'sys.path.insert(0, str(ROOT))\n'
            '\n'
            'import torch\n'
            'from torch.utils.data import DataLoader, random_split\n'
            'from torchvision import datasets, transforms\n'
            'from shared.utils import seed_everything, get_device, ensure_dir\n'
            'from shared.cv import build_timm_model, train_model, evaluate_model\n'
            '\n'
            'PROJECT_DIR = Path(__file__).resolve().parent\n'
            'DATA_DIR    = PROJECT_DIR / "data"\n'
            'OUTPUT_DIR  = PROJECT_DIR / "outputs"\n'
            '\n'
            'CLASSES = ["T-shirt/top","Trouser","Pullover","Dress","Coat",\n'
            '           "Sandal","Shirt","Sneaker","Bag","Ankle boot"]\n'
            'MODEL   = "efficientnet_b0.ra_in1k"\n'
            'EPOCHS  = 10\n'
            '\n'
            '\n'
            'def get_data():\n'
            '    tf = transforms.Compose([\n'
            '        transforms.Grayscale(3),  # 1ch -> 3ch for timm\n'
            '        transforms.Resize(224),\n'
            '        transforms.ToTensor(),\n'
            '        transforms.Normalize([0.485]*3, [0.229]*3),\n'
            '    ])\n'
            '    train_full = datasets.FashionMNIST(str(DATA_DIR), train=True, download=True, transform=tf)\n'
            '    test_ds    = datasets.FashionMNIST(str(DATA_DIR), train=False, download=True, transform=tf)\n'
            '    n_val = int(0.15 * len(train_full))\n'
            '    train_ds, val_ds = random_split(train_full, [len(train_full)-n_val, n_val])\n'
            '    kw = dict(batch_size=64, num_workers=4, pin_memory=torch.cuda.is_available())\n'
            '    return (DataLoader(train_ds, shuffle=True, **kw),\n'
            '            DataLoader(val_ds, **kw),\n'
            '            DataLoader(test_ds, **kw), CLASSES)\n'
            '\n'
            '\n'
            'def main():\n'
            '    seed_everything(42)\n'
            '    device = get_device()\n'
            '    ensure_dir(OUTPUT_DIR)\n'
            '    train_dl, val_dl, test_dl, classes = get_data()\n'
            '    model = build_timm_model(MODEL, num_classes=10)\n'
            '    model = train_model(model, train_dl, val_dl, epochs=EPOCHS,\n'
            '                        lr=1e-4, device=device, output_dir=OUTPUT_DIR)\n'
            '    evaluate_model(model, test_dl, classes, device=device, output_dir=OUTPUT_DIR)\n'
            '\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
        ),
    ),

    # ── P49  Sentiment Analysis Flask App ────────────────────────────────────
    dict(
        num=49,
        folder="Deep Learning Projects 49 - Build_Sentiment_Analysis_Flask_Web_App",
        title="IMDB Sentiment Analysis",
        type="nlp",
        dataset="IMDB 50K Movie Reviews",
        links=[
            "https://ai.stanford.edu/~amaas/data/sentiment/",
            "https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        ],
        kaggle="lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        hf_model="distilbert-base-uncased",
        num_labels=2,
        get_data_body=(
            '    csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    if not csvs:\n'
            '        kaggle_download(KAGGLE, DATA_DIR)\n'
            '        csvs = list(DATA_DIR.rglob("*.csv"))\n'
            '    df = pd.read_csv(csvs[0])\n'
            '    lab_map = {"positive": 1, "negative": 0}\n'
            '    df["_label"] = df["sentiment"].map(lab_map)\n'
            '    from sklearn.model_selection import train_test_split\n'
            '    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_label"])\n'
            '    return (tr["review"].tolist(), tr["_label"].tolist(),\n'
            '            te["review"].tolist(), te["_label"].tolist(),\n'
            '            ["negative", "positive"])'
        ),
    ),

    # ── P50  COVID-19 CT Scans ───────────────────────────────────────────────
    dict(
        num=50,
        folder="Deep Learning Projects 50 - COVID-19 Lung CT Scans",
        title="COVID-19 Lung CT Scan Classification",
        type="cv",
        dataset="SARS-CoV-2 CT Scan Dataset",
        links=["https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset"],
        kaggle="plameneduardo/sarscov2-ctscan-dataset",
        model="convnext_tiny.fb_in22k_ft_in1k",
        num_classes=2,
        epochs=15,
        get_data_body=_cv_kaggle(),
    ),
]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. RENDERING HELPERS                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

TYPE_TO_TPL = {
    "cv":             CV_TPL,
    "tabular_cls":    TAB_CLS_TPL,
    "tabular_reg":    TAB_REG_TPL,
    "tabular_cluster": TAB_CLUST_TPL,
    "nlp":            NLP_TPL,
}

TYPE_TO_DESC = {
    "cv":              ("timm (ConvNeXt / EfficientNet / Swin)", "Image Classification", "N/A"),
    "tabular_cls":     ("PyCaret AutoML", "Tabular Classification", "PyCaret compare_models"),
    "tabular_reg":     ("PyCaret AutoML", "Tabular Regression", "PyCaret compare_models"),
    "tabular_cluster": ("PyCaret Clustering", "Unsupervised Clustering", "PyCaret KMeans"),
    "nlp":             ("HuggingFace Transformers", "Text Classification", "N/A"),
    "audio":           ("CNN on mel-spectrograms", "Audio Classification", "N/A"),
    "custom":          ("see run.py", "see run.py", "N/A"),
}


def render_run_py(c: dict) -> str:
    """Return the full run.py content for project *c*."""
    if c["type"] == "custom":
        return c["custom_run"]

    tpl = TYPE_TO_TPL[c["type"]]
    kwargs = dict(
        num=c["num"],
        title=c["title"],
        dataset=c.get("dataset", ""),
        model=c.get("model", ""),
        hf_model=c.get("hf_model", "distilbert-base-uncased"),
        dataset_repr=repr(c.get("dataset", "")),
        links_repr=repr(c.get("links", [])),
        kaggle_repr=repr(c.get("kaggle", "")),
        model_repr=repr(c.get("model", "")),
        hf_model_repr=repr(c.get("hf_model", "distilbert-base-uncased")),
        num_classes=c.get("num_classes", 0),
        num_labels=c.get("num_labels", 0),
        epochs=c.get("epochs", 15),
        target_repr=repr(c.get("target", "")),
        get_data_body=c.get("get_data_body", "    return DATA_DIR"),
    )
    return tpl.format(**kwargs)


def render_readme(c: dict) -> str:
    """Return README.md content for project *c*."""
    links_md = "\n".join(f"- {l}" for l in c.get("links", []))
    model_desc, task_desc, automl_desc = TYPE_TO_DESC.get(
        c["type"], ("custom", "custom", "N/A"))
    return README_TPL.format(
        num=c["num"],
        title=c["title"],
        dataset=c.get("dataset", ""),
        links_md=links_md,
        model_desc=c.get("model", model_desc),
        task_desc=task_desc,
        automl_desc=automl_desc,
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. MAIN                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    print("=" * 60)
    print("  Modernize Workspace — PyTorch-only stack")
    print("=" * 60)

    # ── Global files ─────────────────────────────────────────────────────────
    print("\n[1/3] Writing global configuration files ...")
    w("requirements.txt", REQUIREMENTS_TXT)
    w("environment.yml", ENVIRONMENT_YML)
    w("configs/global_config.yaml", GLOBAL_CONFIG_YAML)
    w("scripts/check_gpu.py", CHECK_GPU_PY)

    # ── Per-project files ────────────────────────────────────────────────────
    print(f"\n[2/3] Generating run.py + README.md for {len(PROJECTS)} projects ...")
    ok, skip = 0, 0
    for c in PROJECTS:
        folder = ROOT / c["folder"]
        if not folder.is_dir():
            print(f"  [SKIP] {c['folder']}  (directory not found)")
            skip += 1
            continue
        try:
            w(f"{c['folder']}/run.py", render_run_py(c))
            w(f"{c['folder']}/README.md", render_readme(c))
            # ensure outputs/ exists
            (folder / "outputs").mkdir(parents=True, exist_ok=True)
            ok += 1
        except Exception as exc:
            print(f"  [FAIL] P{c['num']}: {exc}")
            skip += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[3/3] Summary")
    print(f"  Global files : 4 written")
    print(f"  Projects     : {ok} generated, {skip} skipped")
    print(f"  Shared module: shared/ (already in place)")
    print(f"\n{'=' * 60}")
    print(f"  DONE — workspace modernized to PyTorch-only stack")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
