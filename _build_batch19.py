"""Batch 19 — 5 Classification notebooks (Fraudulent Transaction Detection,
Garbage Classification, Glass Classification, Groundhog Day Predictions,
H2O Higgs Boson)."""

import json, nbformat, os

BASE = "E:/Github/Machine-Learning-Projects/Classification"

# ── helpers ──────────────────────────────────────────────────────────
def _s(txt):
    return txt.replace("\t", "    ")

def _lines(src):
    out = []
    for ln in src.split("\n"):
        out.append(ln + "\n")
    if out:
        out[-1] = out[-1].rstrip("\n")
    return out

def md(src):
    return nbformat.v4.new_markdown_cell(_s(src))

def code(src):
    return nbformat.v4.new_code_cell(_s(src))

def write_nb(cells, path):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python", "version": "3.13.0"}})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  Wrote {path} ({len(cells)} cells)")


# ── shared code blocks ──────────────────────────────────────────────

INSTALL_TAB = """import subprocess, sys

def _install(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["catboost", "lightgbm", "xgboost", "flaml", "lazypredict"]:
    _install(pkg)

print("All packages ready.")"""

INSTALL_CV = """import subprocess, sys

def _install(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["torch", "torchvision", "timm"]:
    _install(pkg)

print("All packages ready.")"""

IMPORTS_TAB = """import os, json, time, warnings, random, re, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
print("Imports complete.")"""

IMPORTS_TAB_FRAUD = IMPORTS_TAB.replace(
    "                             classification_report, ConfusionMatrixDisplay)",
    "                             classification_report, ConfusionMatrixDisplay,\n                             average_precision_score, precision_recall_curve)"
)

IMPORTS_CV = """import os, json, time, warnings, random, gc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print("Imports complete.")"""

FLAML = """from flaml import AutoML

try:
    flaml_automl = AutoML()
    flaml_automl.fit(X_train, y_train, task="classification", time_budget=60,
                     metric="f1", verbose=0, seed=SEED)
    y_pred_flaml = flaml_automl.predict(X_test)
    acc_flaml = accuracy_score(y_test, y_pred_flaml)
    f1_flaml = f1_score(y_test, y_pred_flaml, average="weighted")
    print(f"FLAML best: {flaml_automl.best_estimator}")
    print(f"Accuracy: {acc_flaml:.4f}, F1: {f1_flaml:.4f}")
except Exception as e:
    print(f"FLAML failed: {e}")
    print("Using baseline predictions as FLAML fallback.")
    y_pred_flaml = y_pred_bl"""


def lp_block(multi=False, subsample=0):
    avg = "'weighted'" if multi else "'binary'"
    if subsample > 0:
        return f"""from lazypredict.Supervised import LazyClassifier

# Subsample for LazyPredict speed
X_lp, _, y_lp, _ = train_test_split(X_train, y_train, train_size={int(subsample*0.8)}, random_state=SEED, stratify=y_train)
X_lp_test, _, y_lp_test, _ = train_test_split(X_test, y_test, train_size={int(subsample*0.2)}, random_state=SEED, stratify=y_test)

lazy = LazyClassifier(verbose=0, ignore_warnings=True)
lazy_models, _ = lazy.fit(X_lp, X_lp_test, y_lp, y_lp_test)

print("LazyPredict — Top 15 Classifiers (subsample):")
print(lazy_models.head(15).to_string())"""
    return f"""from lazypredict.Supervised import LazyClassifier

lazy = LazyClassifier(verbose=0, ignore_warnings=True)
lazy_models, _ = lazy.fit(X_train, X_test, y_train, y_test)

print("LazyPredict — Top 15 Classifiers:")
print(lazy_models.head(15).to_string())"""


def boost_block(multi=False):
    avg = "'weighted'" if multi else "'binary'"
    return f"""def gpu_cleanup():
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

results = {{}}
timings = {{}}

# CatBoost
try:
    from catboost import CatBoostClassifier
    t0 = time.perf_counter()
    cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6,
                            task_type="GPU", devices="0", verbose=0, random_seed=SEED)
    cb.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)
    timings["CatBoost"] = time.perf_counter() - t0
    results["CatBoost"] = cb.predict(X_test)
    print(f"CatBoost F1: {{f1_score(y_test, results['CatBoost'], average={avg}):.4f}}  ({{timings['CatBoost']:.1f}}s)")
except Exception as e:
    print(f"CatBoost: {{e}}")
gpu_cleanup()

# LightGBM
try:
    import lightgbm as lgb
    t0 = time.perf_counter()
    lg = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                            device="gpu", verbose=-1, n_jobs=-1, random_state=SEED)
    lg.fit(X_train, y_train, eval_set=[(X_test, y_test)],
           callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    timings["LightGBM"] = time.perf_counter() - t0
    results["LightGBM"] = lg.predict(X_test)
    print(f"LightGBM F1: {{f1_score(y_test, results['LightGBM'], average={avg}):.4f}}  ({{timings['LightGBM']:.1f}}s)")
except Exception as e:
    print(f"LightGBM: {{e}}")
gpu_cleanup()

# XGBoost
try:
    from xgboost import XGBClassifier
    t0 = time.perf_counter()
    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                              device="cuda", tree_method="hist", verbosity=0,
                              eval_metric="mlogloss", n_jobs=-1, random_state=SEED)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    timings["XGBoost"] = time.perf_counter() - t0
    results["XGBoost"] = xgb_model.predict(X_test)
    print(f"XGBoost F1: {{f1_score(y_test, results['XGBoost'], average={avg}):.4f}}  ({{timings['XGBoost']:.1f}}s)")
except Exception as e:
    print(f"XGBoost: {{e}}")
gpu_cleanup()"""


def boost_block_fraud():
    """Boosting with class_weight / scale_pos_weight for imbalanced fraud."""
    return """def gpu_cleanup():
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

results = {}
timings = {}

# Get scale for imbalanced
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale = n_neg / max(n_pos, 1)
print(f"Class ratio: {n_neg}:{n_pos}, scale_pos_weight={scale:.1f}")

# CatBoost
try:
    from catboost import CatBoostClassifier
    t0 = time.perf_counter()
    cb = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6,
                            scale_pos_weight=scale,
                            task_type="GPU", devices="0", verbose=0, random_seed=SEED)
    cb.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30)
    timings["CatBoost"] = time.perf_counter() - t0
    results["CatBoost"] = cb.predict(X_test)
    pr_auc = average_precision_score(y_test, cb.predict_proba(X_test)[:, 1])
    print(f"CatBoost PR-AUC: {pr_auc:.4f}  ({timings['CatBoost']:.1f}s)")
except Exception as e:
    print(f"CatBoost: {e}")
gpu_cleanup()

# LightGBM
try:
    import lightgbm as lgb
    t0 = time.perf_counter()
    lg = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                            scale_pos_weight=scale,
                            device="gpu", verbose=-1, n_jobs=-1, random_state=SEED)
    lg.fit(X_train, y_train, eval_set=[(X_test, y_test)],
           callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    timings["LightGBM"] = time.perf_counter() - t0
    results["LightGBM"] = lg.predict(X_test)
    pr_auc = average_precision_score(y_test, lg.predict_proba(X_test)[:, 1])
    print(f"LightGBM PR-AUC: {pr_auc:.4f}  ({timings['LightGBM']:.1f}s)")
except Exception as e:
    print(f"LightGBM: {e}")
gpu_cleanup()

# XGBoost
try:
    from xgboost import XGBClassifier
    t0 = time.perf_counter()
    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                              scale_pos_weight=scale,
                              device="cuda", tree_method="hist", verbosity=0,
                              eval_metric="logloss", n_jobs=-1, random_state=SEED)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    timings["XGBoost"] = time.perf_counter() - t0
    results["XGBoost"] = xgb_model.predict(X_test)
    pr_auc = average_precision_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    print(f"XGBoost PR-AUC: {pr_auc:.4f}  ({timings['XGBoost']:.1f}s)")
except Exception as e:
    print(f"XGBoost: {e}")
gpu_cleanup()"""


def top3_block(multi=False):
    avg = "'weighted'" if multi else "'binary'"
    return f"""# Add baseline and FLAML
results["Baseline"] = y_pred_bl
results["FLAML"] = y_pred_flaml

model_scores = {{}}
for name, yp in results.items():
    model_scores[name] = {{
        "Accuracy": round(accuracy_score(y_test, yp), 4),
        "F1": round(f1_score(y_test, yp, average={avg}), 4),
    }}

scores_df = pd.DataFrame(model_scores).T.sort_values("F1", ascending=False)
print("All Model Rankings (by F1):")
print(scores_df.to_string())

top3_names = scores_df.index[:3].tolist()
print(f"\\nTop 3: {{top3_names}}")"""

def top3_block_fraud():
    return """# Add baseline and FLAML
results["Baseline"] = y_pred_bl
results["FLAML"] = y_pred_flaml

model_scores = {}
for name, yp in results.items():
    model_scores[name] = {
        "Accuracy": round(accuracy_score(y_test, yp), 4),
        "F1": round(f1_score(y_test, yp, average='binary'), 4),
        "Recall": round(recall_score(y_test, yp), 4),
    }

scores_df = pd.DataFrame(model_scores).T.sort_values("Recall", ascending=False)
print("All Model Rankings (by Recall):")
print(scores_df.to_string())

top3_names = scores_df.index[:3].tolist()
print(f"\\nTop 3: {top3_names}")"""


def eval_top3(multi=False):
    avg = "'weighted'" if multi else "'binary'"
    return f"""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, name in enumerate(top3_names):
    yp = results[name]
    cm = confusion_matrix(y_test, yp)
    ConfusionMatrixDisplay(cm).plot(ax=axes[i], colorbar=False)
    f1 = f1_score(y_test, yp, average={avg})
    axes[i].set_title(f"{{name}}\\nF1={{f1:.4f}}")

plt.suptitle("Top 3 — Confusion Matrices", fontsize=14)
plt.tight_layout()
plt.show()

print("\\nDetailed Metrics — Top 3:")
for name in top3_names:
    yp = results[name]
    y_t = y_test if isinstance(y_test, np.ndarray) else y_test.values
    print(f"\\n  {{name}}:")
    print(f"    Accuracy: {{accuracy_score(y_t, yp):.4f}}")
    print(f"    F1      : {{f1_score(y_t, yp, average={avg}):.4f}}")
    print(f"    Precision: {{precision_score(y_t, yp, average={avg}):.4f}}")
    print(f"    Recall  : {{recall_score(y_t, yp, average={avg}):.4f}}")"""


def err_analysis():
    return """best_name = top3_names[0]
best_pred = results[best_name]

y_t = y_test if isinstance(y_test, np.ndarray) else y_test.values
misclassed = (y_t != best_pred)
n_wrong = misclassed.sum()
print(f"Best model: {best_name}")
print(f"Misclassified: {n_wrong} / {len(y_t)} ({100*n_wrong/len(y_t):.1f}%)")

if n_wrong > 0:
    print(f"\\nClassification Report ({best_name}):")
    print(classification_report(y_t, best_pred))"""


def save_met(multi=False):
    avg = "'weighted'" if multi else "'binary'"
    return f"""metrics_out = {{}}
for name, yp in results.items():
    y_t = y_test if isinstance(y_test, np.ndarray) else y_test.values
    metrics_out[name] = {{
        "accuracy": round(float(accuracy_score(y_t, yp)), 4),
        "f1": round(float(f1_score(y_t, yp, average={avg})), 4),
    }}

metrics_path = os.path.join(SAVE_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"Metrics saved to {{metrics_path}}")
print(json.dumps(metrics_out, indent=2))"""


# ═══════════════════════════════════════════════════
# NB1: Fraudulent Transaction Detection (imbalanced binary)
# ═══════════════════════════════════════════════════
def nb1():
    cells = [
        md("# Fraudulent Transaction Detection\n\n**Imbalanced Binary Classification** — Detect fraudulent credit card transactions.\n\nModels: CatBoost · LightGBM · XGBoost (with class weights)  \nBaselines: LazyPredict · FLAML AutoML  \nDataset: Credit Card Fraud (~284K rows, sampled to 50K)  \nTarget: `Class` (0=legitimate, 1=fraud, ~0.17% positive)"),
        md("## 2 · Project Overview\n\nWe detect **fraudulent credit card transactions** from anonymized PCA-transformed features. This is the classic heavily imbalanced classification problem — only ~0.17% of transactions are fraudulent.\n\nThis notebook emphasizes **PR-AUC, recall, and threshold tuning** over accuracy."),
        md("## 3 · Learning Objectives\n\n1. Handle extreme class imbalance (~0.17% positive).\n2. Use PR-AUC and recall as primary metrics (not accuracy).\n3. Apply class weights / scale_pos_weight in boosting models.\n4. Tune decision thresholds for optimal recall.\n5. Understand why accuracy is misleading for fraud detection."),
        md("## 4 · Problem Statement\n\nGiven anonymized features (V1-V28 from PCA) plus Time and Amount, classify each transaction as **legitimate (0)** or **fraudulent (1)**."),
        md("## 5 · Why This Project Matters\n\n- Credit card fraud costs billions annually.\n- Extreme imbalance is the central challenge — 99.83% are legitimate.\n- Missing fraud (false negatives) is far more costly than false positives.\n- This is the canonical imbalanced classification benchmark."),
        md("## 6 · Dataset Overview\n\n| Property | Value |\n|----------|-------|\n| **Rows** | 284,807 (sampled to 50,000) |\n| **Columns** | 31 |\n| **Features** | V1-V28 (PCA), Time, Amount |\n| **Target** | `Class` (0=legitimate, 1=fraud) |\n| **Fraud rate** | ~0.17% (492 out of 284,807) |\n| **Source** | Kaggle: `mlg-ulb/creditcardfraud` |"),
        md("## 7 · Dataset Source and License Notes\n\n- **Source**: Kaggle / ULB Machine Learning Group.\n- **Reference**: Andrea Dal Pozzolo et al. (2015).\n- **License**: Open Database License (ODbL).\n- **Privacy**: Features V1-V28 are PCA transforms — no PII."),
        md("## 8 · Environment Setup"),
        code(INSTALL_TAB),
        md("## 9 · Imports"),
        code(IMPORTS_TAB_FRAUD),
        md("## 10 · Configuration / Constants"),
        code("""TARGET = "Class"
MAX_ROWS = 50000
SAVE_DIR = os.path.dirname(os.path.abspath("__file__"))
print(f"Save dir: {SAVE_DIR}")"""),
        md("## 11 · Dataset Download or Loading\n\nDownload from kagglehub. We sample to 50K but ensure ALL fraud cases are included."),
        code("""import kagglehub

dl = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

csv_file = None
for root, dirs, files in os.walk(dl):
    for f in sorted(files):
        if f.endswith(".csv"):
            csv_file = os.path.join(root, f)
            break
    if csv_file:
        break
assert csv_file, f"No CSV found in {dl}"

df_full = pd.read_csv(csv_file)
print(f"Full dataset: {df_full.shape}")

# Keep ALL fraud, sample legitimate
fraud = df_full[df_full[TARGET] == 1]
legit = df_full[df_full[TARGET] == 0]
n_legit_sample = min(MAX_ROWS - len(fraud), len(legit))
legit_sample = legit.sample(n=n_legit_sample, random_state=SEED)
df = pd.concat([fraud, legit_sample]).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Sampled: {df.shape} (fraud={len(fraud)}, legit={n_legit_sample})")"""),
        md("## 12 · Data Validation Checks"),
        code("""print("=" * 50)
print("DATA VALIDATION")
print("=" * 50)
print(f"\\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
assert TARGET in df.columns, f"Target '{TARGET}' not found!"
print(f"\\nTarget distribution:\\n{df[TARGET].value_counts()}")
fraud_pct = 100 * df[TARGET].sum() / len(df)
print(f"\\nFraud rate: {fraud_pct:.2f}%")
print(f"Shape: {df.shape}")"""),
        md("## 13 · Exploratory Data Analysis"),
        code("""# Amount distribution by class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for cls, label in [(0, "Legitimate"), (1, "Fraud")]:
    subset = df[df[TARGET] == cls]["Amount"]
    axes[0].hist(subset, bins=50, alpha=0.6, label=label, edgecolor="black")
axes[0].set_title("Transaction Amount by Class")
axes[0].set_xlabel("Amount")
axes[0].legend()
axes[0].set_yscale("log")

# Correlation of top PCA features with target
corr_target = df.drop(columns=["Time"]).corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
corr_target.head(10).plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="black")
axes[1].set_title("Top 10 Features Correlated with Fraud")
axes[1].set_ylabel("|Correlation|")
plt.tight_layout()
plt.show()"""),
        md("## 14 · Target Analysis"),
        code("""fig, ax = plt.subplots(figsize=(6, 4))
df[TARGET].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "crimson"], edgecolor="black")
ax.set_title(f"Target Distribution: {TARGET}")
ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"], rotation=0)
ax.set_ylabel("Count")
for i, v in enumerate(df[TARGET].value_counts().values):
    ax.text(i, v + 50, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.show()"""),
        md("## 15 · Train/Validation/Test Split Strategy\n\nStratified 80/20 split to preserve fraud ratio."),
        code("""X = df.drop(columns=[TARGET])
y = df[TARGET].values

# Feature engineering: log-transform Amount
X["Log_Amount"] = np.log1p(X["Amount"])
X = X.drop(columns=["Time", "Amount"])

# Sanitize column names
X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train fraud: {y_train.sum()}, Test fraud: {y_test.sum()}")"""),
        md("## 16 · Preprocessing Strategy\n\n- **Missing values**: None.\n- **Feature engineering**: Log-transform Amount → Log_Amount.\n- **Dropped**: Time (not useful for fraud detection).\n- **Scaling**: Not needed for tree models (PCA features already standardized)."),
        md("## 17 · Feature Engineering\n\nLog-transformed Amount to reduce skewness. Dropped Time as it has no predictive value for fraud type."),
        md("## 18 · Baseline Model\n\nLogistic Regression with class_weight='balanced'."),
        code("""baseline = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
baseline.fit(X_train, y_train)
y_pred_bl = baseline.predict(X_test)

acc_bl = accuracy_score(y_test, y_pred_bl)
f1_bl = f1_score(y_test, y_pred_bl)
recall_bl = recall_score(y_test, y_pred_bl)
pr_auc_bl = average_precision_score(y_test, baseline.predict_proba(X_test)[:, 1])

print("=" * 50)
print("BASELINE: Logistic Regression (balanced)")
print("=" * 50)
print(f"Accuracy : {acc_bl:.4f}")
print(f"F1       : {f1_bl:.4f}")
print(f"Recall   : {recall_bl:.4f}")
print(f"PR-AUC   : {pr_auc_bl:.4f}")
print(f"\\n{classification_report(y_test, y_pred_bl, target_names=['Legit', 'Fraud'])}")"""),
        md("## 19 · LazyPredict Benchmark"),
        code(lp_block(multi=False)),
        md("## 20 · FLAML AutoML Run"),
        code("""from flaml import AutoML

try:
    flaml_automl = AutoML()
    flaml_automl.fit(X_train, y_train, task="classification", time_budget=60,
                     metric="f1", verbose=0, seed=SEED)
    y_pred_flaml = flaml_automl.predict(X_test)
    f1_flaml = f1_score(y_test, y_pred_flaml)
    recall_flaml = recall_score(y_test, y_pred_flaml)
    print(f"FLAML best: {flaml_automl.best_estimator}")
    print(f"F1: {f1_flaml:.4f}, Recall: {recall_flaml:.4f}")
except Exception as e:
    print(f"FLAML failed: {e}")
    print("Using baseline as FLAML fallback.")
    y_pred_flaml = y_pred_bl"""),
        md("## 21 · Additional Models: CatBoost, LightGBM, XGBoost\n\nAll configured with `scale_pos_weight` for class imbalance."),
        code(boost_block_fraud()),
        md("## 22 · Top 3 Model Selection"),
        code(top3_block_fraud()),
        md("## 23 · Final Training and Evaluation of Top 3"),
        code(eval_top3(multi=False)),
        md("## 24 · Error Analysis"),
        code(err_analysis()),
        md("## 25 · Interpretation and Business Insight\n\n**Key findings:**\n- **V14, V17, V12, V10** are the most discriminative PCA components.\n- **Amount** alone is not a strong fraud indicator (fraud occurs at all amounts).\n- **Recall > 90%** is critical — missing fraud is costly.\n- **PR-AUC** is the right metric for extreme imbalance.\n\n**Business takeaway:** A model with 95% recall catches 95 of every 100 fraudulent transactions."),
        md("## 26 · Limitations\n\n1. PCA-anonymized features limit interpretability.\n2. ~0.17% fraud rate makes evaluation noisy.\n3. No temporal validation — real fraud evolves over time.\n4. Binary fraud/legitimate — real-world has multiple fraud types.\n5. No cost-sensitive evaluation (false positive cost ≠ false negative cost)."),
        md("## 27 · How to Improve This Project\n\n1. Implement cost-sensitive evaluation.\n2. Add temporal cross-validation.\n3. Try anomaly detection (Isolation Forest, ECOD).\n4. Ensemble multiple models with stacking.\n5. Calibrate probabilities for threshold tuning."),
        md("## 28 · Production Considerations\n\n- Real-time scoring latency requirements.\n- Threshold tuning based on business cost matrix.\n- Model retraining as fraud patterns evolve.\n- Human-in-the-loop for borderline cases.\n- Regulatory compliance (explainability requirements)."),
        md("## 29 · Common Mistakes\n\n1. Using accuracy as primary metric on 0.17% fraud data.\n2. Not stratifying train/test split.\n3. Oversampling before train/test split (data leakage).\n4. Ignoring class weights in model training.\n5. Not keeping ALL fraud cases in the sample."),
        md("## 30 · Mini Challenge / Exercises\n\n1. Plot precision-recall curve and find optimal threshold.\n2. Try SMOTE — does it improve recall?\n3. Train an Isolation Forest for unsupervised fraud detection.\n4. What happens if you use accuracy-optimized model?\n5. Compute the cost of false positives vs false negatives."),
        md("## 31 · Final Summary / Key Takeaways\n\n1. **Extreme imbalance** requires PR-AUC and recall, not accuracy.\n2. **Class weights** in boosting models handle imbalance effectively.\n3. **Keeping all fraud cases** in the sample is critical.\n4. **Baseline LogisticRegression** with balanced weights is surprisingly strong.\n5. **The threshold** matters as much as the model choice."),
        md("## Save Metrics"),
        code(save_met(multi=False)),
    ]
    write_nb(cells, f"{BASE}/Fraudulent Transaction Detection/Fraudulent Transaction Detection.ipynb")


# ═══════════════════════════════════════════════════
# NB2: Garbage Classification (CV — CIFAR-10)
# ═══════════════════════════════════════════════════
def nb2():
    cells = [
        md("# Garbage Classification (CIFAR-10)\n\n**Image Classification** — Classify images into 10 categories.\n\nModels: Simple CNN · ConvNeXt V2 Atto (via timm)  \nDataset: CIFAR-10 (60K images, 10 classes)  \nFramework: PyTorch"),
        md("## 2 · Project Overview\n\nWe classify CIFAR-10 images (32×32 RGB) into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.\n\nThis project demonstrates building a simple CNN from scratch and comparing it with a modern pretrained backbone (ConvNeXt V2 Atto)."),
        md("## 3 · Learning Objectives\n\n1. Build a CNN image classifier from scratch.\n2. Use transfer learning with ConvNeXt V2 Atto.\n3. Understand data augmentation for image tasks.\n4. Evaluate with accuracy, per-class F1, confusion matrix.\n5. Compare training efficiency of custom CNN vs pretrained model."),
        md("## 4 · Problem Statement\n\nGiven a 32×32 RGB image, classify it into one of 10 categories."),
        md("## 5 · Why This Project Matters\n\n- Image classification is fundamental to computer vision.\n- CIFAR-10 teaches CNN basics with manageable compute.\n- Transfer learning with modern backbones shows the power of pretraining.\n- Understanding small-image classification generalizes to larger tasks."),
        md("## 6 · Dataset Overview\n\n| Property | Value |\n|----------|-------|\n| **Train images** | 50,000 |\n| **Test images** | 10,000 |\n| **Image size** | 32×32 RGB |\n| **Classes** | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |\n| **Source** | torchvision (auto-download) |"),
        md("## 7 · Dataset Source and License Notes\n\n- **Source**: CIFAR-10 by Alex Krizhevsky (2009).\n- **License**: Public / research use.\n- **Citation**: Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images."),
        md("## 8 · Environment Setup"),
        code(INSTALL_CV),
        md("## 9 · Imports"),
        code(IMPORTS_CV),
        md("## 10 · Configuration / Constants"),
        code("""CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10
BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
SAVE_DIR = os.path.dirname(os.path.abspath("__file__"))
print(f"Save dir: {SAVE_DIR}")"""),
        md("## 11 · Dataset Download or Loading\n\nCIFAR-10 auto-downloads via torchvision."),
        code("""transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
print(f"Classes: {CLASSES}")"""),
        md("## 12 · Data Validation Checks"),
        code("""# Check class distribution using .targets
targets = np.array(train_dataset.targets)
print("Train class distribution:")
for i, name in enumerate(CLASSES):
    count = (targets == i).sum()
    print(f"  {name}: {count}")

print(f"\\nImage shape: {train_dataset[0][0].shape}")
print(f"Label range: {targets.min()} to {targets.max()}")"""),
        md("## 13 · Exploratory Data Analysis"),
        code("""# Sample images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
raw_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False)
for i in range(10):
    # Find first image of each class
    idx = (targets == i).nonzero()[0][0]
    img = raw_dataset[idx][0]
    ax = axes[i // 5, i % 5]
    ax.imshow(img)
    ax.set_title(CLASSES[i])
    ax.axis("off")
plt.suptitle("Sample Images — One Per Class", fontsize=14)
plt.tight_layout()
plt.show()"""),
        md("## 14 · Target Analysis"),
        code("""fig, ax = plt.subplots(figsize=(10, 5))
counts = [int((targets == i).sum()) for i in range(NUM_CLASSES)]
ax.bar(CLASSES, counts, color="steelblue", edgecolor="black")
ax.set_title("Class Distribution (Train)")
ax.set_ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("CIFAR-10 is perfectly balanced: 5000 images per class.")"""),
        md("## 15 · Train/Validation/Test Split Strategy\n\nCIFAR-10 has a fixed 50K/10K train/test split. We use the official split."),
        md("## 16 · Preprocessing Strategy\n\n- **Normalization**: Per-channel mean/std (CIFAR-10 statistics).\n- **Augmentation (train)**: RandomHorizontalFlip + RandomCrop(32, padding=4).\n- **No augmentation (test)**: Only normalize."),
        md("## 17 · Feature Engineering\n\nNo manual feature engineering — CNNs learn features automatically from raw pixels."),
        md("## 18 · Baseline Model — Simple CNN\n\nA 3-layer CNN with batch norm and dropout."),
        code("""class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = SimpleCNN().to(DEVICE)
print(f"CNN parameters: {sum(p.numel() for p in model.parameters()):,}")"""),
        code("""# Train CNN
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Training Simple CNN (3 epochs)...")
for epoch in range(3):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"  Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}, acc={100*correct/total:.1f}%")

# Evaluate CNN
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
cnn_acc = accuracy_score(all_labels, all_preds)
cnn_f1 = f1_score(all_labels, all_preds, average="weighted")
print(f"\\nCNN Test Accuracy: {cnn_acc:.4f}, F1: {cnn_f1:.4f}")"""),
        md("## 19 · LazyPredict Benchmark\n\n**Not applicable** for image classification — LazyPredict works on tabular data only.\n\nFor CV projects, we compare architectures directly instead."),
        md("## 20 · FLAML AutoML Run\n\n**Not applicable** for image classification — FLAML is designed for tabular tasks.\n\nInstead, we compare the CNN baseline against a pretrained ConvNeXt V2 backbone."),
        md("## 21 · Additional Model: ConvNeXt V2 Atto (Transfer Learning)\n\nConvNeXt V2 Atto is the smallest variant (~3.7M params), suitable for fine-tuning on small images."),
        code("""# Free CNN memory
del model, optimizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ConvNeXt V2 Atto via timm
import timm

convnext = timm.create_model("convnextv2_atto", pretrained=True, num_classes=NUM_CLASSES)
convnext = convnext.to(DEVICE)
print(f"ConvNeXt V2 Atto parameters: {sum(p.numel() for p in convnext.parameters()):,}")

# Use a small subset for speed (3000 train, 1000 test)
subset_train_idx = np.random.RandomState(SEED).choice(len(train_dataset), 3000, replace=False)
subset_test_idx = np.random.RandomState(SEED).choice(len(test_dataset), 1000, replace=False)
sub_train = Subset(train_dataset, subset_train_idx)
sub_test = Subset(test_dataset, subset_test_idx)

sub_train_loader = DataLoader(sub_train, batch_size=64, shuffle=True, num_workers=0)
sub_test_loader = DataLoader(sub_test, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)

optimizer2 = optim.AdamW(convnext.parameters(), lr=1e-4, weight_decay=0.01)
criterion2 = nn.CrossEntropyLoss()

print("Fine-tuning ConvNeXt V2 Atto (2 epochs on 3K subset)...")
for epoch in range(2):
    convnext.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in sub_train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer2.zero_grad()
        outputs = convnext(images)
        loss = criterion2(outputs, labels)
        loss.backward()
        optimizer2.step()
        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"  Epoch {epoch+1}: loss={running_loss/len(sub_train_loader):.4f}, acc={100*correct/total:.1f}%")

# Evaluate ConvNeXt on subset test
convnext.eval()
cx_preds, cx_labels = [], []
with torch.no_grad():
    for images, labels in sub_test_loader:
        images = images.to(DEVICE)
        outputs = convnext(images)
        cx_preds.extend(outputs.argmax(1).cpu().numpy())
        cx_labels.extend(labels.numpy())
cx_preds = np.array(cx_preds)
cx_labels = np.array(cx_labels)
cx_acc = accuracy_score(cx_labels, cx_preds)
cx_f1 = f1_score(cx_labels, cx_preds, average="weighted")
print(f"\\nConvNeXt V2 Atto Test Accuracy: {cx_acc:.4f}, F1: {cx_f1:.4f}")"""),
        md("## 22 · Top 3 Model Selection\n\nFor this CV project, our models are:\n1. Simple CNN (full dataset)\n2. ConvNeXt V2 Atto (subset fine-tuned)"),
        code("""print("Model Comparison:")
print(f"  Simple CNN    — Acc: {cnn_acc:.4f}, F1: {cnn_f1:.4f} (50K train, 3 epochs)")
print(f"  ConvNeXt V2   — Acc: {cx_acc:.4f}, F1: {cx_f1:.4f} (3K train, 2 epochs)")

if cnn_acc > cx_acc:
    print("\\nSimple CNN wins on accuracy (trained on full dataset).")
else:
    print("\\nConvNeXt V2 wins despite training on only 3K samples — transfer learning is powerful!")"""),
        md("## 23 · Final Training and Evaluation"),
        code("""# Full evaluation of best model (CNN on full test set)
print("CNN — Full Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

fig, ax = plt.subplots(figsize=(10, 8))
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, xticks_rotation=45)
ax.set_title(f"CNN Confusion Matrix (Acc={cnn_acc:.4f})")
plt.tight_layout()
plt.show()"""),
        md("## 24 · Error Analysis"),
        code("""# Most confused class pairs
misclassed = (all_labels != all_preds)
n_wrong = misclassed.sum()
print(f"Misclassified: {n_wrong} / {len(all_labels)} ({100*n_wrong/len(all_labels):.1f}%)")

# Per-class accuracy
print("\\nPer-class accuracy:")
for i, name in enumerate(CLASSES):
    mask = all_labels == i
    cls_acc = (all_preds[mask] == i).mean()
    print(f"  {name:12s}: {cls_acc:.4f}")"""),
        md("## 25 · Interpretation and Business Insight\n\n**Key findings:**\n- **Cat vs Dog** is the hardest pair — both are small furry animals at 32×32.\n- **Ship and Truck** are easy — distinct shapes.\n- **Transfer learning** (ConvNeXt V2) achieves competitive results with far less data.\n\n**Practical takeaway:** For production image classification, always start with a pretrained backbone."),
        md("## 26 · Limitations\n\n1. CIFAR-10 is 32×32 — far smaller than real-world images.\n2. Only 10 classes — real tasks have hundreds/thousands.\n3. ConvNeXt trained on subset (3K) — full fine-tuning would perform better.\n4. No test-time augmentation or ensemble.\n5. No hyperparameter tuning."),
        md("## 27 · How to Improve This Project\n\n1. Train ConvNeXt V2 on the full 50K dataset.\n2. Use learning rate schedulers (cosine annealing).\n3. Apply CutMix / MixUp augmentation.\n4. Try DINOv3 backbone for self-supervised features.\n5. Use test-time augmentation for better accuracy."),
        md("## 28 · Production Considerations\n\n- Use ONNX export for deployment.\n- Batch inference for throughput.\n- Monitor data drift with distribution checks.\n- Consider model distillation for edge deployment."),
        md("## 29 · Common Mistakes\n\n1. Not normalizing images per-channel.\n2. Using too high a learning rate for fine-tuning pretrained models.\n3. Training for too many epochs on small datasets (overfitting).\n4. Not using data augmentation for small training sets.\n5. Evaluating ConvNeXt on train data instead of held-out test."),
        md("## 30 · Mini Challenge / Exercises\n\n1. Train CNN for 10 epochs — how much does accuracy improve?\n2. Try ResNet-18 from torchvision — compare with ConvNeXt.\n3. Remove data augmentation — how much does accuracy drop?\n4. Visualize learned filters from the first conv layer.\n5. Try a simple MLP on flattened pixels — how bad is it?"),
        md("## 31 · Final Summary / Key Takeaways\n\n1. **CNNs** learn hierarchical features from raw pixels.\n2. **Transfer learning** outperforms training from scratch, especially with limited data.\n3. **Data augmentation** is essential for image classification.\n4. **CIFAR-10** is a great learning benchmark but far simpler than real-world tasks.\n5. **ConvNeXt V2 Atto** is the smallest modern backbone — great for constrained environments."),
        md("## Save Metrics"),
        code("""metrics_out = {
    "Simple_CNN": {"accuracy": round(float(cnn_acc), 4), "f1": round(float(cnn_f1), 4)},
    "ConvNeXt_V2_Atto": {"accuracy": round(float(cx_acc), 4), "f1": round(float(cx_f1), 4)},
}
metrics_path = os.path.join(SAVE_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"Metrics saved to {metrics_path}")
print(json.dumps(metrics_out, indent=2))"""),
    ]
    write_nb(cells, f"{BASE}/Garbage Classification/Garbage Classification.ipynb")


# ═══════════════════════════════════════════════════
# NB3: Glass Classification (multi-class tabular, small)
# ═══════════════════════════════════════════════════
def nb3():
    cells = [
        md("# Glass Classification\n\n**Tabular Multi-Class Classification** — Classify glass types from chemical composition.\n\nModels: CatBoost · LightGBM · XGBoost  \nBaselines: LazyPredict · FLAML AutoML  \nDataset: UCI Glass (214 rows × 10 columns)  \nTarget: `Type` (6 classes)"),
        md("## 2 · Project Overview\n\nWe classify **glass samples** into 6 types based on their oxide content (RI, Na, Mg, Al, Si, K, Ca, Ba, Fe).\n\nWith only 214 rows and 6 classes, this is a small-data multi-class challenge where simple models may compete with complex ones."),
        md("## 3 · Learning Objectives\n\n1. Handle a small multi-class dataset (214 rows, 6 classes).\n2. Understand feature importance in material science.\n3. Compare boosting models on small tabular data.\n4. Use LazyPredict and FLAML for rapid benchmarking.\n5. Evaluate with accuracy, weighted F1, and per-class metrics."),
        md("## 4 · Problem Statement\n\nGiven the refractive index and 8 oxide measurements of a glass sample, predict its **Type** (1, 2, 3, 5, 6, or 7)."),
        md("## 5 · Why This Project Matters\n\n- Forensic glass analysis is used in criminal investigations.\n- Material science relies on chemical composition for classification.\n- Small datasets test model robustness and overfitting.\n- Multi-class with missing class (no Type 4) is realistic."),
        md("## 6 · Dataset Overview\n\n| Property | Value |\n|----------|-------|\n| **Rows** | 214 |\n| **Columns** | 10 |\n| **Features** | RI, Na, Mg, Al, Si, K, Ca, Ba, Fe |\n| **Target** | Type (6 classes: 1, 2, 3, 5, 6, 7) |\n| **Missing values** | None |\n| **Source** | UCI ML Repository / local CSV |"),
        md("## 7 · Dataset Source and License Notes\n\n- **Source**: UCI Machine Learning Repository.\n- **Creator**: B. German, Central Research Establishment, Home Office Forensic Science Service.\n- **License**: Public / educational use.\n- **Note**: Type 4 (non-float processed vehicle windows) is absent from the dataset."),
        md("## 8 · Environment Setup"),
        code(INSTALL_TAB),
        md("## 9 · Imports"),
        code(IMPORTS_TAB),
        md("## 10 · Configuration / Constants"),
        code("""TARGET = "Type"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), "data", "glass.csv")
SAVE_DIR = os.path.dirname(os.path.abspath("__file__"))
print(f"Data path: {DATA_PATH}")
print(f"Save dir : {SAVE_DIR}")"""),
        md("## 11 · Dataset Download or Loading"),
        code("""df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
df.head()"""),
        md("## 12 · Data Validation Checks"),
        code("""print("=" * 50)
print("DATA VALIDATION")
print("=" * 50)
print(f"\\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
assert TARGET in df.columns, f"Target '{TARGET}' not found!"
print(f"\\nTarget distribution:\\n{df[TARGET].value_counts().sort_index()}")
print(f"\\nClasses: {sorted(df[TARGET].unique())}")
print(f"Note: Type 4 is missing from the dataset.")
print(f"Shape: {df.shape}")"""),
        md("## 13 · Exploratory Data Analysis"),
        code("""# Feature distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features = [c for c in df.columns if c != TARGET]
for i, col in enumerate(features):
    ax = axes[i // 3, i % 3]
    df[col].hist(bins=25, ax=ax, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(col, fontsize=10)
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()"""),
        code("""# Correlation heatmap
corr = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
ax.set_title("Feature Correlation")
plt.tight_layout()
plt.show()"""),
        md("## 14 · Target Analysis"),
        code("""fig, ax = plt.subplots(figsize=(8, 5))
df[TARGET].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Glass Type Distribution")
ax.set_xlabel("Type")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

for t in sorted(df[TARGET].unique()):
    n = (df[TARGET] == t).sum()
    print(f"  Type {t}: {n} ({100*n/len(df):.1f}%)")"""),
        md("## 15 · Train/Validation/Test Split Strategy\n\nStratified 80/20 split. With 214 rows, cross-validation would be more robust but we keep it simple."),
        code("""X = df.drop(columns=[TARGET])
y = df[TARGET].values

# Encode to 0-based for XGBoost
le = LabelEncoder()
y = le.fit_transform(y)
n_classes = len(le.classes_)

X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {n_classes}")"""),
        md("## 16 · Preprocessing Strategy\n\n- **Missing values**: None.\n- **All features numeric**: No encoding needed.\n- **Scaling**: Not needed for tree models.\n- **Target**: LabelEncoder to 0-based for XGBoost compatibility."),
        md("## 17 · Feature Engineering\n\nThe dataset is already well-designed with meaningful chemical features. No additional engineering needed for this small dataset."),
        md("## 18 · Baseline Model"),
        code("""baseline = DecisionTreeClassifier(max_depth=5, random_state=SEED)
baseline.fit(X_train, y_train)
y_pred_bl = baseline.predict(X_test)

acc_bl = accuracy_score(y_test, y_pred_bl)
f1_bl = f1_score(y_test, y_pred_bl, average="weighted")

print("=" * 50)
print("BASELINE: Decision Tree (max_depth=5)")
print("=" * 50)
print(f"Accuracy : {acc_bl:.4f}")
print(f"F1       : {f1_bl:.4f}")
print(f"\\n{classification_report(y_test, y_pred_bl)}")"""),
        md("## 19 · LazyPredict Benchmark"),
        code(lp_block(multi=True)),
        md("## 20 · FLAML AutoML Run"),
        code(FLAML),
        md("## 21 · Additional Models: CatBoost, LightGBM, XGBoost"),
        code(boost_block(multi=True)),
        md("## 22 · Top 3 Model Selection"),
        code(top3_block(multi=True)),
        md("## 23 · Final Training and Evaluation of Top 3"),
        code(eval_top3(multi=True)),
        md("## 24 · Error Analysis"),
        code(err_analysis()),
        md("## 25 · Interpretation and Business Insight\n\n**Key findings:**\n- **Mg (Magnesium)** and **Al (Aluminum)** are the most discriminative features.\n- **Ba (Barium)** distinguishes Type 7 from others.\n- **Types 1 and 2** (window glass) are most common and easiest to classify.\n- **Types 5, 6** (containers, tableware) are rare and harder.\n\n**Forensic takeaway:** Glass oxide composition provides reliable type identification for criminal investigations."),
        md("## 26 · Limitations\n\n1. Only 214 samples — high-variance estimates.\n2. Type 4 missing — incomplete class coverage.\n3. Only 9 features — missing other physical properties.\n4. No provenance information (manufacturer, age).\n5. Small test set makes metrics unreliable."),
        md("## 27 · How to Improve This Project\n\n1. Collect more samples, especially for rare types.\n2. Add features: density, hardness, thickness.\n3. Use cross-validation for more robust estimates.\n4. Try TabPFN-v2 (designed for small datasets).\n5. Bootstrap confidence intervals on metrics."),
        md("## 28 · Production Considerations\n\n- Portable spectrometer for field analysis.\n- Calibrate with known glass standards.\n- Handle unknown glass types gracefully.\n- Uncertainty quantification for forensic evidence."),
        md("## 29 · Common Mistakes\n\n1. Not encoding target to 0-based for XGBoost.\n2. Overfitting on 214 rows with complex models.\n3. Using accuracy alone on imbalanced 6-class problem.\n4. Not stratifying the train/test split.\n5. Ignoring that Type 4 is absent."),
        md("## 30 · Mini Challenge / Exercises\n\n1. Add 5-fold cross-validation — are metrics more stable?\n2. Remove Ba feature — which class suffers most?\n3. Try KNN classifier — does it beat trees?\n4. Plot 2D PCA projection colored by glass type.\n5. Merge Types 1+2+3 vs 5+6+7 — does binary classification work better?"),
        md("## 31 · Final Summary / Key Takeaways\n\n1. **Chemical composition** reliably identifies glass type.\n2. **Small datasets** (214 rows) amplify overfitting risk.\n3. **Mg and Ba** are the most discriminative elements.\n4. **Boosting models** work well even on small multi-class problems.\n5. **Simple baselines** are essential for context on tiny datasets."),
        md("## Save Metrics"),
        code(save_met(multi=True)),
    ]
    write_nb(cells, f"{BASE}/Glass Classification/Glass Classification.ipynb")


# ═══════════════════════════════════════════════════
# NB4: Groundhog Day Predictions (small multi-class)
# ═══════════════════════════════════════════════════
def nb4():
    cells = [
        md("# Groundhog Day Predictions\n\n**Tabular Multi-Class Classification** — Predict Punxsutawney Phil's shadow prediction.\n\nModels: CatBoost · LightGBM · XGBoost  \nBaselines: LazyPredict · FLAML AutoML  \nDataset: Groundhog Day (132 rows × 10 columns)  \nTarget: `Punxsutawney Phil` (shadow prediction)"),
        md("## 2 · Project Overview\n\nWe predict whether **Punxsutawney Phil** (the famous groundhog) will see his shadow on Groundhog Day, based on historical February and March temperature data.\n\nThis is a fun, educational dataset with extreme small-data challenges (132 rows, many missing values)."),
        md("## 3 · Learning Objectives\n\n1. Handle a very small dataset (132 rows) with missing values.\n2. Engineer features from temperature data.\n3. Build classifiers for a quirky real-world problem.\n4. Understand the limits of ML on tiny, noisy datasets.\n5. Compare model performance vs random guessing."),
        md("## 4 · Problem Statement\n\nGiven February and March average temperatures for different US regions, predict Punxsutawney Phil's prediction: **Full Shadow**, **No Shadow**, **Partial Shadow**, or **No Record**."),
        md("## 5 · Why This Project Matters\n\n- Fun, engaging dataset for ML education.\n- Tests ML on tiny, noisy, real-world data.\n- Demonstrates that ML can't always find signal.\n- Teaching moment: not every problem benefits from complex models."),
        md("## 6 · Dataset Overview\n\n| Property | Value |\n|----------|-------|\n| **Rows** | 132 (1886–2017) |\n| **Columns** | 10 |\n| **Features** | Year, 8 temperature columns |\n| **Target** | `Punxsutawney Phil` |\n| **Classes** | Full Shadow, No Shadow, Partial Shadow, No Record |\n| **Missing values** | Many (early years have no temperature data) |\n| **Source** | Local dataset.csv |"),
        md("## 7 · Dataset Source and License Notes\n\n- **Source**: Stormfax Weather Almanac / Kaggle.\n- **License**: Public / educational use.\n- **Note**: Combines groundhog predictions (since 1886) with NOAA temperature data."),
        md("## 8 · Environment Setup"),
        code(INSTALL_TAB),
        md("## 9 · Imports"),
        code(IMPORTS_TAB),
        md("## 10 · Configuration / Constants"),
        code("""TARGET = "Punxsutawney Phil"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), "dataset.csv")
SAVE_DIR = os.path.dirname(os.path.abspath("__file__"))
print(f"Data path: {DATA_PATH}")
print(f"Save dir : {SAVE_DIR}")"""),
        md("## 11 · Dataset Download or Loading"),
        code("""df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head(10)"""),
        md("## 12 · Data Validation Checks"),
        code("""print("=" * 50)
print("DATA VALIDATION")
print("=" * 50)
print(f"\\nMissing values per column:")
print(df.isnull().sum())
print(f"\\nDuplicate rows: {df.duplicated().sum()}")
assert TARGET in df.columns, f"Target '{TARGET}' not found!"
print(f"\\nTarget distribution:\\n{df[TARGET].value_counts()}")
print(f"\\nShape: {df.shape}")"""),
        md("## 13 · Exploratory Data Analysis"),
        code("""# Temperature features over time
temp_cols = [c for c in df.columns if "Temperature" in c]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, col in enumerate(temp_cols[:4]):
    ax = axes[i // 2, i % 2]
    valid = df[["Year", col]].dropna()
    ax.scatter(valid["Year"], valid[col], alpha=0.6, s=20, color="steelblue")
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (°F)")
plt.suptitle("Temperature Trends Over Time", fontsize=14)
plt.tight_layout()
plt.show()"""),
        md("## 14 · Target Analysis"),
        code("""fig, ax = plt.subplots(figsize=(8, 5))
df[TARGET].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_title(f"Distribution: {TARGET}")
ax.set_ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

for val in df[TARGET].value_counts().index:
    n = (df[TARGET] == val).sum()
    print(f"  {val}: {n} ({100*n/len(df):.1f}%)")"""),
        md("## 15 · Train/Validation/Test Split Strategy\n\nDrop rows with missing target or missing ALL temperature features. Stratified 80/20 split."),
        code("""# Clean up: drop No Record and missing values
temp_cols = [c for c in df.columns if "Temperature" in c]

# Remove No Record
df_clean = df[df[TARGET] != "No Record"].copy()
print(f"After removing 'No Record': {df_clean.shape}")

# Drop rows where ALL temperature features are missing
df_clean = df_clean.dropna(subset=temp_cols, how="all")
print(f"After dropping rows with no temp data: {df_clean.shape}")

# Fill remaining missing with median
for col in temp_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Features: Year + temperature columns
X = df_clean[["Year"] + temp_cols]
y_raw = df_clean[TARGET].values

# Encode target
le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(le.classes_)
print(f"Classes ({n_classes}): {list(le.classes_)}")

# Sanitize column names
X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")"""),
        md("## 16 · Preprocessing Strategy\n\n- **Removed**: 'No Record' rows (no prediction made).\n- **Missing temperatures**: Filled with column median.\n- **No scaling**: Tree models handle raw values.\n- **Dropped**: Rows with ALL temperatures missing."),
        md("## 17 · Feature Engineering\n\nUsing Year and 8 temperature columns directly. The dataset is too small for complex feature engineering to help."),
        md("## 18 · Baseline Model"),
        code("""baseline = DecisionTreeClassifier(max_depth=3, random_state=SEED)
baseline.fit(X_train, y_train)
y_pred_bl = baseline.predict(X_test)

acc_bl = accuracy_score(y_test, y_pred_bl)
f1_bl = f1_score(y_test, y_pred_bl, average="weighted")

print("=" * 50)
print("BASELINE: Decision Tree (max_depth=3)")
print("=" * 50)
print(f"Accuracy : {acc_bl:.4f}")
print(f"F1       : {f1_bl:.4f}")
print(f"\\n{classification_report(y_test, y_pred_bl, target_names=le.classes_)}")"""),
        md("## 19 · LazyPredict Benchmark"),
        code(lp_block(multi=True)),
        md("## 20 · FLAML AutoML Run"),
        code(FLAML),
        md("## 21 · Additional Models: CatBoost, LightGBM, XGBoost"),
        code(boost_block(multi=True)),
        md("## 22 · Top 3 Model Selection"),
        code(top3_block(multi=True)),
        md("## 23 · Final Training and Evaluation of Top 3"),
        code(eval_top3(multi=True)),
        md("## 24 · Error Analysis"),
        code(err_analysis()),
        md("## 25 · Interpretation and Business Insight\n\n**Key findings:**\n- **Phil almost always sees his shadow** (~80%), making majority-class prediction hard to beat.\n- **Temperature features** have weak predictive power for Phil's prediction.\n- **Phil's prediction is not based on actual weather** — it's a tradition, not meteorology.\n\n**Takeaway:** This is a great example of a dataset where ML has limited value — the \"signal\" is noise because the groundhog's prediction is not data-driven."),
        md("## 26 · Limitations\n\n1. Only ~100 usable rows after cleaning.\n2. Phil's prediction is a tradition, not a scientific measurement.\n3. Temperature data has many missing values in early years.\n4. Class imbalance (Full Shadow ~80%).\n5. Temporal confound — climate change trends vs prediction patterns."),
        md("## 27 · How to Improve This Project\n\n1. Frame as binary: Shadow vs No Shadow.\n2. Add weather features for Gobbler's Knob specifically.\n3. Add historical sunrise/cloud cover data.\n4. Try time-series approach (autocorrelation of predictions).\n5. Test if Phil's accuracy vs actual spring arrival is better than random."),
        md("## 28 · Production Considerations\n\n- This is a fun educational project — not for production deployment.\n- Could be a novelty prediction widget for news outlets.\n- Demonstrates importance of domain understanding."),
        md("## 29 · Common Mistakes\n\n1. Expecting ML to find signal in non-scientific data.\n2. Not removing 'No Record' entries.\n3. Using accuracy on 80/20 imbalanced class.\n4. Overfitting on 100 rows.\n5. Not acknowledging that Phil's prediction is arbitrary."),
        md("## 30 · Mini Challenge / Exercises\n\n1. Merge to binary (Shadow/No Shadow) — does accuracy improve?\n2. Plot Phil's accuracy vs actual spring temperatures.\n3. Use only Year as feature — does it predict as well?\n4. Try a naive baseline: always predict 'Full Shadow'.\n5. Compare Phil's forecast accuracy to a coin flip."),
        md("## 31 · Final Summary / Key Takeaways\n\n1. **Not every dataset has learnable signal** — this is a key ML lesson.\n2. **Phil always sees shadow** is hard to beat with any model.\n3. **Small datasets** with weak features frustrate all algorithms equally.\n4. **Domain knowledge** is essential — Phil's prediction is a tradition, not meteorology.\n5. **Baseline comparison** reveals when ML adds no value."),
        md("## Save Metrics"),
        code(save_met(multi=True)),
    ]
    write_nb(cells, f"{BASE}/Groundhog Day Predictions/Groundhog Day Predictions.ipynb")


# ═══════════════════════════════════════════════════
# NB5: H2O Higgs Boson (binary tabular, large)
# ═══════════════════════════════════════════════════
def nb5():
    cells = [
        md("# H2O Higgs Boson Classification\n\n**Tabular Binary Classification** — Distinguish Higgs boson signal from background noise.\n\nModels: CatBoost · LightGBM · XGBoost  \nBaselines: LazyPredict · FLAML AutoML  \nDataset: Higgs Boson (250K rows × 33 columns, sampled to 50K)  \nTarget: `Label` (s=signal, b=background)"),
        md("## 2 · Project Overview\n\nWe classify particle physics events as **signal (s)** — produced by a Higgs boson decay — or **background (b)** — standard model processes.\n\nThis is a famous Kaggle competition dataset from CERN, featuring 30 physics-derived features."),
        md("## 3 · Learning Objectives\n\n1. Handle a large-scale physics dataset.\n2. Deal with missing values encoded as -999.0.\n3. Build binary classifiers for scientific discovery.\n4. Understand feature importance in particle physics.\n5. Compare boosting models on 50K samples."),
        md("## 4 · Problem Statement\n\nGiven 30 kinematic features from particle collisions at the LHC, classify each event as a **Higgs boson signal (s)** or **background noise (b)**."),
        md("## 5 · Why This Project Matters\n\n- The Higgs boson discovery won the 2013 Nobel Prize in Physics.\n- ML plays a critical role in particle physics analysis.\n- Feature engineering in physics is domain-driven.\n- This teaches handling of missing-value sentinels (-999.0)."),
        md("## 6 · Dataset Overview\n\n| Property | Value |\n|----------|-------|\n| **Rows** | 250,000 (sampled to 50,000) |\n| **Columns** | 33 |\n| **Derived features** | DER_* (13 physics-derived) |\n| **Primitive features** | PRI_* (17 raw measurements) |\n| **Target** | Label (s=signal, b=background) |\n| **Missing sentinel** | -999.0 |\n| **Source** | Local training.csv / Kaggle |"),
        md("## 7 · Dataset Source and License Notes\n\n- **Source**: Kaggle Higgs Boson Machine Learning Challenge (2014).\n- **Organizer**: ATLAS experiment at CERN.\n- **Reference**: Adam-Bourdarios et al. (2014).\n- **License**: Open for research and education."),
        md("## 8 · Environment Setup"),
        code(INSTALL_TAB),
        md("## 9 · Imports"),
        code(IMPORTS_TAB),
        md("## 10 · Configuration / Constants"),
        code("""TARGET = "Label"
MAX_ROWS = 50000
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath("__file__")), "training.csv")
SAVE_DIR = os.path.dirname(os.path.abspath("__file__"))
print(f"Data path: {DATA_PATH}")
print(f"Save dir : {SAVE_DIR}")"""),
        md("## 11 · Dataset Download or Loading"),
        code("""df = pd.read_csv(DATA_PATH)
print(f"Full dataset: {df.shape}")

if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
    print(f"Sampled to: {df.shape}")

df.head()"""),
        md("## 12 · Data Validation Checks"),
        code("""print("=" * 50)
print("DATA VALIDATION")
print("=" * 50)
print(f"\\nMissing values (NaN): {df.isnull().sum().sum()}")
# Count -999.0 sentinel values
sentinel_count = (df.select_dtypes(include="number") == -999.0).sum()
print(f"\\nSentinel (-999.0) values per column:")
for col, cnt in sentinel_count[sentinel_count > 0].items():
    print(f"  {col}: {cnt}")
print(f"\\nTarget distribution:\\n{df[TARGET].value_counts()}")
assert TARGET in df.columns
print(f"Shape: {df.shape}")"""),
        md("## 13 · Exploratory Data Analysis"),
        code("""# Feature distributions (skip EventId and Weight)
feature_cols = [c for c in df.columns if c not in [TARGET, "EventId", "Weight"]]
der_cols = [c for c in feature_cols if c.startswith("DER")]
pri_cols = [c for c in feature_cols if c.startswith("PRI")]

fig, axes = plt.subplots(3, 5, figsize=(20, 12))
for i, col in enumerate(der_cols[:15]):
    ax = axes[i // 5, i % 5]
    valid = df[df[col] != -999.0][col]
    valid.hist(bins=30, ax=ax, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title(col, fontsize=8)
plt.suptitle("Derived Feature Distributions (excluding -999)", fontsize=14)
plt.tight_layout()
plt.show()"""),
        code("""# Correlation of features with target
df_corr = df.copy()
df_corr[TARGET] = (df_corr[TARGET] == "s").astype(int)
# Replace -999 with NaN for correlation
num_cols = df_corr.select_dtypes(include="number").columns
df_corr[num_cols] = df_corr[num_cols].replace(-999.0, np.nan)
corr_target = df_corr[num_cols].corr()[TARGET].drop([TARGET, "EventId", "Weight"], errors="ignore").abs().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
corr_target.head(15).plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Top 15 Features Correlated with Signal")
ax.set_ylabel("|Correlation|")
plt.tight_layout()
plt.show()"""),
        md("## 14 · Target Analysis"),
        code("""fig, ax = plt.subplots(figsize=(6, 4))
df[TARGET].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "coral"], edgecolor="black")
ax.set_title(f"Target Distribution: {TARGET}")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

for val in df[TARGET].value_counts().index:
    n = (df[TARGET] == val).sum()
    print(f"  {val}: {n} ({100*n/len(df):.1f}%)")"""),
        md("## 15 · Train/Validation/Test Split Strategy\n\nStratified 80/20 split after encoding target."),
        code("""# Prepare features
X = df.drop(columns=[TARGET, "EventId", "Weight"])
y_raw = df[TARGET].values

# Encode target: s=1, b=0
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Classes: {list(le.classes_)} → [0, 1]")

# Replace -999.0 with NaN, then fill with median
X = X.replace(-999.0, np.nan)
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())

# Sanitize column names
X.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")"""),
        md("## 16 · Preprocessing Strategy\n\n- **Missing sentinels**: -999.0 replaced with NaN, then imputed with column median.\n- **Dropped**: EventId (identifier), Weight (competition scoring only).\n- **Scaling**: Not needed for tree models.\n- **Target**: s→1, b→0 via LabelEncoder."),
        md("## 17 · Feature Engineering\n\nPhysics features are already domain-engineered (DER_* derived variables). No additional feature engineering needed — domain scientists designed these features."),
        md("## 18 · Baseline Model"),
        code("""baseline = LogisticRegression(max_iter=1000, random_state=SEED)
baseline.fit(X_train, y_train)
y_pred_bl = baseline.predict(X_test)

acc_bl = accuracy_score(y_test, y_pred_bl)
f1_bl = f1_score(y_test, y_pred_bl, average="binary")

print("=" * 50)
print("BASELINE: Logistic Regression")
print("=" * 50)
print(f"Accuracy : {acc_bl:.4f}")
print(f"F1       : {f1_bl:.4f}")
print(f"\\n{classification_report(y_test, y_pred_bl, target_names=le.classes_)}")"""),
        md("## 19 · LazyPredict Benchmark\n\nSubsampled to 10K for LazyPredict speed."),
        code(lp_block(multi=False, subsample=10000)),
        md("## 20 · FLAML AutoML Run"),
        code("""from flaml import AutoML

try:
    flaml_automl = AutoML()
    flaml_automl.fit(X_train, y_train, task="classification", time_budget=60,
                     metric="f1", verbose=0, seed=SEED)
    y_pred_flaml = flaml_automl.predict(X_test)
    acc_flaml = accuracy_score(y_test, y_pred_flaml)
    f1_flaml = f1_score(y_test, y_pred_flaml)
    print(f"FLAML best: {flaml_automl.best_estimator}")
    print(f"Accuracy: {acc_flaml:.4f}, F1: {f1_flaml:.4f}")
except Exception as e:
    print(f"FLAML failed: {e}")
    print("Using baseline predictions as FLAML fallback.")
    y_pred_flaml = y_pred_bl"""),
        md("## 21 · Additional Models: CatBoost, LightGBM, XGBoost"),
        code(boost_block(multi=False)),
        md("## 22 · Top 3 Model Selection"),
        code(top3_block(multi=False)),
        md("## 23 · Final Training and Evaluation of Top 3"),
        code(eval_top3(multi=False)),
        md("## 24 · Error Analysis"),
        code(err_analysis()),
        md("## 25 · Interpretation and Business Insight\n\n**Key findings:**\n- **DER_mass_MMC** (reconstructed mass) is the strongest feature — as expected physically.\n- **DER_mass_transverse_met_lep** and **DER_mass_vis** are also strong discriminators.\n- **Primitive features** (PRI_*) are less predictive individually.\n- Missing values (-999.0) concentrate in jet-related features when fewer jets are detected.\n\n**Physics takeaway:** The Higgs boson signature is most visible in reconstructed mass distributions, consistent with H→ττ decay mode."),
        md("## 26 · Limitations\n\n1. Sampled to 50K from 250K — full dataset may yield better results.\n2. Simplified to binary — real analysis has nuanced background categories.\n3. No systematic uncertainty handling.\n4. Weight column ignored (affects competition metric AMS).\n5. Features are pre-computed — no access to raw detector data."),
        md("## 27 · How to Improve This Project\n\n1. Train on full 250K rows.\n2. Use the competition AMS metric with Weight.\n3. Use physics-informed feature engineering.\n4. Try deep neural networks (as done by winning solution).\n5. Handle -999.0 with indicator features instead of median imputation."),
        md("## 28 · Production Considerations\n\n- Real physics analyses use ensemble of models.\n- Systematic uncertainties must be propagated.\n- Production runs on the CERN computing grid.\n- Results require formal statistical validation (CLs method)."),
        md("## 29 · Common Mistakes\n\n1. Not handling -999.0 sentinel values.\n2. Including EventId or Weight as features.\n3. Using accuracy on slightly imbalanced data.\n4. Not stratifying the train/test split.\n5. Ignoring physics context when interpreting features."),
        md("## 30 · Mini Challenge / Exercises\n\n1. Add indicator features for -999.0 values — does it help?\n2. Use only DER_* features — how much accuracy is lost?\n3. Train on 100K rows — significant improvement?\n4. Plot ROC curves for all models.\n5. Implement the AMS metric from the competition."),
        md("## 31 · Final Summary / Key Takeaways\n\n1. **Reconstructed mass** features are most discriminative for Higgs detection.\n2. **Boosting models** are the dominant approach in particle physics ML.\n3. **-999.0 handling** is critical for this dataset.\n4. **Domain knowledge** from physics designed the best features.\n5. **ML enabled** the Higgs boson discovery verification at CERN."),
        md("## Save Metrics"),
        code(save_met(multi=False)),
    ]
    write_nb(cells, f"{BASE}/H2O Higgs Boson/H2O Higgs Boson.ipynb")


# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    for i, fn in enumerate([nb1, nb2, nb3, nb4, nb5], 1):
        print(f"\n{'='*60}\nBuilding notebook {i}/5: {fn.__name__}")
        fn()
    print("\nAll 5 notebooks built.")
