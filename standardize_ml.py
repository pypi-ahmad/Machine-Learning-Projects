#!/usr/bin/env python3
"""
Phase 5 — ML Standardization Script
=====================================
Transforms eligible ML project notebooks to use:
  STEP 1: LazyPredict — baseline model comparison
  STEP 2: PyCaret    — final pipeline (setup, compare_models, finalize_model)

Reads audit data (phase1, phase2, phase3) to identify eligible projects,
then rewrites each notebook to:
  - KEEP:    imports, data loading, EDA, preprocessing (through train_test_split)
  - REMOVE:  custom model training, manual pipelines, redundant evaluation
  - REPLACE: standardized LazyPredict + PyCaret cells

Usage:
    python standardize_ml.py                    # all eligible projects
    python standardize_ml.py --project P001     # single project
    python standardize_ml.py --dry-run          # preview only
"""

from __future__ import annotations
import argparse
import copy
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Audit data loaders ──────────────────────────────────────────────────────

def load_phase1_inventory() -> list[dict]:
    with open(ROOT / "audit_phase1" / "project_inventory.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_phase2_detail() -> list[dict]:
    with open(ROOT / "audit_phase2" / "phase2_detail.json", encoding="utf-8") as f:
        return json.load(f)

def load_phase3_status() -> list[dict]:
    with open(ROOT / "audit_phase3" / "phase3_dataset_status.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ── Project eligibility ─────────────────────────────────────────────────────

# Classification models (names as they appear in audit)
CLASSIFICATION_MODELS = {
    "LogisticRegression", "RandomForestClassifier", "DecisionTreeClassifier",
    "KNeighborsClassifier", "SVM", "SVC", "GaussianNB", "MultinomialNB",
    "BernoulliNB", "NaiveBayes", "AdaBoostClassifier", "GradientBoostingClassifier",
    "BaggingClassifier", "ExtraTreesClassifier", "XGBClassifier", "XGBoost",
    "LGBMClassifier", "CatBoostClassifier", "CatBoost", "MLPClassifier",
    "VotingClassifier", "StackingClassifier",
}

REGRESSION_MODELS = {
    "LinearRegression", "Ridge", "Lasso", "ElasticNet",
    "RandomForestRegressor", "DecisionTreeRegressor",
    "GradientBoostingRegressor", "AdaBoostRegressor",
    "XGBRegressor", "LGBMRegressor", "CatBoostRegressor",
    "SVR", "KNeighborsRegressor", "MLPRegressor",
    "BaggingRegressor", "ExtraTreesRegressor",
}

CLUSTERING_MODELS = {
    "KMeans", "DBSCAN", "AgglomerativeClustering", "GaussianMixture",
    "MiniBatchKMeans", "SpectralClustering",
}

# Slug corrections from generate_tests.py
SLUG_CORRECTIONS = {
    "time_series_forecasting": "time_series_forecastings",
}


def classify_project(p2: dict, ml_type_hint: str = "") -> dict | None:
    """Determine if project is eligible and its task type.
    
    Args:
        p2: phase2_detail entry with 'custom_ml_to_replace', 'model_training_snippets' etc.
        ml_type_hint: ml_type string from phase1 (e.g. 'classification|cv')
    """
    models = p2.get("custom_ml_to_replace", [])
    model_snippets = p2.get("model_training_snippets", [])
    project_name = p2.get("project", "")

    # Skip conceptual, EDA-only
    if "(conceptual" in project_name.lower():
        return None
    if not models and not model_snippets:
        return None

    # Determine task — model evidence takes priority, hint is fallback
    task = None
    has_clf = any(m in CLASSIFICATION_MODELS for m in models)
    has_reg = any(m in REGRESSION_MODELS for m in models)
    has_clust = any(m in CLUSTERING_MODELS for m in models)

    ml_lower = ml_type_hint.lower()

    # 1) Clear model evidence: only classifiers → classification, only regressors → regression
    if has_clf and not has_reg and not has_clust:
        task = "classification"
    elif has_reg and not has_clf and not has_clust:
        task = "regression"
    elif has_clust and not has_clf and not has_reg:
        task = "clustering"
    # 2) Mixed models — use hint to disambiguate
    elif "classification" in ml_lower:
        task = "classification"
    elif "regression" in ml_lower:
        task = "regression"
    elif "clustering" in ml_lower:
        task = "clustering"
    # 3) Fallback: prefer the majority model type
    elif has_clf:
        clf_count = sum(1 for m in models if m in CLASSIFICATION_MODELS)
        reg_count = sum(1 for m in models if m in REGRESSION_MODELS)
        task = "classification" if clf_count >= reg_count else "regression"
    elif has_reg:
        task = "regression"
    elif has_clust:
        task = "clustering"
    else:
        # No recognized models — use hint alone
        if "classification" in ml_lower:
            task = "classification"
        elif "regression" in ml_lower:
            task = "regression"
        elif "clustering" in ml_lower:
            task = "clustering"
        else:
            return None  # can't determine

    # Skip deep learning / CV / PyTorch
    dl_keywords = ["keras", "tensorflow", "conv2d", "lstm", "pytorch", "torchvision",
                    "dataloader", "cifar", "imagefolder"]
    all_models_lower = " ".join(models).lower() + " " + project_name.lower()
    if any(kw in all_models_lower for kw in dl_keywords):
        return None
    if "cv" in ml_lower and task not in ("classification", "regression"):
        return None

    return {"task": task, "models": models}


# ── Cell classification ─────────────────────────────────────────────────────

# Patterns that indicate a cell is MODEL code (pre-compiled for performance)
MODEL_PATTERNS = [re.compile(p) for p in [
    r"\.fit\s*\(",                                    # .fit(
    r"\.predict\s*\(",                                # .predict(
    r"\.predict_proba\s*\(",                          # .predict_proba(
    r"from\s+sklearn\.\w+\s+import\s+\w+(Classifier|Regressor|NB|SVM|SVC|SVR)",
    r"from\s+sklearn\.linear_model\s+import",
    r"from\s+sklearn\.tree\s+import",
    r"from\s+sklearn\.ensemble\s+import",
    r"from\s+sklearn\.neighbors\s+import",
    r"from\s+sklearn\.svm\s+import",
    r"from\s+sklearn\.naive_bayes\s+import",
    r"from\s+sklearn\.neural_network\s+import",
    r"from\s+xgboost\s+import",
    r"from\s+catboost\s+import",
    r"from\s+lightgbm\s+import",
    r"import\s+xgboost",
    r"import\s+catboost",
    r"import\s+lightgbm",
    r"LogisticRegression\s*\(",
    r"RandomForest(Classifier|Regressor)\s*\(",
    r"DecisionTree(Classifier|Regressor)\s*\(",
    r"(KNeighbors|KNN)(Classifier|Regressor)\s*\(",
    r"GradientBoosting(Classifier|Regressor)\s*\(",
    r"(SVC|SVR|SVM)\s*\(",
    r"(GaussianNB|MultinomialNB|BernoulliNB)\s*\(",
    r"AdaBoost(Classifier|Regressor)\s*\(",
    r"XGB(Classifier|Regressor)\s*\(",
    r"LGBM(Classifier|Regressor)\s*\(",
    r"CatBoost(Classifier|Regressor)\s*\(",
    r"BaggingClassifier\s*\(",
    r"ExtraTrees(Classifier|Regressor)\s*\(",
    r"VotingClassifier\s*\(",
    r"StackingClassifier\s*\(",
    r"MLPClassifier\s*\(",
    r"LinearRegression\s*\(",
    r"Ridge\s*\(",
    r"Lasso\s*\(",
    r"ElasticNet\s*\(",
]]

# Patterns that indicate EVALUATION code (pre-compiled)
EVAL_PATTERNS = [re.compile(p) for p in [
    r"accuracy_score\s*\(",
    r"classification_report\s*\(",
    r"confusion_matrix\s*\(",
    r"f1_score\s*\(",
    r"precision_score\s*\(",
    r"recall_score\s*\(",
    r"roc_auc_score\s*\(",
    r"roc_curve\s*\(",
    r"mean_squared_error\s*\(",
    r"mean_absolute_error\s*\(",
    r"r2_score\s*\(",
    r"from\s+sklearn\.metrics\s+import",
    r"cross_val_score\s*\(",
    r"GridSearchCV\s*\(",
    r"RandomizedSearchCV\s*\(",
    r"\.score\s*\(\s*X_test",          # model.score(X_test, ...)
]]

# Patterns that indicate PREPROCESSING (should be kept, pre-compiled)
PREPROCESS_PATTERNS = [re.compile(p) for p in [
    r"train_test_split\s*\(",
    r"StandardScaler|MinMaxScaler|RobustScaler",
    r"LabelEncoder|OneHotEncoder|OrdinalEncoder",
    r"fit_transform\s*\(",
    r"get_dummies\s*\(",
    r"\.fillna\s*\(",
    r"\.dropna\s*\(",
    r"\.drop\s*\(",
    r"\.replace\s*\(",
    r"\.map\s*\(",
    r"\.apply\s*\(",
    r"CountVectorizer|TfidfVectorizer",
    r"SimpleImputer",
]]

# Patterns that indicate DATA LOADING (pre-compiled)
DATA_LOAD_PATTERNS = [re.compile(p) for p in [
    r"pd\.read_csv\s*\(",
    r"pd\.read_excel\s*\(",
    r"pd\.read_json\s*\(",
    r"pd\.read_table\s*\(",
    r"pd\.read_parquet\s*\(",
    r"load_\w+\s*\(",              # sklearn load_iris() etc.
]]

# Pure EDA patterns (just viewing data, no transform, pre-compiled)
EDA_PATTERNS = [re.compile(p) for p in [
    r"\.head\s*\(",
    r"\.tail\s*\(",
    r"\.describe\s*\(",
    r"\.info\s*\(",
    r"\.shape\s*$",
    r"\.dtypes\s*$",
    r"\.columns\s*$",
    r"\.value_counts\s*\(",
    r"\.isnull\(\)\.sum\s*\(",
    r"\.nunique\s*\(",
    r"sns\.",
    r"plt\.",
    r"\.plot\s*\(",
    r"\.hist\s*\(",
    r"\.corr\s*\(",
]]

# Imports that are MODEL-related (should be removed in standardized version)
MODEL_IMPORT_MODULES = {
    "sklearn.linear_model", "sklearn.tree", "sklearn.ensemble",
    "sklearn.neighbors", "sklearn.svm", "sklearn.naive_bayes",
    "sklearn.neural_network", "xgboost", "catboost", "lightgbm",
    "sklearn.metrics",
}


def classify_cell(source: str) -> str:
    """Classify a code cell into: import, data_load, eda, preprocess, model, eval, other."""
    stripped = source.strip()
    if not stripped:
        return "other"

    lines = stripped.split("\n")
    non_comment = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if not non_comment:
        return "other"

    # Check for model patterns first (most specific)
    is_model = any(p.search(source) for p in MODEL_PATTERNS)
    is_eval = any(p.search(source) for p in EVAL_PATTERNS)
    is_preprocess = any(p.search(source) for p in PREPROCESS_PATTERNS)
    is_data_load = any(p.search(source) for p in DATA_LOAD_PATTERNS)
    is_eda = any(p.search(source) for p in EDA_PATTERNS)

    # If cell has .fit() or model class instantiation → MODEL
    if is_model and not is_preprocess:
        return "model"
    if is_eval and not is_preprocess:
        return "eval"

    # Check if it's purely imports
    all_import = all(
        l.strip().startswith(("import ", "from ", "#", "%", "!"))
        or not l.strip()
        for l in lines
    )
    if all_import and any(l.strip().startswith(("import ", "from ")) for l in lines):
        # Check if it imports model-related modules
        source_lower = source.lower()
        if any(mod in source_lower for mod in MODEL_IMPORT_MODULES):
            return "model_import"
        return "import"

    if is_data_load:
        return "data_load"
    if is_preprocess:
        return "preprocess"
    if is_eda and not is_model:
        return "eda"

    # Model-related assignment without .fit() (e.g., "model = LogisticRegression()")
    if is_model:
        return "model"

    return "other"


def find_split_point(cells: list[dict]) -> int:
    """Find the cell index where model code begins (first model/eval cell after preprocessing).
    Returns the index of the first cell to REMOVE."""
    last_preprocess_idx = -1
    first_model_idx = -1

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        cat = classify_cell(source)

        if cat in ("preprocess", "data_load", "import"):
            last_preprocess_idx = i
        elif cat in ("model", "eval", "model_import"):
            if first_model_idx == -1:
                first_model_idx = i

    if first_model_idx == -1:
        # No model cells found — append at end
        return len(cells)

    # Use the first model cell as split point
    # But check if there's EDA between last_preprocess and first_model — keep that EDA
    split = first_model_idx

    # Walk backward from first_model — if preceding cells are EDA/markdown about model, include them
    return split


def extract_target_info(cells: list[dict]) -> dict:
    """Extract target column, feature definitions, and data loading from notebook cells."""
    info = {
        "target_col": None,
        "target_var": None,
        "target_mode": "__unknown__",
        "feature_var": None,
        "df_var": "df",
        "read_csv_line": None,
        "has_train_test_split": False,
        "split_test_size": "0.2",
        "split_random_state": "42",
    }

    all_source = ""
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        all_source += source + "\n"

    # Find DataFrame variable name
    for m in re.finditer(r"(\w+)\s*=\s*pd\.read_(?:csv|excel|json)\s*\(", all_source):
        info["df_var"] = m.group(1)
        break
    # Also check for sns.load_dataset / sklearn datasets
    for m in re.finditer(r"(\w+)\s*=\s*sns\.load_dataset\s*\(", all_source):
        info["df_var"] = m.group(1)
        break

    # Find target column — multiple patterns (priority order)

    # Pattern 1: y = df['col'] or Y = df['col'] (direct dict access)
    for m in re.finditer(
        r"([yY](?:_train|_test)?|target|label)\s*=\s*(\w+)\[(?:\"|')(\w+)(?:\"|')\]",
        all_source
    ):
        info["target_var"] = m.group(1)
        info["target_col"] = m.group(3)
        break

    # Pattern 2: y = np.array(df['col']) or y = pd.get_dummies(df['col']) — wrapped
    if not info["target_col"]:
        for m in re.finditer(
            r"([yY](?:_train|_test)?|target|label)\s*=\s*\w+\.\w+\(\s*(\w+)\[(?:\"|')(\w+)(?:\"|')\]",
            all_source
        ):
            info["target_var"] = m.group(1)
            info["target_col"] = m.group(3)
            break

    # Pattern 3: X = df.drop('col', ...) — the dropped column is target
    if not info["target_col"]:
        for m in re.finditer(
            r"[xX]\s*=\s*\w+\.drop\(\s*(?:\[?\s*)?['\"](\w+)['\"]",
            all_source
        ):
            info["target_col"] = m.group(1)
            break

    # Pattern 4: X = df.drop(columns=['col'])
    if not info["target_col"]:
        for m in re.finditer(
            r"\.drop\(\s*columns\s*=\s*\[?\s*['\"](\w+)['\"]",
            all_source
        ):
            info["target_col"] = m.group(1)
            break

    # Pattern 5: .drop(labels='col')
    if not info["target_col"]:
        for m in re.finditer(
            r"\.drop\(\s*labels\s*=\s*['\"](\w+)['\"]",
            all_source
        ):
            info["target_col"] = m.group(1)
            break

    # Pattern 6: y = df.iloc[:, -1] — infer last column from context
    if not info["target_col"]:
        if re.search(r"[yY]\s*=\s*\w+\.iloc\s*\[\s*:?\s*,\s*-?\s*1\s*\]", all_source):
            info["target_var"] = "y"
            info["has_train_test_split"] = True  # These notebooks always split
            # Try to find column name from df[['col1', ..., 'target']] or df.columns
            # Look for the last element in column selection list
            for m_cols in re.finditer(
                r"\w+\s*=\s*\w+\[\s*\[([^\]]+)\]\s*\]",
                all_source
            ):
                cols_str = m_cols.group(1)
                cols = re.findall(r"['\"](\w+)['\"]", cols_str)
                if cols:
                    info["target_col"] = cols[-1]  # Last column = target
                    break
            # Fallback: look for common column name patterns in df/data definition
            if not info["target_col"]:
                # Search for column added like df['target'] = dataset.target
                for m_added in re.finditer(r"\w+\[(?:\"|')(\w+)(?:\"|')\]\s*=\s*\w+\.target", all_source):
                    info["target_col"] = m_added.group(1)
                    break

    # Pattern 7: target in train_test_split args: train_test_split(X, df['col'])
    if not info["target_col"]:
        for m in re.finditer(
            r"train_test_split\s*\([^,]+,\s*\w+\[(?:\"|')(\w+)(?:\"|')\]",
            all_source
        ):
            info["target_col"] = m.group(1)
            break

    # Pattern 8: sklearn dataset — target = dataset.target, X = dataset.data
    if not info["target_col"]:
        if re.search(r"(?:target|y)\s*=\s*\w+\.target\b", all_source):
            info["target_col"] = "__sklearn_target__"
            info["target_var"] = "target"

    # Pattern 9: integer column reference — Y = df[60]
    if not info["target_col"]:
        for m in re.finditer(
            r"[yY]\s*=\s*\w+\[\s*(\d+)\s*\]",
            all_source
        ):
            info["target_col"] = m.group(1)  # integer column
            break

    # Detect train_test_split parameters
    for m in re.finditer(r"train_test_split\s*\([^)]*test_size\s*=\s*([\d.]+)", all_source):
        info["has_train_test_split"] = True
        info["split_test_size"] = m.group(1)
    for m in re.finditer(r"train_test_split\s*\([^)]*random_state\s*=\s*(\d+)", all_source):
        info["split_random_state"] = m.group(1)
    if "train_test_split" in all_source:
        info["has_train_test_split"] = True

    # Derive target_mode from what was detected
    if info["target_col"] and info["target_col"] not in ("__sklearn_target__",):
        info["target_mode"] = "__column__"
    elif info["target_var"]:
        info["target_mode"] = "__inferred_from_var__"
    else:
        info["target_mode"] = "__unknown__"

    return info


# ── Standardized cell generators ────────────────────────────────────────────

def make_markdown_cell(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source,
    }


def make_code_cell(source: str) -> dict:
    """Create a code cell."""
    lines = source.split("\n")
    # Ensure each line (except last) ends with \n
    source_lines = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        source_lines.append(lines[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def generate_standardized_cells(task: str, target_info: dict) -> list[dict]:
    """Generate the standardized LazyPredict + PyCaret replacement cells."""
    target_col = target_info["target_col"]
    target_mode = target_info.get("target_mode", "__unknown__")
    df_var = target_info["df_var"]
    test_size = target_info["split_test_size"]
    random_state = target_info["split_random_state"]
    has_split = target_info["has_train_test_split"]
    
    # Determine whether to use existing X_train/y_train directly
    use_existing_vars = target_mode == "__inferred_from_var__" or has_split
    needs_reconstruction = target_mode == "__inferred_from_var__"

    cells = []

    # ── Separator markdown ──
    cells.append(make_markdown_cell(
        "---\n"
        "## Standardized ML Pipeline\n"
        "*Auto-generated by Phase 5 ML Standardization*\n\n"
        "**STEP 1** — LazyPredict baseline comparison  \n"
        "**STEP 2** — PyCaret automated pipeline"
    ))

    # ── STEP 1: LazyPredict ──
    cells.append(make_markdown_cell(
        "### STEP 1 — LazyPredict: Baseline Model Comparison\n"
        "Run all sklearn-compatible models to find the best baseline."
    ))

    if task == "classification":
        if target_info["has_train_test_split"]:
            lazy_code = f"""import warnings
warnings.filterwarnings('ignore')

from lazypredict.Supervised import LazyClassifier

# Use existing train/test split from preprocessing above
# Ensure numeric-only for LazyPredict (handles mixed types)
try:
    X_train_lp = X_train.select_dtypes(include=['number']).fillna(0)
    X_test_lp = X_test.select_dtypes(include=['number']).fillna(0)
except AttributeError:
    X_train_lp, X_test_lp = X_train, X_test

# Run LazyPredict
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models_df, predictions_df = clf.fit(X_train_lp, X_test_lp, y_train, y_test)

print("=" * 60)
print("LazyPredict — Classification Baseline Results")
print("=" * 60)
models_df"""
        else:
            lazy_code = f"""import warnings
warnings.filterwarnings('ignore')

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

# Prepare features and target
target_col = '{target_col}'
X = {df_var}.drop(columns=[target_col]).select_dtypes(include=['number']).fillna(0)
y = {df_var}[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)

# Run LazyPredict
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models_df, predictions_df = clf.fit(X_train, X_test, y_train, y_test)

print("=" * 60)
print("LazyPredict — Classification Baseline Results")
print("=" * 60)
models_df"""
    elif task == "regression":
        if target_info["has_train_test_split"]:
            lazy_code = f"""import warnings
warnings.filterwarnings('ignore')

from lazypredict.Supervised import LazyRegressor

# Use existing train/test split from preprocessing above
# Ensure numeric-only for LazyPredict (handles mixed types)
try:
    X_train_lp = X_train.select_dtypes(include=['number']).fillna(0)
    X_test_lp = X_test.select_dtypes(include=['number']).fillna(0)
except AttributeError:
    X_train_lp, X_test_lp = X_train, X_test

# Run LazyPredict
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models_df, predictions_df = reg.fit(X_train_lp, X_test_lp, y_train, y_test)

print("=" * 60)
print("LazyPredict — Regression Baseline Results")
print("=" * 60)
models_df"""
        else:
            lazy_code = f"""import warnings
warnings.filterwarnings('ignore')

from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split

# Prepare features and target
target_col = '{target_col}'
X = {df_var}.drop(columns=[target_col]).select_dtypes(include=['number']).fillna(0)
y = {df_var}[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state={random_state}
)

# Run LazyPredict
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models_df, predictions_df = reg.fit(X_train, X_test, y_train, y_test)

print("=" * 60)
print("LazyPredict — Regression Baseline Results")
print("=" * 60)
models_df"""
    elif task == "clustering":
        lazy_code = f"""import warnings
warnings.filterwarnings('ignore')

# Note: LazyPredict doesn't support clustering directly.
# We'll use sklearn for quick comparison instead.
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd

# Prepare features (numeric only)
X = {df_var}.select_dtypes(include=['number']).dropna()

results = {{}}
for name, model in [('KMeans', KMeans(n_clusters=3, random_state=42, n_init=10)),
                     ('Agglomerative', AgglomerativeClustering(n_clusters=3))]:
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    results[name] = {{'Silhouette Score': round(score, 4)}}

results_df = pd.DataFrame(results).T.sort_values('Silhouette Score', ascending=False)
print("=" * 60)
print("Clustering — Baseline Results")
print("=" * 60)
results_df"""

    cells.append(make_code_cell(lazy_code))

    # ── LazyPredict visualization ──
    cells.append(make_markdown_cell(
        "#### Top Models Visualization"
    ))

    if task in ("classification", "regression"):
        metric = "Accuracy" if task == "classification" else "R-Squared"
        viz_code = f"""import matplotlib.pyplot as plt

top_n = min(15, len(models_df))
fig, ax = plt.subplots(figsize=(10, 6))
models_df['{metric}'].head(top_n).plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('{metric}')
ax.set_title(f'Top {{top_n}} Models — {metric}')
ax.invert_yaxis()
plt.tight_layout()
plt.show()"""
        cells.append(make_code_cell(viz_code))

    # ── STEP 2: PyCaret ──
    cells.append(make_markdown_cell(
        "### STEP 2 — PyCaret: Automated ML Pipeline\n"
        "Full pipeline with automated preprocessing, model comparison, tuning, and finalization.\n\n"
        "> **Note:** PyCaret requires Python 3.9–3.11. "
        "Install with: `pip install pycaret`"
    ))

    # ── PyCaret code generation (shared logic for clf/reg) ──
    _PYCARET_MODULES = {
        "classification": ("pycaret.classification", "clf_setup", "Accuracy",
                           ["confusion_matrix", "auc", "feature"]),
        "regression": ("pycaret.regression", "reg_setup", "R2",
                        ["residuals", "error", "feature"]),
    }

    if task in _PYCARET_MODULES:
        module, var_name, sort_metric, eval_plots = _PYCARET_MODULES[task]
        if needs_reconstruction:
            pycaret_setup_code = f"""import sys
import pandas as pd

# PyCaret setup
try:
    from {module} import setup, compare_models, finalize_model, predict_model, save_model, plot_model
except ImportError:
    print("PyCaret not installed. Install with: pip install pycaret")
    print("Requires Python 3.9-3.11")
    raise SystemExit("PyCaret required for STEP 2")

# Safety check — target variable must exist
assert y_train is not None, "y_train is None — cannot proceed"

# Reconstruct DataFrame for PyCaret (needs column-named target)
# IMPORTANT: Do NOT mutate original X_train / y_train
if isinstance(X_train, pd.DataFrame):
    df_train = X_train.copy()
else:
    df_train = pd.DataFrame(X_train)
df_train['__target__'] = y_train

if isinstance(X_test, pd.DataFrame):
    df_test = X_test.copy()
else:
    df_test = pd.DataFrame(X_test)
df_test['__target__'] = y_test

_full_df = pd.concat([df_train, df_test], ignore_index=True)

# Configure PyCaret session
{var_name} = setup(
    data=_full_df,
    target='__target__',
    session_id={random_state},
    silent=True,
    verbose=False,
    fold=5,
)
print("PyCaret setup complete.")"""
        else:
            pycaret_setup_code = f"""import sys

# PyCaret setup
try:
    from {module} import setup, compare_models, finalize_model, predict_model, save_model, plot_model
except ImportError:
    print("PyCaret not installed. Install with: pip install pycaret")
    print("Requires Python 3.9-3.11")
    raise SystemExit("PyCaret required for STEP 2")

# Configure PyCaret session
{var_name} = setup(
    data={df_var},
    target='{target_col}',
    session_id={random_state},
    verbose=False,
    fold=5,
)
print("PyCaret setup complete.")"""

        pycaret_compare_code = f"""# Compare all models
best_model = compare_models(sort='{sort_metric}', n_select=1)
print(f"\\nBest model: {{best_model}}")"""

        pycaret_finalize_code = """# Finalize the best model (retrain on full dataset)
final_model = finalize_model(best_model)
print(f"Finalized model: {final_model}")"""

        plot_lines = "\n".join(
            f"    plot_model(best_model, plot='{p}', save=True)" for p in eval_plots
        )
        pycaret_eval_code = f"""# Model evaluation plots
try:
{plot_lines}
except Exception as e:
    print(f"Plot generation note: {{e}}")"""

        pycaret_save_code = """# Save the final pipeline
save_model(final_model, 'final_model_pipeline')
print("Model saved as 'final_model_pipeline.pkl'")"""

    elif task == "clustering":
        pycaret_setup_code = f"""import sys

# PyCaret setup
try:
    from pycaret.clustering import setup, create_model, assign_model, plot_model, save_model
except ImportError:
    print("PyCaret not installed. Install with: pip install pycaret")
    print("Requires Python 3.9-3.11")
    raise SystemExit("PyCaret required for STEP 2")

# Configure PyCaret session — clustering (no target)
clust_setup = setup(
    data={df_var}.select_dtypes(include=['number']).dropna(),
    session_id=42,
    verbose=False,
)
print("PyCaret clustering setup complete.")"""

        pycaret_compare_code = """# Create KMeans model
kmeans = create_model('kmeans', num_clusters=3)
print(kmeans)"""

        pycaret_finalize_code = """# Assign cluster labels
clustered_df = assign_model(kmeans)
print(clustered_df.head())"""

        pycaret_eval_code = """# Cluster visualization
try:
    plot_model(kmeans, plot='cluster', save=True)
    plot_model(kmeans, plot='elbow', save=True)
except Exception as e:
    print(f"Plot generation note: {e}")"""

        pycaret_save_code = """# Save the clustering pipeline
save_model(kmeans, 'clustering_pipeline')
print("Clustering model saved as 'clustering_pipeline.pkl'")"""

    cells.append(make_code_cell(pycaret_setup_code))
    cells.append(make_code_cell(pycaret_compare_code))
    cells.append(make_code_cell(pycaret_finalize_code))

    cells.append(make_markdown_cell(
        "#### Model Evaluation"
    ))
    cells.append(make_code_cell(pycaret_eval_code))

    cells.append(make_markdown_cell(
        "#### Save Model Pipeline"
    ))
    cells.append(make_code_cell(pycaret_save_code))

    return cells


# ── Notebook transformer ────────────────────────────────────────────────────

def transform_notebook(nb_path: Path, task: str, target_info: dict,
                       backup: bool = True) -> dict:
    """Transform a notebook to use standardized ML pipeline.
    
    Returns stats dict with transformation details.
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    if not cells:
        return {"status": "SKIP", "reason": "empty notebook"}

    # Classify each cell
    cell_categories = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            cell_categories.append(("markdown", i))
            continue
        source = "".join(cell.get("source", []))
        cat = classify_cell(source)
        cell_categories.append((cat, i))

    # Find the split point
    split_idx = find_split_point(cells)

    # Count what we're removing
    removed_cells = []
    kept_cells = []
    for i, cell in enumerate(cells):
        if i < split_idx:
            kept_cells.append(i)
        else:
            cat = cell_categories[i][0]
            if cat in ("model", "eval", "model_import"):
                removed_cells.append((i, cat))
            elif cat == "markdown":
                # Check if markdown is about models — remove
                source = "".join(cell.get("source", []))
                if any(kw in source.lower() for kw in
                       ["model", "classifier", "regressor", "prediction",
                        "accuracy", "evaluation", "training", "logistic",
                        "random forest", "decision tree", "svm", "naive bayes",
                        "xgboost", "confusion matrix"]):
                    removed_cells.append((i, "markdown_model"))
                else:
                    kept_cells.append(i)
            elif cat in ("eda", "other"):
                # EDA after model section — likely model-related plotting
                removed_cells.append((i, cat))
            else:
                kept_cells.append(i)

    # For the cells we keep after split_idx, only keep data_load/preprocess/import/eda
    # that appear before the first model cell
    final_kept = [i for i in kept_cells if i < split_idx]

    # Also keep any standalone preprocessing/data cells after split (e.g., loading test data)
    # but NOT model-dependent cells
    for i in kept_cells:
        if i >= split_idx:
            cat = cell_categories[i][0]
            if cat in ("data_load", "import", "preprocess"):
                # Check if this is a secondary data load (test set) — skip for standardized version
                source = "".join(cells[i].get("source", []))
                if "test" in source.lower() and "read_csv" in source.lower():
                    # Secondary test set load — we'll let PyCaret handle splits
                    removed_cells.append((i, "secondary_data_load"))
                    continue
                final_kept.append(i)

    # Build new notebook
    new_cells = [cells[i] for i in sorted(final_kept)]

    # Generate standardized cells
    std_cells = generate_standardized_cells(task, target_info)

    # Append standardized cells
    new_cells.extend(std_cells)

    # Backup original
    if backup:
        backup_path = nb_path.with_suffix(".ipynb.bak")
        if not backup_path.exists():
            shutil.copy2(nb_path, backup_path)

    # Write transformed notebook
    nb_new = copy.deepcopy(nb)
    nb_new["cells"] = new_cells
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_new, f, indent=1, ensure_ascii=False)

    return {
        "status": "OK",
        "original_cells": len(cells),
        "kept_cells": len(final_kept),
        "removed_cells": len(removed_cells),
        "added_cells": len(std_cells),
        "final_cells": len(new_cells),
        "task": task,
        "target_col": target_info["target_col"],
        "removed_detail": [(cat, i) for i, cat in removed_cells],
    }


# ── Main orchestrator ───────────────────────────────────────────────────────

def find_notebook(project_dir: str) -> Path | None:
    """Find the primary .ipynb file in a project directory."""
    proj_path = ROOT / project_dir
    if not proj_path.exists():
        return None
    notebooks = list(proj_path.glob("*.ipynb"))
    if not notebooks:
        return None
    # Prefer notebooks not named "Untitled" if there are multiple
    named = [n for n in notebooks if "untitled" not in n.stem.lower()]
    if named:
        return named[0]
    return notebooks[0]


def build_project_map(p1_rows: list[dict]) -> dict:
    """Map project number → project directory name."""
    pmap = {}
    for row in p1_rows:
        raw = row.get("project", "").strip()  # P1 uses 'project' column
        num_match = re.match(r"Machine Learning Project\s+(\d+)", raw)
        if num_match:
            pnum = int(num_match.group(1))
            # Find actual directory — use regex to match exact project number
            dirs = [d for d in ROOT.iterdir() if d.is_dir()
                    and re.match(rf"Machine Learning Project\s+{pnum}\b", d.name)]
            if dirs:
                pmap[pnum] = dirs[0].name
        else:
            # Handle non-"Machine Learning" projects (e.g. "Project 1- SONAR...")
            bare_match = re.match(r"^Project\s+(\d+)", raw)
            if bare_match:
                pnum = 200 + int(bare_match.group(1))  # 200+N to avoid collisions
                dirs = [d for d in ROOT.iterdir() if d.is_dir()
                        and re.match(rf"^Project\s+{bare_match.group(1)}\b", d.name)]
                if dirs:
                    pmap[pnum] = dirs[0].name
    return pmap


def extract_number(project_name: str) -> int | None:
    m = re.search(r"Machine Learning Project\s+(\d+)", project_name)
    if m:
        return int(m.group(1))
    # Also handle "Project N" (without "Machine Learning") — map to 200+N to avoid collisions
    m2 = re.match(r"^Project\s+(\d+)", project_name)
    if m2:
        return 200 + int(m2.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Phase 5 — ML Standardization")
    parser.add_argument("--project", type=str, help="Single project, e.g. P001")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no writes")
    parser.add_argument("--no-backup", action="store_true", help="Don't create .bak files")
    args = parser.parse_args()

    print("Phase 5 — ML Standardization")
    print("=" * 60)

    # Load audit data
    p1_rows = load_phase1_inventory()
    p2_details = load_phase2_detail()
    p3_rows = load_phase3_status()

    # Build P3 lookup by slug
    p3_by_slug = {}
    for row in p3_rows:
        slug = row.get("slug", "").strip()
        if slug:
            p3_by_slug[slug] = row

    # Build P1 lookup by project number
    p1_by_num = {}
    for row in p1_rows:
        pnum = extract_number(row.get("project", ""))
        if pnum:
            p1_by_num[pnum] = row

    # Build project number → directory map
    pmap = build_project_map(p1_rows)

    # Process each project
    stats = {"ok": 0, "skip": 0, "error": 0, "details": []}
    target_project_num = None
    if args.project:
        m = re.match(r"P?(\d+)", args.project, re.I)
        if m:
            target_project_num = int(m.group(1))

    for p2 in p2_details:
        pnum = extract_number(p2.get("project", ""))
        if pnum is None:
            continue
        if target_project_num and pnum != target_project_num:
            continue

        project_name = p2.get("project", "")

        # Get P1 data
        p1_match = p1_by_num.get(pnum, {})
        ml_type_hint = p1_match.get("ml_type_declared_from_names", "")

        # Classify using P2 data (has model info)
        result = classify_project(p2, ml_type_hint)
        if result is None:
            if target_project_num:
                print(f"  P{pnum:03d}: SKIP — not eligible (EDA/conceptual/CV/DL)")
            stats["skip"] += 1
            continue

        task = result["task"]
        models = result["models"]

        # Find notebook directory
        project_dir = pmap.get(pnum)
        if not project_dir:
            if target_project_num:
                print(f"  P{pnum:03d}: SKIP — directory not found")
            stats["skip"] += 1
            continue

        nb_path = find_notebook(project_dir)
        if not nb_path:
            if target_project_num:
                print(f"  P{pnum:03d}: SKIP — no notebook found")
            stats["skip"] += 1
            continue

        # Extract target info from notebook
        try:
            with open(nb_path, encoding="utf-8") as f:
                nb_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  P{pnum:03d}: ERROR — can't parse notebook JSON")
            stats["error"] += 1
            continue

        target_info = extract_target_info(nb_data.get("cells", []))

        if not target_info["target_col"] and task != "clustering":
            # Try to infer from phase2 model training snippets
            for snippet in p2.get("model_training_snippets", []):
                for m_target in re.finditer(r"\.fit\([^,]+,\s*(\w+)", snippet):
                    candidate = m_target.group(1)
                    if candidate.startswith("y"):
                        target_info["target_var"] = candidate
                        break

            # Try harder — read all notebook source for target patterns
            if not target_info["target_col"]:
                all_source = ""
                for cell in nb_data.get("cells", []):
                    if cell.get("cell_type") == "code":
                        all_source += "".join(cell.get("source", [])) + "\n"

                for pattern in [
                    # y = df['salary'] or target = df['Price']
                    r"(?:y|Y|target|label|output)\s*=\s*\w+\[[\"\'](\w+)[\"\']",
                    # y = np.array(df['col']) or y = pd.get_dummies(df['col'])
                    r"(?:y|Y|target|label)\s*=\s*\w+\.\w+\(\s*\w+\[[\"\'](\w+)[\"\']",
                    # df.drop('Price', ...) or df.drop(['Price'], ...)
                    r"\.drop\(\s*\[?\s*[\"\'](\w+)[\"\']",
                    # .drop(labels='col') or .drop(columns=['col'])
                    r"\.drop\(\s*(?:labels|columns)\s*=\s*\[?\s*[\"\'](\w+)[\"\']",
                    # target = 'column_name'
                    r"target\s*=\s*[\"\'](\w+)[\"\']",
                    # integer column: Y = df[60]
                    r"[yY]\s*=\s*\w+\[\s*(\d+)\s*\]",
                ]:
                    for m_target in re.finditer(pattern, all_source, re.M):
                        candidate = m_target.group(1)
                        if candidate.lower() in ("id", "index", "unnamed", "axis",
                                                  "columns", "errors", "inplace"):
                            continue
                        target_info["target_col"] = candidate
                        break
                    if target_info["target_col"]:
                        break

        # Allow transformation if target_col OR target_var is known
        has_target = bool(target_info["target_col"]) or bool(target_info["target_var"])
        if not has_target and task != "clustering":
            print(f"  P{pnum:03d}: SKIP — can't determine target ({task})")
            stats["skip"] += 1
            stats["details"].append({
                "project": pnum, "status": "SKIP",
                "reason": "missing_target_var"
            })
            continue

        # Derive target_mode if not already set by extract_target_info
        if target_info.get("target_mode") == "__unknown__":
            if target_info["target_col"]:
                target_info["target_mode"] = "__column__"
            elif target_info["target_var"]:
                target_info["target_mode"] = "__inferred_from_var__"

        # Transform
        label = f"P{pnum:03d}"
        models_str = ", ".join(models[:3]) + ("..." if len(models) > 3 else "")

        if args.dry_run:
            if target_info["target_mode"] == "__inferred_from_var__":
                display_target = f"{target_info['target_var'] or 'y'} (inferred)"
            elif target_info["target_col"]:
                display_target = target_info["target_col"]
            else:
                display_target = "None"
            print(f"  {label}: {task:15s} | target={display_target!r:20s} | "
                  f"replacing: {models_str}")
            stats["ok"] += 1
            stats["details"].append({
                "project": pnum, "status": "DRY_RUN", "task": task,
                "target": target_info["target_col"], "models": models_str,
                "notebook": nb_path.name,
            })
            continue

        try:
            xform = transform_notebook(
                nb_path, task, target_info,
                backup=not args.no_backup
            )
            if xform["status"] == "OK":
                print(f"  {label}: OK  | {task:15s} | "
                      f"kept={xform['kept_cells']:2d} removed={xform['removed_cells']:2d} "
                      f"added={xform['added_cells']:2d} | target={target_info['target_col']!r}")
                stats["ok"] += 1
                stats["details"].append({
                    "project": pnum, "status": "OK", "task": task,
                    "target": target_info["target_col"],
                    "notebook": nb_path.name,
                    **xform,
                })
            else:
                print(f"  {label}: SKIP — {xform.get('reason', 'unknown')}")
                stats["skip"] += 1
        except Exception as e:
            print(f"  {label}: ERROR — {e}")
            stats["error"] += 1
            stats["details"].append({
                "project": pnum, "status": "ERROR", "error": str(e)
            })

    # Summary
    print()
    print("=" * 60)
    print(f"Results: OK={stats['ok']}  Skip={stats['skip']}  Error={stats['error']}")
    total = stats["ok"] + stats["skip"] + stats["error"]
    print(f"Total projects processed: {total}")

    # Write report
    report_path = ROOT / "audit_phase5" / "standardization_report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
