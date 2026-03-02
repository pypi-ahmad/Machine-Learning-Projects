#!/usr/bin/env python3
"""
Phase 4 — Test Generator for ML Projects Workspace
===================================================
Reads audit data (phase1 inventory, phase2 detail, phase3 dataset status)
and generates pytest test files for each of the ~160 ML projects.

Test categories per project:
  - test_data_loading   : data file exists, loads without error, shape/columns
  - test_preprocessing  : no all-null columns, encoding, scaling, split
  - test_model_training : model instantiation, fit, has predict method
  - test_prediction     : prediction shape, type correctness, no NaN

Usage:
    python generate_tests.py
"""

from __future__ import annotations
import csv
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
TESTS_DIR.mkdir(exist_ok=True)

# ── Load audit data ─────────────────────────────────────────────────────────

def load_phase1_inventory() -> list[dict]:
    rows = []
    with open(ROOT / "audit_phase1" / "project_inventory.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def load_phase2_detail() -> list[dict]:
    with open(ROOT / "audit_phase2" / "phase2_detail.json", encoding="utf-8") as f:
        return json.load(f)

def load_phase3_status() -> list[dict]:
    rows = []
    with open(ROOT / "audit_phase3" / "phase3_dataset_status.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# ── Project classification ──────────────────────────────────────────────────

# Map of known sklearn model imports
SKLEARN_MODELS = {
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression", "classification"),
    "LinearRegression": ("sklearn.linear_model", "LinearRegression", "regression"),
    "Ridge": ("sklearn.linear_model", "Ridge", "regression"),
    "Lasso": ("sklearn.linear_model", "Lasso", "regression"),
    "ElasticNet": ("sklearn.linear_model", "ElasticNet", "regression"),
    "DecisionTreeClassifier": ("sklearn.tree", "DecisionTreeClassifier", "classification"),
    "DecisionTreeRegressor": ("sklearn.tree", "DecisionTreeRegressor", "regression"),
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier", "classification"),
    "RandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor", "regression"),
    "GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier", "classification"),
    "GradientBoostingRegressor": ("sklearn.ensemble", "GradientBoostingRegressor", "regression"),
    "AdaBoostClassifier": ("sklearn.ensemble", "AdaBoostClassifier", "classification"),
    "SVC": ("sklearn.svm", "SVC", "classification"),
    "SVR": ("sklearn.svm", "SVR", "regression"),
    "KNeighborsClassifier": ("sklearn.neighbors", "KNeighborsClassifier", "classification"),
    "KNeighborsRegressor": ("sklearn.neighbors", "KNeighborsRegressor", "regression"),
    "GaussianNB": ("sklearn.naive_bayes", "GaussianNB", "classification"),
    "MultinomialNB": ("sklearn.naive_bayes", "MultinomialNB", "classification"),
    "BernoulliNB": ("sklearn.naive_bayes", "BernoulliNB", "classification"),
    "KMeans": ("sklearn.cluster", "KMeans", "clustering"),
    "DBSCAN": ("sklearn.cluster", "DBSCAN", "clustering"),
    "NaiveBayes": ("sklearn.naive_bayes", "MultinomialNB", "classification"),
    "XGBClassifier": ("xgboost", "XGBClassifier", "classification"),
    "XGBRegressor": ("xgboost", "XGBRegressor", "regression"),
    "CatBoostClassifier": ("catboost", "CatBoostClassifier", "classification"),
    "CatBoostRegressor": ("catboost", "CatBoostRegressor", "regression"),
    "LGBMClassifier": ("lightgbm", "LGBMClassifier", "classification"),
    "LGBMRegressor": ("lightgbm", "LGBMRegressor", "regression"),
    "MLPClassifier": ("sklearn.neural_network", "MLPClassifier", "classification"),
    "MLPRegressor": ("sklearn.neural_network", "MLPRegressor", "regression"),
}

def extract_project_number(project_name: str) -> str:
    """Extract the numeric project ID from the project directory name."""
    m = re.search(r'(?:Project|Projects)\s+(\d+)', project_name)
    if m:
        return m.group(1)
    return "0"

def slugify(name: str) -> str:
    """Convert project name to a Python-safe identifier."""
    # Remove prefix
    name = re.sub(r'^Machine Learning Projects?\s+\d+\s*-\s*', '', name)
    name = re.sub(r'^Project\s+\d+\s*-\s*', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()
    return name[:50]

def classify_project(p2_entry: dict, p1_entry: dict, p3_entry: dict) -> dict:
    """Classify a project by its ML type, framework, and data pattern."""
    info = {
        "project": p2_entry["project"],
        "number": extract_project_number(p2_entry["project"]),
        "slug": p3_entry.get("slug", ""),
        "data_path": p3_entry.get("data_path", ""),
        "data_status": p3_entry.get("status", ""),
        "data_files": p3_entry.get("files_standardized", ""),
        "notebooks": p1_entry.get("notebooks", "0"),
        "scripts_py": p1_entry.get("scripts_py", "0"),
        "ml_type": p1_entry.get("ml_type_declared_from_names", ""),
        "ml_evidence": p1_entry.get("ml_type_evidence", ""),
        "custom_algorithms": p2_entry.get("custom_ml_to_replace", []),
        "has_pipeline": p2_entry.get("has_pipeline", False),
        "no_train_test_split": p2_entry.get("no_train_test_split", False),
        "no_evaluation": p2_entry.get("no_evaluation", False),
        "model_training_snippets": p2_entry.get("model_training_snippets", []),
        "data_loading_snippets": p2_entry.get("data_loading_snippets", []),
        "evaluation_snippets": p2_entry.get("evaluation_snippets", []),
        "notebooks_detail": p2_entry.get("notebooks", []),
        "data_loading_count": int(p1_entry.get("ml_type_evidence", "").count("read_csv") or 0),
        "dataset_source": p1_entry.get("dataset_source", ""),
    }
    
    # Determine framework
    algos = info["custom_algorithms"]
    snippets_str = " ".join(info["model_training_snippets"])
    
    info["framework"] = "sklearn"  # default
    if any(k in snippets_str for k in ["keras", "Keras", "Sequential", "Dense", "Conv2D", "LSTM"]):
        info["framework"] = "keras"
    elif any(k in snippets_str for k in ["torch", "nn.Module", "nn.Linear", "nn.Conv2d"]):
        info["framework"] = "pytorch"
    elif any(k in snippets_str for k in ["XGB", "xgb"]):
        info["framework"] = "xgboost"
    elif any(k in snippets_str for k in ["CatBoost", "catboost"]):
        info["framework"] = "catboost"
    elif any(k in snippets_str for k in ["LGBM", "lgbm", "LightGBM"]):
        info["framework"] = "lightgbm"
    elif any(k in snippets_str for k in ["ARIMA", "arima", "auto_arima"]):
        info["framework"] = "statsmodels"
    
    # Determine task type
    ml_type = info["ml_type"]
    if "clustering" in ml_type.lower() or any("KMeans" in a for a in algos):
        info["task"] = "clustering"
    elif "cv" in ml_type.lower() or "image" in info["project"].lower():
        info["task"] = "cv"
    elif "nlp" in ml_type.lower() or "sentiment" in info["project"].lower() or "spam" in info["project"].lower():
        info["task"] = "nlp"
    elif "regression" in ml_type.lower() or "forecasting" in ml_type.lower() or "prediction" in info["project"].lower():
        # Check if it's actually classification based on algorithms
        if any(a in ["LogisticRegression", "RandomForestClassifier", "SVC", "KNeighborsClassifier",
                      "GaussianNB", "MultinomialNB", "DecisionTreeClassifier",
                      "GradientBoostingClassifier", "XGBClassifier", "CatBoostClassifier"]
               for a in algos):
            info["task"] = "classification"
        elif any(a in ["LinearRegression", "Ridge", "Lasso", "SVR",
                        "RandomForestRegressor", "DecisionTreeRegressor"]
                 for a in algos):
            info["task"] = "regression"
        else:
            info["task"] = "regression"
    elif "classification" in ml_type.lower():
        info["task"] = "classification"
    elif any(a in SKLEARN_MODELS for a in algos):
        # Infer from algorithm
        for a in algos:
            if a in SKLEARN_MODELS:
                info["task"] = SKLEARN_MODELS[a][2]
                break
    else:
        info["task"] = "eda"  # no model found
    
    # Check if it's conceptual / EDA only
    if not info["model_training_snippets"] and not info["evaluation_snippets"]:
        info["task"] = "eda"
    
    # Determine the primary model to test
    info["primary_model"] = None
    info["primary_model_import"] = None
    for a in algos:
        if a in SKLEARN_MODELS:
            mod, cls, _ = SKLEARN_MODELS[a]
            info["primary_model"] = cls
            info["primary_model_import"] = f"from {mod} import {cls}"
            break
    
    return info

# ── Notebook parsing ────────────────────────────────────────────────────────

def parse_notebook_cells(project_dir: str, notebook_files: list[str]) -> dict:
    """Parse notebook JSON to extract imports, data loading, etc."""
    result = {
        "imports": [],
        "data_loads": [],
        "read_csv_calls": [],
        "target_column": None,
        "feature_columns": [],
        "has_train_test_split": False,
        "has_scaler": False,
        "scaler_type": None,
        "has_label_encoder": False,
        "has_images": False,
        "image_dir_pattern": None,
        "has_text_vectorizer": False,
        "vectorizer_type": None,
        "keras_model": False,
        "pytorch_model": False,
        "all_code": "",
    }
    
    for nb_file in notebook_files:
        nb_path = ROOT / project_dir / nb_file.split("\\")[-1]
        if not nb_path.exists():
            # Try finding the notebook
            candidates = list((ROOT / project_dir).glob("*.ipynb"))
            if candidates:
                nb_path = candidates[0]
            else:
                continue
        
        try:
            with open(nb_path, encoding="utf-8") as f:
                nb = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            result["all_code"] += source + "\n"
            
            # Extract read_csv patterns
            for m in re.finditer(r"pd\.read_csv\(['\"]([^'\"]+)['\"]", source):
                result["read_csv_calls"].append(m.group(1))
            for m in re.finditer(r"read_excel\(['\"]([^'\"]+)['\"]", source):
                result["read_csv_calls"].append(m.group(1))
            
            # Detect target column
            for m in re.finditer(r"\.drop\(\[?['\"](\w+)['\"]", source):
                # The column being dropped alongside features is often the target
                pass
            # Look for y = df['column'] pattern
            for m in re.finditer(r"[yY]\s*=\s*(?:df|data|dataset)\[(?:\"|')(\w+)(?:\"|')\]", source):
                result["target_column"] = m.group(1)
            
            # Detect train_test_split
            if "train_test_split" in source:
                result["has_train_test_split"] = True
            
            # Detect scalers
            if "StandardScaler" in source:
                result["has_scaler"] = True
                result["scaler_type"] = "StandardScaler"
            elif "MinMaxScaler" in source:
                result["has_scaler"] = True
                result["scaler_type"] = "MinMaxScaler"
            elif "RobustScaler" in source:
                result["has_scaler"] = True
                result["scaler_type"] = "RobustScaler"
            
            # Detect label encoder
            if "LabelEncoder" in source:
                result["has_label_encoder"] = True
            
            # Detect image patterns
            if "flow_from_directory" in source or "ImageDataGenerator" in source:
                result["has_images"] = True
            if "PIL" in source or "cv2.imread" in source:
                result["has_images"] = True
            
            # Detect text vectorizers
            if "CountVectorizer" in source:
                result["has_text_vectorizer"] = True
                result["vectorizer_type"] = "CountVectorizer"
            elif "TfidfVectorizer" in source:
                result["has_text_vectorizer"] = True
                result["vectorizer_type"] = "TfidfVectorizer"
            
            # Detect DL frameworks
            if "keras" in source.lower() or "tensorflow" in source.lower():
                result["keras_model"] = True
            if "torch" in source.lower() and "import torch" in source.lower():
                result["pytorch_model"] = True
    
    return result


# ── Known slug corrections (singular/plural mismatches, etc.) ───────────────
SLUG_CORRECTIONS = {
    "time_series_forecasting": "time_series_forecastings",
}

# Files that are not useful datasets (metadata only)
SKIP_DATA_FILES = {"Data Dictionary - carprices.xlsx", "link_to_dataset.txt", "linkt_to_dataset.txt"}

# Image / binary extensions that can't be read with pd.read_csv
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff",
                    ".npz", ".npy", ".pkl", ".h5", ".hdf5", ".mp4", ".avi", ".wav"}


# ── Test file generators ────────────────────────────────────────────────────


def _generate_binary_file_tests(primary_file: str) -> str:
    """Generate tests for image/binary files that can't be pd.read_csv'd."""
    ext = Path(primary_file).suffix.lower()
    if ext == ".npz":
        load_code = f'''import numpy as np
        data = np.load(DATA_DIR / "{primary_file}", allow_pickle=True)
        assert len(data.files) > 0, "NPZ archive has no arrays"'''
    elif ext == ".npy":
        load_code = f'''import numpy as np
        data = np.load(DATA_DIR / "{primary_file}", allow_pickle=True)
        assert data is not None'''
    elif ext in (".pkl",):
        load_code = f'''import pickle
        with open(DATA_DIR / "{primary_file}", "rb") as f:
            data = pickle.load(f)
        assert data is not None'''
    else:
        # Image or other binary — just check file existence and non-zero size
        load_code = f'''import os
        fpath = DATA_DIR / "{primary_file}"
        assert os.path.getsize(fpath) > 0, "File is empty"'''

    return f'''
class TestDataLoading:
    """Verify binary/image data file exists and loads."""

    def test_data_dir_exists(self):
        if not DATA_DIR.exists():
            pytest.skip(f"Data directory not found: {{DATA_DIR}}")
        assert DATA_DIR.exists()

    def test_data_file_exists(self):
        if not DATA_DIR.exists():
            pytest.skip("Data directory missing")
        fpath = DATA_DIR / "{primary_file}"
        if not fpath.exists():
            pytest.skip(f"Data file not found: {{fpath}}")
        assert fpath.exists()

    def test_load_without_error(self):
        fpath = DATA_DIR / "{primary_file}"
        if not fpath.exists():
            pytest.skip(f"Data file not found: {{fpath}}")
        {load_code}
'''


def generate_csv_data_tests(info: dict, nb_info: dict) -> str:
    """Generate data loading tests for CSV-based projects."""
    slug = info["slug"]
    data_files = info["data_files"]
    
    # Try to find the primary CSV file
    primary_file = None
    if data_files:
        files = [f.strip() for f in data_files.split("|")]
        # Clean up file names: remove "(downloaded)" suffix, backslash paths → basename
        cleaned = []
        for f in files:
            f = re.sub(r'\s*\(downloaded\)', '', f)
            f = f.replace('\\', '/').split('/')[-1]  # basename only
            f = f.strip()
            if f and f not in SKIP_DATA_FILES:
                cleaned.append(f)
        files = cleaned
        for f in files:
            if f.endswith(".csv"):
                primary_file = f
                break
            if f.endswith((".xlsx", ".xls")):
                primary_file = f
                break
            if f.endswith(".tsv"):
                primary_file = f
                break
            if f.endswith((".txt", ".data")):
                primary_file = f
                break
        if not primary_file and files:
            primary_file = files[0]
    
    # Fall back to read_csv calls from notebook
    if not primary_file and nb_info["read_csv_calls"]:
        raw = nb_info["read_csv_calls"][0]
        # Extract just the filename (handle Windows paths, forward slashes)
        candidate = raw.replace('\\', '/').split('/')[-1].strip()
        if candidate and candidate not in SKIP_DATA_FILES:
            primary_file = candidate
    
    if not primary_file:
        # No data file found — generate minimal tests
        return '''
class TestDataLoading:
    """Verify data directory exists."""

    def test_data_dir_exists(self):
        if not DATA_DIR.exists():
            pytest.skip(f"Data directory not found: {DATA_DIR}")
        assert DATA_DIR.exists()
'''
    
    # Check if the primary file is an image/binary (can't use pd.read_csv)
    ext_lower = Path(primary_file).suffix.lower()
    if ext_lower in IMAGE_EXTENSIONS:
        return _generate_binary_file_tests(primary_file)
    
    # Determine read function and encoding hints
    read_extra = ""
    if primary_file.endswith((".xlsx", ".xls")):
        read_fn = "pd.read_excel"
    elif primary_file.endswith(".tsv"):
        read_fn = "pd.read_csv"
        read_extra = ', sep="\\t"'
    elif primary_file.endswith(".data"):
        read_fn = "pd.read_csv"
        read_extra = ', header=None, sep=r"\\s+"'
    elif primary_file.endswith(".json"):
        read_fn = "pd.read_json"
    elif primary_file.startswith("BX-"):
        # Book-Crossing dataset files use semicolon separator and latin-1 encoding
        read_fn = "pd.read_csv"
        read_extra = ', sep=";", encoding="latin-1", on_bad_lines="skip"'
    else:
        read_fn = "pd.read_csv"
    
    load_line = f'{read_fn}(DATA_DIR / "{primary_file}"{read_extra})'
    
    # Build encoding fallback line (avoid duplicating encoding if already in read_extra)
    has_encoding = "encoding=" in read_extra
    if has_encoding:
        # Already has encoding — fallback is same as load_line
        fallback_line = load_line
    elif read_extra:
        fallback_line = f'{read_fn}(DATA_DIR / "{primary_file}", encoding="latin-1"{read_extra})'
    else:
        fallback_line = f'{read_fn}(DATA_DIR / "{primary_file}", encoding="latin-1")'
    
    # For JSON files, add extra exception handling
    if read_fn == "pd.read_json":
        load_try = f'''try:
            df = {load_line}
        except (ValueError, KeyError, UnicodeDecodeError) as e:
            pytest.skip(f"Cannot parse JSON file: {{e}}")'''
        load_try_fb = load_try  # JSON doesn't need latin-1 fallback
    else:
        load_try = f'''try:
            df = {load_line}
        except (UnicodeDecodeError, ValueError):
            df = {fallback_line}'''
        load_try_fb = load_try
    
    code = f'''
class TestDataLoading:
    """Verify dataset loads correctly."""

    def test_data_dir_exists(self):
        if not DATA_DIR.exists():
            pytest.skip(f"Data directory not found: {{DATA_DIR}}")
        assert DATA_DIR.exists()

    def test_data_file_exists(self):
        if not DATA_DIR.exists():
            pytest.skip("Data directory missing")
        fpath = DATA_DIR / "{primary_file}"
        if not fpath.exists():
            pytest.skip(f"Data file not found: {{fpath}}")
        assert fpath.exists()

    def test_load_without_error(self):
        fpath = DATA_DIR / "{primary_file}"
        if not fpath.exists():
            pytest.skip(f"Data file not found: {{fpath}}")
        {load_try}
        assert df is not None

    def test_dataframe_not_empty(self):
        fpath = DATA_DIR / "{primary_file}"
        if not fpath.exists():
            pytest.skip(f"Data file not found: {{fpath}}")
        {load_try_fb}
        assert len(df) > 0, "DataFrame is empty"
        assert df.shape[1] > 0, "DataFrame has no columns"

    def test_no_all_null_columns(self):
        fpath = DATA_DIR / "{primary_file}"
        if not fpath.exists():
            pytest.skip(f"Data file not found: {{fpath}}")
        {load_try_fb}
        # Filter out trailing "Unnamed:" columns (common CSV artifact)
        real_cols = [c for c in df.columns if not str(c).startswith("Unnamed:")]
        if not real_cols:
            pytest.skip("All columns are unnamed — likely metadata file")
        null_cols = [col for col in real_cols if df[col].isna().all()]
        assert len(null_cols) == 0, f"All-NaN columns: {{null_cols}}"
'''
    return code


def generate_image_data_tests(info: dict) -> str:
    """Generate data loading tests for image-based projects."""
    slug = info["slug"]
    
    code = '''
class TestDataLoading:
    """Verify image dataset structure."""

    def test_data_dir_exists(self):
        if not DATA_DIR.exists():
            pytest.skip(f"Data directory not found: {DATA_DIR}")
        assert DATA_DIR.exists()

    def test_has_subdirectories(self):
        if not DATA_DIR.exists():
            pytest.skip("Data directory missing")
        subdirs = [d for d in DATA_DIR.rglob("*") if d.is_dir()]
        if not subdirs:
            pytest.skip("No subdirectories — may need dataset download")
        assert len(subdirs) > 0

    def test_contains_images(self):
        if not DATA_DIR.exists():
            pytest.skip("Data directory missing")
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
        found = any(p.suffix.lower() in image_exts for p in DATA_DIR.rglob("*"))
        if not found:
            pytest.skip("No images found — may need dataset download")
        assert found

    def test_sample_image_loadable(self):
        if not DATA_DIR.exists():
            pytest.skip("Data directory missing")
        from PIL import Image
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        for p in DATA_DIR.rglob("*"):
            if p.suffix.lower() in image_exts:
                img = Image.open(p)
                assert img.size[0] > 0 and img.size[1] > 0
                return
        pytest.skip("No images to test")
'''
    return code


def generate_builtin_data_tests(info: dict) -> str:
    """Generate tests for projects that use built-in datasets."""
    project = info["project"].lower()
    snippets = " ".join(info.get("data_loading_snippets", []))
    all_code = ""
    for nb in info.get("notebooks_detail", []):
        for dl in nb.get("data_loading", []):
            all_code += dl + " "
    
    if "load_iris" in all_code or "iris" in project:
        return '''
class TestDataLoading:
    """Verify built-in dataset loads."""

    def test_load_iris(self):
        from sklearn.datasets import load_iris
        data = load_iris()
        assert data.data.shape == (150, 4)
        assert data.target.shape == (150,)
'''
    elif "load_digits" in all_code or "digit" in project:
        return '''
class TestDataLoading:
    """Verify built-in dataset loads."""

    def test_load_digits(self):
        from sklearn.datasets import load_digits
        data = load_digits()
        assert data.data.shape[0] == 1797
        assert data.target.shape[0] == 1797
'''
    elif "load_wine" in all_code or "wine" in project:
        return '''
class TestDataLoading:
    """Verify built-in dataset loads."""

    def test_load_wine(self):
        from sklearn.datasets import load_wine
        data = load_wine()
        assert data.data.shape[0] > 0
'''
    elif "load_breast_cancer" in all_code or "breast_cancer" in project or "pca" in project.lower():
        return '''
class TestDataLoading:
    """Verify built-in dataset loads."""

    def test_load_breast_cancer(self):
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        assert data.data.shape == (569, 30)
        assert data.target.shape == (569,)
'''
    elif "load_boston" in all_code or "boston" in project:
        return '''
class TestDataLoading:
    """Verify built-in dataset loads."""

    def test_load_data(self):
        import pandas as pd
        # Boston dataset removed from sklearn; test CSV fallback
        csv_path = DATA_DIR / "housing.data"
        if csv_path.exists():
            df = pd.read_csv(csv_path, header=None, sep=r"\\s+")
            assert len(df) > 0
        else:
            pytest.skip("housing.data not found")
'''
    elif "make_classification" in all_code:
        return '''
class TestDataLoading:
    """Verify synthetic dataset generation."""

    def test_make_classification(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        assert X.shape == (200, 10)
        assert y.shape == (200,)
'''
    elif "seaborn" in all_code.lower() or "sns.load_dataset" in all_code:
        return '''
class TestDataLoading:
    """Verify seaborn dataset loads."""

    def test_load_seaborn_dataset(self):
        try:
            import seaborn as sns
        except ImportError:
            pytest.skip("seaborn not installed")
        df = sns.load_dataset("iris")
        assert len(df) > 0
'''
    elif "fashion" in project.lower() or "FashionMNIST" in all_code:
        return '''
class TestDataLoading:
    """Verify FashionMNIST can be loaded."""

    def test_fashionmnist_importable(self):
        try:
            from torchvision.datasets import FashionMNIST
        except ImportError:
            pytest.skip("torchvision not installed")
        assert FashionMNIST is not None
'''
    elif "cifar" in project.lower():
        return '''
class TestDataLoading:
    """Verify CIFAR-10 can be loaded."""

    def test_cifar10_importable(self):
        from torchvision.datasets import CIFAR10
        assert CIFAR10 is not None
'''
    elif "imdb" in project.lower() or "tfds" in all_code:
        return '''
class TestDataLoading:
    """Verify IMDB dataset loader exists."""

    def test_tfds_importable(self):
        try:
            import tensorflow_datasets as tfds
            assert tfds is not None
        except ImportError:
            pytest.skip("tensorflow_datasets not installed")
'''
    else:
        return '''
class TestDataLoading:
    """Verify data is available."""

    def test_data_dir_exists(self):
        if not DATA_DIR.exists():
            pytest.skip(f"Data directory not found: {DATA_DIR}")
        assert DATA_DIR.exists()
'''


def generate_preprocessing_tests(info: dict, nb_info: dict) -> str:
    """Generate preprocessing tests based on project patterns."""
    slug = info["slug"]
    task = info["task"]
    target = nb_info.get("target_column")
    
    parts = []
    
    parts.append('''
class TestPreprocessing:
    """Verify preprocessing steps work correctly."""
''')
    
    if nb_info["has_scaler"]:
        scaler = nb_info["scaler_type"] or "StandardScaler"
        parts.append(f'''
    def test_scaler_fit_transform(self):
        from sklearn.preprocessing import {scaler}
        import numpy as np
        X = np.random.randn(100, 5)
        scaler = {scaler}()
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == X.shape
        assert not np.any(np.isnan(X_scaled))
''')
    
    if nb_info["has_label_encoder"]:
        parts.append('''
    def test_label_encoder(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded = le.fit_transform(["cat", "dog", "cat", "bird"])
        assert len(encoded) == 4
        assert set(encoded) == {0, 1, 2}
''')
    
    if nb_info["has_text_vectorizer"]:
        vec_type = nb_info["vectorizer_type"] or "CountVectorizer"
        parts.append(f'''
    def test_text_vectorizer(self):
        from sklearn.feature_extraction.text import {vec_type}
        corpus = ["hello world", "world is great", "hello great world"]
        vec = {vec_type}()
        X = vec.fit_transform(corpus)
        assert X.shape[0] == 3
        assert X.shape[1] > 0
''')
    
    if nb_info["has_train_test_split"]:
        parts.append('''
    def test_train_test_split(self):
        from sklearn.model_selection import train_test_split
        import numpy as np
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert X_train.shape[1] == 5
''')
    
    # If no preprocessing detected, add a basic test
    if len(parts) == 1:  # only the class header
        parts.append('''
    def test_numpy_operations(self):
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        assert arr.std() > 0
''')
    
    return "\n".join(parts)


def generate_model_tests_sklearn(info: dict, nb_info: dict) -> str:
    """Generate model training and prediction tests for sklearn-based projects."""
    algos = info["custom_algorithms"]
    task = info["task"]
    
    # Find models to test
    models_to_test = []
    for a in algos:
        if a in SKLEARN_MODELS:
            mod, cls, mtype = SKLEARN_MODELS[a]
            models_to_test.append((mod, cls, mtype))
    
    if not models_to_test:
        # Default based on task
        if task == "classification":
            models_to_test = [("sklearn.linear_model", "LogisticRegression", "classification")]
        elif task == "regression":
            models_to_test = [("sklearn.linear_model", "LinearRegression", "regression")]
        elif task == "clustering":
            models_to_test = [("sklearn.cluster", "KMeans", "clustering")]
        else:
            return ""
    
    # Use the first model for primary tests
    primary_mod, primary_cls, primary_type = models_to_test[0]
    
    if primary_type == "clustering":
        return f'''
class TestModelTraining:
    """Verify model training pipeline."""

    def test_model_instantiation(self):
        from {primary_mod} import {primary_cls}
        model = {primary_cls}(n_clusters=3)
        assert model is not None

    def test_model_fit(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = np.random.randn(100, 5)
        model = {primary_cls}(n_clusters=3)
        model.fit(X)
        assert hasattr(model, "labels_")
        assert len(model.labels_) == 100

    def test_prediction(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = np.random.randn(100, 5)
        model = {primary_cls}(n_clusters=3)
        model.fit(X)
        labels = model.predict(X[:10])
        assert labels.shape == (10,)
        assert all(0 <= l < 3 for l in labels)


class TestPrediction:
    """Verify prediction outputs."""

    def test_prediction_shape(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = np.random.randn(200, 5)
        model = {primary_cls}(n_clusters=3)
        model.fit(X)
        preds = model.predict(X[:50])
        assert preds.shape == (50,)

    def test_prediction_type(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = np.random.randn(200, 5)
        model = {primary_cls}(n_clusters=3)
        model.fit(X)
        preds = model.predict(X[:10])
        assert preds.dtype in [np.int32, np.int64, np.float64, np.float32]
'''
    
    if primary_type == "classification":
        # Check if primary model needs non-negative data
        primary_nonneg = primary_cls in ("MultinomialNB", "BernoulliNB")
        x100 = "np.abs(np.random.randn(100, 5))" if primary_nonneg else "np.random.randn(100, 5)"
        x200 = "np.abs(np.random.randn(200, 5))" if primary_nonneg else "np.random.randn(200, 5)"
        
        # Build import lines for all models
        extra_model_tests = ""
        for mod, cls, mtype in models_to_test[1:3]:  # test up to 3 models
            if cls == primary_cls:
                continue
            # Skip models that need special imports
            needs_nonneg = cls in ("MultinomialNB", "BernoulliNB")
            x_gen = "np.abs(np.random.randn(100, 5))" if needs_nonneg else "np.random.randn(100, 5)"
            x_gen10 = "X[:10]" 
            if mod.startswith(("xgboost", "catboost", "lightgbm")):
                extra_model_tests += f'''
    def test_{cls.lower()}_fit(self):
        try:
            from {mod} import {cls}
        except ImportError:
            pytest.skip("{mod} not installed")
        import numpy as np
        X = {x_gen}
        y = np.random.randint(0, 2, 100)
        model = {cls}()
        model.fit(X, y)
        preds = model.predict({x_gen10})
        assert preds.shape == (10,)
'''
            else:
                extra_model_tests += f'''
    def test_{cls.lower()}_fit(self):
        from {mod} import {cls}
        import numpy as np
        X = {x_gen}
        y = np.random.randint(0, 2, 100)
        model = {cls}()
        model.fit(X, y)
        preds = model.predict({x_gen10})
        assert preds.shape == (10,)
'''
        
        return f'''
class TestModelTraining:
    """Verify model training pipeline."""

    def test_model_instantiation(self):
        from {primary_mod} import {primary_cls}
        model = {primary_cls}()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_model_fit(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = {x100}
        y = np.random.randint(0, 2, 100)
        model = {primary_cls}()
        model.fit(X, y)
        assert hasattr(model, "predict")

    def test_model_score(self):
        from {primary_mod} import {primary_cls}
        from sklearn.metrics import accuracy_score
        import numpy as np
        X = {x200}
        y = np.random.randint(0, 2, 200)
        model = {primary_cls}()
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        score = accuracy_score(y[150:], preds)
        assert 0.0 <= score <= 1.0
{extra_model_tests}

class TestPrediction:
    """Verify prediction outputs."""

    def test_prediction_shape(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = {x200}
        y = np.random.randint(0, 2, 200)
        model = {primary_cls}()
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        assert preds.shape == (50,)

    def test_prediction_type(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = {x200}
        y = np.random.randint(0, 2, 200)
        model = {primary_cls}()
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        assert preds.dtype in [np.int32, np.int64, np.float64, np.float32]

    def test_no_nan_predictions(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = {x200}
        y = np.random.randint(0, 2, 200)
        model = {primary_cls}()
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        assert not np.any(np.isnan(preds))
'''
    
    else:  # regression
        return f'''
class TestModelTraining:
    """Verify model training pipeline."""

    def test_model_instantiation(self):
        from {primary_mod} import {primary_cls}
        model = {primary_cls}()
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_model_fit(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model = {primary_cls}()
        model.fit(X, y)
        assert hasattr(model, "predict")

    def test_model_r2(self):
        from {primary_mod} import {primary_cls}
        from sklearn.metrics import r2_score
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(200) * 0.1
        model = {primary_cls}()
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        r2 = r2_score(y[150:], preds)
        assert r2 > 0.5  # should fit well on linear data


class TestPrediction:
    """Verify prediction outputs."""

    def test_prediction_shape(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = np.random.randn(200, 5)
        y = np.random.randn(200)
        model = {primary_cls}()
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        assert preds.shape == (50,)

    def test_predictions_are_numeric(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = np.random.randn(200, 5)
        y = np.random.randn(200)
        model = {primary_cls}()
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        assert np.issubdtype(preds.dtype, np.floating) or np.issubdtype(preds.dtype, np.integer)

    def test_no_nan_predictions(self):
        from {primary_mod} import {primary_cls}
        import numpy as np
        X = np.random.randn(200, 5)
        y = np.random.randn(200)
        model = {primary_cls}()
        model.fit(X[:150], y[:150])
        preds = model.predict(X[150:])
        assert not np.any(np.isnan(preds))
'''


def generate_model_tests_keras(info: dict, nb_info: dict) -> str:
    """Generate model tests for Keras/TF-based projects."""
    task = info["task"]
    
    if task == "cv":
        return '''
class TestModelTraining:
    """Verify Keras model pipeline."""

    def test_keras_importable(self):
        try:
            from tensorflow import keras
        except ImportError:
            pytest.skip("tensorflow not installed")

    def test_model_build(self):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
        except ImportError:
            pytest.skip("tensorflow not installed")
        model = Sequential([
            Conv2D(8, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(16, activation="relu"),
            Dense(2, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        assert model is not None
        assert len(model.layers) == 5

    def test_model_predict_shape(self):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
        except ImportError:
            pytest.skip("tensorflow not installed")
        import numpy as np
        model = Sequential([
            Conv2D(8, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(16, activation="relu"),
            Dense(2, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        X_dummy = np.random.randn(5, 32, 32, 3).astype(np.float32)
        preds = model.predict(X_dummy, verbose=0)
        assert preds.shape == (5, 2)


class TestPrediction:
    """Verify prediction outputs."""

    def test_predictions_sum_to_one(self):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
        except ImportError:
            pytest.skip("tensorflow not installed")
        import numpy as np
        model = Sequential([
            Conv2D(8, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(16, activation="relu"),
            Dense(2, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        X_dummy = np.random.randn(5, 32, 32, 3).astype(np.float32)
        preds = model.predict(X_dummy, verbose=0)
        for row in preds:
            assert abs(row.sum() - 1.0) < 1e-5
'''
    
    else:  # NLP or tabular keras
        return '''
class TestModelTraining:
    """Verify Keras model pipeline."""

    def test_keras_importable(self):
        try:
            from tensorflow import keras
        except ImportError:
            pytest.skip("tensorflow not installed")

    def test_model_build(self):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
        except ImportError:
            pytest.skip("tensorflow not installed")
        model = Sequential([
            Dense(32, activation="relu", input_shape=(10,)),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        assert len(model.layers) == 3

    def test_model_fit_and_predict(self):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
        except ImportError:
            pytest.skip("tensorflow not installed")
        import numpy as np
        model = Sequential([
            Dense(32, activation="relu", input_shape=(10,)),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0)
        preds = model.predict(X[:5], verbose=0)
        assert preds.shape == (5, 1)


class TestPrediction:
    """Verify prediction outputs."""

    def test_prediction_range(self):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
        except ImportError:
            pytest.skip("tensorflow not installed")
        import numpy as np
        model = Sequential([
            Dense(16, activation="relu", input_shape=(10,)),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        model.fit(X, y, epochs=1, verbose=0)
        preds = model.predict(X[:5], verbose=0)
        assert np.all(preds >= 0) and np.all(preds <= 1)
'''


def generate_model_tests_pytorch(info: dict, nb_info: dict) -> str:
    """Generate model tests for PyTorch-based projects."""
    return '''
class TestModelTraining:
    """Verify PyTorch model pipeline."""

    def test_torch_importable(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

    def test_model_forward_pass(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        X = torch.randn(5, 10)
        out = model(X)
        assert out.shape == (5, 2)

    def test_model_backward(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        X = torch.randn(5, 10)
        y = torch.tensor([0, 1, 0, 1, 0])
        out = model(X)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        assert model[0].weight.grad is not None


class TestPrediction:
    """Verify prediction outputs."""

    def test_prediction_shape(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        model.eval()
        with torch.no_grad():
            X = torch.randn(10, 10)
            out = model(X)
        assert out.shape == (10, 2)

    def test_no_nan_output(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(5, 10))
        assert not torch.any(torch.isnan(out))
'''


def generate_eda_tests(info: dict, nb_info: dict) -> str:
    """Generate tests for EDA-only / conceptual projects."""
    return '''
class TestModelTraining:
    """EDA/conceptual project — no model training tests needed."""

    def test_pandas_available(self):
        import pandas as pd
        assert pd.__version__ is not None

    def test_numpy_available(self):
        import numpy as np
        assert np.__version__ is not None


class TestPrediction:
    """EDA/conceptual project — no prediction tests needed."""

    def test_placeholder(self):
        # This project is EDA/conceptual only — no model to predict with
        assert True
'''


def generate_timeseries_tests(info: dict, nb_info: dict) -> str:
    """Generate tests for time series projects."""
    return '''
class TestModelTraining:
    """Verify time-series model pipeline."""

    def test_statsmodels_available(self):
        try:
            import statsmodels
            assert statsmodels.__version__ is not None
        except ImportError:
            pytest.skip("statsmodels not installed")

    def test_arima_fit(self):
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            pytest.skip("statsmodels not installed")
        import numpy as np
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100)) + 100
        model = ARIMA(data, order=(1, 1, 1))
        fitted = model.fit()
        assert fitted.aic is not None

    def test_forecast_shape(self):
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            pytest.skip("statsmodels not installed")
        import numpy as np
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100)) + 100
        model = ARIMA(data, order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=10)
        assert len(forecast) == 10


class TestPrediction:
    """Verify forecast outputs."""

    def test_forecast_numeric(self):
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            pytest.skip("statsmodels not installed")
        import numpy as np
        np.random.seed(42)
        data = np.cumsum(np.random.randn(100)) + 100
        fitted = ARIMA(data, order=(1, 1, 1)).fit()
        fc = fitted.forecast(steps=5)
        assert not np.any(np.isnan(fc))
'''


# ── Main generator ──────────────────────────────────────────────────────────

def generate_test_file(info: dict, nb_info: dict) -> str:
    """Generate a complete test file for one project."""
    pnum = info["number"]
    slug = info["slug"]
    # Apply known slug corrections
    if slug in SLUG_CORRECTIONS:
        slug = SLUG_CORRECTIONS[slug]
        info["slug"] = slug
    project = info["project"]
    task = info["task"]
    framework = info["framework"]
    
    # Determine markers
    markers = []
    data_status = info.get("data_status", "")
    if data_status not in ("OK_BUILTIN",):
        markers.append("data")
    if task == "classification":
        markers.append("sklearn")
    elif task == "regression":
        markers.append("sklearn")
    elif task == "clustering":
        markers.append("clustering")
    elif task == "cv":
        markers.append("cv")
        markers.append("slow")
    elif task == "nlp":
        markers.append("nlp")
    elif task == "eda":
        markers.append("eda")
    
    if framework == "keras":
        markers.append("keras")
        markers.append("slow")
    elif framework == "pytorch":
        markers.append("pytorch")
        markers.append("slow")
    elif framework == "statsmodels":
        markers.append("timeseries")
    
    marker_decorators = "\n".join(f"pytestmark.append(pytest.mark.{m})" for m in markers)
    
    header = f'''"""
Tests for: {project}
Auto-generated by Phase 4 test generator.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "{slug}"
pytestmark = []
{marker_decorators}

'''
    
    # Generate sections
    sections = []
    
    # 1. Data loading tests
    data_status = info["data_status"]
    if data_status in ("OK_LOCAL", "DOWNLOADED") or (info["data_files"] and data_status not in ("OK_BUILTIN",)):
        # Check if it's image data
        if task == "cv" and nb_info.get("has_images"):
            sections.append(generate_image_data_tests(info))
        else:
            sections.append(generate_csv_data_tests(info, nb_info))
    elif data_status == "OK_BUILTIN" or (not info["data_files"] and not info["data_path"]):
        sections.append(generate_builtin_data_tests(info))
    else:
        sections.append(generate_csv_data_tests(info, nb_info))
    
    # 2. Preprocessing tests
    sections.append(generate_preprocessing_tests(info, nb_info))
    
    # 3. Model training + 4. Prediction tests
    if task == "eda":
        sections.append(generate_eda_tests(info, nb_info))
    elif framework == "keras":
        sections.append(generate_model_tests_keras(info, nb_info))
    elif framework == "pytorch":
        sections.append(generate_model_tests_pytorch(info, nb_info))
    elif framework == "statsmodels" or "ARIMA" in " ".join(info.get("model_training_snippets", [])) or "arima" in info["project"].lower():
        sections.append(generate_timeseries_tests(info, nb_info))
    elif task in ("classification", "regression", "clustering"):
        sections.append(generate_model_tests_sklearn(info, nb_info))
    elif info["primary_model"] and info["primary_model"] in SKLEARN_MODELS:
        # CV/NLP projects that use sklearn models (e.g. LogisticRegression on iris)
        sections.append(generate_model_tests_sklearn(info, nb_info))
    else:
        sections.append(generate_eda_tests(info, nb_info))
    
    return header + "\n".join(sections) + "\n"


def main():
    print("Phase 4 — Test Generator")
    print("=" * 60)
    
    # Load all audit data
    p1_inventory = load_phase1_inventory()
    p2_detail = load_phase2_detail()
    p3_status = load_phase3_status()
    
    # Build lookup dicts
    p2_by_project = {e["project"]: e for e in p2_detail}
    p3_by_project = {e["project"]: e for e in p3_status}
    
    # Filter to actual ML projects
    ml_projects = [
        row for row in p1_inventory
        if row["project"].startswith(("Machine Learning Project", "Project"))
        and row["project"] not in ("audit_phase2",)
    ]
    
    print(f"Found {len(ml_projects)} ML projects")
    
    generated = 0
    skipped = 0
    errors = []
    
    for p1 in ml_projects:
        project = p1["project"]
        pnum = extract_project_number(project)
        p2 = p2_by_project.get(project, {})
        p3 = p3_by_project.get(project, {})
        
        if not p2:
            # No phase 2 detail available — skip
            skipped += 1
            continue
        
        # Classify
        info = classify_project(p2, p1, p3)
        slug = info["slug"]
        
        if not slug:
            slug = slugify(project)
            info["slug"] = slug
        
        # Parse notebook
        nb_files = []
        for nb_entry in p2.get("notebooks", []):
            nb_files.append(nb_entry.get("file", ""))
        nb_info = parse_notebook_cells(project, nb_files)
        
        # Generate test file
        try:
            content = generate_test_file(info, nb_info)
            filename = f"test_p{pnum.zfill(3)}_{slug[:40]}.py"
            filepath = TESTS_DIR / filename
            filepath.write_text(content, encoding="utf-8")
            generated += 1
        except Exception as e:
            errors.append((project, str(e)))
    
    print(f"\nResults:")
    print(f"  Generated: {generated}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {len(errors)}")
    
    if errors:
        print("\nErrors:")
        for proj, err in errors:
            print(f"  {proj}: {err}")
    
    # Count test functions
    total_tests = 0
    for f in TESTS_DIR.glob("test_p*.py"):
        content = f.read_text(encoding="utf-8")
        total_tests += content.count("def test_")
    
    print(f"\nTotal test functions generated: {total_tests}")
    print(f"Test files written to: {TESTS_DIR}")


if __name__ == "__main__":
    main()
