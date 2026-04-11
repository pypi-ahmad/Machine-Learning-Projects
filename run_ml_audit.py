"""
production-grade ML Audit System
=================================
Discovers, executes, scores and reports on every .ipynb in the repository.

Scoring rubric (0-100):
  execution       20  – notebook ran without errors
  pipeline        20  – has all key ML pipeline stages
  performance     30  – primary metric meets difficulty threshold
  code_quality    10  – no dead-code / hardcoded paths
  reproducibility 10  – random seed fixed, deterministic flag
  mlops           10  – mlflow logs present OR results serialised

Verdicts: PRODUCTION_READY ≥90 | NEEDS_IMPROVEMENT 80-89 | WEAK 70-79 | FAILED <70
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import mlflow
import nbformat
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

# ── configuration ─────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).parent.resolve()
REPORTS_DIR     = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

MLFLOW_URI            = f"sqlite:///{REPO_ROOT}/mlflow.db"
DEFAULT_KERNEL        = os.environ.get("ML_AUDIT_KERNEL", "ml-projects")
OLLAMA_CHAT_MODEL     = os.environ.get("ML_AUDIT_OLLAMA_MODEL",       "llama3.1:8b")
OLLAMA_EMBED_MODEL    = os.environ.get("ML_AUDIT_OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe:latest")
MAX_WORKERS           = int(os.environ.get("ML_AUDIT_WORKERS", "1"))   # >1 for parallel

# per-category execution timeout (seconds)
TIMEOUT_MAP = {
    "computer vision": 2400,
    "deep learning":   2400,
    "nlp":             1800,
    "reinforcement":   1800,
    "time series":     1200,
    "classification":  1200,
    "regression":      1200,
    "clustering":      900,
    "recommendation":  900,
    "anomaly":         900,
    "data analysis":   600,
    "100_local":       600,
    "advance":         600,
    "rag":             600,
    "genai":           600,
    "agentic":         600,
    "langgraph":       600,
    "llamaindex":      600,
    "crewai":          600,
    "fine tuning":     900,
    "default":         600,
}

# ── Python 2→3 compatibility patches ─────────────────────────────────────────
PY2_FIXES = [
    # StringIO moved to io module in Python 3
    ("from StringIO import StringIO",          "from io import StringIO"),
    ("from StringIO import BytesIO",            "from io import BytesIO"),
    ("import StringIO\n",                       "from io import StringIO\n"),
    # cPickle → pickle
    ("import cPickle\n",                        "import pickle as cPickle\n"),
    # urllib2 → urllib.request
    ("import urllib2\n",                        "import urllib.request as urllib2\n"),
    # __future__ lines (must be first statement; in Jupyter middle-of-cell is SyntaxError)
    ("from __future__ import division\n",       ""),
    ("from __future__ import print_function\n", ""),
    ("from __future__ import unicode_literals\n", ""),
    # seaborn API changes: factorplot→catplot (deprecated seaborn <0.12)
    ("sns.factorplot(",                         "sns.catplot("),
    ("seaborn.factorplot(",                     "seaborn.catplot("),
    # pandas deprecated .ix indexer → .loc
    (".ix[",                                    ".loc["),
    # pandas Panel removed in 0.25
    ("pd.Panel(",                               "None  # pd.Panel removed; "),
    # matplotlib deprecated squeeze kwarg pattern  
    ("plt.tight_layout()",                      "plt.tight_layout()"),
]
LC_REPLACEMENTS = [
    (r"from langchain\.prompts import (ChatPromptTemplate[^\n]*)",
     r"from langchain_core.prompts import \1"),
    (r"from langchain\.prompts import (PromptTemplate[^\n]*)",
     r"from langchain_core.prompts import \1"),
    (r"from langchain\.schema import Document",
     "from langchain_core.documents import Document"),
    (r"from langchain\.schema import (BaseRetriever[^\n]*)",
     r"from langchain_core.retrievers import \1"),
    (r"from langchain\.chat_models import (ChatOpenAI[^\n]*)",
     r"from langchain_openai import \1"),
    (r"from langchain\.llms import (OpenAI[^\n]*)",
     r"from langchain_openai import \1"),
    (r"from langchain\.embeddings import (OpenAIEmbeddings[^\n]*)",
     r"from langchain_openai import \1"),
    (r"from langchain\.vectorstores import Chroma",
     "from langchain_community.vectorstores import Chroma"),
    (r"from langchain\.vectorstores import FAISS",
     "from langchain_community.vectorstores import FAISS"),
]

# Package aliases: what the error says → what pip calls it
PKG_MAP = {
    "sklearn":         "scikit-learn",
    "PIL":             "Pillow",
    "cv2":             "opencv-python",
    "yaml":            "PyYAML",
    "bs4":             "beautifulsoup4",
    "dotenv":          "python-dotenv",
    "pydantic_settings":"pydantic-settings",
    "google.colab":    None,   # skip – Colab-only
    "tqdm":            "tqdm",
    "rich":            "rich",
    "lightgbm":        "lightgbm",
    "xgboost":         "xgboost",
    "catboost":        "catboost",
    "optuna":          "optuna",
    "shap":            "shap",
    "plotly":          "plotly",
    "seaborn":         "seaborn",
}

# Metrics that indicate good performance in notebook output text
METRIC_PATTERNS = [
    re.compile(r"(?:accuracy|acc)[:\s=]+([0-9]+\.[0-9]+)", re.I),
    re.compile(r"(?:f1(?:[_-]score)?)[:\s=]+([0-9]+\.[0-9]+)", re.I),
    re.compile(r"(?:roc[_-]?auc|auc)[:\s=]+([0-9]+\.[0-9]+)", re.I),
    re.compile(r"r[\^²]2?[:\s=]+([0-9]+\.[0-9]+)", re.I),
    re.compile(r"(?:mape)[:\s=]+([0-9]+\.[0-9]+)", re.I),
    re.compile(r"(?:silhouette)[:\s=]+([0-9]+\.[0-9]+)", re.I),
    re.compile(r"(?:rmse|mae)[:\s=]+([0-9]+\.[0-9]+)", re.I),
    # classification_report format: "accuracy   0.9312" or "weighted avg   0.93"
    re.compile(r"accuracy\s+([0-9]+\.[0-9]+)", re.I),
    re.compile(r"weighted\s+avg\s+[0-9.]+\s+[0-9.]+\s+([0-9]+\.[0-9]+)", re.I),
    # Keras/PyTorch training output: "val_acc: 0.9312" or "val_accuracy: 0.93"
    re.compile(r"val[_-]acc(?:uracy)?[:\s]+([0-9]+\.[0-9]+)", re.I),
    re.compile(r"test[_-]acc(?:uracy)?[:\s=]+([0-9]+\.[0-9]+)", re.I),
    # Common print formats: "Accuracy: 93.12%" or "Acc = 0.93"
    re.compile(r"[Aa]cc(?:uracy)?[:\s=]+([0-9]+\.[0-9]+)\s*%?"),
    # Loss metrics (inverted: lower is better, but presence indicates training happened)
    re.compile(r"(?:val_loss|test_loss)[:\s=]+([0-9]+\.[0-9]+)", re.I),
    # Cross-validation score
    re.compile(r"(?:cv[_-]?score|cross[_-]?val)[:\s=]+([0-9]+\.[0-9]+)", re.I),
]

mlflow.set_tracking_uri(MLFLOW_URI)


# ── helpers ────────────────────────────────────────────────────────────────────

def _resolve_timeout(nb_path: Path) -> int:
    p = str(nb_path).lower()
    for key, t in TIMEOUT_MAP.items():
        if key in p:
            return t
    return TIMEOUT_MAP["default"]


def estimate_difficulty(nb_path: Path) -> str:
    p = str(nb_path).lower()
    if any(k in p for k in ("deep learning", "computer vision", "reinforcement",
                             "100_local", "agentic", "advance", "fine tuning")):
        return "HARD"
    if any(k in p for k in ("nlp", "time series", "recommendation")):
        return "MEDIUM"
    return "EASY"


def _is_canonical(nb_path: Path) -> bool:
    """True for the human-readable 'project' notebook, False for Kaggle-style copies."""
    name = nb_path.stem
    # reject plain test artefacts and version copies
    if name.lower() in {"untitled", "test_run", "executed_check"}:
        return False
    # reject versioned copies: name(1).ipynb, name(2).ipynb, name_v2.ipynb etc.
    if re.search(r"\(\d+\)$|_v\d+$", name):
        return False
    # reject purely snake_case Kaggle copies when a Title Case sibling exists
    folder = nb_path.parent
    if "_" in name and name == name.lower():
        siblings = [p.stem for p in folder.glob("*.ipynb")
                    if p != nb_path and p.stem.lower() not in {"untitled", "test_run"}]
        if any(s[0].isupper() for s in siblings):
            return False
    return True


def _infer_category(nb_path: Path) -> str:
    parts = str(nb_path).lower().split(os.sep)
    top   = parts[1] if len(parts) > 1 else parts[0]
    return top


def _normalize_source(source: str) -> str:
    """Apply compatibility patches to a cell source string."""
    for old, new in PY2_FIXES:
        source = source.replace(old, new)
    for pat, repl in LC_REPLACEMENTS:
        source = re.sub(pat, repl, source)
    # Patch heavy Ollama model names → audit-safe defaults
    source = re.sub(r'(?<=model=")[^"]*qwen[^"]*(?=")', OLLAMA_CHAT_MODEL, source)
    source = re.sub(r'(?<=model=")[^"]*nomic-embed-text-v2[^"]*(?=")', OLLAMA_EMBED_MODEL, source)
    source = re.sub(r'(?<=model=")[^"]*nomic-embed[^"]*(?=")', OLLAMA_EMBED_MODEL, source)
    source = re.sub(r'(?<="ollama/)[^"]*qwen[^"]*(?=")', OLLAMA_CHAT_MODEL, source)
    # Wrap bare surprise imports — scikit-surprise has no Python 3.13 wheel
    if "from surprise import" in source and "try:" not in source.split("from surprise import")[0].split("\n")[-1]:
        source = re.sub(
            r"^(from surprise import .+)$",
            r"try:\n    \1\nexcept ImportError:\n    pass  # scikit-surprise unavailable on this Python",
            source, flags=re.MULTILINE,
        )
    return source


def _detect_primary_df_name(nb: nbformat.NotebookNode) -> str:
    """Detect the primary DataFrame variable name used in this notebook."""
    code = "\n".join(c.get("source", "") for c in nb.cells if c.get("cell_type") == "code")
    # Common patterns: 'train_df', 'data_df', 'df_train', etc.
    for candidate in ("train_df", "data", "dataset", "df_train", "raw_df", "data_df"):
        if re.search(rf"^{re.escape(candidate)}\s*=", code, re.M):
            return candidate
    return "df"


def normalize_notebook(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    primary_df = _detect_primary_df_name(nb)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = _normalize_source(cell.get("source", ""))
        # If this cell uses bare 'df' but repo primary var is something else,
        # prepend an alias so the cell resolves correctly
        if primary_df != "df" and re.search(r"\bdf\b", src) and "df =" not in src:
            src = f"df = {primary_df} if '{primary_df}' in dir() else None  # audit alias\n" + src
        cell["source"] = src
    return nb


def _extract_metrics(nb: nbformat.NotebookNode) -> dict[str, float]:
    """Scan cell outputs for printed numeric metrics."""
    metrics: dict[str, float] = {}
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            text = output.get("text", "") or "".join(output.get("data", {}).get("text/plain", []))
            for pat in METRIC_PATTERNS:
                m = pat.search(text)
                if m:
                    val = float(m.group(1))
                    key = pat.pattern.split("(?:")[1].split(")")[0].split("|")[0].strip("(?:")
                    metrics[key] = max(metrics.get(key, 0.0), val)
    return metrics


def _notebook_type(nb_path: Path) -> str:
    """Classify notebook into execution profile for category-aware scoring."""
    p = str(nb_path).lower()
    if any(k in p for k in ("advance rag", "rag\\", "rag/", "genai", "agentic",
                             "crewai", "langgraph", "llamaindex", "100_local",
                             "agents with tools", "fine tuning")):
        return "LLM"
    if any(k in p for k in ("data analysis", "exploratory")):
        return "EDA"
    if "recommendation" in p:
        return "RECOMMENDER"
    if any(k in p for k in ("conceptual", "lecture", "tutorial", "intro")):
        return "CONCEPTUAL"
    return "ML"


def _score_notebook(ran: bool, nb: nbformat.NotebookNode, nb_path: Path, metrics: dict) -> dict:
    """Return a detailed score breakdown."""
    scores: dict[str, Any] = {
        "execution":       0,
        "pipeline":        0,
        "performance":     0,
        "code_quality":    0,
        "reproducibility": 0,
        "mlops":           0,
        "total":           0,
        "issues":          [],
    }
    if not ran:
        return scores

    # Count cell-level errors (allow_errors mode)
    cell_err_count = sum(
        1 for cell in nb.cells
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    )
    total_code = max(1, sum(1 for c in nb.cells if c.get("cell_type") == "code"))
    err_ratio  = cell_err_count / total_code

    # Execution: 20  (deduct proportionally for cell errors)
    scores["execution"] = max(0, int(20 * (1 - err_ratio)))

    # ── pipeline completeness: 20 ──────────────────────────────────────────────
    code_text = "\n".join(
        cell.get("source", "") for cell in nb.cells if cell.get("cell_type") == "code"
    )
    nb_type = _notebook_type(nb_path)

    if nb_type == "EDA":
        pipeline_checks = {
            "data_load":   bool(re.search(r"(read_csv|load_dataset|pd\.DataFrame|pd\.read_|fetch_)", code_text, re.I)),
            "clean":       bool(re.search(r"(fillna|dropna|isnull|duplicated|astype|rename|replace)", code_text, re.I)),
            "visualise":   bool(re.search(r"(plt\.|px\.|go\.|seaborn|sns\.|\.plot\(|matplotlib)", code_text, re.I)),
            "stats":       bool(re.search(r"(describe\(\)|corr\(\)|value_counts|groupby|mean\(|median\(|std\()", code_text, re.I)),
            "insights":    bool(re.search(r"(print\(|display\(|\.head\(|\.tail\(|f'|f\")", code_text, re.I)),
        }
    elif nb_type == "LLM":
        pipeline_checks = {
            "imports":     bool(re.search(r"(langchain|ollama|openai|llm|chatollama|llama)", code_text, re.I)),
            "model_setup": bool(re.search(r"(ChatOllama|ChatOpenAI|llm\s*=|model\s*=)", code_text, re.I)),
            "chain":       bool(re.search(r"(chain|agent|invoke|pipe|run\(|arun\(|graph)", code_text, re.I)),
            "output":      bool(re.search(r"(print\(|display\(|response|result\s*=|answer)", code_text, re.I)),
        }
    elif nb_type == "RECOMMENDER":
        pipeline_checks = {
            "data_load":   bool(re.search(r"(read_csv|load_dataset|pd\.DataFrame|pd\.read_|fetch_)", code_text, re.I)),
            "matrix":      bool(re.search(r"(pivot|pivot_table|matrix|sparse|csr_matrix|TfidfVectorizer|cosine_similarity)", code_text, re.I)),
            "model":       bool(re.search(r"(fit\(|NMF|SVD|surprise|ALS|collaborative|content.based|cosin)", code_text, re.I)),
            "recommend":   bool(re.search(r"(recommend|top_n|similar|suggest)", code_text, re.I)),
        }
    elif nb_type == "CONCEPTUAL":
        pipeline_checks = {
            "explains":    bool(re.search(r"(print\(|display\(|Markdown|HTML\()", code_text, re.I)),
            "code":        bool(re.search(r"(def |class |import )", code_text, re.I)),
            "output":      bool(re.search(r"(\.show\(|plt\.|visuali|plot)", code_text, re.I)),
        }
    else:  # ML (default)
        pipeline_checks = {
            "data_load":  bool(re.search(r"(read_csv|load_dataset|pd\.DataFrame|pd\.read_|datasets\.load|fetch_|download)", code_text, re.I)),
            "preprocess": bool(re.search(r"(train_test_split|StandardScaler|LabelEncoder|fillna|dropna|Pipeline|ColumnTransformer|tokenize|transform)", code_text, re.I)),
            "model":      bool(re.search(r"(\.fit\(|\.train|model\.compile|AutoML|FLAML|TabPFN|CatBoost|XGBoost|RandomForest|LightGBM)", code_text, re.I)),
            "evaluate":   bool(re.search(r"(accuracy_score|f1_score|roc_auc_score|r2_score|mean_squared|silhouette|classification_report|confusion_matrix|evaluate|\.score\()", code_text, re.I)),
            "visualise":  bool(re.search(r"(plt\.|px\.|go\.|seaborn|sns\.|\.plot\()", code_text, re.I)),
        }

    missing = [k for k, v in pipeline_checks.items() if not v]
    pipeline_score = int(20 * (1 - len(missing) / max(len(pipeline_checks), 1)))
    scores["pipeline"] = pipeline_score
    if missing:
        scores["issues"].append(f"Missing pipeline stages: {missing}")

    # ── performance: 30 ───────────────────────────────────────────────────────
    difficulty  = estimate_difficulty(nb_path)
    nb_type     = _notebook_type(nb_path)
    # Base thresholds (for accuracy/F1/AUC-type metrics)
    base_thresholds = {"EASY": 0.90, "MEDIUM": 0.80, "HARD": 0.70}
    # Metric-specific thresholds (override base where metric range differs)
    METRIC_THRESHOLDS = {
        "silhouette": 0.30,   # good clustering = 0.3-0.7, 0.5 is excellent
        "rmse":       None,    # skip — lower-is-better, hard to threshold
        "mae":        None,    # skip — lower-is-better
        "mape":       None,    # handled separately via inversion
        "r2":         0.60,   # R² ≥ 0.6 reasonable regression
    }
    threshold   = base_thresholds[difficulty]
    best_metric = None
    best_val    = 0.0
    for k, v in metrics.items():
        # Normalise: accuracy/F1/AUC are 0-1; R² can be negative; MAPE is %
        if k in ("mape",) and v > 0:
            # lower is better – invert for scoring
            normed = max(0.0, 1.0 - v / 100)
        elif k in ("val_loss", "test_loss", "val_loss_pattern", "test_loss_pattern"):
            # loss metrics: skip for accuracy comparison; handled via training_detected flag
            continue
        elif METRIC_THRESHOLDS.get(k.lower()) is None and k.lower() in ("rmse", "mae"):
            # Skip lower-is-better metrics without domain threshold
            continue
        elif v > 1.0:
            normed = v / 100  # possibly printed as percentage
        else:
            normed = v
        # Use metric-specific threshold for normalization
        metric_threshold = METRIC_THRESHOLDS.get(k.lower(), threshold)
        # For silhouette / r2: compare against their own threshold for perf_ratio
        if normed > best_val:
            best_val    = normed
            best_metric = k

    if best_metric:
        # Use metric-specific threshold for performance ratio calculation
        effective_threshold = METRIC_THRESHOLDS.get(best_metric.lower(), threshold)
        if effective_threshold is None:
            effective_threshold = threshold
        perf_ratio      = min(best_val / effective_threshold, 1.0)
        scores["performance"] = int(30 * perf_ratio)
        if best_val < effective_threshold:
            scores["issues"].append(
                f"Metric {best_metric}={best_val:.4f} below threshold {effective_threshold:.2f} for {difficulty}"
            )
    else:
        # Category-based fallback: EDA/LLM/RECOMMENDER/CONCEPTUAL notebooks get performance
        # credit for successful execution and rich output (plots, summaries, text results).
        # ML notebooks also get partial credit if training evidence is present.
        has_text_outputs = any(
            any(o.get("output_type") in ("stream", "display_data", "execute_result")
                for o in cell.get("outputs", []))
            for cell in nb.cells if cell.get("cell_type") == "code"
        )
        # Check if training happened (DL notebooks often only log loss, not accuracy to stdout)
        has_training_evidence = bool(re.search(
            r"(val_loss|test_loss|Epoch\s+\d+|fit\(|\.train\(|\.fit_transform\()",
            code_text, re.I))
        if nb_type in ("EDA", "LLM", "RECOMMENDER", "CONCEPTUAL") and has_text_outputs:
            scores["performance"] = 25   # 83% of 30 for non-metric notebooks
        elif nb_type in ("EDA", "LLM", "RECOMMENDER", "CONCEPTUAL"):
            scores["performance"] = 20
        elif has_training_evidence and has_text_outputs:
            scores["performance"] = 20   # ML notebook trained something, partial credit
        else:
            scores["performance"] = 0
            scores["issues"].append("No numeric performance metric found in outputs")

    # ── code quality: 10 ──────────────────────────────────────────────────────
    cq = 10
    if re.search(r'["\'](?:[A-Z]:\\|/home/|/Users/)[^"\']+["\']', code_text):
        cq -= 3
        scores["issues"].append("Hardcoded absolute path detected")
    if re.search(r"TODO|FIXME|HACK|XXX", code_text):
        cq -= 2
        scores["issues"].append("TODO/FIXME comments found")
    scores["code_quality"] = max(cq, 0)

    # ── reproducibility: 10 ───────────────────────────────────────────────────
    has_seed = bool(re.search(r"(random_state|seed|np\.random\.seed|torch\.manual_seed|set_seed)", code_text, re.I))
    scores["reproducibility"] = 10 if has_seed else 5
    if not has_seed:
        scores["issues"].append("No random seed found")

    # ── MLOps: 10 ─────────────────────────────────────────────────────────────
    has_mlflow = bool(re.search(r"mlflow\.(log|set|start)", code_text, re.I))
    has_serial  = bool(re.search(r"(joblib\.dump|pickle\.dump|\.save\(|torch\.save|json\.dump|to_csv)", code_text, re.I))
    scores["mlops"] = 10 if (has_mlflow or has_serial) else 5

    scores["total"] = (scores["execution"] + scores["pipeline"] + scores["performance"]
                       + scores["code_quality"] + scores["reproducibility"] + scores["mlops"])
    return scores


def _install_package(pkg: str) -> bool:
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            timeout=120,
        )
        return True
    except Exception:
        return False


# ── core execution ────────────────────────────────────────────────────────────

def run_notebook(
    nb_path: Path,
    *,
    kernel: str = DEFAULT_KERNEL,
    attempt: int = 1,
) -> dict[str, Any]:
    nb_path   = Path(nb_path).resolve()
    stem      = nb_path.stem
    rel_path  = str(nb_path.relative_to(REPO_ROOT))
    safe_stem = re.sub(r"[^\w\-]", "_", stem)[:80]
    report_path = REPORTS_DIR / f"{safe_stem}_report.json"

    # Terminate any stale MLflow run
    while mlflow.active_run():
        mlflow.end_run()

    difficulty = estimate_difficulty(nb_path)
    report: dict[str, Any] = {
        "notebook_name":  rel_path,
        "status":         "FAIL",
        "score":          0,
        "difficulty":     difficulty,
        "execution":      {"ran_successfully": False, "errors": [], "attempt": attempt},
        "pipeline":       {"complete": False, "missing_steps": []},
        "performance":    {"metric_name": None, "value": None, "threshold": None, "meets_threshold": False},
        "baseline_comparison": {"baseline_score": None, "model_score": None,
                                "improvement_percent": None, "better_than_baseline": None},
        "code_quality":   {"issues": [], "score": 0},
        "reproducibility":{"seed_fixed": False, "deterministic": False},
        "mlops":          {"mlflow_logging": False, "artifacts_saved": False},
        "final_verdict":  "FAILED",
    }

    print(f"[{attempt}] >> {rel_path}", flush=True)

    try:
        nb = nbformat.read(nb_path, as_version=4)
        nb = normalize_notebook(nb)
    except Exception as exc:
        report["execution"]["errors"].append(f"Parse error: {exc}")
        _save_report(report, report_path)
        return report

    timeout = _resolve_timeout(nb_path)
    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name=kernel,
        allow_errors=True,   # record errors but keep running remaining cells
    )

    mlflow.set_experiment(safe_stem[:120])
    run_tags = {"notebook": rel_path, "attempt": str(attempt), "difficulty": difficulty}

    start = time.time()
    ran   = False
    err_msg = ""
    executed_nb = nb
    cell_errors: list[str] = []

    try:
        with mlflow.start_run(run_name=f"audit_a{attempt}", tags=run_tags):
            ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})
            ran = True
            executed_nb = nb
            # Collect per-cell errors even when allow_errors=True
            for cell in executed_nb.cells:
                for output in cell.get("outputs", []):
                    if output.get("output_type") == "error":
                        cell_errors.append(f"{output.get('ename','')}: {output.get('evalue','')}")
            elapsed = time.time() - start
            mlflow.log_metric("execution_time_s", elapsed)
            mlflow.log_metric("cell_errors", len(cell_errors))

    except CellExecutionError as exc:
        err_msg = str(exc)[:2000]
        elapsed = time.time() - start
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - start
    finally:
        while mlflow.active_run():
            mlflow.end_run()

    # ── auto-fix: missing pip package ─────────────────────────────────────────
    if not ran and attempt <= 2:
        m = re.search(r"No module named '([^']+)'", err_msg)
        if m:
            raw_pkg = m.group(1).split(".")[0]
            pip_pkg = PKG_MAP.get(raw_pkg, raw_pkg)
            if pip_pkg is None:
                report["execution"]["errors"].append(
                    f"Colab-only import '{raw_pkg}' – cannot fix automatically"
                )
            else:
                print(f"  → auto-installing {pip_pkg} …", flush=True)
                if _install_package(pip_pkg):
                    return run_notebook(nb_path, kernel=kernel, attempt=attempt + 1)
                else:
                    report["execution"]["errors"].append(f"pip install {pip_pkg} failed")

    report["execution"]["ran_successfully"] = ran
    report["execution"]["cell_errors"]      = cell_errors[:10]  # keep first 10
    if err_msg:
        report["execution"]["errors"].append(err_msg)
    if ran:
        report["status"] = "PASS" if not cell_errors else "PASS_WITH_WARNINGS"

    # ── extract metrics, score ─────────────────────────────────────────────────
    metrics = _extract_metrics(executed_nb) if ran else {}
    breakdown = _score_notebook(ran, executed_nb, nb_path, metrics)

    score = breakdown["total"]
    report["score"] = score

    # populate structured fields
    code_text = "\n".join(
        c.get("source", "") for c in executed_nb.cells if c.get("cell_type") == "code"
    )
    report["pipeline"]["complete"]       = breakdown["pipeline"] >= 16
    report["pipeline"]["missing_steps"]  = [i for i in breakdown.get("issues", [])
                                             if "Missing pipeline" in i]
    if metrics:
        best_k = max(metrics, key=lambda k: metrics[k])
        report["performance"]["metric_name"]    = best_k
        report["performance"]["value"]          = metrics[best_k]
        diff_thresh = {"EASY": 0.90, "MEDIUM": 0.80, "HARD": 0.70}[difficulty]
        report["performance"]["threshold"]      = diff_thresh
        report["performance"]["meets_threshold"]= metrics[best_k] >= diff_thresh
    report["code_quality"]["score"]     = breakdown["code_quality"]
    report["code_quality"]["issues"]    = [i for i in breakdown["issues"]
                                            if "Hardcoded" in i or "TODO" in i]
    report["reproducibility"]["seed_fixed"] = (breakdown["reproducibility"] == 10)
    report["mlops"]["mlflow_logging"]   = bool(re.search(r"mlflow\.", code_text, re.I))
    report["mlops"]["artifacts_saved"]  = bool(re.search(
        r"(joblib\.dump|pickle\.dump|\.save\(|torch\.save|json\.dump)", code_text, re.I))

    if score >= 90:
        report["final_verdict"] = "PRODUCTION_READY"
    elif score >= 80:
        report["final_verdict"] = "NEEDS_IMPROVEMENT"
    elif score >= 70:
        report["final_verdict"] = "WEAK"
    else:
        report["final_verdict"] = "FAILED"

    # ── log summary to MLflow ─────────────────────────────────────────────────
    while mlflow.active_run():
        mlflow.end_run()
    try:
        mlflow.set_experiment(safe_stem[:120])
        with mlflow.start_run(run_name="audit_summary"):
            mlflow.log_metric("score",            score)
            mlflow.log_metric("execution_time_s", elapsed)
            mlflow.log_metric("pipeline_score",   breakdown["pipeline"])
            mlflow.log_metric("performance_score",breakdown["performance"])
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, v)
                except Exception:
                    pass
            mlflow.log_text(json.dumps(report, indent=2), "report.json")
    except Exception:
        pass
    finally:
        while mlflow.active_run():
            mlflow.end_run()

    _save_report(report, report_path)
    status_icon = "PASS" if ran else "FAIL"
    print(f"  [{status_icon}] score={score}  verdict={report['final_verdict']}", flush=True)
    return report


def _save_report(report: dict, path: Path) -> None:
    try:
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"  [warn] could not write report {path}: {exc}", flush=True)


# ── notebook discovery ─────────────────────────────────────────────────────────

SKIP_DIRS = {"venv", ".venv", ".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}

def discover_notebooks(root: Path, canonical_only: bool = True) -> list[Path]:
    notebooks: list[Path] = []
    for nb_path in sorted(root.rglob("*.ipynb")):
        if any(part in SKIP_DIRS for part in nb_path.parts):
            continue
        if canonical_only and not _is_canonical(nb_path):
            continue
        notebooks.append(nb_path)
    return notebooks


# ── summary / leaderboard ─────────────────────────────────────────────────────

def _write_summary(results: list[dict]) -> None:
    passed  = [r for r in results if r["status"] in ("PASS", "PASS_WITH_WARNINGS")]
    failed  = [r for r in results if r["status"] not in ("PASS", "PASS_WITH_WARNINGS")]
    scores  = [r["score"] for r in results]

    summary = {
        "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_notebooks":    len(results),
        "passed":             len(passed),
        "failed":             len(failed),
        "pass_rate_pct":      round(100 * len(passed) / max(len(results), 1), 1),
        "avg_score":          round(float(np.mean(scores)), 2) if scores else 0.0,
        "production_ready":   sum(1 for r in results if r["final_verdict"] == "PRODUCTION_READY"),
        "needs_improvement":  sum(1 for r in results if r["final_verdict"] == "NEEDS_IMPROVEMENT"),
        "weak":               sum(1 for r in results if r["final_verdict"] == "WEAK"),
        "failed_cnt":         sum(1 for r in results if r["final_verdict"] == "FAILED"),
        "ranking_asc":        [{"notebook": r["notebook_name"], "score": r["score"],
                                 "verdict": r["final_verdict"]}
                                for r in sorted(results, key=lambda x: x["score"])],
    }
    (REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    leaderboard = [
        {"rank": i + 1, "notebook": r["notebook_name"],
         "score": r["score"], "verdict": r["final_verdict"],
         "primary_metric": r["performance"].get("metric_name"),
         "primary_value":  r["performance"].get("value")}
        for i, r in enumerate(sorted(results, key=lambda x: x["score"], reverse=True))
    ]
    (REPORTS_DIR / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2), encoding="utf-8"
    )
    print(f"\n{'='*60}", flush=True)
    print(f"  AUDIT COMPLETE | total={len(results)}  pass={len(passed)}  fail={len(failed)}", flush=True)
    print(f"  avg_score={summary['avg_score']}  PRODUCTION_READY={summary['production_ready']}", flush=True)
    print(f"  reports → {REPORTS_DIR}", flush=True)
    print(f"{'='*60}\n", flush=True)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ML Audit – run every notebook end-to-end")
    parser.add_argument("notebooks",    nargs="*",  help="Specific notebooks to run (default: all)")
    parser.add_argument("--kernel",     default=DEFAULT_KERNEL)
    parser.add_argument("--all",        action="store_true", help="Include Kaggle-style copies too")
    parser.add_argument("--workers",    type=int, default=MAX_WORKERS,
                        help="Parallel workers (1 = serial, safe for GPU)")
    parser.add_argument("--category",  default=None,
                        help="Restrict to notebooks whose path contains this string")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip notebooks that already have a PASS report")
    args = parser.parse_args()

    if args.notebooks:
        notebooks = [Path(p).resolve() for p in args.notebooks]
    else:
        notebooks = discover_notebooks(REPO_ROOT, canonical_only=not args.all)

    if args.category:
        cat_lower = args.category.lower()
        # Match only against the top-level repo directory (parts[0])
        notebooks = [
            n for n in notebooks
            if cat_lower in n.relative_to(REPO_ROOT).parts[0].lower()
        ]

    if args.resume:
        def _has_pass(n: Path) -> bool:
            stem = re.sub(r"[^\w\-]", "_", n.stem)[:80]
            rp   = REPORTS_DIR / f"{stem}_report.json"
            if not rp.exists():
                return False
            try:
                return json.loads(rp.read_text())["status"] == "PASS"
            except Exception:
                return False
        notebooks = [n for n in notebooks if not _has_pass(n)]

    print(f">> Discovered {len(notebooks)} notebooks to audit", flush=True)

    results: list[dict] = []

    def _run(nb: Path) -> dict:
        return run_notebook(nb, kernel=args.kernel)

    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run, nb): nb for nb in notebooks}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    print(f"  [error] {futures[fut]}: {exc}", flush=True)
    else:
        for nb in notebooks:
            results.append(_run(nb))

    _write_summary(results)


if __name__ == "__main__":
    main()


# ── backwards-compat shim ─────────────────────────────────────────────────────
# estimate_difficulty is defined earlier in the module; this block is superseded.
