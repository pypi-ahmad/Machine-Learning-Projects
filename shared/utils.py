#!/usr/bin/env python3
"""
Shared utilities for the Deep Learning Projects monorepo.

Every project imports from here instead of duplicating boilerplate.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ── Repo root (two levels up from shared/utils.py) ──────────
REPO_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════
#  Device helpers
# ═════════════════════════════════════════════════════════════

def get_device(verbose: bool = True) -> "torch.device":
    """Return the best available torch device (CUDA > MPS > CPU)."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            name = torch.cuda.get_device_name(0)
            logger.info("Using CUDA device: %s", name)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        if verbose:
            logger.info("Using CPU device")
    return device


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set seeds for reproducibility across numpy, random, and torch.

    If *deterministic* is True, enables ``torch.use_deterministic_algorithms``
    (some ops may fall back to slower kernels; failures are caught and warned).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                logger.warning(
                    "torch.use_deterministic_algorithms(True) not fully supported "
                    "— some ops may remain non-deterministic"
                )
    except ImportError:
        pass


# ═════════════════════════════════════════════════════════════
#  Safe import check (subprocess-based, catches native crashes)
# ═════════════════════════════════════════════════════════════

def safe_import_available(module_name: str) -> bool:
    """Check if a Python module can be imported without crashing the current process.

    Some native extensions (e.g. torchaudio) can call exit() or segfault
    at import time if their native libraries are misconfigured, which
    cannot be caught by Python try/except.  This function spawns a
    sub-process that attempts the import and returns True only if it
    succeeds cleanly.
    """
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True, timeout=30,
    )
    return result.returncode == 0


# ═════════════════════════════════════════════════════════════
#  Dataset prompt
# ═════════════════════════════════════════════════════════════

def dataset_prompt(
    name: str,
    official_links: List[str],
    notes: str = "",
) -> None:
    """Print a clear, user-facing dataset information block."""
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  DATASET: {name}")
    print(sep)
    for link in official_links:
        print(f"  -> {link}")
    if notes:
        print(f"\n  Note: {notes}")
    print(sep + "\n")


def kaggle_prompt(dataset_slug: str) -> None:
    """Print instructions for setting up Kaggle API credentials."""
    print("\n" + "=" * 60)
    print("  KAGGLE CREDENTIALS REQUIRED")
    print("=" * 60)
    print(f"  Dataset : https://www.kaggle.com/datasets/{dataset_slug}")
    print()
    print("  Option A  (environment variable, recommended):")
    print("    1. Go to https://www.kaggle.com/settings -> 'Create New Token'")
    print("    2. Set env var  KAGGLE_API_TOKEN=<your-token>")
    print("    3. Re-run this script.")
    print()
    print("  Option B  (kaggle.json file):")
    print("    1. Go to https://www.kaggle.com/settings -> 'Create New Token'")
    print("    2. Save the downloaded kaggle.json to:")
    if sys.platform == "win32":
        print(r"         %USERPROFILE%\.kaggle\kaggle.json")
    else:
        print("         ~/.kaggle/kaggle.json")
    print("    3. Re-run this script.")
    print("=" * 60 + "\n")


def _kaggle_cli_download(slug: str, data_dir: Path) -> bool:
    """Try downloading via the official kaggle CLI. Returns True on success."""
    import shutil
    import subprocess
    import zipfile

    if not shutil.which("kaggle"):
        return False

    dest = data_dir / slug.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Some versions don't support --unzip; retry without it and unzip manually
        cmd_no_unzip = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)]
        result = subprocess.run(cmd_no_unzip, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("kaggle CLI failed: %s", result.stderr.strip())
            return False
        # manually unzip any .zip files
        for zf in dest.glob("*.zip"):
            try:
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(dest)
                zf.unlink()
            except zipfile.BadZipFile:
                pass
    return True


def download_kaggle_dataset(
    slug: str,
    data_dir: Path,
    *,
    dataset_name: str = "",
) -> Path:
    """Download a Kaggle dataset. Prefers the kaggle CLI (supports
    KAGGLE_API_TOKEN env var) and falls back to opendatasets."""
    dataset_prompt(
        name=dataset_name or slug.split("/")[-1].replace("-", " ").title(),
        official_links=[f"https://www.kaggle.com/datasets/{slug}"],
        notes="Requires Kaggle API credentials.",
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Strategy 1: kaggle CLI (supports KAGGLE_API_TOKEN natively) ----------
    if _kaggle_cli_download(slug, data_dir):
        sub = data_dir / slug.split("/")[-1]
        return sub if sub.exists() else data_dir

    # --- Strategy 2: opendatasets library -------------------------------------
    try:
        import opendatasets as od
        od.download(
            f"https://www.kaggle.com/datasets/{slug}",
            data_dir=str(data_dir),
        )
    except Exception as exc:
        logger.error("Failed to download dataset: %s", exc)
        kaggle_prompt(slug)
        raise RuntimeError(f"Dataset download failed for {slug}: {exc}") from exc

    # opendatasets puts files in data_dir/<slug-last-part>/
    sub = data_dir / slug.split("/")[-1]
    return sub if sub.exists() else data_dir


# ═════════════════════════════════════════════════════════════
#  File / directory helpers
# ═════════════════════════════════════════════════════════════

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_paths(project_file: str) -> Dict[str, Path]:
    """Return standard paths relative to the project's run.py location.

    Usage (inside any run.py):
        paths = project_paths(__file__)
        data_dir   = paths["data"]
        output_dir = paths["outputs"]
    """
    root = Path(project_file).resolve().parent
    data_dir = ensure_dir(root / "data")
    output_dir = ensure_dir(root / "outputs")
    return {
        "root": root,
        "data": data_dir,
        "outputs": output_dir,
    }


# ═════════════════════════════════════════════════════════════
#  Metrics & reporting  —  CLASSIFICATION
# ═════════════════════════════════════════════════════════════

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute standard classification metrics and return a dict."""
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                metrics["auc"] = float(roc_auc_score(y_true, y_prob.ravel()))
            else:
                metrics["auc"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
                )
        except (ValueError, IndexError):
            metrics["auc"] = None
    # Guard: target_names must match number of unique classes in data
    if labels is not None:
        n_unique = len(set(y_true) | set(y_pred))
        if len(labels) != n_unique:
            logger.warning("label count (%d) != unique classes (%d); "
                           "dropping target_names from classification_report", len(labels), n_unique)
            labels = None
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    metrics["classification_report"] = report
    return metrics


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Compute metrics, save confusion matrix plot + JSON report."""
    output_dir = ensure_dir(output_dir)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, labels)

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(cm) * 0.8), max(5, len(cm) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels or "auto", yticklabels=labels or "auto")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{prefix + ' — ' if prefix else ''}Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix + '_' if prefix else ''}confusion_matrix.png", dpi=150)
    plt.close(fig)

    # JSON report
    report_path = output_dir / f"{prefix + '_' if prefix else ''}metrics.json"
    saveable = {k: v for k, v in metrics.items() if k != "classification_report"}
    report_path.write_text(json.dumps(saveable, indent=2))

    # Text report
    txt_path = output_dir / f"{prefix + '_' if prefix else ''}classification_report.txt"
    txt_path.write_text(metrics["classification_report"])

    logger.info("Accuracy : %.4f", metrics["accuracy"])
    logger.info("F1 macro : %.4f", metrics["macro_f1"])
    if metrics.get("auc") is not None:
        logger.info("AUC      : %.4f", metrics["auc"])
    logger.info("Reports saved to %s", output_dir)

    return metrics


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    labels: Optional[List[str]] = None,
    prefix: str = "",
) -> None:
    """Save a standalone confusion matrix plot."""
    output_dir = ensure_dir(output_dir)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(cm) * 0.8), max(5, len(cm) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels or "auto", yticklabels=labels or "auto")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{prefix + ' — ' if prefix else ''}Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix + '_' if prefix else ''}confusion_matrix.png", dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════
#  Metrics & reporting  —  REGRESSION
# ═════════════════════════════════════════════════════════════

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
    return metrics


def save_regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    prefix: str = "",
) -> Dict[str, float]:
    """Compute regression metrics, save scatter plot + JSON."""
    output_dir = ensure_dir(output_dir)
    metrics = compute_regression_metrics(y_true, y_pred)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=10)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{prefix + ' — ' if prefix else ''}Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix + '_' if prefix else ''}scatter.png", dpi=150)
    plt.close(fig)

    # JSON report
    report_path = output_dir / f"{prefix + '_' if prefix else ''}metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2))

    logger.info("MAE  : %.4f", metrics["mae"])
    logger.info("RMSE : %.4f", metrics["rmse"])
    logger.info("R²   : %.4f", metrics["r2"])
    logger.info("Reports saved to %s", output_dir)

    return metrics


# ═════════════════════════════════════════════════════════════
#  Python version guard
# ═════════════════════════════════════════════════════════════

PYCARET_SUPPORTED_VERSIONS = [(3, 9), (3, 10), (3, 11)]


def pycaret_python_ok() -> bool:
    """Return True if the running Python is supported by PyCaret."""
    return (sys.version_info.major, sys.version_info.minor) in PYCARET_SUPPORTED_VERSIONS


def assert_supported_python(*, for_pycaret: bool = False) -> None:
    """Print a user-friendly message if Python version is unsupported.

    Does NOT raise or crash — callers should check pycaret_python_ok()
    and choose a fallback path instead.
    """
    if for_pycaret and not pycaret_python_ok():
        supported = ", ".join(f"{ma}.{mi}" for ma, mi in PYCARET_SUPPORTED_VERSIONS)
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        logger.warning(
            "PyCaret requires Python %s but you have %s. "
            "Falling back to LazyPredict / sklearn baseline.",
            supported, current,
        )


# ═════════════════════════════════════════════════════════════
#  Missing-dependency helper
# ═════════════════════════════════════════════════════════════

def missing_dependency_metrics(
    output_dir: Path,
    missing: List[str],
    install_cmd: str = "",
) -> None:
    """Write metrics.json for a missing optional dependency, then exit(0)."""
    msg = f"Missing optional dependencies: {', '.join(missing)}"
    if install_cmd:
        msg += f"\n  Install with: {install_cmd}"
        print(f"\n  {msg}\n")
    write_metrics(
        {
            "status": "missing_dependency",
            "missing": missing,
            "install_cmd": install_cmd,
            "message": msg,
        },
        output_dir,
    )
    logger.warning("Missing dependency -> metrics.json written. Exiting cleanly.")
    sys.exit(0)


# ═════════════════════════════════════════════════════════════
#  PyCaret helper (with version guard)
# ═════════════════════════════════════════════════════════════

def run_pycaret_classification(
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
    *,
    session_id: int = 42,
    top_n: int = 3,
) -> Any:
    """Run PyCaret classification pipeline — compare, tune, evaluate, save."""
    assert_supported_python(for_pycaret=True)
    from pycaret.classification import (
        setup as cls_setup,
        compare_models,
        tune_model,
        evaluate_model,
        save_model,
        predict_model,
        pull,
    )

    output_dir = ensure_dir(output_dir)

    cls_setup(data=df, target=target, session_id=session_id, verbose=False)
    best_models = compare_models(n_select=top_n, sort="AUC")
    comparison_df = pull()
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    logger.info("Model comparison:\n%s", comparison_df.to_string())

    best = best_models[0] if isinstance(best_models, list) else best_models
    tuned = tune_model(best, optimize="AUC")
    save_model(tuned, str(output_dir / "best_model"))
    logger.info("Best model saved to %s", output_dir / "best_model.pkl")

    return tuned


def run_pycaret_regression(
    df: pd.DataFrame,
    target: str,
    output_dir: Path,
    *,
    session_id: int = 42,
    top_n: int = 3,
) -> Any:
    """Run PyCaret regression pipeline — compare, tune, evaluate, save."""
    assert_supported_python(for_pycaret=True)
    from pycaret.regression import (
        setup as reg_setup,
        compare_models,
        tune_model,
        save_model,
        pull,
    )

    output_dir = ensure_dir(output_dir)

    reg_setup(data=df, target=target, session_id=session_id, verbose=False)
    best_models = compare_models(n_select=top_n, sort="RMSE")
    comparison_df = pull()
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    logger.info("Model comparison:\n%s", comparison_df.to_string())

    best = best_models[0] if isinstance(best_models, list) else best_models
    tuned = tune_model(best, optimize="RMSE")
    save_model(tuned, str(output_dir / "best_model"))
    logger.info("Best model saved to %s", output_dir / "best_model.pkl")

    return tuned


# ═════════════════════════════════════════════════════════════
#  run_tabular_auto — PyCaret -> LazyPredict -> sklearn fallback
# ═════════════════════════════════════════════════════════════

def _fallback_sklearn_classification(X_train, X_test, y_train, y_test):
    """Minimal sklearn baseline: LogisticRegression + RandomForest."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_te = le.transform(y_test)

    results = {}
    for name, clf in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        clf.fit(X_train, y_tr)
        preds = clf.predict(X_test)
        acc = float(accuracy_score(y_te, preds))
        mf1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
        wf1 = float(f1_score(y_te, preds, average="weighted", zero_division=0))
        results[name] = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1, "model": clf}

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best = results[best_name]
    logger.info("sklearn fallback best: %s (acc=%.4f)", best_name, best["accuracy"])
    return best_name, best["model"], results


def _fallback_sklearn_regression(X_train, X_test, y_train, y_test):
    """Minimal sklearn baseline: LinearRegression + RandomForest."""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    results = {}
    for name, reg in [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("GradientBoosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        results[name] = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds)),
            "model": reg,
        }

    best_name = max(results, key=lambda k: results[k]["r2"])
    best = results[best_name]
    logger.info("sklearn fallback best: %s (R2=%.4f)", best_name, best["r2"])
    return best_name, best["model"], results


# ─── Split-aware sklearn evaluation (train→val→test) ────────

def _split_sklearn_classification(X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    """Train sklearn classifiers on *train*, select best on *val*, report on *test*."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    # Fit encoder on ALL classes to avoid unseen-label errors on val/test
    le = LabelEncoder()
    all_y = np.concatenate([np.asarray(y_train), np.asarray(y_val), np.asarray(y_test)])
    le.fit(all_y)
    y_tr = le.transform(y_train)
    y_v = le.transform(y_val)
    y_te = le.transform(y_test)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    trained, val_scores = {}, {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_tr)
        trained[name] = clf
        val_scores[name] = float(accuracy_score(y_v, clf.predict(X_val)))
        logger.info("  %s  val_acc=%.4f", name, val_scores[name])

    best_name = max(val_scores, key=val_scores.get)
    best = trained[best_name]
    logger.info("Selected %s (val_acc=%.4f)", best_name, val_scores[best_name])

    tp = best.predict(X_test)
    acc = float(accuracy_score(y_te, tp))
    mf1 = float(f1_score(y_te, tp, average="macro", zero_division=0))
    wf1 = float(f1_score(y_te, tp, average="weighted", zero_division=0))
    try:
        proba = best.predict_proba(X_test) if hasattr(best, "predict_proba") else None
        if proba is not None:
            auc_val = float(
                roc_auc_score(y_te, proba[:, 1])
                if proba.shape[1] == 2
                else roc_auc_score(y_te, proba, multi_class="ovr", average="weighted")
            )
        else:
            auc_val = None
    except Exception:
        auc_val = None

    save_confusion_matrix(y_te, tp, output_dir, labels=[str(c) for c in le.classes_])
    pd.DataFrame([
        {"model": n, "val_accuracy": val_scores[n]} for n in candidates
    ]).to_csv(output_dir / "model_comparison.csv", index=False)

    try:
        import joblib
        joblib.dump(best, output_dir / "best_model.joblib")
        logger.info("Best model checkpoint saved → %s", output_dir / "best_model.joblib")
    except ImportError:
        pass

    return {
        "accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1, "auc": auc_val,
        "best_model": best_name, "n_test": len(y_te), "split": "test",
        "engine": "sklearn", "status": "ok",
    }


def _split_sklearn_regression(X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    """Train sklearn regressors on *train*, select best on *val*, report on *test*."""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    trained, val_scores = {}, {}
    for name, reg in candidates.items():
        reg.fit(X_train, y_train)
        trained[name] = reg
        val_scores[name] = float(r2_score(y_val, reg.predict(X_val)))
        logger.info("  %s  val_r2=%.4f", name, val_scores[name])

    best_name = max(val_scores, key=val_scores.get)
    best = trained[best_name]
    logger.info("Selected %s (val_r2=%.4f)", best_name, val_scores[best_name])

    tp = best.predict(X_test)
    met = {
        "mae": float(mean_absolute_error(y_test, tp)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, tp))),
        "r2": float(r2_score(y_test, tp)),
        "best_model": best_name, "n_test": len(y_test), "split": "test",
        "engine": "sklearn", "status": "ok",
    }

    pd.DataFrame([
        {"model": n, "val_r2": val_scores[n]} for n in candidates
    ]).to_csv(output_dir / "model_comparison.csv", index=False)

    try:
        import joblib
        joblib.dump(best, output_dir / "best_model.joblib")
    except ImportError:
        pass

    return met


def _run_tabular_with_splits(splits: dict, output_dir: Path, *, task: str = "classification") -> Dict[str, Any]:
    """Route to classification or regression with explicit train/val/test."""
    output_dir = ensure_dir(Path(output_dir))
    keys = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    Xtr, ytr, Xv, yv, Xte, yte = (splits[k] for k in keys)
    if task == "classification":
        return _split_sklearn_classification(Xtr, ytr, Xv, yv, Xte, yte, output_dir)
    return _split_sklearn_regression(Xtr, ytr, Xv, yv, Xte, yte, output_dir)


def run_tabular_auto(
    df: pd.DataFrame = None,
    target: str = None,
    output_dir: Path = None,
    *,
    task: str = "classification",
    session_id: int = 42,
    splits: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Auto-ML for tabular data with 3-tier fallback.

    Tries: PyCaret -> LazyPredict -> sklearn baseline.
    Always returns a standardised metrics dict and writes metrics.json.

    If *splits* is provided (dict with keys X_train, y_train, X_val, y_val,
    X_test, y_test), bypasses internal split and evaluates on the TEST
    portion only.  This is the recommended path for ``--mode full``.

    Parameters
    ----------
    task : 'classification' or 'regression'
    splits : optional pre-split data dict
    """
    output_dir = ensure_dir(output_dir) if output_dir else None

    # ── explicit splits → proper train / val / test evaluation ──
    if splits is not None:
        return _run_tabular_with_splits(splits, output_dir, task=task)

    # ── legacy path (backward-compat, used in smoke mode) ──────
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target])
    y = df[target]

    # Encode categoricals for sklearn / lazypredict
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=session_id
    )

    engine_used = "none"
    metrics: Dict[str, Any] = {}

    # --- Tier 1: PyCaret ---------------------------------------------------
    if pycaret_python_ok():
        try:
            if task == "classification":
                run_pycaret_classification(df, target, output_dir, session_id=session_id)
                # read back comparison csv for metrics
                comp = pd.read_csv(output_dir / "model_comparison.csv")
                metrics = {
                    "accuracy": float(comp["Accuracy"].iloc[0]) if "Accuracy" in comp.columns else None,
                    "macro_f1": float(comp["F1"].iloc[0]) if "F1" in comp.columns else None,
                    "weighted_f1": None,
                    "auc": float(comp["AUC"].iloc[0]) if "AUC" in comp.columns else None,
                }
            else:
                run_pycaret_regression(df, target, output_dir, session_id=session_id)
                comp = pd.read_csv(output_dir / "model_comparison.csv")
                metrics = {
                    "mae": float(comp["MAE"].iloc[0]) if "MAE" in comp.columns else None,
                    "rmse": float(comp["RMSE"].iloc[0]) if "RMSE" in comp.columns else None,
                    "r2": float(comp["R2"].iloc[0]) if "R2" in comp.columns else None,
                }
            engine_used = "pycaret"
            logger.info("PyCaret succeeded.")
        except Exception as exc:
            logger.warning("PyCaret failed: %s — trying LazyPredict.", exc)
    else:
        assert_supported_python(for_pycaret=True)  # prints warning
        logger.info("PyCaret not available for this Python version.")

    # --- Tier 2: LazyPredict -----------------------------------------------
    if engine_used == "none":
        try:
            if task == "classification":
                from lazypredict.Supervised import LazyClassifier
                clf = LazyClassifier(verbose=0, ignore_warnings=True)
                models_df, _ = clf.fit(X_train, X_test, y_train, y_test)
                models_df.to_csv(output_dir / "model_comparison.csv", index=True)
                logger.info("LazyPredict comparison:\n%s", models_df.head(5).to_string())
                best_row = models_df.iloc[0]
                metrics = {
                    "accuracy": float(best_row["Accuracy"]) if "Accuracy" in models_df.columns else None,
                    "macro_f1": float(best_row["F1 Score"]) if "F1 Score" in models_df.columns else None,
                    "weighted_f1": None,
                    "auc": float(best_row["ROC AUC"]) if "ROC AUC" in models_df.columns else None,
                }
            else:
                from lazypredict.Supervised import LazyRegressor
                reg = LazyRegressor(verbose=0, ignore_warnings=True)
                models_df, _ = reg.fit(X_train, X_test, y_train, y_test)
                models_df.to_csv(output_dir / "model_comparison.csv", index=True)
                logger.info("LazyPredict comparison:\n%s", models_df.head(5).to_string())
                best_row = models_df.iloc[0]
                metrics = {
                    "mae": float(best_row["MAE"]) if "MAE" in models_df.columns else None,
                    "rmse": float(best_row["RMSE"]) if "RMSE" in models_df.columns else None,
                    "r2": float(best_row["R-Squared"]) if "R-Squared" in models_df.columns else None,
                }
            engine_used = "lazypredict"
            logger.info("LazyPredict succeeded.")
        except Exception as exc:
            logger.warning("LazyPredict failed: %s — falling back to sklearn.", exc)

    # --- Tier 3: sklearn baseline ------------------------------------------
    if engine_used == "none":
        try:
            if task == "classification":
                best_name, model, all_results = _fallback_sklearn_classification(
                    X_train, X_test, y_train, y_test
                )
                # Build comparison csv
                rows = [{k: v for k, v in vals.items() if k != "model"}
                        for nm, vals in all_results.items()]
                comp_df = pd.DataFrame(rows, index=list(all_results.keys()))
                comp_df.to_csv(output_dir / "model_comparison.csv")

                best_res = all_results[best_name]
                preds = model.predict(X_test)
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_te = le.fit_transform(y_test)
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test)
                        if proba.shape[1] == 2:
                            auc_val = float(roc_auc_score(y_te, proba[:, 1]))
                        else:
                            auc_val = float(roc_auc_score(y_te, proba, multi_class="ovr", average="weighted"))
                    else:
                        auc_val = None
                except Exception:
                    auc_val = None
                metrics = {
                    "accuracy": best_res["accuracy"],
                    "macro_f1": best_res["macro_f1"],
                    "weighted_f1": best_res["weighted_f1"],
                    "auc": auc_val,
                }
            else:
                best_name, model, all_results = _fallback_sklearn_regression(
                    X_train, X_test, y_train, y_test
                )
                rows = [{k: v for k, v in vals.items() if k != "model"}
                        for nm, vals in all_results.items()]
                comp_df = pd.DataFrame(rows, index=list(all_results.keys()))
                comp_df.to_csv(output_dir / "model_comparison.csv")
                best_res = all_results[best_name]
                metrics = {
                    "mae": best_res["mae"],
                    "rmse": best_res["rmse"],
                    "r2": best_res["r2"],
                }
            engine_used = "sklearn"
            logger.info("sklearn baseline succeeded.")
        except Exception as exc:
            logger.error("All engines failed: %s", exc)
            metrics = {"status": "error", "error": str(exc)}
            engine_used = "failed"

    metrics["engine"] = engine_used
    if "status" not in metrics:
        metrics["status"] = "ok"
    return metrics


# ═════════════════════════════════════════════════════════════
#  Logging setup
# ═════════════════════════════════════════════════════════════

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ═════════════════════════════════════════════════════════════
#  Common CLI argument parser
# ═════════════════════════════════════════════════════════════

def common_argparser(description: str = "Run project") -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-loaded with the standard project flags.

    Standard flags:
        --smoke-test   Run 1 epoch on a tiny subset (CI / quick check).
        --download-only Download dataset and exit.
        --epochs N     Override the default number of training epochs.
        --batch-size N Override the default batch size.
        --device DEV   Force a specific torch device (cuda / cpu / mps).
        --no-amp       Disable automatic mixed precision.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 1 epoch on a tiny subset for CI / quick check")
    p.add_argument("--download-only", action="store_true",
                   help="Download the dataset and exit without training")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override default number of epochs")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override default batch size")
    p.add_argument("--device", type=str, default=None,
                   choices=["cuda", "cpu", "mps"],
                   help="Force a specific device")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable automatic mixed precision")
    return p


def resolve_device(args_device: Optional[str] = None) -> "torch.device":
    """Resolve device from CLI --device flag or auto-detect."""
    import torch
    if args_device:
        return torch.device(args_device)
    return get_device(verbose=True)


# ═════════════════════════════════════════════════════════════
#  Standardised metrics I/O
# ═════════════════════════════════════════════════════════════

def write_metrics(metrics: Dict[str, Any], output_dir: Path,
                  filename: str = "metrics.json") -> Path:
    """Write a metrics dictionary to JSON.  Always includes 'status' key."""
    output_dir = ensure_dir(output_dir)
    if "status" not in metrics:
        metrics["status"] = "ok"
    p = output_dir / filename
    p.write_text(json.dumps(metrics, indent=2, default=str))
    logger.info("Metrics written → %s", p)
    return p


def dataset_missing_metrics(
    output_dir: Path,
    dataset_name: str,
    official_links: List[str],
) -> None:
    """Write a metrics.json indicating the dataset is missing, then exit."""
    write_metrics(
        {
            "status": "dataset_missing",
            "dataset": dataset_name,
            "official_links": official_links,
            "message": "Dataset not found. See links above to download manually.",
        },
        output_dir,
    )
    logger.warning("Dataset missing — metrics.json written with links. Exiting.")
    sys.exit(0)


# ╔═══════════════════════════════════════════════════════════╗
# ║  PATH A — Quality / Correctness Infrastructure           ║
# ║  parse_common_args · run_metadata · GPU budget · splits  ║
# ║  split manifests · save_metrics · EarlyStopping · RecSys ║
# ╚═══════════════════════════════════════════════════════════╝

from dataclasses import dataclass


# ═════════════════════════════════════════════════════════════
#  A1: Standardised CLI
# ═════════════════════════════════════════════════════════════

def parse_common_args(project_name: Optional[str] = None) -> argparse.Namespace:
    """Return a *parsed* Namespace with every standard project flag.

    Flags
    -----
    --mode {smoke,full}   Run profile (default: smoke)
    --seed INT            Random seed (default: 42)
    --gpu-mem-gb FLOAT    GPU budget in GB (default: 4)
    --batch-size INT      Override batch size
    --grad-accum INT      Gradient-accumulation steps (default: auto)
    --img-size INT        Override image size (CV)
    --epochs INT          Override epoch count
    --patience INT        Early-stopping patience
    --no-amp              Disable AMP
    --device {auto,cpu,cuda,mps}
    --num-workers INT     DataLoader workers (default: 0)
    --download-only       Download dataset and exit
    --smoke-test          (deprecated) alias for --mode smoke
    """
    p = argparse.ArgumentParser(
        description=project_name or "Deep Learning Project",
    )
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                   help="Run profile: smoke (fast CI) or full (real evaluation)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--gpu-mem-gb", type=float, default=4.0, dest="gpu_mem_gb",
                   help="GPU memory budget in GB (default 4)")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                   help="Override batch size")
    p.add_argument("--grad-accum", type=int, default=None, dest="grad_accum",
                   help="Gradient accumulation steps (default: auto)")
    p.add_argument("--img-size", type=int, default=None, dest="img_size",
                   help="Override image size (CV projects)")
    p.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    p.add_argument("--patience", type=int, default=None,
                   help="Early stopping patience")
    p.add_argument("--no-amp", action="store_true", dest="no_amp",
                   help="Disable automatic mixed precision")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                   default="auto", help="Device selection")
    p.add_argument("--num-workers", type=int, default=0, dest="num_workers",
                   help="DataLoader num_workers")
    p.add_argument("--download-only", action="store_true", dest="download_only",
                   help="Download dataset and exit")
    # Backward compat
    p.add_argument("--smoke-test", action="store_true", dest="smoke_test",
                   help="(deprecated) same as --mode smoke")
    args = p.parse_args()

    # --smoke-test implies --mode smoke (unless user explicitly passed --mode full)
    if args.smoke_test and args.mode != "full":
        args.mode = "smoke"
    return args


def resolve_device_from_args(args) -> "torch.device":
    """Resolve torch device from *args.device* (``auto`` or explicit)."""
    import torch
    dev = getattr(args, "device", "auto")
    if dev == "auto":
        return get_device(verbose=True)
    return torch.device(dev)


def run_metadata(args) -> Dict[str, Any]:
    """Capture environment metadata dict for reproducibility tagging."""
    import platform
    meta: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mode": getattr(args, "mode", "unknown"),
        "seed": getattr(args, "seed", 42),
    }
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import subprocess as _sp
        r = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True, timeout=5, cwd=str(REPO_ROOT))
        if r.returncode == 0:
            meta["git_commit"] = r.stdout.strip()
    except Exception:
        pass
    return meta


# ═════════════════════════════════════════════════════════════
#  A2: GPU Budget Enforcement
# ═════════════════════════════════════════════════════════════

def configure_cuda_allocator() -> None:
    """Set safe CUDA memory-allocator env vars (idempotent)."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def get_gpu_mem_bytes(gb: float) -> int:
    """Convert GB → bytes."""
    return int(gb * 1024 ** 3)


def current_cuda_peak_bytes() -> int:
    """Return ``torch.cuda.max_memory_allocated()`` or 0."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    return 0


def enforce_gpu_budget_step(budget_bytes: int, hard_fail: bool = False) -> bool:
    """Return *True* if peak GPU usage is within budget, *False* otherwise.

    If *hard_fail* is True, raises ``RuntimeError`` on overshoot.
    """
    peak = current_cuda_peak_bytes()
    if peak > budget_bytes:
        msg = f"GPU peak {peak / 1e9:.2f} GB exceeds budget {budget_bytes / 1e9:.2f} GB"
        if hard_fail:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False
    return True


def auto_batch_and_accum(
    budget_gb: float,
    requested_batch: int,
    min_batch: int = 1,
    target_effective_batch: Optional[int] = None,
) -> tuple:
    """Return ``(batch_size, grad_accum_steps)`` respecting *budget_gb*.

    Heuristic: if budget ≤ 4 GB, halve the batch size and compensate
    with gradient accumulation to preserve the effective batch.
    """
    effective = target_effective_batch or requested_batch
    batch = requested_batch
    accum = 1
    if budget_gb <= 4.0 and batch > min_batch * 2:
        batch = max(min_batch, batch // 2)
        accum = max(1, effective // batch)
    return batch, accum


# ═════════════════════════════════════════════════════════════
#  A3: Dataset Fingerprint + Splits
# ═════════════════════════════════════════════════════════════

def dataset_fingerprint(path) -> Dict[str, Any]:
    """Stable-ish fingerprint of a dataset directory."""
    p = Path(path)
    if not p.exists():
        return {"file_count": 0, "total_bytes": 0, "exists": False}
    files = [f for f in p.rglob("*") if f.is_file()
             and "__pycache__" not in str(f) and ".git" not in f.parts]
    total = sum(f.stat().st_size for f in files)
    return {
        "file_count": len(files),
        "total_bytes": total,
        "exists": True,
        "top_level": sorted(e.name for e in p.iterdir())[:20],
    }


def make_tabular_splits(
    df: pd.DataFrame,
    target_col: str,
    task: str = "classification",
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    """Split a DataFrame into train / val / test.

    Returns ``(X_train, y_train, X_val, y_val, X_test, y_test)``.
    Classification uses stratified splits when class counts allow it.
    Categorical columns are ordinal-encoded for sklearn compatibility.
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ordinal-encode object / category columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes

    strat = y if (task == "classification" and y.value_counts().min() >= 2) else None

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat,
    )

    strat_tv = y_tv if (strat is not None and y_tv.value_counts().min() >= 2) else None
    val_of_tv = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_of_tv, random_state=seed, stratify=strat_tv,
    )

    logger.info("Tabular split: train=%d  val=%d  test=%d", len(y_train), len(y_val), len(y_test))
    return X_train, y_train, X_val, y_val, X_test, y_test


@dataclass
class SplitPaths:
    """File lists per split."""
    train: List[str]
    val: List[str]
    test: List[str]


def index_files_for_split(
    root_dir,
    exts: tuple = (".jpg", ".jpeg", ".png", ".wav", ".flac", ".mp3"),
    ignore_dirs: tuple = ("__pycache__", "__MACOSX", ".git"),
) -> List[str]:
    """Return sorted list of *relative* file paths suitable for splitting."""
    root = Path(root_dir)
    out: List[str] = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in exts:
            if not any(ig in f.parts for ig in ignore_dirs):
                out.append(str(f.relative_to(root)))
    return out


def make_file_splits(
    files: List[str],
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
    group_fn=None,
) -> SplitPaths:
    """Split a file list into train / val / test.

    If *group_fn* is provided (``file → group_key``), splits by group
    to prevent data leakage (e.g. same speaker in train+test).
    """
    from sklearn.model_selection import train_test_split

    if group_fn is not None:
        groups = [group_fn(f) for f in files]
        ugroups = sorted(set(groups))
        rng = np.random.RandomState(seed)
        rng.shuffle(ugroups)
        n_te = max(1, int(len(ugroups) * test_size))
        n_va = max(1, int(len(ugroups) * val_size))
        te_set = set(ugroups[:n_te])
        va_set = set(ugroups[n_te:n_te + n_va])
        train = [f for f, g in zip(files, groups) if g not in te_set and g not in va_set]
        val   = [f for f, g in zip(files, groups) if g in va_set]
        test  = [f for f, g in zip(files, groups) if g in te_set]
    else:
        tv, test = train_test_split(files, test_size=test_size, random_state=seed)
        train, val = train_test_split(tv, test_size=val_size / (1 - test_size),
                                      random_state=seed)
    return SplitPaths(train=train, val=val, test=test)


def write_split_manifest(
    output_dir,
    *,
    dataset_fp: Dict[str, Any],
    split_method: str,
    seed: int,
    counts: Dict[str, int],
    extras: Optional[Dict] = None,
) -> Path:
    """Write ``outputs/split_manifest.json``."""
    output_dir = ensure_dir(Path(output_dir))
    manifest: Dict[str, Any] = {
        "dataset_fingerprint": dataset_fp,
        "split_method": split_method,
        "seed": seed,
        "counts": counts,
    }
    if extras:
        manifest.update(extras)
    p = output_dir / "split_manifest.json"
    p.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Split manifest → %s", p)
    return p


# ═════════════════════════════════════════════════════════════
#  A4: Metrics Writers + Schema Validation
# ═════════════════════════════════════════════════════════════

METRICS_SCHEMAS: Dict[str, List[str]] = {
    "classification": ["accuracy", "macro_f1", "weighted_f1"],
    "regression":     ["mae", "rmse", "r2"],
    "rl":             ["avg_reward", "num_eval_episodes"],
    "recsys":         ["recall_at_k", "ndcg_at_k"],
    "gan":            ["epochs_ran"],
    "audio":          [],
    "association":    [],
}


def validate_metrics_schema(
    metrics: Dict[str, Any],
    task_type: str,
    mode: str = "smoke",
) -> List[str]:
    """Return list of *missing* required keys for *task_type*."""
    required = METRICS_SCHEMAS.get(task_type, [])
    missing = [k for k in required if k not in metrics]
    if missing:
        msg = "Missing required metrics for %s: %s"
        if mode == "full":
            logger.error(msg, task_type, missing)
        else:
            logger.warning(msg + " (smoke — OK)", task_type, missing)
    return missing


def _metrics_to_markdown(metrics: Dict[str, Any]) -> str:
    """Render a metrics dict as readable Markdown."""
    lines = ["# Metrics Report", ""]
    lines.append(f"**Status:** {metrics.get('status', 'unknown')}\n")

    _meta_keys = {
        "status", "engine", "mode", "seed", "python_version", "torch_version",
        "cuda_available", "cuda_device", "platform", "git_commit", "task",
        "split", "classification_report", "run_metadata",
    }
    numeric = {k: v for k, v in metrics.items()
               if k not in _meta_keys and not isinstance(v, (dict, list))}
    meta = {k: v for k, v in metrics.items() if k in _meta_keys and k != "status"}

    if numeric:
        lines += ["## Results", "", "| Metric | Value |", "|--------|-------|"]
        for k, v in numeric.items():
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        lines.append("")

    if meta:
        lines += ["## Metadata", ""]
        for k, v in meta.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    return "\n".join(lines)


def save_metrics(
    output_dir,
    metrics: Dict[str, Any],
    task_type: str = "",
    mode: str = "smoke",
) -> Path:
    """Write ``metrics.json`` **and** ``metrics.md``.

    Validates schema when *task_type* is given.
    """
    output_dir = ensure_dir(Path(output_dir))
    if "status" not in metrics:
        metrics["status"] = "ok"

    if task_type:
        validate_metrics_schema(metrics, task_type, mode=mode)

    json_path = output_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2, default=str))
    md_path = output_dir / "metrics.md"
    md_path.write_text(_metrics_to_markdown(metrics))

    logger.info("Metrics written → %s", json_path)
    return json_path


# ═════════════════════════════════════════════════════════════
#  Early Stopping
# ═════════════════════════════════════════════════════════════

class EarlyStopping:
    """Call each epoch with the monitored metric.

    Returns *True* once *patience* epochs elapse without improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.best_score is None:
            self.best_score = metric
            return False
        improved = (
            metric < self.best_score - self.min_delta
            if self.mode == "min"
            else metric > self.best_score + self.min_delta
        )
        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.should_stop = False


# ═════════════════════════════════════════════════════════════
#  RecSys Ranking Metrics
# ═════════════════════════════════════════════════════════════

def compute_recsys_metrics(
    y_true_per_user: Dict,
    y_scores_per_user: Dict,
    k: int = 10,
) -> Dict[str, Any]:
    """Compute Recall@K, NDCG@K, MAP@K for recommendation evaluation.

    Parameters
    ----------
    y_true_per_user : dict[user → list of relevant item ids]
    y_scores_per_user : dict[user → list of (item_id, score) sorted desc]
    """
    recalls, ndcgs, aps = [], [], []
    for user, relevant in y_true_per_user.items():
        if user not in y_scores_per_user:
            continue
        recommended = [item for item, _ in y_scores_per_user[user][:k]]
        rel_set = set(relevant)

        hits = len(set(recommended) & rel_set)
        recalls.append(hits / min(len(rel_set), k) if rel_set else 0.0)

        dcg = sum(1 / np.log2(i + 2) for i, it in enumerate(recommended) if it in rel_set)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        n_hits, psum = 0, 0.0
        for i, it in enumerate(recommended):
            if it in rel_set:
                n_hits += 1
                psum += n_hits / (i + 1)
        aps.append(psum / min(len(rel_set), k) if rel_set else 0.0)

    return {
        "k": k,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k":   float(np.mean(ndcgs))   if ndcgs   else 0.0,
        "map_at_k":    float(np.mean(aps))      if aps     else 0.0,
        "n_users_test": len(recalls),
    }
