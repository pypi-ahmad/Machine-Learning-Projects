"""Text classification trainer (single-label & multi-label).

Uses HuggingFace Trainer with DeBERTa-v3-base by default.
Includes K-Fold cross-validation (LogReg on TF-IDF) for variance estimates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.logger import get_logger
from utils.training_common import (
    SEED,
    build_training_args,
    cleanup_gpu,
    compute_class_weights_list,
    ensure_output_dirs,
    get_device,
    is_imbalanced,
    plot_confusion_matrix,
    save_json,
    seed_everything,
)

logger = get_logger(__name__)


# ======================================================================
# Custom Trainer with class weights
# ======================================================================

class _WeightedTrainer:
    """Mixin-free approach: we build a custom Trainer class at call time
    so imports are deferred."""
    pass


def _make_weighted_trainer(class_weights_tensor, focal_gamma: float = 0.0, label_smoothing: float = 0.0):
    """Build a Trainer subclass with optional class weights + focal loss.

    Parameters
    ----------
    class_weights_tensor : Tensor | None
        Per-class weights for cross-entropy (already clamped & normed).
    focal_gamma : float
        If > 0, use focal loss modulation ``(1-p_t)^gamma``.
    label_smoothing : float
        Label smoothing factor (applied before focal modulation).
    """
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            wt = class_weights_tensor.to(logits.device) if class_weights_tensor is not None else None

            # Standard cross-entropy (with optional weight + label smoothing)
            ce = torch.nn.functional.cross_entropy(
                logits, labels,
                weight=wt,
                label_smoothing=label_smoothing,
                reduction="none",
            )

            if focal_gamma > 0:
                # Focal loss modulation: multiply by (1 - p_t)^gamma
                probs = torch.softmax(logits.detach(), dim=-1)
                p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                focal_weight = (1.0 - p_t).clamp(min=0.0, max=1.0) ** focal_gamma
                ce = ce * focal_weight

            loss = ce.mean()
            # Guard against NaN/Inf to prevent CUDA kernel crashes
            if not torch.isfinite(loss):
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


# ======================================================================
# HF Dataset builder
# ======================================================================

def _make_dataset(texts, labels, tokenizer, max_length: int):
    from datasets import Dataset as HFDataset

    ds = HFDataset.from_dict({"text": texts, "label": labels})

    def tok(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=max_length,
        )

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


def _make_multilabel_dataset(texts, label_matrix, tokenizer, max_length: int):
    from datasets import Dataset as HFDataset

    ds = HFDataset.from_dict({
        "text": texts,
        "labels": [row.tolist() for row in label_matrix],
    })

    def tok(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=max_length,
        )

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds = ds.with_format("torch")
    return ds


# ======================================================================
# Metrics helpers
# ======================================================================

def _compute_metrics_single(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def _compute_metrics_multilabel(eval_pred):
    from sklearn.metrics import f1_score
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)).numpy() > 0.5).astype(int)
    return {
        "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(labels, preds, average="samples", zero_division=0)),
    }


def _full_test_metrics(y_true, y_pred, y_proba, labels_list):
    from sklearn.metrics import (
        accuracy_score, classification_report, f1_score,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    # ROC-AUC
    try:
        from sklearn.metrics import roc_auc_score
        if y_proba is not None:
            if len(labels_list) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
    except Exception:
        pass
    return metrics


# ======================================================================
# K-Fold (LogReg on TF-IDF) for variance estimate
# ======================================================================

def _run_kfold_logreg(texts, labels, n_splits: int = 5, seed: int = SEED) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    vec = TfidfVectorizer(max_features=10_000, sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(texts)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_f1s: list[float] = []

    for train_idx, val_idx in skf.split(X, labels):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = np.array(labels)[train_idx], np.array(labels)[val_idx]
        lr = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
        lr.fit(X_tr, y_tr)
        preds = lr.predict(X_val)
        fold_f1s.append(float(f1_score(y_val, preds, average="weighted", zero_division=0)))

    return {
        "n_splits": n_splits,
        "fold_f1s": fold_f1s,
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
    }


# ======================================================================
# Threshold tuning for binary classification
# ======================================================================

def _find_optimal_threshold(y_true, y_proba_pos):
    from sklearn.metrics import f1_score
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.2, 0.8, 0.02):
        preds = (y_proba_pos >= t).astype(int)
        f = f1_score(y_true, preds, average="binary", zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return best_t, best_f1


# ======================================================================
# MAIN: Single-label text classification
# ======================================================================

def train_text_classifier(
    slug: str,
    df: pd.DataFrame,
    text_col: str,
    target_col: str,
    *,
    model_name: str = "microsoft/deberta-v3-base",
    max_length: int = 256,
    batch_size: int = 8,
    grad_accum: int = 2,
    epochs: int = 5,
    lr: float = 2e-5,
    patience: int = 2,
    seed: int = SEED,
    max_samples: int = 50_000,
    run_kfold: bool = True,
    kfold_n: int = 5,
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.05,
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = "cosine",
    force: bool = False,
) -> dict:
    """Train a transformer classifier and return metrics dict."""
    seed_everything(seed)
    dirs = ensure_output_dirs(slug)
    metrics_file = dirs["metrics"] / "phase2_metrics.json"
    if metrics_file.exists() and not force:
        logger.info("Phase 2 already done for %s -- loading cached results", slug)
        return json.loads(metrics_file.read_text())

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer

    # ---- Data prep --------------------------------------------------------
    df = df[[text_col, target_col]].dropna().reset_index(drop=True)
    df[text_col] = df[text_col].astype(str)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    le = LabelEncoder()
    df["_label"] = le.fit_transform(df[target_col].astype(str))
    num_labels = len(le.classes_)
    label_names = le.classes_.tolist()

    texts_all = df[text_col].tolist()
    labels_all = df["_label"].tolist()

    train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["_label"], random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df["_label"], random_state=seed)
    logger.info("[%s] train=%d  val=%d  test=%d  classes=%d", slug, len(train_df), len(val_df), len(test_df), num_labels)

    result: dict = {
        "slug": slug,
        "task": "classification",
        "model_name": model_name,
        "num_labels": num_labels,
        "label_names": label_names,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "max_length": max_length,
        "status": "running",
    }

    # ---- K-Fold (fast, LogReg) -------------------------------------------
    if run_kfold and len(df) >= 2000:
        logger.info("[%s] Running %d-fold CV (LogReg on TF-IDF)...", slug, kfold_n)
        kfold_summary = _run_kfold_logreg(texts_all, labels_all, n_splits=kfold_n, seed=seed)
        result["kfold_summary"] = kfold_summary
        logger.info("[%s] K-Fold F1: %.4f +/- %.4f", slug, kfold_summary["mean_f1"], kfold_summary["std_f1"])

    # ---- Tokenize ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = _make_dataset(train_df[text_col].tolist(), train_df["_label"].tolist(), tokenizer, max_length)
    val_ds = _make_dataset(val_df[text_col].tolist(), val_df["_label"].tolist(), tokenizer, max_length)
    test_ds = _make_dataset(test_df[text_col].tolist(), test_df["_label"].tolist(), tokenizer, max_length)

    # ---- Model ------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )

    # ---- Class weights? ---------------------------------------------------
    imbalanced = is_imbalanced(labels_all)
    result["class_imbalanced"] = imbalanced

    cw_tensor = None
    TrainerCls = Trainer
    if imbalanced:
        logger.info("[%s] Imbalanced classes detected -- using weighted loss (clamped)", slug)
        cw = compute_class_weights_list(labels_all, num_labels)
        cw_tensor = torch.tensor(cw, dtype=torch.float32)

    # Always use custom trainer: supports focal loss + label smoothing
    use_focal = focal_gamma > 0 or num_labels >= 20
    actual_gamma = focal_gamma if focal_gamma > 0 else (1.5 if num_labels >= 20 else 0.0)
    if imbalanced or use_focal or label_smoothing > 0:
        TrainerCls = _make_weighted_trainer(
            cw_tensor,
            focal_gamma=actual_gamma,
            label_smoothing=label_smoothing,
        )
        result["focal_gamma"] = actual_gamma
        result["label_smoothing"] = label_smoothing

    # ---- Training args ----------------------------------------------------
    args = build_training_args(
        dirs["checkpoints"],
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        patience=patience,
        seed=seed,
        max_grad_norm=max_grad_norm,
        label_smoothing_factor=0.0,  # handled in custom trainer
        lr_scheduler_type=lr_scheduler_type,
    )

    trainer = TrainerCls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics_single,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    # ---- Train ------------------------------------------------------------
    logger.info("[%s] Training %s  (bs=%d, accum=%d, epochs=%d) ...", slug, model_name, batch_size, grad_accum, epochs)
    train_out = trainer.train()
    result["train_runtime_sec"] = train_out.metrics.get("train_runtime", 0)

    # ---- Validate ---------------------------------------------------------
    val_metrics = trainer.evaluate(val_ds)
    result["val_metrics"] = {k.replace("eval_", ""): v for k, v in val_metrics.items() if isinstance(v, (int, float))}

    # ---- Test -------------------------------------------------------------
    test_out = trainer.predict(test_ds)
    y_true = test_df["_label"].values
    y_pred = np.argmax(test_out.predictions, axis=-1)
    y_proba = None
    try:
        from scipy.special import softmax as sp_softmax
        y_proba = sp_softmax(test_out.predictions, axis=-1)
    except Exception:
        pass

    test_met = _full_test_metrics(y_true, y_pred, y_proba, label_names)
    result["test_metrics"] = test_met
    logger.info("[%s] Test: acc=%.4f  F1w=%.4f  F1m=%.4f", slug, test_met["accuracy"], test_met["f1_weighted"], test_met["f1_macro"])

    # ---- Threshold tuning (binary) ----------------------------------------
    if num_labels == 2 and y_proba is not None:
        t, f1_at_t = _find_optimal_threshold(y_true, y_proba[:, 1])
        result["optimal_threshold"] = t
        result["f1_at_threshold"] = f1_at_t

    # ---- Confusion matrix -------------------------------------------------
    try:
        plot_confusion_matrix(y_true, y_pred, list(range(num_labels)), dirs["figures"] / "confusion_matrix.png")
    except Exception as exc:
        logger.warning("Could not plot confusion matrix: %s", exc)

    # ---- Save model + tokenizer -------------------------------------------
    save_dir = dirs["artifacts"] / "model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    result["model_path"] = str(save_dir)

    # ---- Save metrics -----------------------------------------------------
    result["status"] = "OK"
    save_json(result, metrics_file)
    save_json(result, dirs["metrics"] / "phase2_full_results.json")

    # ---- Cleanup ----------------------------------------------------------
    del model, trainer
    cleanup_gpu()

    return result


# ======================================================================
# MAIN: Multi-label text classification
# ======================================================================

def train_multilabel_classifier(
    slug: str,
    df: pd.DataFrame,
    text_col: str,
    target_cols: list[str],
    *,
    model_name: str = "microsoft/deberta-v3-base",
    max_length: int = 256,
    batch_size: int = 8,
    grad_accum: int = 2,
    epochs: int = 5,
    lr: float = 2e-5,
    patience: int = 2,
    seed: int = SEED,
    max_samples: int = 50_000,
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = "cosine",
    force: bool = False,
) -> dict:
    """Train a multi-label transformer classifier with pos_weight + threshold tuning."""
    seed_everything(seed)
    dirs = ensure_output_dirs(slug)
    metrics_file = dirs["metrics"] / "phase2_metrics.json"
    if metrics_file.exists() and not force:
        logger.info("Phase 2 already done for %s", slug)
        return json.loads(metrics_file.read_text())

    from sklearn.model_selection import train_test_split
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer

    # ---- Data prep --------------------------------------------------------
    cols = [text_col] + target_cols
    df = df[cols].dropna(subset=[text_col]).fillna(0).reset_index(drop=True)
    df[text_col] = df[text_col].astype(str)
    for c in target_cols:
        df[c] = df[c].astype(int)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    label_matrix = df[target_cols].values.astype(np.float32)

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=seed)
    logger.info("[%s] Multilabel: train=%d val=%d test=%d labels=%d", slug, len(train_df), len(val_df), len(test_df), len(target_cols))

    result: dict = {
        "slug": slug,
        "task": "multilabel_classification",
        "model_name": model_name,
        "label_names": target_cols,
        "num_labels": len(target_cols),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "status": "running",
    }

    # ---- Compute pos_weight (clamped) for BCEWithLogitsLoss ---------------
    train_labels = train_df[target_cols].values.astype(np.float32)
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    # pos_weight = neg / pos, clamped to [1.0, 20.0]
    raw_pw = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
    pos_weight = np.clip(raw_pw, 1.0, 20.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32)
    logger.info("[%s] pos_weight (clamped): %s", slug, [round(float(v), 2) for v in pos_weight])
    result["pos_weight"] = [round(float(v), 2) for v in pos_weight]

    # ---- Tokenize ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = _make_multilabel_dataset(train_df[text_col].tolist(), train_df[target_cols].values.astype(np.float32), tokenizer, max_length)
    val_ds = _make_multilabel_dataset(val_df[text_col].tolist(), val_df[target_cols].values.astype(np.float32), tokenizer, max_length)
    test_ds = _make_multilabel_dataset(test_df[text_col].tolist(), test_df[target_cols].values.astype(np.float32), tokenizer, max_length)

    # ---- Model with custom BCEWithLogitsLoss trainer ----------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(target_cols),
        problem_type="multi_label_classification",
    )

    # Custom trainer with pos_weight
    class MultilabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            pw = pos_weight_tensor.to(logits.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=pw,
            )
            return (loss, outputs) if return_outputs else loss

    args = build_training_args(
        dirs["checkpoints"],
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        patience=patience,
        metric_for_best="f1_micro",
        seed=seed,
        max_grad_norm=max_grad_norm,
        label_smoothing_factor=0.0,  # not used for multilabel
        lr_scheduler_type=lr_scheduler_type,
    )

    trainer = MultilabelTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics_multilabel,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    logger.info("[%s] Training multi-label %s ...", slug, model_name)
    train_out = trainer.train()
    result["train_runtime_sec"] = train_out.metrics.get("train_runtime", 0)

    # ---- Val + Test -------------------------------------------------------
    val_m = trainer.evaluate(val_ds)
    result["val_metrics"] = {k.replace("eval_", ""): v for k, v in val_m.items() if isinstance(v, (int, float))}

    # ---- Per-label threshold tuning on validation -------------------------
    val_out = trainer.predict(val_ds)
    val_logits = torch.tensor(val_out.predictions)
    val_probs = torch.sigmoid(val_logits).numpy()
    val_true = val_df[target_cols].values.astype(int)

    from sklearn.metrics import f1_score as _f1

    optimal_thresholds = []
    for i, col in enumerate(target_cols):
        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.02):
            preds_t = (val_probs[:, i] >= t).astype(int)
            f = _f1(val_true[:, i], preds_t, average="binary", zero_division=0)
            if f > best_f:
                best_f = f
                best_t = float(t)
        optimal_thresholds.append(best_t)
        logger.info("[%s]   label=%s  best_threshold=%.2f  val_f1=%.4f", slug, col, best_t, best_f)

    result["optimal_thresholds"] = dict(zip(target_cols, optimal_thresholds))

    # ---- Test with tuned thresholds ---------------------------------------
    test_out = trainer.predict(test_ds)
    test_probs = torch.sigmoid(torch.tensor(test_out.predictions)).numpy()
    y_true = test_df[target_cols].values.astype(int)

    # Apply per-label thresholds
    preds = np.zeros_like(y_true, dtype=int)
    for i in range(len(target_cols)):
        preds[:, i] = (test_probs[:, i] >= optimal_thresholds[i]).astype(int)

    from sklearn.metrics import classification_report, f1_score
    test_met = {
        "f1_micro": float(f1_score(y_true, preds, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "per_label": {},
    }
    for i, col in enumerate(target_cols):
        f1 = f1_score(y_true[:, i], preds[:, i], average="binary", zero_division=0)
        test_met["per_label"][col] = float(f1)
    result["test_metrics"] = test_met
    logger.info("[%s] Test F1(micro)=%.4f F1(macro)=%.4f", slug, test_met["f1_micro"], test_met["f1_macro"])

    # ---- Save per-label CSV -----------------------------------------------
    per_label_rows = []
    for i, col in enumerate(target_cols):
        per_label_rows.append({
            "label": col,
            "threshold": optimal_thresholds[i],
            "f1": test_met["per_label"][col],
            "pos_weight": float(pos_weight[i]),
            "train_pos_count": int(pos_counts[i]),
            "train_neg_count": int(neg_counts[i]),
        })
    per_label_df = pd.DataFrame(per_label_rows)
    per_label_df.to_csv(dirs["metrics"] / "per_label.csv", index=False)

    # ---- Save -------------------------------------------------------------
    save_dir = dirs["artifacts"] / "model"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    result["model_path"] = str(save_dir)
    result["status"] = "OK"
    save_json(result, metrics_file)

    del model, trainer
    cleanup_gpu()
    return result


# ======================================================================
# Orchestrator entry point
# ======================================================================

def run_project(project_slug, project_dir, raw_paths, processed_dir, outputs_dir, config, force=False):
    """Unified entry point called by the Phase 2.1 orchestrator."""
    df = config["df"]
    text_col = config.get("text_col", "text")
    _KEYS = {"model_name","max_length","batch_size","grad_accum","epochs","patience","lr",
             "max_samples","run_kfold","kfold_n","seed",
             "focal_gamma","label_smoothing","max_grad_norm","lr_scheduler_type"}
    kw = {k: v for k, v in config.items() if k in _KEYS}

    if config.get("multilabel"):
        target_cols = config["target_cols"]
        _ML = {"model_name","max_length","batch_size","grad_accum","epochs","patience","lr",
               "max_samples","seed","max_grad_norm","lr_scheduler_type"}
        kw2 = {k: v for k, v in config.items() if k in _ML}
        r = train_multilabel_classifier(project_slug, df, text_col, target_cols, force=force, **kw2)
    else:
        target_col = config.get("target_col", "label")
        r = train_text_classifier(project_slug, df, text_col, target_col, force=force, **kw)

    n = r.get("n_train", 0) + r.get("n_val", 0) + r.get("n_test", 0)
    return {
        "status": r.get("status", "UNKNOWN"),
        "model_name": r.get("model_name", config.get("model_name", "")),
        "dataset_size": n,
        "main_metrics": r.get("test_metrics", {}),
        "val_metrics": r.get("val_metrics", {}),
        "training_mode": "full",
        "train_runtime_sec": r.get("train_runtime_sec", 0),
        "notes": "",
        "full_result": r,
    }
