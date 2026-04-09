#!/usr/bin/env python3
"""
News Chatbot — News Category Classification
==============================================
Fine-tunes DistilBERT on a news-category dataset to classify articles into
topic categories based on their headline and short description.

Dataset: https://www.kaggle.com/datasets/rmisra/news-category-dataset
Run:     python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from shared.utils import (
    download_kaggle_dataset,
    set_seed,
    setup_logging,
    project_paths,
    save_classification_report,
    parse_common_args,
    save_metrics,
    dataset_missing_metrics,
    resolve_device_from_args,
    run_metadata,
    dataset_fingerprint,
    write_split_manifest,
    EarlyStopping,
    auto_batch_and_accum,
    get_gpu_mem_bytes,
    enforce_gpu_budget_step,
    configure_cuda_allocator,
)

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "rmisra/news-category-dataset"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
SEED = 42
MAX_SAMPLES = 30_000
MIN_CLASS_COUNT = 2  # minimum viable for stratified split

TEXT_CANDIDATES = [
    "headline", "short_description", "title", "text",
    "description", "content", "message", "body",
]
LABEL_CANDIDATES = [
    "category", "label", "class", "topic", "section",
    "tag", "type", "subject",
]


# ═════════════════════════════════════════════════════════════
#  Dataset wrapper
# ═════════════════════════════════════════════════════════════
class TextDataset(Dataset):
    """PyTorch Dataset wrapping pre-tokenised text."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ═════════════════════════════════════════════════════════════
#  Column detection
# ═════════════════════════════════════════════════════════════
def _find_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if kind == "text" and obj_cols:
        avg_lens = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
        return max(avg_lens, key=avg_lens.get)
    if kind == "label":
        cat_cols = {c: df[c].nunique() for c in obj_cols}
        if cat_cols:
            return min(cat_cols, key=cat_cols.get)
    raise ValueError(f"Cannot auto-detect {kind} column among {list(df.columns)}")


# ═════════════════════════════════════════════════════════════
#  Data loading
# ═════════════════════════════════════════════════════════════
def _load_json_lines(path: Path) -> pd.DataFrame:
    """Load a JSON-lines file (one JSON object per line)."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def get_data(data_dir: Path) -> pd.DataFrame:
    ds_path = download_kaggle_dataset(
        KAGGLE_SLUG, data_dir,
        dataset_name="News Category Dataset",
    )

    # Try JSON files first (this dataset is typically JSON-lines)
    jsons = sorted(ds_path.glob("*.json")) + sorted(data_dir.glob("**/*.json"))
    csvs = sorted(ds_path.glob("*.csv")) + sorted(data_dir.glob("**/*.csv"))

    df = None
    if jsons:
        chosen = max(jsons, key=lambda f: f.stat().st_size)
        try:
            df = pd.read_json(chosen)
        except ValueError:
            # JSON-lines format
            df = _load_json_lines(chosen)
        logger.info("Loaded %d rows from %s", len(df), chosen.name)
    elif csvs:
        chosen = max(csvs, key=lambda f: f.stat().st_size)
        df = pd.read_csv(chosen)
        logger.info("Loaded %d rows from %s", len(df), chosen.name)
    else:
        raise FileNotFoundError(f"No JSON/CSV files found in {ds_path}")

    return df


def preprocess(df: pd.DataFrame):
    """Build text (headline + description) and label arrays."""
    # Detect label column
    label_col = _find_column(df, LABEL_CANDIDATES, "label")
    logger.info("Label column: '%s'", label_col)

    # Build combined text
    cols_lower = {c.lower().strip(): c for c in df.columns}
    headline_col = cols_lower.get("headline")
    desc_col = cols_lower.get("short_description") or cols_lower.get("description")

    if headline_col and desc_col:
        df["_text"] = df[headline_col].astype(str) + " " + df[desc_col].astype(str)
        logger.info("Combined '%s' + '%s' into text", headline_col, desc_col)
    elif headline_col:
        df["_text"] = df[headline_col].astype(str)
    else:
        text_col = _find_column(df, TEXT_CANDIDATES, "text")
        df["_text"] = df[text_col].astype(str)

    df = df.dropna(subset=["_text", label_col]).reset_index(drop=True)

    # Drop rare classes
    counts = df[label_col].value_counts()
    valid = counts[counts >= MIN_CLASS_COUNT].index
    df = df[df[label_col].isin(valid)].reset_index(drop=True)
    logger.info("Kept %d rows after dropping classes with < %d samples", len(df), MIN_CLASS_COUNT)

    # Limit samples
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=SEED).reset_index(drop=True)
        logger.info("Sampled down to %d rows", MAX_SAMPLES)

    texts = df["_text"].tolist()
    le = LabelEncoder()
    labels = le.fit_transform(df[label_col].astype(str))
    logger.info("Classes (%d): %s", len(le.classes_), le.classes_[:20])
    return texts, labels, le


# ═════════════════════════════════════════════════════════════
#  Model
# ═════════════════════════════════════════════════════════════
def build_model(num_labels: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels,
    )
    model.to(device)
    return tokenizer, model


def tokenize(texts, tokenizer):
    return tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt",
    )


# ═════════════════════════════════════════════════════════════
#  Training
# ═════════════════════════════════════════════════════════════
@torch.no_grad()
def _val_loss(model, loader, device, use_amp=True):
    """Compute average validation loss."""
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**batch)
        total += outputs.loss.item() * batch["labels"].size(0)
        count += batch["labels"].size(0)
    return total / count if count else float("inf")


def train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR,
                use_amp=True, grad_accum=1, patience=3, output_dir=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stop = EarlyStopping(patience=patience, mode="min")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * grad_accum * batch["labels"].size(0)
            total += batch["labels"].size(0)

        # Validation
        val_loss = _val_loss(model, val_loader, device, use_amp)
        logger.info("Epoch %d/%d — train_loss=%.4f  val_loss=%.4f",
                     epoch, epochs, total_loss / total, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if output_dir:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        if early_stop(val_loss):
            logger.info("Early stopping at epoch %d", epoch)
            break


# ═════════════════════════════════════════════════════════════
#  Evaluation
# ═════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════
def main():
    args = parse_common_args("News Chatbot — News Category Classification")
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    paths = project_paths(__file__)
    device = resolve_device_from_args(args)

    # -- download-only gate
    if args.download_only:
        try:
            get_data(paths["data"])
            logger.info("Download complete.")
        except Exception as e:
            logger.error("Download failed: %s", e)
        sys.exit(0)

    # 1. Data
    try:
        df = get_data(paths["data"])
    except (FileNotFoundError, Exception) as exc:
        logger.error("Dataset error: %s", exc)
        dataset_missing_metrics(
            paths["outputs"],
            "News Category Dataset",
            ["https://www.kaggle.com/datasets/rmisra/news-category-dataset"],
        )
        return

    epochs = args.epochs or EPOCHS
    use_amp = not args.no_amp and device.type == "cuda"

    if args.mode == "smoke":
        df = df.sample(n=min(200, len(df)), random_state=args.seed)
        epochs = 1
        logger.info("SMOKE TEST: 1 epoch, %d rows", len(df))

    texts, labels, le = preprocess(df)

    # 2. Split (70/15/15)
    try:
        train_idx, temp_idx = train_test_split(
            range(len(texts)), test_size=0.3, random_state=args.seed, stratify=labels,
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=args.seed,
            stratify=[labels[i] for i in temp_idx],
        )
    except ValueError:
        logger.warning("Stratified split failed (too few samples per class) — falling back to random split")
        train_idx, temp_idx = train_test_split(
            range(len(texts)), test_size=0.3, random_state=args.seed,
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=args.seed,
        )

    write_split_manifest(
        paths["outputs"],
        dataset_fp=dataset_fingerprint(paths["data"]),
        split_method="stratified_random",
        seed=args.seed,
        counts={"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    )

    X_train = [texts[i] for i in train_idx]
    y_train = labels[train_idx]
    X_val = [texts[i] for i in val_idx]
    y_val = labels[val_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = labels[test_idx]

    # 3. GPU budget
    batch_size, grad_accum = auto_batch_and_accum(
        args.gpu_mem_gb, args.batch_size or BATCH_SIZE, min_batch=2,
    )
    budget_bytes = get_gpu_mem_bytes(args.gpu_mem_gb)
    patience = args.patience or 3

    # 4. Tokenize + dataloaders
    tokenizer, model = build_model(num_labels=len(le.classes_), device=device)
    train_enc = tokenize(X_train, tokenizer)
    val_enc = tokenize(X_val, tokenizer)
    test_enc = tokenize(X_test, tokenizer)

    train_ds = TextDataset(train_enc, y_train)
    val_ds = TextDataset(val_enc, y_val)
    test_ds = TextDataset(test_enc, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # 5. Train (with AMP fallback)
    try:
        train_model(model, train_loader, val_loader, device, epochs=epochs,
                     use_amp=use_amp, grad_accum=grad_accum, patience=patience,
                     output_dir=paths["outputs"])
    except RuntimeError as exc:
        if use_amp:
            logger.warning("AMP training failed (%s), retrying without AMP ...", exc)
            use_amp = False
            tokenizer, model = build_model(num_labels=len(le.classes_), device=device)
            try:
                train_model(model, train_loader, val_loader, device, epochs=epochs,
                             use_amp=False, grad_accum=grad_accum, patience=patience,
                             output_dir=paths["outputs"])
            except RuntimeError as exc2:
                logger.error("Training failed: %s", exc2)
                save_metrics(paths["outputs"], {"status": "error", "error": str(exc2)[:300]},
                             task_type="classification", mode=args.mode)
                return
        else:
            logger.error("Training failed: %s", exc)
            save_metrics(paths["outputs"], {"status": "error", "error": str(exc)[:300]},
                         task_type="classification", mode=args.mode)
            return

    # 6. Evaluate on test set (load best checkpoint)
    best_ckpt = paths["outputs"] / "best_model.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    preds, actuals = evaluate(model, test_loader, device)
    label_names = [str(c) for c in le.classes_]
    metrics = save_classification_report(
        actuals, preds, paths["outputs"],
        labels=label_names, prefix="news_category",
    )

    # Save model weights
    model.save_pretrained(paths["outputs"] / "model")
    tokenizer.save_pretrained(paths["outputs"] / "model")
    logger.info("Model saved to %s", paths["outputs"] / "model")

    metrics["run_metadata"] = run_metadata(args)
    metrics["n_test"] = len(test_idx)
    metrics["split"] = "test"
    save_metrics(paths["outputs"], metrics, task_type="classification", mode=args.mode)
    logger.info("Done -- accuracy %.4f  |  F1-macro %.4f",
                metrics["accuracy"], metrics["macro_f1"])


if __name__ == "__main__":
    main()
