"""
Shared NLP utilities — HuggingFace Transformers (PyTorch backend).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              f1_score)

from .utils import ensure_dir, save_metrics, plot_confusion_matrix


# ═══════════════════════════════════════════════════════════════════════════════
# Model builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_hf_classifier(model_name: str = "distilbert-base-uncased",
                         num_labels: int = 2):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )
    print(f"  HF model : {model_name}  |  labels = {num_labels}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenization -> torch Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def tokenize_texts(texts: list[str], labels: list[int], tokenizer,
                   max_length: int = 256):
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt")
    return _TextDataset(enc, labels)


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace Trainer wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def train_hf_classifier(model, train_ds, val_ds, output_dir: str | Path, *,
                         epochs: int = 3, batch_size: int = 16, lr: float = 2e-5,
                         use_amp: bool = True, max_steps: int = -1,
                         gradient_checkpointing: bool = False,
                         early_stopping_patience: int | None = None,
                         grad_accum_steps: int = 1):
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

    output_dir = ensure_dir(output_dir)

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("  [NLP] Gradient checkpointing enabled")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        }

    load_best = (max_steps <= 0) and (early_stopping_patience is not None
                                       or epochs > 1)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=epochs,
        max_steps=max_steps,
        weight_decay=0.01,
        load_best_model_at_end=load_best,
        metric_for_best_model="f1",
        fp16=use_amp and torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
    )

    callbacks = []
    if early_stopping_patience is not None and max_steps <= 0:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience))
        print(f"  [NLP] Early stopping enabled (patience={early_stopping_patience})")

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    return trainer


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_hf_classifier(trainer, test_ds, class_names: list[str],
                            output_dir: str | Path) -> dict:
    output_dir = ensure_dir(output_dir)

    out   = trainer.predict(test_ds)
    preds  = np.argmax(out.predictions, axis=-1)
    labels = out.label_ids

    acc         = accuracy_score(labels, preds)
    macro_f1    = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report      = classification_report(labels, preds,
                                        target_names=class_names, zero_division=0)

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  Weighted F1   : {weighted_f1:.4f}")
    print(f"\n{report}")

    plot_confusion_matrix(labels, preds, class_names, output_dir)
    (output_dir / "classification_report.txt").write_text(report)
    metrics = {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
    save_metrics(metrics, output_dir)
    return metrics
