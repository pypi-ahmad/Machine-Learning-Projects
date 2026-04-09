"""English-to-French translation trainer (NLLB-200-distilled-600M + LoRA).

Loads tab-separated parallel corpus, fine-tunes with Seq2SeqTrainer,
evaluates with BLEU and chrF.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.training_common import (
    SEED, WORKSPACE, cleanup_gpu, ensure_output_dirs,
    get_amp_dtype, save_json, seed_everything,
)

logger = get_logger(__name__)

SRC_LANG = "eng_Latn"
TGT_LANG = "fra_Latn"


# ======================================================================
# Data loader
# ======================================================================

def _load_parallel_corpus(raw_dir: Path) -> pd.DataFrame:
    """Load eng-fra parallel text (tab-separated or line-based)."""
    rows: list[dict] = []

    for f in raw_dir.rglob("*fra*"):
        if f.suffix in (".txt", ".tsv"):
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    eng, fra = parts[0].strip(), parts[1].strip()
                    if eng and fra:
                        rows.append({"en": eng, "fr": fra})

    if not rows:
        # Try any .txt file
        for f in sorted(raw_dir.glob("*.txt")):
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    eng, fra = parts[0].strip(), parts[1].strip()
                    if eng and fra:
                        rows.append({"en": eng, "fr": fra})
            if rows:
                break

    # Also try CSVs
    if not rows:
        for f in raw_dir.rglob("*.csv"):
            try:
                df = pd.read_csv(f, encoding="utf-8")
                for ec in ("en", "English", "english", "source"):
                    for fc in ("fr", "French", "french", "target"):
                        if ec in df.columns and fc in df.columns:
                            df = df[[ec, fc]].dropna()
                            df.columns = ["en", "fr"]
                            return df
            except Exception:
                pass

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["en", "fr"])


# ======================================================================
# MAIN
# ======================================================================

def train_translator(
    slug: str,
    raw_dir: str | Path | None = None,
    df: pd.DataFrame | None = None,
    *,
    model_name: str = "facebook/nllb-200-distilled-600M",
    max_length: int = 128,
    batch_size: int = 4,
    grad_accum: int = 4,
    epochs: int = 3,
    lr: float = 2e-5,
    lora_r: int = 16,
    seed: int = SEED,
    max_samples: int = 50_000,
    force: bool = False,
) -> dict:
    """Fine-tune NLLB for EN->FR translation with LoRA."""
    seed_everything(seed)
    dirs = ensure_output_dirs(slug)
    metrics_file = dirs["metrics"] / "phase2_metrics.json"
    if metrics_file.exists() and not force:
        logger.info("Phase 2 already done for %s", slug)
        return json.loads(metrics_file.read_text())

    from datasets import Dataset as HFDataset
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer,
        DataCollatorForSeq2Seq, EarlyStoppingCallback,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )

    # ---- Data -------------------------------------------------------------
    if df is None:
        if raw_dir is None:
            raw_dir = WORKSPACE / "data" / slug / "raw"
        raw_dir = Path(raw_dir)
        df = _load_parallel_corpus(raw_dir)

    if df.empty:
        return {"slug": slug, "status": "ERROR", "error": "No parallel corpus found"}

    # Standardize column names
    if "en" not in df.columns:
        df.columns = ["en", "fr"] + list(df.columns[2:])
    df = df[["en", "fr"]].dropna().reset_index(drop=True)

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.05, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=seed)
    logger.info("[%s] train=%d  val=%d  test=%d", slug, len(train_df), len(val_df), len(test_df))

    result: dict = {
        "slug": slug,
        "task": "translation",
        "model_name": model_name,
        "direction": "en -> fr",
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "status": "running",
    }

    # ---- Tokenizer --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    def preprocess(examples):
        inputs = tokenizer(
            examples["en"], max_length=max_length, truncation=True, padding="max_length",
        )
        targets = tokenizer(
            text_target=examples["fr"], max_length=max_length, truncation=True, padding="max_length",
        )
        inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in lbl]
            for lbl in targets["input_ids"]
        ]
        return inputs

    train_ds = HFDataset.from_pandas(train_df[["en", "fr"]]).map(preprocess, batched=True, remove_columns=["en", "fr"])
    val_ds = HFDataset.from_pandas(val_df[["en", "fr"]]).map(preprocess, batched=True, remove_columns=["en", "fr"])
    test_ds = HFDataset.from_pandas(test_df[["en", "fr"]]).map(preprocess, batched=True, remove_columns=["en", "fr"])

    # ---- Model + LoRA -----------------------------------------------------
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        result["lora"] = True
    except Exception as exc:
        logger.warning("LoRA unavailable (%s); full fine-tune", exc)
        result["lora"] = False

    model.config.use_cache = False

    # ---- Args -------------------------------------------------------------
    amp = get_amp_dtype()
    args = Seq2SeqTrainingArguments(
        output_dir=str(dirs["checkpoints"]),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=(amp == "fp16"),
        bf16=(amp == "bf16"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=max_length,
        logging_steps=100,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
    )

    # ---- Metrics ----------------------------------------------------------
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        import sacrebleu
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])
        return {"bleu": bleu.score, "chrf": chrf.score}

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ---- Train + Evaluate -------------------------------------------------
    logger.info("[%s] Training %s (LoRA=%s) ...", slug, model_name, result.get("lora"))
    train_out = trainer.train()
    result["train_runtime_sec"] = train_out.metrics.get("train_runtime", 0)

    val_m = trainer.evaluate(val_ds)
    result["val_metrics"] = {k.replace("eval_", ""): v for k, v in val_m.items() if isinstance(v, (int, float))}

    test_m = trainer.evaluate(test_ds)
    result["test_metrics"] = {k.replace("eval_", ""): v for k, v in test_m.items() if isinstance(v, (int, float))}
    logger.info("[%s] Test: BLEU=%.2f  chrF=%.2f",
                slug, result["test_metrics"].get("bleu", 0), result["test_metrics"].get("chrf", 0))

    # ---- Sample translations ----------------------------------------------
    samples = []
    for _, row in test_df.head(10).iterrows():
        inp = tokenizer(row["en"], max_length=max_length, truncation=True, return_tensors="pt").to(model.device)
        gen = model.generate(**inp, max_length=max_length, num_beams=5,
                             forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG))
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        samples.append({"source": row["en"], "reference": row["fr"], "predicted": pred})
    result["sample_translations"] = samples

    # ---- Save -------------------------------------------------------------
    save_path = dirs["artifacts"] / "model"
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    result["model_path"] = str(save_path)
    result["status"] = "OK"
    save_json(result, metrics_file)

    del model, trainer
    cleanup_gpu()
    return result


def run_project(project_slug, project_dir, raw_paths, processed_dir, outputs_dir, config, force=False):
    """Unified entry point called by the Phase 2.1 orchestrator."""
    _KEYS = {"model_name","max_length","batch_size","grad_accum","epochs","lr","lora_r","max_samples","seed"}
    kw = {k: v for k, v in config.items() if k in _KEYS}
    raw_dir = config.get("raw_dir", project_dir)
    r = train_translator(project_slug, raw_dir=raw_dir, force=force, **kw)
    n = r.get("n_train", 0) + r.get("n_val", 0) + r.get("n_test", 0)
    return {
        "status": r.get("status", "UNKNOWN"),
        "model_name": r.get("model_name", config.get("model_name", "")),
        "dataset_size": n,
        "main_metrics": r.get("test_metrics", {}),
        "val_metrics": r.get("val_metrics", {}),
        "training_mode": "LoRA" if r.get("lora") else "full",
        "train_runtime_sec": r.get("train_runtime_sec", 0),
        "notes": "",
        "full_result": r,
    }
