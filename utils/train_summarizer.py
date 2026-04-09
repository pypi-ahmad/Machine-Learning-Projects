"""Abstractive summarization trainer (BART-large-CNN + LoRA).

Loads article-summary pairs from the BBC dataset raw directory,
fine-tunes with HF Seq2SeqTrainer, evaluates with ROUGE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.training_common import (
    SEED, WORKSPACE, cleanup_gpu, ensure_output_dirs,
    get_amp_dtype, save_json, seed_everything,
)

logger = get_logger(__name__)


# ======================================================================
# BBC data loader (Articles / Summaries folder layout)
# ======================================================================

def _load_bbc_pairs(raw_dir: Path) -> pd.DataFrame:
    """Find article-summary pairs from the BBC News Summary dataset."""
    rows: list[dict] = []

    # Try nested structure: raw_dir/BBC News Summary/{News Articles,Articles}/...
    candidates = [
        raw_dir,
        raw_dir / "BBC News Summary",
        raw_dir / "BBC News Summary" / "BBC News Summary",
        raw_dir / "bbc-news-summary",
    ]
    articles_names = ["News Articles", "Articles", "news_articles"]
    summaries_names = ["Summaries", "summaries"]

    for candidate in candidates:
        articles_dir = None
        summaries_dir = None
        for a in articles_names:
            d = candidate / a
            if d.is_dir():
                articles_dir = d
                break
        for s in summaries_names:
            d = candidate / s
            if d.is_dir():
                summaries_dir = d
                break
        if articles_dir and summaries_dir:
            for cat_dir in sorted(articles_dir.iterdir()):
                if not cat_dir.is_dir():
                    continue
                category = cat_dir.name
                for art_file in sorted(cat_dir.glob("*.txt")):
                    sum_file = summaries_dir / category / art_file.name
                    if sum_file.exists():
                        try:
                            art = art_file.read_text(encoding="utf-8", errors="replace").strip()
                            summ = sum_file.read_text(encoding="utf-8", errors="replace").strip()
                            if art and summ:
                                rows.append({"article": art, "summary": summ, "category": category})
                        except Exception:
                            pass
            if rows:
                logger.info("Loaded %d article-summary pairs from %s", len(rows), articles_dir)
                break

    # Fall back: look for CSV files
    if not rows:
        for csv_file in raw_dir.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
                # Look for article/summary-like columns
                for acol in ("article", "Articles", "text", "document"):
                    for scol in ("summary", "Summaries", "highlights", "abstract"):
                        if acol in df.columns and scol in df.columns:
                            df = df[[acol, scol]].dropna()
                            df.columns = ["article", "summary"]
                            return df
            except Exception:
                pass

    if not rows:
        logger.warning("No article-summary pairs found in %s", raw_dir)
        return pd.DataFrame(columns=["article", "summary"])

    return pd.DataFrame(rows)


# ======================================================================
# MAIN: Summarization trainer
# ======================================================================

def train_summarizer(
    slug: str,
    raw_dir: str | Path | None = None,
    df: pd.DataFrame | None = None,
    *,
    article_col: str = "article",
    summary_col: str = "summary",
    model_name: str = "facebook/bart-large-cnn",
    max_input_length: int = 1024,
    max_target_length: int = 128,
    batch_size: int = 2,
    grad_accum: int = 8,
    epochs: int = 3,
    lr: float = 3e-5,
    lora_r: int = 16,
    seed: int = SEED,
    max_samples: int = 10_000,
    force: bool = False,
) -> dict:
    """Fine-tune BART for abstractive summarization with LoRA."""
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
        df = _load_bbc_pairs(raw_dir)
    else:
        df = df[[article_col, summary_col]].dropna().reset_index(drop=True)
        df.columns = ["article", "summary"]
        article_col, summary_col = "article", "summary"

    if df.empty:
        return {"slug": slug, "status": "ERROR", "error": "No article-summary pairs loaded"}

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.10, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=0.10, random_state=seed)
    logger.info("[%s] train=%d  val=%d  test=%d", slug, len(train_df), len(val_df), len(test_df))

    result: dict = {
        "slug": slug,
        "task": "summarization",
        "model_name": model_name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "status": "running",
    }

    # ---- Tokenizer + Tokenize --------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(examples):
        inputs = tokenizer(
            examples["article"], max_length=max_input_length, truncation=True , padding="max_length",
        )
        targets = tokenizer(
            text_target=examples["summary"], max_length=max_target_length, truncation=True, padding="max_length",
        )
        inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in lbl]
            for lbl in targets["input_ids"]
        ]
        return inputs

    train_ds = HFDataset.from_pandas(train_df[["article", "summary"]]).map(preprocess, batched=True, remove_columns=["article", "summary"])
    val_ds = HFDataset.from_pandas(val_df[["article", "summary"]]).map(preprocess, batched=True, remove_columns=["article", "summary"])
    test_ds = HFDataset.from_pandas(test_df[["article", "summary"]]).map(preprocess, batched=True, remove_columns=["article", "summary"])

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
        logger.warning("LoRA not available (%s), full fine-tune", exc)
        result["lora"] = False

    model.config.use_cache = False  # required for gradient checkpointing

    # ---- Training args ----------------------------------------------------
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
        metric_for_best_model="rouge1",
        greater_is_better=True,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        logging_steps=50,
        report_to="none",
        seed=seed,
        dataloader_num_workers=0,
    )

    # ---- ROUGE compute_metrics -------------------------------------------
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # rouge
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1, r2, rl = [], [], []
        for pred, ref in zip(decoded_preds, decoded_labels):
            s = scorer.score(ref, pred)
            r1.append(s["rouge1"].fmeasure)
            r2.append(s["rouge2"].fmeasure)
            rl.append(s["rougeL"].fmeasure)
        return {"rouge1": np.mean(r1), "rouge2": np.mean(r2), "rougeL": np.mean(rl)}

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

    # ---- Train ------------------------------------------------------------
    logger.info("[%s] Training %s (LoRA=%s, bs=%d, accum=%d) ...", slug, model_name, result.get("lora"), batch_size, grad_accum)
    train_out = trainer.train()
    result["train_runtime_sec"] = train_out.metrics.get("train_runtime", 0)

    # ---- Evaluate ---------------------------------------------------------
    val_m = trainer.evaluate(val_ds)
    result["val_metrics"] = {k.replace("eval_", ""): v for k, v in val_m.items() if isinstance(v, (int, float))}

    test_m = trainer.evaluate(test_ds)
    result["test_metrics"] = {k.replace("eval_", ""): v for k, v in test_m.items() if isinstance(v, (int, float))}
    logger.info("[%s] Test: R1=%.4f  R2=%.4f  RL=%.4f",
                slug, result["test_metrics"].get("rouge1", 0),
                result["test_metrics"].get("rouge2", 0),
                result["test_metrics"].get("rougeL", 0))

    # ---- Generate samples -------------------------------------------------
    sample_rows = test_df.head(5)
    samples = []
    for _, row in sample_rows.iterrows():
        inp = tokenizer(row["article"], max_length=max_input_length, truncation=True, return_tensors="pt").to(model.device)
        gen = model.generate(**inp, max_length=max_target_length, num_beams=4, length_penalty=2.0)
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)
        samples.append({"reference": row["summary"][:300], "generated": pred[:300]})
    result["sample_predictions"] = samples

    # ---- Save model -------------------------------------------------------
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
    _KEYS = {"model_name","max_input_length","max_target_length","batch_size","grad_accum","epochs","lr","lora_r","max_samples","seed"}
    kw = {k: v for k, v in config.items() if k in _KEYS}
    raw_dir = config.get("raw_dir", project_dir)
    r = train_summarizer(project_slug, raw_dir=raw_dir, force=force, **kw)
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
