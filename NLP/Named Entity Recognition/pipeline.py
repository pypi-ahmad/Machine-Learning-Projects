"""
Modern NER / Entity Extraction Pipeline (April 2026)

Primary model : GLiNER (urchade/gliner_large-v2.1) — zero-shot NER that
                generalises to arbitrary entity types without fine-tuning.
Supervised    : HuggingFace token classification with ModernBERT when
                labelled data is available.
Baseline      : spaCy NER (en_core_web_sm) for quick comparison.

Evaluated with seqeval (entity-level precision / recall / F1).
All results + per-entity metrics exported to metrics.json.

Compute: GPU recommended for transformer models; GLiNER runs on CPU in
         ~1 min for small corpora.  spaCy baseline is CPU-only.
Data: Auto-downloaded at runtime.
"""
import os, json, time, warnings, re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

LABELS = ["person", "location", "organization", "miscellaneous"]
TEXT_COL = "tokens"
TAG_COL = "ner_tags"

# CoNLL BIO tag mapping (used when tag_col contains int IDs)
CONLL_TAG_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("conll2003", split="train").to_pandas()
    print(f"Dataset: {len(df)} samples")
    return df


def prepare_sentences(df):
    """Convert raw dataset rows into (tokens_list, tags_list) pairs."""
    sentences, tags_all = [], []
    for _, row in df.iterrows():
        toks = row[TEXT_COL]
        tags = row.get(TAG_COL)
        # Tokens may be a list or a space-separated string
        if isinstance(toks, str):
            toks = toks.split()
        if isinstance(toks, (list, np.ndarray)):
            toks = [str(t) for t in toks]
        else:
            continue
        # Tags: list of ints (CoNLL) or list of BIO strings
        if tags is not None:
            if isinstance(tags, str):
                tags = tags.split()
            if isinstance(tags, (list, np.ndarray)) and len(tags) == len(toks):
                if all(isinstance(t, (int, np.integer)) for t in tags):
                    tags = [CONLL_TAG_NAMES[int(t)] if int(t) < len(CONLL_TAG_NAMES) else "O" for t in tags]
                tags = [str(t) for t in tags]
            else:
                tags = ["O"] * len(toks)
        else:
            tags = ["O"] * len(toks)
        sentences.append(toks)
        tags_all.append(tags)
    return sentences, tags_all


# ═══════════════════════════════════════════════════════════════
# PRIMARY: GLiNER zero-shot NER
# ═══════════════════════════════════════════════════════════════
def run_gliner(sentences, gold_tags, labels):
    from gliner import GLiNER
    model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
    pred_tags_all = []
    t0 = time.perf_counter()
    for toks in sentences:
        text = " ".join(toks)
        entities = model.predict_entities(text[:2048], labels, threshold=0.35)
        # Map character-span entities back to token-level BIO tags
        tag_seq = ["O"] * len(toks)
        char_offsets = []
        pos = 0
        for tok in toks:
            start = text.find(tok, pos)
            if start == -1:
                start = pos
            char_offsets.append((start, start + len(tok)))
            pos = start + len(tok)
        for ent in entities:
            ent_start, ent_end = ent["start"], ent["end"]
            lbl = ent["label"].upper().replace(" ", "_")
            first = True
            for i, (cs, ce) in enumerate(char_offsets):
                if cs >= ent_start and ce <= ent_end + 1:
                    tag_seq[i] = f"B-{lbl}" if first else f"I-{lbl}"
                    first = False
        pred_tags_all.append(tag_seq)
    elapsed = time.perf_counter() - t0
    return pred_tags_all, elapsed


# ═══════════════════════════════════════════════════════════════
# SUPERVISED: Token classification with ModernBERT
# ═══════════════════════════════════════════════════════════════
def run_supervised(sentences, gold_tags):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
    from sklearn.model_selection import train_test_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Build label vocab from gold tags
    all_labels = sorted(set(t for seq in gold_tags for t in seq))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    n_labels = len(all_labels)

    idx = list(range(len(sentences)))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)
    tr_sents = [sentences[i] for i in tr_idx]
    tr_tags = [gold_tags[i] for i in tr_idx]
    te_sents = [sentences[i] for i in te_idx]
    te_tags = [gold_tags[i] for i in te_idx]

    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=n_labels, id2label=id2label, label2id=label2id,
    ).to(device)

    MAX_LEN = 256

    class NERDataset(Dataset):
        def __init__(self, sents, tag_seqs):
            self.items = []
            for toks, tags in zip(sents, tag_seqs):
                enc = tokenizer(toks, is_split_into_words=True, truncation=True,
                                padding="max_length", max_length=MAX_LEN, return_tensors="pt")
                word_ids = enc.word_ids()
                label_ids = []
                prev_word = None
                for wid in word_ids:
                    if wid is None:
                        label_ids.append(-100)
                    elif wid != prev_word:
                        label_ids.append(label2id.get(tags[wid], 0) if wid < len(tags) else 0)
                    else:
                        label_ids.append(-100)
                    prev_word = wid
                enc = {k: v.squeeze(0) for k, v in enc.items()}
                enc["labels"] = torch.tensor(label_ids, dtype=torch.long)
                self.items.append(enc)
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    train_ds = NERDataset(tr_sents, tr_tags)
    test_ds = NERDataset(te_sents, te_tags)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_loader) * 3), len(train_loader) * 3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    t0 = time.perf_counter()

    for epoch in range(3):
        model.train(); total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            total_loss += loss.item()
        print(f"  [ModernBERT-NER] Epoch {epoch+1}/3, Loss: {total_loss/len(train_loader):.4f}")

    elapsed = time.perf_counter() - t0

    # Predict on test set
    model.eval()
    pred_tags_all, true_tags_all = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            for pred_seq, label_seq in zip(preds, labels):
                p, t = [], []
                for pi, li in zip(pred_seq, label_seq):
                    if li != -100:
                        p.append(id2label.get(int(pi), "O"))
                        t.append(id2label.get(int(li), "O"))
                pred_tags_all.append(p)
                true_tags_all.append(t)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    model.save_pretrained(os.path.join(save_dir, "modernbert_ner_model"))
    return pred_tags_all, true_tags_all, elapsed


# ═══════════════════════════════════════════════════════════════
# BASELINE: spaCy NER
# ═══════════════════════════════════════════════════════════════
def run_spacy_baseline(sentences, labels):
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # spaCy label -> our label mapping (best-effort)
    SPACY_MAP = {
        "PERSON": "PER", "NORP": "MISC", "FAC": "LOC", "ORG": "ORG",
        "GPE": "LOC", "LOC": "LOC", "PRODUCT": "MISC", "EVENT": "MISC",
        "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
        "DATE": "MISC", "TIME": "MISC", "PERCENT": "MISC", "MONEY": "MISC",
        "QUANTITY": "MISC", "ORDINAL": "MISC", "CARDINAL": "MISC",
        "KEYWORD": "KEYWORD", "KEYPHRASE": "KEYPHRASE",
        "TOPIC": "TOPIC", "ENTITY": "ENTITY",
    }
    pred_tags_all = []
    t0 = time.perf_counter()
    for toks in sentences:
        text = " ".join(toks)
        doc = nlp(text)
        tag_seq = ["O"] * len(toks)
        # Map spaCy char-span entities to token indices
        char_offsets = []
        pos = 0
        for tok in toks:
            start = text.find(tok, pos)
            if start == -1:
                start = pos
            char_offsets.append((start, start + len(tok)))
            pos = start + len(tok)
        for ent in doc.ents:
            lbl = SPACY_MAP.get(ent.label_, "MISC")
            first = True
            for i, (cs, ce) in enumerate(char_offsets):
                if cs >= ent.start_char and ce <= ent.end_char + 1:
                    tag_seq[i] = f"B-{lbl}" if first else f"I-{lbl}"
                    first = False
        pred_tags_all.append(tag_seq)
    elapsed = time.perf_counter() - t0
    return pred_tags_all, elapsed


# ═══════════════════════════════════════════════════════════════
# EVALUATION: seqeval entity-level metrics
# ═══════════════════════════════════════════════════════════════
def normalise_tags(pred_tags, gold_tags):
    """Ensure pred and gold have the same length per sentence."""
    out_p, out_g = [], []
    for p, g in zip(pred_tags, gold_tags):
        min_len = min(len(p), len(g))
        out_p.append(p[:min_len])
        out_g.append(g[:min_len])
    return out_p, out_g


def evaluate(pred_tags, gold_tags, model_name):
    from seqeval.metrics import classification_report as seq_report
    from seqeval.metrics import f1_score as seq_f1, precision_score as seq_p, recall_score as seq_r
    pred_tags, gold_tags = normalise_tags(pred_tags, gold_tags)
    p = seq_p(gold_tags, pred_tags, zero_division=0)
    r = seq_r(gold_tags, pred_tags, zero_division=0)
    f1 = seq_f1(gold_tags, pred_tags, zero_division=0)
    print()
    print(f"=== {model_name} entity-level metrics ===")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(seq_report(gold_tags, pred_tags, zero_division=0))
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def plot_entity_counts(pred_tags, title, save_path):
    """Bar chart of predicted entity type counts."""
    counts = {}
    for seq in pred_tags:
        for tag in seq:
            if tag.startswith("B-"):
                lbl = tag[2:]
                counts[lbl] = counts.get(lbl, 0) + 1
    if not counts:
        return
    labels_sorted = sorted(counts.keys())
    vals = [counts[l] for l in labels_sorted]
    fig, ax = plt.subplots(figsize=(max(6, len(labels_sorted) * 0.8), 5))
    ax.bar(labels_sorted, vals, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Entity Type")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run_eda(df, save_dir):
    """Dataset summary for NER / entity extraction tasks."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Dataset size: {len(df)} rows")

    summary_rows = []
    for col in df.columns:
        series = df[col]
        summary_rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "n_unique": int(series.nunique(dropna=True)) if series.dtype != "object" else None,
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "eda_summary.csv"), index=False)

    if TEXT_COL in df.columns:
        token_lengths = []
        for value in df[TEXT_COL].head(5000):
            if isinstance(value, str):
                token_lengths.append(len(value.split()))
            elif isinstance(value, (list, np.ndarray)):
                token_lengths.append(len(value))
        if token_lengths:
            print(
                f"Token length: mean={np.mean(token_lengths):.1f}, "
                f"median={np.median(token_lengths):.1f}, max={max(token_lengths)}"
            )

    if TAG_COL in df.columns:
        entity_counts = {}
        for value in df[TAG_COL].head(5000):
            tags = value if isinstance(value, (list, np.ndarray)) else str(value).split()
            for tag in tags:
                tag_str = str(tag)
                if tag_str == "O":
                    continue
                if isinstance(tag, (int, np.integer)) and int(tag) < len(CONLL_TAG_NAMES):
                    tag_str = CONLL_TAG_NAMES[int(tag)]
                entity_counts[tag_str] = entity_counts.get(tag_str, 0) + 1
        if entity_counts:
            top_items = sorted(entity_counts.items(), key=lambda item: item[1], reverse=True)[:12]
            labels = [item[0] for item in top_items]
            values = [item[1] for item in top_items]
            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
            ax.bar(labels, values, color="steelblue")
            ax.set_title("Top Entity Tags in Dataset")
            ax.set_ylabel("Count")
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, "eda_entity_tags.png"), dpi=100, bbox_inches="tight")
            plt.close(fig)

    print("Summary statistics saved to eda_summary.csv")
    print("EDA complete.")


def main():
    print("=" * 60)
    print("NER / ENTITY EXTRACTION  GLiNER + ModernBERT + spaCy")
    print(f"Target labels: {LABELS}")
    print("=" * 60)
    df = load_data()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    run_eda(df, save_dir)
    sentences, gold_tags = prepare_sentences(df)
    if len(sentences) > 5000:
        sentences, gold_tags = sentences[:5000], gold_tags[:5000]
    print(f"Prepared {len(sentences)} sentences")
    metrics_out = {}

    # -- GLiNER (primary) --
    print()
    print("-- GLiNER Zero-Shot NER --")
    try:
        gliner_preds, gliner_time = run_gliner(sentences, gold_tags, LABELS)
        m = evaluate(gliner_preds, gold_tags, "GLiNER")
        m["time_s"] = round(gliner_time, 1)
        metrics_out["GLiNER"] = m
        plot_entity_counts(gliner_preds, "GLiNER  Predicted Entities",
                           os.path.join(save_dir, "entities_gliner.png"))
        print(f"  Time: {gliner_time:.1f}s")
    except Exception as e:
        print(f"GLiNER failed: {e}")

    # -- Supervised ModernBERT (if gold tags available) --
    has_gold = any(any(t != "O" for t in seq) for seq in gold_tags)
    if has_gold:
        print()
        print("-- ModernBERT Token Classification (supervised) --")
        try:
            sup_preds, sup_golds, sup_time = run_supervised(sentences, gold_tags)
            m = evaluate(sup_preds, sup_golds, "ModernBERT-NER")
            m["time_s"] = round(sup_time, 1)
            metrics_out["ModernBERT-NER"] = m
            plot_entity_counts(sup_preds, "ModernBERT-NER  Predicted Entities",
                               os.path.join(save_dir, "entities_modernbert.png"))
            print(f"  Time: {sup_time:.1f}s")
        except Exception as e:
            print(f"ModernBERT-NER failed: {e}")
    else:
        print()
        print("-- Skipping supervised ModernBERT (no gold BIO tags) --")

    # -- spaCy baseline --
    print()
    print("-- spaCy Baseline NER --")
    try:
        spacy_preds, spacy_time = run_spacy_baseline(sentences, LABELS)
        m = evaluate(spacy_preds, gold_tags, "spaCy")
        m["time_s"] = round(spacy_time, 1)
        metrics_out["spaCy"] = m
        plot_entity_counts(spacy_preds, "spaCy  Predicted Entities",
                           os.path.join(save_dir, "entities_spacy.png"))
        print(f"  Time: {spacy_time:.1f}s")
    except Exception as e:
        print(f"spaCy failed: {e}")

    # -- Summary --
    if metrics_out:
        best_name = max(metrics_out, key=lambda k: metrics_out[k].get("f1", 0))
        print()
        print(f"Best: {best_name} (F1: {metrics_out[best_name].get('f1', 0):.4f})")

    # Save metrics
    out_path = os.path.join(save_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print()
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()
