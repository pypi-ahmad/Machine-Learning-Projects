"""
Modern NLP Classification Pipeline (April 2026)
Models: ModernBERT + XLM-RoBERTa fine-tuned + GLiNER NER
Data: Auto-downloaded at runtime
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib; matplotlib.use("Agg")

warnings.filterwarnings("ignore")

TARGET = "label"
TEXT_COL = "text"
MAX_LEN, BATCH_SIZE, EPOCHS, LR = 256, 16, 3, 2e-5

MODELS = [
    ("answerdotai/ModernBERT-base", "ModernBERT"),
    ("FacebookAI/xlm-roberta-base", "XLM-R"),
]


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("cardiffnlp/tweet_eval", "sentiment", split="train").to_pandas()
    # Auto-detect text column
    text_col = TEXT_COL
    if text_col not in df.columns:
        candidates = [c for c in df.columns if df[c].dtype == "object" and df[c].str.len().mean() > 20]
        text_col = candidates[0] if candidates else df.select_dtypes("object").columns[0]
    target = TARGET if TARGET in df.columns else df.columns[-1]
    df = df[[text_col, target]].dropna()
    df.columns = ["text", "label"]
    print(f"Dataset: {len(df)} samples")
    return df


def train_transformer(df, model_name, display_name):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    le = LabelEncoder(); df["label_id"] = le.fit_transform(df["label"])
    n_classes = len(le.classes_)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                          stratify=df["label_id"] if n_classes < 50 else None)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes).to(device)

    class DS(Dataset):
        def __init__(self, texts, labels):
            self.enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.labels[i]}

    train_loader = DataLoader(DS(train_df["text"].tolist(), train_df["label_id"].tolist()), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(DS(test_df["text"].tolist(), test_df["label_id"].tolist()), batch_size=BATCH_SIZE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_loader) * EPOCHS), len(train_loader) * EPOCHS)

    for epoch in range(EPOCHS):
        model.train(); total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); total_loss += loss.item()
        print(f"  [{display_name}] Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.extend(torch.argmax(model(**batch).logits, dim=-1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    print(f"\n✓ {display_name} — Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(classification_report(labels, preds, target_names=le.classes_.astype(str), zero_division=0))
    model.save_pretrained(os.path.join(os.path.dirname(__file__), f"{display_name.lower().replace('-','_')}_model"))
    return acc, f1


def run_gliner(df):
    """Zero-shot NER with GLiNER on a sample of texts."""
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained("urchade/gliner_base")
        sample_labels = ["person", "location", "organization", "date", "money", "product"]
        for i, text in enumerate(df["text"].head(10)):
            entities = model.predict_entities(text[:512], sample_labels, threshold=0.4)
            if entities:
                ent_str = ", ".join(f"{e['text']}({e['label']})" for e in entities[:5])
                print(f"  [{i+1}] {ent_str}")
        print("✓ GLiNER NER complete")
    except Exception as e:
        print(f"✗ GLiNER: {e}")


def main():
    print("=" * 60)
    print("NLP CLASSIFICATION — ModernBERT + XLM-R + GLiNER")
    print("=" * 60)
    df = load_data()
    best_acc, best_name = 0, ""
    for model_name, display_name in MODELS:
        try:
            acc, f1 = train_transformer(df.copy(), model_name, display_name)
            if acc > best_acc:
                best_acc, best_name = acc, display_name
        except Exception as e:
            print(f"✗ {display_name}: {e}")
    print(f"\n🏆 Best: {best_name} (Accuracy: {best_acc:.4f})")
    print("\n— GLiNER Zero-Shot NER —")
    run_gliner(df)


if __name__ == "__main__":
    main()
