"""
Modern NLP Classification Pipeline (April 2026)
Models: ModernBERT (English) + XLM-RoBERTa (multilingual) + GLiNER (zero-shot NER)
        TF-IDF + Naive Bayes as baseline
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

TARGET = "ner_tags"
TEXT_COL = "tokens"
MAX_LEN, BATCH_SIZE, EPOCHS, LR = 256, 16, 3, 2e-5

MODELS = [
    ("answerdotai/ModernBERT-base", "ModernBERT"),
    ("FacebookAI/xlm-roberta-base", "XLM-R"),
]


def load_data():
    from datasets import load_dataset as _hf_load
    df = _hf_load("conll2003", split="train").to_pandas()
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


# ═══════════════════════════════════════════════════════════════
# BASELINE: TF-IDF + Naive Bayes / Logistic Regression
# ═══════════════════════════════════════════════════════════════
def run_tfidf_baseline(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    le = LabelEncoder(); y = le.fit_transform(df["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(df["text"], y, test_size=0.2, random_state=42,
                                                stratify=y if len(le.classes_) < 50 else None)
    for name, clf in [("Naive Bayes", MultinomialNB()), ("LogReg", LogisticRegression(max_iter=1000, n_jobs=-1))]:
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="weighted")
        print(f"  [Baseline] {name} — Accuracy: {acc:.4f}, F1: {f1:.4f}")


# ═══════════════════════════════════════════════════════════════
# PRIMARY: ModernBERT / XLM-R fine-tuned classifier
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# GLiNER: Zero-shot NER on text samples
# ═══════════════════════════════════════════════════════════════
def run_gliner(df):
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained("urchade/gliner_base")
        sample_labels = ["person", "location", "organization", "date", "money", "product", "event"]
        for i, text in enumerate(df["text"].head(10)):
            entities = model.predict_entities(text[:512], sample_labels, threshold=0.4)
            if entities:
                ent_str = ", ".join(f"{e['text']}({e['label']})" for e in entities[:5])
                print(f"  [{i+1}] {ent_str}")
        print("✓ GLiNER zero-shot NER complete")
    except Exception as e:
        print(f"✗ GLiNER: {e}")


# ═══════════════════════════════════════════════════════════════
# EMBEDDING SIMILARITY: BGE-M3 / Qwen3-Embedding
# ═══════════════════════════════════════════════════════════════
def run_embedding_similarity(df):
    """Embedding-based retrieval/similarity with BGE-M3 and Qwen3-Embedding."""
    texts = df["text"].dropna().head(200).tolist()
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        # BGE-M3
        model = SentenceTransformer("BAAI/bge-m3")
        embs = model.encode(texts, batch_size=32, show_progress_bar=True)
        sim = cosine_similarity(embs)
        # Show top-3 similar texts for first 3 samples
        for i in range(min(3, len(texts))):
            top_idx = np.argsort(sim[i])[-4:-1][::-1]
            print(f"  Text {i+1} most similar to: {[idx for idx in top_idx]}")
        print(f"✓ BGE-M3: {len(texts)} texts embedded (dim={embs.shape[1]})")

        # Qwen3-Embedding
        try:
            qwen = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
            qwen_embs = qwen.encode(texts[:100], batch_size=16, show_progress_bar=True)
            print(f"✓ Qwen3-Embedding: {len(qwen_embs)} texts embedded (dim={qwen_embs.shape[1]})")
        except Exception as e:
            print(f"✗ Qwen3-Embedding: {e}")
    except Exception as e:
        print(f"✗ Embedding similarity: {e}")


def main():
    print("=" * 60)
    print("NLP CLASSIFICATION — ModernBERT + XLM-R | TF-IDF baseline | GLiNER NER")
    print("=" * 60)
    df = load_data()

    # Baseline first
    print("\n— TF-IDF / Naive Bayes Baseline —")
    run_tfidf_baseline(df)

    # Primary transformer models
    best_acc, best_name = 0, ""
    for model_name, display_name in MODELS:
        try:
            acc, f1 = train_transformer(df.copy(), model_name, display_name)
            if acc > best_acc:
                best_acc, best_name = acc, display_name
        except Exception as e:
            print(f"✗ {display_name}: {e}")
    print(f"\n🏆 Best: {best_name} (Accuracy: {best_acc:.4f})")

    # Zero-shot NER
    print("\n— GLiNER Zero-Shot NER —")
    run_gliner(df)

    # Embedding similarity
    print("\n— Embedding Similarity (BGE-M3 / Qwen3-Embedding) —")
    run_embedding_similarity(df)


if __name__ == "__main__":
    main()
