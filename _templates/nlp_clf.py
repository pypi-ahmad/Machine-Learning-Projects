"""NLP Classification pipeline template: ModernBERT / XLM-R — April 2026"""
import textwrap


def generate(project_path, config):
    target = config.get("target", "label")
    text_col = config.get("text_col", "text")
    model_name = config.get("model", "answerdotai/ModernBERT-base")

    return textwrap.dedent(f'''\
        """
        Modern NLP Classification Pipeline (April 2026)
        Model: ModernBERT (fine-tuned) with HuggingFace Transformers + Accelerate
        Fallback: XLM-RoBERTa for multilingual tasks
        """
        import os, sys, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        warnings.filterwarnings("ignore")

        TARGET = "{target}"
        TEXT_COL = "{text_col}"
        MODEL_NAME = "{model_name}"
        MAX_LEN = 256
        BATCH_SIZE = 16
        EPOCHS = 3
        LR = 2e-5


        def load_data():
            data_dir = Path(os.path.dirname(__file__))
            csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv"))
            if csv_files:
                sep = "\\t" if csv_files[0].suffix == ".tsv" else ","
                df = pd.read_csv(csv_files[0], sep=sep)
            else:
                raise FileNotFoundError("No CSV/TSV data found.")

            # Auto-detect text column
            text_col = TEXT_COL
            if text_col not in df.columns:
                text_candidates = [c for c in df.columns
                                   if df[c].dtype == "object" and df[c].str.len().mean() > 30]
                if text_candidates:
                    text_col = text_candidates[0]
                else:
                    text_col = df.select_dtypes(include=["object"]).columns[0]

            df = df[[text_col, TARGET]].dropna()
            df.columns = ["text", "label"]
            print(f"Dataset: {{len(df)}} samples, columns: {{text_col}} → text, {{TARGET}} → label")
            print(f"Label distribution:\\n{{df['label'].value_counts()}}")
            return df


        def train_modernbert(df):
            """Fine-tune ModernBERT for text classification."""
            import torch
            from torch.utils.data import DataLoader, Dataset
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers import get_linear_schedule_with_warmup

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {{device}}")

            # Encode labels
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df["label_id"] = le.fit_transform(df["label"])
            n_classes = len(le.classes_)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                                  stratify=df["label_id"] if n_classes < 50 else None)

            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=n_classes
            ).to(device)

            class TextDataset(Dataset):
                def __init__(self, texts, labels):
                    self.encodings = tokenizer(
                        texts, truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="pt"
                    )
                    self.labels = torch.tensor(labels, dtype=torch.long)

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, idx):
                    item = {{k: v[idx] for k, v in self.encodings.items()}}
                    item["labels"] = self.labels[idx]
                    return item

            train_ds = TextDataset(train_df["text"].tolist(), train_df["label_id"].tolist())
            test_ds = TextDataset(test_df["text"].tolist(), test_df["label_id"].tolist())
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
            total_steps = len(train_loader) * EPOCHS
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                         num_training_steps=total_steps)

            # Training
            model.train()
            for epoch in range(EPOCHS):
                total_loss = 0
                for batch in train_loader:
                    batch = {{k: v.to(device) for k, v in batch.items()}}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                print(f"  Epoch {{epoch+1}}/{{EPOCHS}}, Loss: {{total_loss/len(train_loader):.4f}}")

            # Evaluation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = {{k: v.to(device) for k, v in batch.items()}}
                    outputs = model(**batch)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())

            y_pred = np.array(all_preds)
            y_test = np.array(all_labels)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            print(f"\\n✓ ModernBERT — Accuracy: {{acc:.4f}}, F1: {{f1:.4f}}")
            print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str),
                                        zero_division=0))

            # Save model
            save_path = os.path.join(os.path.dirname(__file__), "modernbert_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved to {{save_path}}")

            return {{"accuracy": acc, "f1": f1, "model": model}}


        def main():
            print("=" * 60)
            print("MODERN NLP CLASSIFICATION PIPELINE")
            print(f"Model: {{MODEL_NAME}}")
            print("=" * 60)
            df = load_data()
            results = train_modernbert(df)
            print(f"\\n🏆 Final Accuracy: {{results['accuracy']:.4f}}")


        if __name__ == "__main__":
            main()
    ''')
