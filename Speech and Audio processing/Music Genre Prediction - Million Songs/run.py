#!/usr/bin/env python3
"""
Music Genre Prediction — Audio CNN / PyCaret (PyTorch)
=======================================================
Predict music genres from the Spotify Million Song dataset.

The dataset primarily contains **metadata / audio features** (danceability,
energy, loudness, tempo, etc.) rather than raw audio files.  We therefore
use two approaches:

* **Approach 1 (primary):** PyCaret autoML classification on tabular
  audio features + genre labels.
* **Approach 2 (fallback):** If raw audio files are present, extract
  mel-spectrograms and train a small CNN genre classifier.

Dataset
-------
* Kaggle: https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset

Run
---
    python run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.utils import (
    download_kaggle_dataset,
    set_seed,
    setup_logging,
    project_paths,
    get_device,
    ensure_dir,
    dataset_prompt,
    save_classification_report,
    run_tabular_auto,
    parse_common_args,
    save_metrics,
    run_metadata,
    dataset_fingerprint,
    write_split_manifest,
    make_tabular_splits,
    dataset_missing_metrics,
    missing_dependency_metrics,
    safe_import_available,
    resolve_device_from_args,
    configure_cuda_allocator,
)

logger = logging.getLogger(__name__)

# Check torchaudio availability once (import can hard-crash the process)
_TORCHAUDIO_OK = safe_import_available("torchaudio")

# ── Configuration ────────────────────────────────────────────
KAGGLE_SLUG = "notshrirang/spotify-million-song-dataset"
EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
N_MELS = 64
SPEC_MAX_LEN = 128
SAMPLE_RATE = 22_050


# ═════════════════════════════════════════════════════════════
#  Data discovery
# ═════════════════════════════════════════════════════════════

def find_csv_files(root: Path):
    """Find CSV files under the dataset directory."""
    return sorted(root.rglob("*.csv"))


def find_audio_files(root: Path):
    """Find audio files for the CNN fallback."""
    exts = {".wav", ".mp3", ".flac", ".ogg"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file())


def load_tabular_data(csv_files: list[Path]):
    """Load and concatenate CSV files, return a DataFrame."""
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
            logger.info("  Loaded %s — %d rows, %d cols", f.name, len(df), len(df.columns))
        except Exception as exc:
            logger.warning("  Could not read %s: %s", f.name, exc)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def detect_genre_column(df: pd.DataFrame):
    """Heuristically find the genre / target column."""
    candidates = ["genre", "genres", "track_genre", "music_genre", "label",
                   "class", "category", "target"]
    for col in candidates:
        for c in df.columns:
            if c.lower().strip() == col:
                return c
    # Try columns with few unique values (< 50) that look categorical
    for c in df.columns:
        if df[c].dtype == object and 2 <= df[c].nunique() <= 50:
            return c
    return None


def detect_feature_columns(df: pd.DataFrame, target_col: str):
    """Select numeric feature columns suitable for classification."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove ID-like or irrelevant columns
    drop_pats = ["id", "uri", "url", "track_id", "unnamed"]
    features = [
        c for c in numeric_cols
        if c != target_col and not any(p in c.lower() for p in drop_pats)
    ]
    return features


# ═════════════════════════════════════════════════════════════
#  Approach 1 — PyCaret classification
# ═════════════════════════════════════════════════════════════

def run_pycaret_approach(df: pd.DataFrame, target_col: str, output_dir: Path):
    """Run auto-ML classification pipeline (PyCaret -> LazyPredict -> sklearn)."""
    logger.info("Running tabular auto-ML classification (target=%s) …", target_col)
    # Subsample if too large
    if len(df) > 50_000:
        df = df.sample(50_000, random_state=42).reset_index(drop=True)
        logger.info("  Sub-sampled to %d rows for speed.", len(df))

    metrics = run_tabular_auto(df, target=target_col, output_dir=output_dir, task="classification", session_id=42)
    return metrics


# ═════════════════════════════════════════════════════════════
#  Approach 1b — sklearn fallback (if PyCaret unavailable)
# ═════════════════════════════════════════════════════════════

def run_sklearn_approach(df: pd.DataFrame, target_col: str, feature_cols: list[str], output_dir: Path):
    """Train a Random Forest as a lightweight fallback."""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    logger.info("Running sklearn fallback classification …")

    le = LabelEncoder()
    df = df.dropna(subset=[target_col])
    y = le.fit_transform(df[target_col].astype(str))
    X = df[feature_cols].fillna(0).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = None
    try:
        y_prob = clf.predict_proba(X_test)
    except Exception:
        pass

    labels = list(le.classes_)
    metrics = save_classification_report(y_test, y_pred, output_dir,
                                         y_prob=y_prob, labels=labels,
                                         prefix="genre")

    # Feature importance
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(idx)), importances[idx][::-1])
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances — Genre Classification")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance plot → %s", output_dir / "feature_importance.png")

    return metrics


# ═════════════════════════════════════════════════════════════
#  Approach 2 — CNN on mel spectrograms (if audio files exist)
# ═════════════════════════════════════════════════════════════

class MelDataset(Dataset):
    def __init__(self, specs, labels, max_len=SPEC_MAX_LEN):
        self.specs = specs
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        # pad / crop time dimension
        if spec.shape[-1] >= self.max_len:
            spec = spec[:, :, :self.max_len]
        else:
            spec = torch.nn.functional.pad(spec, (0, self.max_len - spec.shape[-1]))
        return spec, self.labels[idx]


class GenreCNN(nn.Module):
    def __init__(self, n_classes, n_mels=N_MELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def run_cnn_approach(audio_files, output_dir, device, *, use_amp=None, epochs=None):
    """Extract mel specs from audio files grouped by parent-folder genre."""
    if not _TORCHAUDIO_OK:
        logger.warning("torchaudio not available -- skipping CNN approach.")
        return None

    import torchaudio

    # Infer genre from parent folder name
    genre_map: dict[str, list[Path]] = {}
    for fp in audio_files:
        genre = fp.parent.name
        genre_map.setdefault(genre, []).append(fp)

    if len(genre_map) < 2:
        logger.warning("Cannot determine genres from folder structure; skipping CNN.")
        return None

    label_names = sorted(genre_map.keys())
    label2idx = {g: i for i, g in enumerate(label_names)}
    logger.info("CNN approach — %d genres: %s", len(label_names), label_names)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS,
    )

    specs, labels = [], []
    for genre, files in genre_map.items():
        for fp in files[:200]:
            try:
                wav, sr = torchaudio.load(str(fp))
                if sr != SAMPLE_RATE:
                    wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                mel = torch.log1p(mel_transform(wav))
                specs.append(mel)
                labels.append(label2idx[genre])
            except Exception:
                continue

    if len(specs) < 20:
        logger.warning("Too few audio spectrograms (%d) — skipping CNN.", len(specs))
        return None

    labels_t = torch.tensor(labels, dtype=torch.long)
    ds = MelDataset(specs, labels_t)

    # 70 / 15 / 15 split
    n_test = max(1, int(0.15 * len(ds)))
    n_val = max(1, int(0.15 * len(ds)))
    n_train = len(ds) - n_val - n_test
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    cnn_epochs = epochs or EPOCHS
    model = GenreCNN(len(label_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    amp_enabled = use_amp if use_amp is not None else (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(1, cnn_epochs + 1):
        model.train()
        total_loss = 0.0
        for spec_b, lab_b in train_dl:
            spec_b, lab_b = spec_b.to(device), lab_b.to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(spec_b)
                loss = criterion(logits, lab_b)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for spec_b, lab_b in val_dl:
                spec_b, lab_b = spec_b.to(device), lab_b.to(device)
                preds = model(spec_b).argmax(dim=1)
                correct += (preds == lab_b).sum().item()
                total += lab_b.size(0)

        acc = correct / max(total, 1)
        logger.info("CNN Epoch %d/%d — loss=%.4f  val_acc=%.4f",
                     epoch, cnn_epochs, total_loss / max(len(train_dl), 1), acc)

    # ── Evaluate on TEST split ───────────────────────────────
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for spec_b, lab_b in test_dl:
            spec_b, lab_b = spec_b.to(device), lab_b.to(device)
            preds = model(spec_b).argmax(dim=1)
            test_correct += (preds == lab_b).sum().item()
            test_total += lab_b.size(0)

    test_acc = test_correct / max(test_total, 1)
    logger.info("CNN TEST accuracy: %.4f (%d samples)", test_acc, test_total)

    torch.save(model.state_dict(), output_dir / "genre_cnn.pth")
    logger.info("Saved CNN model → %s", output_dir / "genre_cnn.pth")

    return {
        "cnn_test_accuracy": test_acc,
        "cnn_test_samples": test_total,
        "cnn_split": {"train": n_train, "val": n_val, "test": n_test},
    }


# ═════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════

def plot_genre_distribution(df: pd.DataFrame, target_col: str, output_dir: Path):
    """Bar chart of genre class distribution."""
    counts = df[target_col].value_counts().head(25)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title("Genre Distribution (top 25)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "genre_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved genre distribution plot → %s", output_dir / "genre_distribution.png")


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════

def main() -> None:
    setup_logging()
    args = parse_common_args("Music Genre Prediction — PyCaret / sklearn")
    set_seed(args.seed)
    configure_cuda_allocator()

    paths = project_paths(__file__)
    data_dir = paths["data"]
    output_dir = ensure_dir(paths["outputs"])
    device = resolve_device_from_args(args)

    # ── CLI overrides ────────────────────────────────────────
    epochs = args.epochs or EPOCHS
    use_amp = not args.no_amp and device.type == "cuda"

    # ── Download dataset ─────────────────────────────────
    ds_path = download_kaggle_dataset(
        KAGGLE_SLUG, data_dir,
        dataset_name="Spotify Million Song Dataset",
    )

    if args.download_only:
        logger.info("Download complete — exiting (--download-only).")
        sys.exit(0)

    # ── Discover data ────────────────────────────────────────
    csv_files = find_csv_files(ds_path)
    audio_files = find_audio_files(ds_path)
    logger.info("Found %d CSV files, %d audio files", len(csv_files), len(audio_files))

    ds_fp = dataset_fingerprint(ds_path)
    meta = run_metadata(args)

    metrics: dict = {
        "dataset": f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}",
        "csv_files": len(csv_files),
        "audio_files": len(audio_files),
    }

    split_counts: dict = {}

    # ── Approach 1: Tabular classification ───────────────────
    if csv_files:
        df = load_tabular_data(csv_files)
        logger.info("Combined dataframe: %d rows × %d cols", len(df), len(df.columns))
        logger.info("Columns: %s", list(df.columns))

        if args.mode == "smoke":
            df = df.sample(n=min(200, len(df)), random_state=args.seed)
            logger.info("SMOKE TEST: limited to %d rows", len(df))

        target_col = detect_genre_column(df)
        if target_col:
            logger.info("Detected genre/target column: '%s' (%d unique)",
                        target_col, df[target_col].nunique())
            plot_genre_distribution(df, target_col, output_dir)

            feature_cols = detect_feature_columns(df, target_col)
            logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:10])

            # ── Tabular splits ───────────────────────────────
            try:
                X_tr, y_tr, X_va, y_va, X_te, y_te = make_tabular_splits(
                    df.dropna(subset=[target_col]), target_col,
                    task="classification", seed=args.seed,
                )
                split_counts = {
                    "train": len(y_tr), "val": len(y_va), "test": len(y_te),
                }
                splits_dict = {
                    "X_train": X_tr, "y_train": y_tr,
                    "X_val": X_va, "y_val": y_va,
                    "X_test": X_te, "y_test": y_te,
                }
            except Exception as exc:
                logger.warning("make_tabular_splits failed: %s — using legacy path", exc)
                splits_dict = None

            # Run auto-ML (PyCaret -> LazyPredict -> sklearn) with splits
            try:
                auto_metrics = run_tabular_auto(
                    df, target=target_col, output_dir=output_dir,
                    task="classification", session_id=args.seed,
                    splits=splits_dict,
                )
                metrics["approach"] = auto_metrics.pop("engine", "auto")
                metrics.update(auto_metrics)
            except Exception as exc:
                logger.warning("Auto-ML failed (%s) — falling back to sklearn.", exc)
                if feature_cols:
                    cls_metrics = run_sklearn_approach(df, target_col, feature_cols, output_dir)
                    metrics["approach"] = "sklearn"
                    metrics["accuracy"] = cls_metrics.get("accuracy")
                    metrics["macro_f1"] = cls_metrics.get("macro_f1")
                else:
                    logger.error("No numeric feature columns found.")
                    metrics["approach"] = "none — no features"
        else:
            logger.warning("Could not detect a genre / target column in the data.")
            logger.info("Available columns: %s", list(df.columns))
            metrics["approach"] = "none — no target column"
    else:
        logger.warning("No CSV files found in dataset.")
        metrics["approach"] = "none — no CSVs"

    # ── Approach 2: CNN on audio (if present) ────────────────
    if audio_files:
        logger.info("Audio files detected — running CNN spectrogram approach …")
        cnn_result = run_cnn_approach(
            audio_files, output_dir, device,
            use_amp=use_amp, epochs=epochs,
        )
        if cnn_result:
            metrics["cnn_trained"] = True
            metrics["cnn_test_accuracy"] = cnn_result.get("cnn_test_accuracy")
            if not split_counts:
                split_counts = cnn_result.get("cnn_split", {})
        else:
            metrics["cnn_trained"] = False
    else:
        logger.info("No audio files found — CNN approach skipped.")
        metrics["cnn_trained"] = False

    # ── Ensure default metric keys ────────────────────────
    metrics.setdefault("accuracy", 0)
    metrics.setdefault("macro_f1", 0)
    metrics.setdefault("weighted_f1", 0)
    metrics.setdefault("auc", 0)

    # ── Write split manifest ─────────────────────────────
    if not split_counts:
        split_counts = {"train": 0, "val": 0, "test": 0}
    write_split_manifest(
        output_dir,
        dataset_fp=ds_fp,
        split_method="make_tabular_splits 70/15/15 (tabular) + random_split 70/15/15 (CNN)",
        seed=args.seed,
        counts=split_counts,
    )

    # ── Save final metrics ───────────────────────────────
    metrics["split"] = "test"
    metrics["run_metadata"] = meta
    save_metrics(output_dir, metrics, task_type="classification", mode=args.mode)
    logger.info("Done ✓")


if __name__ == "__main__":
    main()
