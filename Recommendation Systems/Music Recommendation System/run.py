#!/usr/bin/env python3
"""Music Recommendation — NCF

Dataset : https://www.kaggle.com/datasets/gauthamvijayaraj/spotify-tracks-dataset-updated-every-week
Approach: If the dataset contains user–track interactions, train Neural
          Collaborative Filtering (NCF).  Otherwise fall back to a
          content-based PyCaret regression predicting track popularity
          from audio features.
Run     : python "Recommendation Systems/6. Music  recommendation system/run.py"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from shared.utils import (
    download_kaggle_dataset,
    set_seed,
    setup_logging,
    project_paths,
    get_device,
    save_regression_report,
    ensure_dir,
    parse_common_args,
    save_metrics,
    dataset_missing_metrics,
    resolve_device_from_args,
    configure_cuda_allocator,
    run_metadata,
    dataset_fingerprint,
    write_split_manifest,
    EarlyStopping,
    auto_batch_and_accum,
    get_gpu_mem_bytes,
    compute_recsys_metrics,
    run_tabular_auto,
    make_tabular_splits,
)

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "gauthamvijayaraj/spotify-tracks-dataset-updated-every-week"
SEED = 42
EMBED_DIM = 64
HIDDEN_DIMS = [128, 64, 32]
BATCH_SIZE = 2048
EPOCHS = 10
LR = 1e-3
SAMPLE_LIMIT = 500_000

# ── Model ────────────────────────────────────────────────────────────────────

class NCF(nn.Module):
    """Neural Collaborative Filtering."""

    def __init__(self, n_users, n_items, embed_dim=64, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        layers = []
        in_dim = embed_dim * 2
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        x = torch.cat([u, i], dim=1)
        return self.fc(x).squeeze(1)


class InteractionDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


# ── Column detection helpers ─────────────────────────────────────────────────

USER_COLS = ["user_id", "userid", "user", "listener_id", "listener"]
ITEM_COLS = [
    "track_id", "trackid", "song_id", "songid", "track", "song",
    "item_id", "itemid", "track_name",
]
RATING_COLS = [
    "rating", "score", "play_count", "playcount", "listen_count",
    "streams", "popularity",
]
POPULARITY_COLS = ["popularity", "track_popularity", "pop", "streams"]

# Audio / content features typically present in Spotify-style datasets
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
    "key", "mode", "time_signature",
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _load_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.rglob("*.csv")) + sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CSV/parquet files under {data_dir}")

    best_file, best_score = files[0], -1
    for f in files:
        try:
            sample = (
                pd.read_csv(f, nrows=5) if f.suffix == ".csv"
                else pd.read_parquet(f).head(5)
            )
        except Exception:
            continue
        score = 0
        if _find_col(sample, USER_COLS):
            score += 2
        if _find_col(sample, ITEM_COLS):
            score += 1
        if _find_col(sample, RATING_COLS):
            score += 1
        if score > best_score:
            best_score = score
            best_file = f

    logger.info("Loading %s (match score %d)", best_file.name, best_score)
    if best_file.suffix == ".csv":
        return pd.read_csv(best_file, nrows=SAMPLE_LIMIT * 2)
    return pd.read_parquet(best_file)


def _has_user_item(df: pd.DataFrame) -> bool:
    """Return True if the dataframe has recognizable user AND item columns."""
    return _find_col(df, USER_COLS) is not None and _find_col(df, ITEM_COLS) is not None


def _prepare_ncf(df: pd.DataFrame):
    """Prepare user-item-rating triples for NCF training."""
    user_col = _find_col(df, USER_COLS)
    item_col = _find_col(df, ITEM_COLS)
    rating_col = _find_col(df, RATING_COLS)

    if rating_col is None:
        logger.info("No explicit rating column — using implicit 1.0")
        df["_rating"] = 1.0
        rating_col = "_rating"

    df = df[[user_col, item_col, rating_col]].dropna()
    if len(df) > SAMPLE_LIMIT:
        df = df.sample(SAMPLE_LIMIT, random_state=SEED)

    le_user = LabelEncoder()
    le_item = LabelEncoder()
    users = le_user.fit_transform(df[user_col].astype(str).values)
    items = le_item.fit_transform(df[item_col].astype(str).values)
    ratings = df[rating_col].values.astype(np.float32)
    return users, items, ratings, le_user.classes_.shape[0], le_item.classes_.shape[0]


# ── Training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for users, items, ratings in loader:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            preds = model(users, items)
            loss = criterion(preds, ratings)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(ratings)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    for users, items, ratings in loader:
        users, items, ratings = users.to(device), items.to(device), ratings.to(device)
        preds = model(users, items)
        preds_all.append(preds.cpu().numpy())
        targets_all.append(ratings.cpu().numpy())
    return np.concatenate(preds_all), np.concatenate(targets_all)


# ── PyCaret fallback ─────────────────────────────────────────────────────────

def _run_pycaret_fallback(df: pd.DataFrame, out_dir: Path, args) -> dict:
    """Content-based: predict popularity from audio features using auto-ML."""
    target_col = _find_col(df, POPULARITY_COLS)
    if target_col is None:
        # Pick first numeric column that could be a reasonable target
        for c in df.select_dtypes(include=[np.number]).columns:
            if df[c].nunique() > 5:
                target_col = c
                break
    if target_col is None:
        raise ValueError("No suitable target column found for regression.")

    # Keep only numeric / audio features + target
    keep = [c for c in df.columns if c.lower() in [a.lower() for a in AUDIO_FEATURES]]
    # Also include any other numeric columns
    for c in df.select_dtypes(include=[np.number]).columns:
        if c not in keep and c != target_col:
            keep.append(c)
    keep = [c for c in keep if c != target_col]
    if not keep:
        raise ValueError("No numeric features found for regression.")

    data = df[keep + [target_col]].dropna()
    if len(data) > 50_000:
        data = data.sample(50_000, random_state=args.seed)
    logger.info("Auto-ML fallback: %d rows, target=%s, %d features", len(data), target_col, len(keep))

    # Use proper train/val/test splits
    X_tr, y_tr, X_va, y_va, X_te, y_te = make_tabular_splits(
        data, target_col, task="regression", seed=args.seed,
    )
    splits = dict(X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va, X_test=X_te, y_test=y_te)
    ds_fp = dataset_fingerprint(out_dir.parent / "data")
    write_split_manifest(
        out_dir,
        dataset_fp=ds_fp,
        split_method="tabular_random",
        seed=args.seed,
        counts={"train": len(y_tr), "val": len(y_va), "test": len(y_te)},
    )
    metrics = run_tabular_auto(data, target=target_col, output_dir=out_dir, task="regression", session_id=args.seed, splits=splits)
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global EPOCHS, BATCH_SIZE, SAMPLE_LIMIT

    args = parse_common_args("Music Recommendation — NCF")
    paths = project_paths(__file__)
    setup_logging()
    set_seed(args.seed, deterministic=True)
    configure_cuda_allocator()
    device = resolve_device_from_args(args)
    logger.info("Device: %s", device)

    if args.mode == "smoke":
        EPOCHS = 1
        SAMPLE_LIMIT = 1000
    else:
        EPOCHS = args.epochs or 20
    BATCH_SIZE = args.batch_size or BATCH_SIZE

    # GPU budget
    BATCH_SIZE, grad_accum = auto_batch_and_accum(args.gpu_mem_gb, BATCH_SIZE)
    budget_bytes = get_gpu_mem_bytes(args.gpu_mem_gb)

    # Data
    data_dir = download_kaggle_dataset(KAGGLE_SLUG, paths["data"])
    if args.download_only:
        logger.info("Download complete — exiting (--download-only).")
        sys.exit(0)

    try:
        df = _load_data(data_dir)
    except Exception as exc:
        logger.warning("Dataset loading failed: %s", exc)
        dataset_missing_metrics(
            paths["outputs"], KAGGLE_SLUG,
            [f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}"],
        )
        return

    logger.info("Loaded dataframe: %s rows, %d cols", f"{len(df):,}", df.shape[1])

    out_dir = ensure_dir(paths["outputs"])

    # ── Decide pathway ───────────────────────────────────────────────────
    if _has_user_item(df):
        logger.info("User-item columns detected — training NCF.")
        users, items, ratings, n_users, n_items = _prepare_ncf(df)
        logger.info("Users=%d  Items=%d  Interactions=%d", n_users, n_items, len(ratings))

        # Dataset fingerprint
        ds_fp = dataset_fingerprint(data_dir)

        # Split — 70/15/15
        idx = np.arange(len(ratings))
        tr_val, te = train_test_split(idx, test_size=0.15, random_state=args.seed)
        tr, va = train_test_split(tr_val, test_size=0.15 / 0.85, random_state=args.seed)

        write_split_manifest(
            paths["outputs"],
            dataset_fp=ds_fp,
            split_method="interaction_random",
            seed=args.seed,
            counts={"train": len(tr), "val": len(va), "test": len(te)},
        )

        train_ds = InteractionDataset(users[tr], items[tr], ratings[tr])
        val_ds = InteractionDataset(users[va], items[va], ratings[va])
        test_ds = InteractionDataset(users[te], items[te], ratings[te])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = NCF(n_users, n_items, EMBED_DIM, HIDDEN_DIMS).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not args.no_amp) else None

        # Early stopping
        patience = args.patience if args.patience is not None else 5
        early_stop = EarlyStopping(patience=patience, mode="min")
        best_val_loss = float("inf")

        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_preds, val_targets = evaluate(model, val_loader, device)
            val_loss = float(np.mean((val_preds - val_targets) ** 2))
            logger.info("Epoch %02d/%d  train_loss=%.4f  val_loss=%.4f", epoch, EPOCHS, loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), out_dir / "ncf_best.pt")

            if early_stop(val_loss):
                logger.info("Early stopping at epoch %d", epoch)
                break

        # Load best checkpoint and evaluate on TEST set
        best_ckpt = out_dir / "ncf_best.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, weights_only=True))
        preds, targets = evaluate(model, test_loader, device)
        reg_metrics = save_regression_report(targets, preds, out_dir)

        # Ranking metrics
        test_users_arr = users[te]
        test_items_arr = items[te]
        median_rating = float(np.median(ratings[te]))
        y_true_per_user: dict = {}
        y_scores_per_user: dict = {}
        for u, i, r, p in zip(test_users_arr, test_items_arr, ratings[te], preds):
            u_int, i_int = int(u), int(i)
            if r >= median_rating:
                y_true_per_user.setdefault(u_int, []).append(i_int)
            y_scores_per_user.setdefault(u_int, []).append((i_int, float(p)))
        for u in y_scores_per_user:
            y_scores_per_user[u].sort(key=lambda x: x[1], reverse=True)
        ranking = compute_recsys_metrics(y_true_per_user, y_scores_per_user, k=10)

        torch.save(model.state_dict(), out_dir / "ncf_model.pt")
        logger.info("NCF model + reports saved to %s", out_dir)
        metrics = {**reg_metrics, **ranking}
    else:
        logger.info("No user-item columns — falling back to auto-ML content-based regression.")
        metrics = _run_pycaret_fallback(df, out_dir, args)

    # Final metrics
    metrics["run_metadata"] = run_metadata(args)
    metrics["n_test"] = metrics.get("n_test", 0)
    metrics["split"] = "test"
    save_metrics(out_dir, metrics, task_type="recsys", mode=args.mode)

    # ── Scatter / metrics plot ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics_file = out_dir / "regression_metrics.json"
        if metrics_file.exists():
            import json
            with open(metrics_file) as f:
                bar_metrics = json.load(f)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(list(bar_metrics.keys()), list(bar_metrics.values()))
            ax.set_title("Music Recommendation — Regression Metrics")
            fig.tight_layout()
            fig.savefig(out_dir / "metrics_bar.png", dpi=150)
            plt.close(fig)
            logger.info("Metrics bar chart saved.")
    except Exception as exc:
        logger.warning("Could not create plot: %s", exc)

    logger.info("Done.")


if __name__ == "__main__":
    main()
