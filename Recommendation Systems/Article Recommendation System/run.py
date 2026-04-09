#!/usr/bin/env python3
"""Article Recommendation — NCF

Dataset : https://www.kaggle.com/datasets/arashnic/mind-news-dataset
Approach: The MIND dataset contains user click behaviors / impressions.
          We extract user–article interactions from the behaviors file
          and train Neural Collaborative Filtering (NCF).
          Evaluated with MAE / RMSE / R².
Run     : python "Recommendation Systems/10 Article Recommendation System/run.py"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import re
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
)

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "arashnic/mind-news-dataset"
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

USER_COLS = ["user_id", "userid", "user", "impression_id", "viewer_id"]
ITEM_COLS = [
    "article_id", "articleid", "news_id", "newsid", "item_id",
    "itemid", "doc_id", "docid",
]
RATING_COLS = ["rating", "score", "click", "label", "clicked"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


# ── MIND-specific behavior parsing ───────────────────────────────────────────

def _parse_mind_behaviors(data_dir: Path) -> pd.DataFrame | None:
    """Parse MIND behaviors.tsv → user-article click interactions.

    The MIND behaviors file is TSV with columns:
    impression_id, user_id, time, history, impressions
    where impressions is like "N12345-1 N12346-0 ..." (newsid-click).
    """
    behavior_files = (
        list(data_dir.rglob("behaviors.tsv"))
        + list(data_dir.rglob("behaviors.csv"))
    )
    if not behavior_files:
        return None

    bf = behavior_files[0]
    logger.info("Parsing MIND behaviors from %s", bf)

    if bf.suffix == ".tsv":
        df = pd.read_csv(
            bf, sep="\t", header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"],
        )
    else:
        df = pd.read_csv(bf)
        # Normalise column names
        cols_lower = {c.lower(): c for c in df.columns}
        if "impressions" not in cols_lower and "impression" not in cols_lower:
            return None

    rows = []
    for _, row in df.iterrows():
        user = row.get("user_id") or row.get("User_ID") or row.iloc[1]
        impressions_str = str(row.get("impressions") or row.get("Impressions") or row.iloc[-1])
        if not impressions_str or impressions_str == "nan":
            continue
        for token in impressions_str.strip().split():
            parts = token.rsplit("-", 1)
            if len(parts) == 2:
                news_id, click = parts[0], parts[1]
                try:
                    rows.append({"user_id": user, "article_id": news_id, "click": int(click)})
                except ValueError:
                    continue
            else:
                # History entries (no click label) → implicit positive
                rows.append({"user_id": user, "article_id": token, "click": 1})

    if not rows:
        return None

    interactions = pd.DataFrame(rows)
    logger.info("Parsed %d interactions from MIND behaviors", len(interactions))
    return interactions


# ── Generic CSV loader ───────────────────────────────────────────────────────

def _load_data(data_dir: Path) -> pd.DataFrame:
    """Try MIND behaviors first, then fall back to generic CSV discovery."""
    mind_df = _parse_mind_behaviors(data_dir)
    if mind_df is not None and len(mind_df) > 0:
        return mind_df

    files = sorted(data_dir.rglob("*.csv")) + sorted(data_dir.rglob("*.parquet")) + sorted(data_dir.rglob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No data files under {data_dir}")

    best_file, best_score = files[0], -1
    for f in files:
        try:
            sep = "\t" if f.suffix == ".tsv" else ","
            sample = (
                pd.read_csv(f, nrows=5, sep=sep) if f.suffix != ".parquet"
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
    sep = "\t" if best_file.suffix == ".tsv" else ","
    if best_file.suffix == ".parquet":
        return pd.read_parquet(best_file)
    return pd.read_csv(best_file, sep=sep)


def _prepare(df: pd.DataFrame):
    user_col = _find_col(df, USER_COLS)
    item_col = _find_col(df, ITEM_COLS)
    rating_col = _find_col(df, RATING_COLS)

    if user_col is None or item_col is None:
        candidates = [c for c in df.columns if df[c].dtype == object or df[c].dtype.kind in "iu"]
        if len(candidates) < 2:
            raise ValueError("Cannot identify user and item columns.")
        user_col, item_col = candidates[0], candidates[1]
        logger.warning("Guessed user=%s, item=%s", user_col, item_col)

    if rating_col is None:
        logger.info("No explicit rating column — using implicit 1.0")
        df["_rating"] = 1.0
        rating_col = "_rating"

    df = df[[user_col, item_col, rating_col]].dropna()
    if len(df) > SAMPLE_LIMIT:
        df = df.sample(SAMPLE_LIMIT, random_state=SEED)
        logger.info("Sampled to %d rows", SAMPLE_LIMIT)

    le_user = LabelEncoder()
    le_item = LabelEncoder()
    users = le_user.fit_transform(df[user_col].astype(str).values)
    items = le_item.fit_transform(df[item_col].astype(str).values)
    ratings = df[rating_col].values.astype(np.float32)
    return users, items, ratings, le_user.classes_.shape[0], le_item.classes_.shape[0]


# ── Training ─────────────────────────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global EPOCHS, BATCH_SIZE, SAMPLE_LIMIT

    args = parse_common_args("Article Recommendation — NCF")
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
    users, items, ratings, n_users, n_items = _prepare(df)
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

    # Model
    model = NCF(n_users, n_items, EMBED_DIM, HIDDEN_DIMS).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and not args.no_amp) else None

    # Early stopping
    patience = args.patience if args.patience is not None else 5
    early_stop = EarlyStopping(patience=patience, mode="min")
    best_val_loss = float("inf")
    out_dir = ensure_dir(paths["outputs"])

    # Train with validation
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
    logger.info("Reports saved to %s", out_dir)

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

    # Save model
    torch.save(model.state_dict(), out_dir / "ncf_model.pt")
    logger.info("Model weights saved.")

    # Scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(targets, preds, alpha=0.3, s=4)
        mn, mx = min(targets.min(), preds.min()), max(targets.max(), preds.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Article Recommendation — Actual vs Predicted")
        fig.tight_layout()
        fig.savefig(out_dir / "scatter.png", dpi=150)
        plt.close(fig)
        logger.info("Scatter plot saved.")
    except ImportError:
        logger.warning("matplotlib not available — skipping scatter plot.")

    # Final metrics
    metrics = {**reg_metrics, **ranking}
    metrics["run_metadata"] = run_metadata(args)
    metrics["n_test"] = len(te)
    metrics["split"] = "test"
    save_metrics(out_dir, metrics, task_type="recsys", mode=args.mode)
    logger.info("Done.")


if __name__ == "__main__":
    main()
