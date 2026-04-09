#!/usr/bin/env python3
"""E-commerce Recommendation — NCF

Dataset : https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
Approach: Neural Collaborative Filtering on implicit-feedback events
          (view → 1, cart → 3, purchase → 5).  Evaluated with MAE / RMSE / R².
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
)

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "mkechinov/ecommerce-behavior-data-from-multi-category-store"
SEED = 42
EMBED_DIM = 64
HIDDEN_DIMS = [128, 64, 32]
BATCH_SIZE = 2048
EPOCHS = 10
LR = 1e-3
SAMPLE_LIMIT = 200_000

EVENT_SCORE = {"view": 1.0, "cart": 3.0, "add_to_cart": 3.0, "purchase": 5.0, "remove_from_cart": 0.5}

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


# ── Data helpers ─────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def _load_data(data_dir: Path) -> pd.DataFrame:
    """Load and merge CSVs from *data_dir*, sample to SAMPLE_LIMIT."""
    files = sorted(data_dir.rglob("*.csv")) + sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CSV/parquet files under {data_dir}")

    chunks: list[pd.DataFrame] = []
    remaining = SAMPLE_LIMIT
    for f in files:
        if remaining <= 0:
            break
        logger.info("Reading %s (need %d more rows)", f.name, remaining)
        try:
            if f.suffix == ".csv":
                chunk = pd.read_csv(f, nrows=remaining)
            else:
                chunk = pd.read_parquet(f)
                if len(chunk) > remaining:
                    chunk = chunk.head(remaining)
        except Exception as exc:
            logger.warning("Skipping %s: %s", f.name, exc)
            continue
        chunks.append(chunk)
        remaining -= len(chunk)

    df = pd.concat(chunks, ignore_index=True)
    logger.info("Loaded %s rows total", f"{len(df):,}")
    return df


def _prepare(df: pd.DataFrame):
    """Return (users_enc, items_enc, ratings, n_users, n_items)."""
    # Identify columns
    user_col = _find_col(df, ["user_id", "userid", "visitor_id", "user_session", "user"])
    item_col = _find_col(df, ["product_id", "productid", "item_id", "itemid", "sku"])
    event_col = _find_col(df, ["event_type", "eventtype", "action", "event"])

    if user_col is None or item_col is None:
        raise ValueError(f"Cannot find user/item columns in {list(df.columns)}")

    # Build rating from event type or fall back to implicit 1
    if event_col is not None:
        df["_rating"] = df[event_col].astype(str).str.lower().str.strip().map(EVENT_SCORE).fillna(1.0)
    else:
        rating_col = _find_col(df, ["rating", "score"])
        if rating_col:
            df["_rating"] = df[rating_col].astype(float)
        else:
            df["_rating"] = 1.0

    df = df[[user_col, item_col, "_rating"]].dropna()
    if len(df) > SAMPLE_LIMIT:
        df = df.sample(SAMPLE_LIMIT, random_state=SEED)

    le_user = LabelEncoder()
    le_item = LabelEncoder()
    users = le_user.fit_transform(df[user_col].astype(str).values)
    items = le_item.fit_transform(df[item_col].astype(str).values)
    ratings = df["_rating"].values.astype(np.float32)
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

    args = parse_common_args("E-commerce Recommendation — NCF")
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
        ax.set_title("E-commerce Recommendation — Actual vs Predicted")
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


if __name__ == "__main__":
    main()
