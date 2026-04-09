#!/usr/bin/env python3
"""
Online Video Game Store — Association Rule Mining
===================================================
Dataset : https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
Run     : python run.py

Groups e-commerce events by user session, treats purchased/viewed product
categories as basket items, then mines association rules.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging

import pandas as pd

try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    from shared.utils import missing_dependency_metrics, project_paths
    missing_dependency_metrics(
        project_paths(__file__)["outputs"],
        missing=["mlxtend"],
        install_cmd="pip install mlxtend",
    )

from shared.utils import (
    parse_common_args,
    configure_cuda_allocator,
    run_metadata,
    save_metrics,
    dataset_fingerprint,
    write_split_manifest,
    dataset_missing_metrics,
    download_kaggle_dataset,
    set_seed,
    setup_logging,
    project_paths,
    ensure_dir,
    dataset_prompt,
)

logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
KAGGLE_SLUG = "mkechinov/ecommerce-behavior-data-from-multi-category-store"
MIN_SUPPORT = 0.01
LIFT_THRESHOLD = 1.0
TOP_N_RULES = 50
MAX_TRANSACTIONS = 100_000    # down-sample if dataset is huge


def _pick_item_column(df: pd.DataFrame) -> str:
    """Choose the best column to use as the 'item' identifier."""
    # Prefer category_code (more interpretable), fall back to product_id
    for candidate in ("category_code", "category", "product_id"):
        matches = [c for c in df.columns if candidate in c.lower()]
        if matches:
            col = matches[0]
            if df[col].notna().sum() > 0:
                return col
    raise KeyError("Cannot find a suitable item column in the dataset.")


def _pick_session_column(df: pd.DataFrame) -> str:
    """Choose the session / transaction grouping column."""
    for candidate in ("user_session", "session_id", "user_id", "order_id"):
        matches = [c for c in df.columns if candidate in c.lower()]
        if matches:
            return matches[0]
    raise KeyError("Cannot find a session/user column in the dataset.")


def load_transactions(data_dir: Path) -> list[list[str]]:
    """Load CSV(s), group by session, and return transaction lists."""
    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Read only needed columns to save memory
    logger.info("Scanning columns in %s ...", csv_files[0].name)
    sample = pd.read_csv(csv_files[0], nrows=5, low_memory=False)
    logger.info("Columns: %s", list(sample.columns))

    session_col = _pick_session_column(sample)
    item_col = _pick_item_column(sample)
    use_cols = [session_col, item_col]
    # Include event_type if present so we can optionally filter
    if "event_type" in sample.columns:
        use_cols.append("event_type")

    frames = []
    for f in csv_files:
        logger.info("Reading %s (cols=%s) ...", f.name, use_cols)
        chunk = pd.read_csv(f, usecols=use_cols, low_memory=False)
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    logger.info("Total rows loaded: %d", len(df))

    # Drop rows where item is missing
    df = df.dropna(subset=[item_col])
    df[item_col] = df[item_col].astype(str).str.strip()

    # Optional: keep only purchase events if the column exists
    if "event_type" in df.columns:
        purchase_df = df[df["event_type"].str.lower() == "purchase"]
        if len(purchase_df) > 1000:
            df = purchase_df
            logger.info("Filtered to purchase events: %d rows", len(df))
        else:
            logger.info("Few purchase events (%d) — using all event types.", len(purchase_df))

    # ── group by session ─────────────────────────────────────────────────
    grouped = df.groupby(session_col)[item_col].apply(lambda s: list(s.unique())).reset_index()
    # Keep only baskets with ≥ 2 items
    grouped = grouped[grouped[item_col].apply(len) >= 2]
    transactions: list[list[str]] = grouped[item_col].tolist()
    logger.info("Sessions with ≥ 2 items: %d", len(transactions))

    # ── down-sample if necessary ─────────────────────────────────────────
    if len(transactions) > MAX_TRANSACTIONS:
        logger.info("Down-sampling from %d to %d transactions.", len(transactions), MAX_TRANSACTIONS)
        import random
        random.seed(42)
        transactions = random.sample(transactions, MAX_TRANSACTIONS)

    return transactions


def encode_transactions(transactions: list[list[str]]) -> pd.DataFrame:
    """One-hot encode via TransactionEncoder."""
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)


def mine_rules(df_encoded: pd.DataFrame, algo: str = "apriori") -> pd.DataFrame:
    """Run frequent-itemset mining and generate association rules."""
    logger.info("Running %s with min_support=%.4f ...", algo, MIN_SUPPORT)
    func = fpgrowth if algo == "fpgrowth" else apriori
    freq = func(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    logger.info("Frequent itemsets found: %d", len(freq))

    if freq.empty:
        logger.warning("No frequent itemsets — try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(freq, metric="lift", min_threshold=LIFT_THRESHOLD)
    logger.info("Association rules generated: %d", len(rules))
    return rules


def save_rules(rules: pd.DataFrame, output_dir: Path, filename: str = "association_rules.csv") -> None:
    """Serialise frozensets and save to CSV."""
    df_out = rules.copy()
    df_out["antecedents"] = df_out["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    df_out["consequents"] = df_out["consequents"].apply(lambda x: ", ".join(sorted(x)))
    out_path = output_dir / filename
    df_out.sort_values("lift", ascending=False).head(TOP_N_RULES).to_csv(out_path, index=False)
    logger.info("Top %d rules saved → %s", TOP_N_RULES, out_path)


def print_summary(rules: pd.DataFrame) -> None:
    """Print summary statistics."""
    if rules.empty:
        print("\n⚠  No rules to display.")
        return
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"  Total rules        : {len(rules)}")
    print(f"  Avg support        : {rules['support'].mean():.4f}")
    print(f"  Avg confidence     : {rules['confidence'].mean():.4f}")
    print(f"  Avg lift           : {rules['lift'].mean():.4f}")
    print(f"  Max lift           : {rules['lift'].max():.4f}")
    print()
    top = rules.sort_values("lift", ascending=False).head(10)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        ant = ", ".join(sorted(r["antecedents"]))
        con = ", ".join(sorted(r["consequents"]))
        print(f"  [{i:>2}]  {ant}  →  {con}   (lift={r['lift']:.2f}, conf={r['confidence']:.2f})")
    print("=" * 70)


def main() -> None:
    setup_logging()
    args = parse_common_args("Online Video Game Store – Association Rule Mining")
    set_seed(args.seed)
    configure_cuda_allocator()
    dirs = project_paths(__file__)
    data_dir = dirs["data"]
    output_dir = ensure_dir(dirs["outputs"])

    # ── download-only ────────────────────────────────────────────────────
    if args.download_only:
        try:
            download_kaggle_dataset(KAGGLE_SLUG, data_dir)
            logger.info("Download complete.")
        except Exception as e:
            logger.error("Download failed: %s", e)
        sys.exit(0)

    # ── download ─────────────────────────────────────────────────────────
    download_kaggle_dataset(KAGGLE_SLUG, data_dir)

    # ── load & encode ────────────────────────────────────────────────────
    try:
        transactions = load_transactions(data_dir)
    except Exception:
        dataset_missing_metrics(
            output_dir, "Online Video Game Store",
            ["https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store"],
        )
        return

    if args.mode == "smoke":
        transactions = transactions[:500]

    df_encoded = encode_transactions(transactions)
    logger.info("Encoded basket matrix: %s", df_encoded.shape)

    # ── apriori ──────────────────────────────────────────────────────────
    rules = mine_rules(df_encoded, algo="apriori")
    if not rules.empty:
        save_rules(rules, output_dir, "association_rules.csv")
        print_summary(rules)

    # ── fpgrowth ─────────────────────────────────────────────────────────
    try:
        rules_fp = mine_rules(df_encoded, algo="fpgrowth")
        if not rules_fp.empty:
            save_rules(rules_fp, output_dir, "association_rules_fpgrowth.csv")
            print("\n[FP-Growth] rules generated:", len(rules_fp))
    except Exception as exc:
        logger.warning("FP-Growth skipped: %s", exc)
        rules_fp = pd.DataFrame()

    # ── split manifest ───────────────────────────────────────────────────
    write_split_manifest(
        output_dir,
        dataset_fp=dataset_fingerprint(data_dir),
        split_method="unsupervised_full",
        seed=args.seed,
        counts={"transactions": len(transactions)},
        extras={"note": "Association rule mining — no train/test split"},
    )

    # ── write metrics ────────────────────────────────────────────────────
    metrics = {
        "num_rules_apriori": len(rules) if not rules.empty else 0,
        "num_rules_fpgrowth": len(rules_fp) if not rules_fp.empty else 0,
    }
    all_rules = rules if not rules.empty else rules_fp
    if len(all_rules) > 0:
        metrics["avg_support"] = float(all_rules["support"].mean())
        metrics["avg_confidence"] = float(all_rules["confidence"].mean())
        metrics["avg_lift"] = float(all_rules["lift"].mean())
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(output_dir, metrics, task_type="association", mode=args.mode)

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
