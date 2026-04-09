#!/usr/bin/env python3
"""
Grocery Store — Association Rule Mining
========================================
Dataset : https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis
Run     : python run.py
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
KAGGLE_SLUG = "aslanahmedov/market-basket-analysis"
MIN_SUPPORT = 0.01
LIFT_THRESHOLD = 1.0
TOP_N_RULES = 50


def load_transactions(data_dir: Path) -> list[list[str]]:
    """Load CSV and return a list of transactions (each a list of item strings)."""
    csv_files = list(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    csv_path = csv_files[0]
    logger.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Raw shape: %s  Columns: %s", df.shape, list(df.columns))

    # Strategy 1: If there's a single column with comma-separated items
    if df.shape[1] == 1:
        col = df.columns[0]
        transactions = [
            [item.strip() for item in str(row).split(",") if item.strip()]
            for row in df[col].dropna()
        ]
        return transactions

    # Strategy 2: If columns represent items (wide / one-hot-ish format)
    # or each row lists items across columns
    transactions = []
    for _, row in df.iterrows():
        items = [str(v).strip() for v in row.dropna().values if str(v).strip() and str(v).strip().lower() != "nan"]
        if items:
            transactions.append(items)

    return transactions


def encode_transactions(transactions: list[list[str]]) -> pd.DataFrame:
    """One-hot encode transactions using TransactionEncoder."""
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    return df_encoded


def mine_rules(df_encoded: pd.DataFrame, algo: str = "apriori") -> pd.DataFrame:
    """Run frequent-itemset mining and generate association rules."""
    logger.info("Running %s with min_support=%.4f ...", algo, MIN_SUPPORT)
    if algo == "fpgrowth":
        freq = fpgrowth(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    else:
        freq = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    logger.info("Frequent itemsets found: %d", len(freq))

    if freq.empty:
        logger.warning("No frequent itemsets — try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(freq, metric="lift", min_threshold=LIFT_THRESHOLD)
    logger.info("Association rules generated: %d", len(rules))
    return rules


def save_rules(rules: pd.DataFrame, output_dir: Path, filename: str = "association_rules.csv") -> None:
    """Serialise antecedents/consequents to strings and save to CSV."""
    df_out = rules.copy()
    df_out["antecedents"] = df_out["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    df_out["consequents"] = df_out["consequents"].apply(lambda x: ", ".join(sorted(x)))
    out_path = output_dir / filename
    df_out.head(TOP_N_RULES).to_csv(out_path, index=False)
    logger.info("Top %d rules saved → %s", TOP_N_RULES, out_path)


def print_summary(rules: pd.DataFrame) -> None:
    """Print summary statistics for the mined rules."""
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
    args = parse_common_args("Grocery Store – Association Rule Mining")
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
            output_dir, "Grocery Store",
            ["https://www.kaggle.com/datasets/aslanahmedov/market-basket-analysis"],
        )
        return

    if args.mode == "smoke":
        transactions = transactions[:500]

    logger.info("Transactions loaded: %d", len(transactions))
    df_encoded = encode_transactions(transactions)
    logger.info("Encoded basket matrix: %s", df_encoded.shape)

    # ── apriori ──────────────────────────────────────────────────────────
    rules_apriori = mine_rules(df_encoded, algo="apriori")
    if not rules_apriori.empty:
        save_rules(rules_apriori, output_dir, "association_rules.csv")
        print_summary(rules_apriori)

    # ── fpgrowth ─────────────────────────────────────────────────────────
    rules_fp = mine_rules(df_encoded, algo="fpgrowth")
    if not rules_fp.empty:
        save_rules(rules_fp, output_dir, "association_rules_fpgrowth.csv")
        print("\n[FP-Growth] rules generated:", len(rules_fp))

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
        "num_rules_apriori": len(rules_apriori) if not rules_apriori.empty else 0,
        "num_rules_fpgrowth": len(rules_fp) if not rules_fp.empty else 0,
    }
    all_rules = rules_apriori if not rules_apriori.empty else rules_fp
    if len(all_rules) > 0:
        metrics["avg_support"] = float(all_rules["support"].mean())
        metrics["avg_confidence"] = float(all_rules["confidence"].mean())
        metrics["avg_lift"] = float(all_rules["lift"].mean())
    metrics["run_metadata"] = run_metadata(args)
    save_metrics(output_dir, metrics, task_type="association", mode=args.mode)

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
