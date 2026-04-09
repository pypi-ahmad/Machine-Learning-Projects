#!/usr/bin/env python3
"""
Online Retail — Association Rule Mining
========================================
Dataset : https://www.kaggle.com/datasets/vijayuv/onlineretail
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
KAGGLE_SLUG = "vijayuv/onlineretail"
MIN_SUPPORT = 0.02
LIFT_THRESHOLD = 1.0
TOP_N_RULES = 50


def load_and_preprocess(data_dir: Path) -> pd.DataFrame:
    """Load the Online Retail transactional data and build a basket matrix."""
    # Try Excel first (original format), fall back to CSV
    xlsx_files = list(data_dir.rglob("*.xlsx")) + list(data_dir.rglob("*.xls"))
    csv_files = list(data_dir.rglob("*.csv"))

    if xlsx_files:
        path = xlsx_files[0]
        logger.info("Loading Excel file: %s", path)
        df = pd.read_excel(path)
    elif csv_files:
        path = csv_files[0]
        logger.info("Loading CSV file: %s", path)
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    else:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    logger.info("Raw shape: %s  Columns: %s", df.shape, list(df.columns))

    # ── basic cleaning ───────────────────────────────────────────────────
    # Standardise column names (handle slight variations)
    df.columns = df.columns.str.strip()
    invoice_col = [c for c in df.columns if "invoice" in c.lower()][0]
    desc_col = [c for c in df.columns if "description" in c.lower() or "stockcode" in c.lower()][0]
    qty_col = [c for c in df.columns if "quantity" in c.lower()]
    qty_col = qty_col[0] if qty_col else None

    # Drop cancellations (invoice starting with 'C')
    df = df[~df[invoice_col].astype(str).str.startswith("C")]
    df = df.dropna(subset=[invoice_col, desc_col])

    # Keep only positive quantities if column exists
    if qty_col:
        df = df[df[qty_col] > 0]

    # ── basket matrix ────────────────────────────────────────────────────
    logger.info("Building basket matrix grouped by %s ...", invoice_col)
    basket = (
        df.groupby([invoice_col, desc_col])[qty_col if qty_col else desc_col]
        .count()
        .unstack()
        .fillna(0)
    )
    # Convert counts → boolean (1/0)
    basket = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)
    logger.info("Basket matrix shape: %s", basket.shape)
    return basket


def mine_rules(basket: pd.DataFrame, algo: str = "apriori") -> pd.DataFrame:
    """Run frequent-itemset mining and generate association rules."""
    logger.info("Running %s with min_support=%.4f ...", algo, MIN_SUPPORT)
    if algo == "fpgrowth":
        freq = fpgrowth(basket, min_support=MIN_SUPPORT, use_colnames=True)
    else:
        freq = apriori(basket, min_support=MIN_SUPPORT, use_colnames=True)
    logger.info("Frequent itemsets found: %d", len(freq))

    if freq.empty:
        logger.warning("No frequent itemsets — try lowering min_support.")
        return pd.DataFrame()

    rules = association_rules(freq, metric="lift", min_threshold=LIFT_THRESHOLD)
    logger.info("Association rules generated: %d", len(rules))
    return rules


def save_rules(rules: pd.DataFrame, output_dir: Path, filename: str = "association_rules.csv") -> None:
    """Serialise antecedents/consequents and save top rules to CSV."""
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
    args = parse_common_args("Online Retail – Association Rule Mining")
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

    # ── load & build basket ──────────────────────────────────────────────
    try:
        basket = load_and_preprocess(data_dir)
    except Exception:
        dataset_missing_metrics(
            output_dir, "Online Retail",
            ["https://www.kaggle.com/datasets/vijayuv/onlineretail"],
        )
        return

    if args.mode == "smoke":
        basket = basket.head(500)

    # ── apriori ──────────────────────────────────────────────────────────
    rules = mine_rules(basket, algo="apriori")
    if not rules.empty:
        save_rules(rules, output_dir, "association_rules.csv")
        print_summary(rules)

    # ── fpgrowth (optional comparison) ───────────────────────────────────
    try:
        rules_fp = mine_rules(basket, algo="fpgrowth")
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
        counts={"transactions": len(basket)},
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
