#!/usr/bin/env python3
"""
Online News — Association Rule Mining
=======================================
Dataset : https://www.kaggle.com/datasets/snapcrack/all-the-news
Run     : python run.py

Extracts keywords from article titles and mines association rules among
co-occurring keywords within the same article.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import re
from collections import Counter

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
KAGGLE_SLUG = "snapcrack/all-the-news"
MIN_SUPPORT = 0.01
LIFT_THRESHOLD = 1.0
TOP_N_RULES = 50
TOP_N_KEYWORDS = 500          # vocabulary size (most frequent words)
MIN_KEYWORD_LEN = 4           # ignore very short tokens
KEYWORDS_PER_ARTICLE = 8      # max keywords kept per article


# ── stop-words (lightweight, no NLTK dependency) ────────────────────────────
_STOP_WORDS = frozenset(
    "a about above after again against all also am an and any are aren arent as at be "
    "because been before being below between both but by can cant could couldn couldnt "
    "did didn didnt do does doesn doesnt doing don dont down during each even few for "
    "from further get gets got had hadn hadnt has hasn hasnt have haven havent having "
    "he her here hers herself him himself his how however i if in into is isn isnt it "
    "its itself just know let like ll make me might more most much must my myself new "
    "no nor not now of off on once one only or other our ours ourselves out over own "
    "people re really right said same say says she should shouldn shouldnt since so "
    "some still such take than that the their theirs them then there these they think "
    "this those through time to too under until up upon us use used very want was wasn "
    "wasnt we well were weren werent what when where which while who whom why will with "
    "without won would wouldn wouldnt you your yours yourself yourselves year years "
    "first last many way going back came come made".split()
)


def tokenise(text: str) -> list[str]:
    """Lowercase tokenisation keeping only alpha tokens."""
    return [t for t in re.findall(r"[a-z]+", text.lower()) if len(t) >= MIN_KEYWORD_LEN and t not in _STOP_WORDS]


def load_transactions(data_dir: Path) -> list[list[str]]:
    """Load news CSVs, extract keyword transactions from titles."""
    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f, usecols=lambda c: c.lower() == "title", low_memory=False))
        except (ValueError, KeyError):
            # File may not have a 'title' column — try reading all and selecting
            tmp = pd.read_csv(f, low_memory=False)
            title_cols = [c for c in tmp.columns if "title" in c.lower()]
            if title_cols:
                frames.append(tmp[[title_cols[0]]].rename(columns={title_cols[0]: "title"}))

    if not frames:
        raise RuntimeError("Could not find a 'title' column in any CSV file.")

    df = pd.concat(frames, ignore_index=True).dropna(subset=["title"])
    logger.info("Articles loaded: %d", len(df))

    # ── build vocabulary from most frequent tokens ───────────────────────
    all_tokens = []
    for title in df["title"]:
        all_tokens.extend(tokenise(str(title)))

    vocab = {word for word, _ in Counter(all_tokens).most_common(TOP_N_KEYWORDS)}
    logger.info("Vocabulary size (top keywords): %d", len(vocab))

    # ── create transactions ──────────────────────────────────────────────
    transactions: list[list[str]] = []
    for title in df["title"]:
        tokens = [t for t in tokenise(str(title)) if t in vocab]
        # deduplicate while keeping order, then trim
        seen: set[str] = set()
        unique: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        if len(unique) >= 2:
            transactions.append(unique[:KEYWORDS_PER_ARTICLE])

    logger.info("Transactions created: %d", len(transactions))
    return transactions


def encode_transactions(transactions: list[list[str]]) -> pd.DataFrame:
    """One-hot encode transactions."""
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_array, columns=te.columns_)


def mine_rules(df_encoded: pd.DataFrame, algo: str = "apriori") -> pd.DataFrame:
    """Frequent-itemset mining → association rules."""
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
    """Save top rules to CSV."""
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
    args = parse_common_args("Online News – Association Rule Mining")
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
            output_dir, "Online News",
            ["https://www.kaggle.com/datasets/snapcrack/all-the-news"],
        )
        return

    if args.mode == "smoke":
        transactions = transactions[:500]

    df_encoded = encode_transactions(transactions)
    logger.info("Encoded matrix: %s", df_encoded.shape)

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
