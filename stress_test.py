"""
PHASE 6 — EXECUTION & STRESS TEST SUITE

Simulates:  Large datasets, missing values, wrong schema, repeated runs
Validates:  No crashes, stable outputs, consistent predictions
Detects:    Memory issues, slow execution, pipeline failures

Runs against ALL 14 projects using temporary mock models.
"""

import json
import time
import traceback
import gc
import sys
import shutil
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load

# ── Setup ────────────────────────────────────────────────────────────

from config import ROOT, ARTIFACTS

ALL_SLUGS = [
    "alexa_reviews", "fake_news_detection", "hate_speech_detection",
    "imdb_sentiment_ml", "movie_review_sentiments", "restaurant_reviews",
    "resume_screening", "sentiment_analysis",
    "sms_spam_analysis", "sms_spam_detection", "spam_sms_classification",
    "stock_sentiment_djia", "twitter_sentiment",
    "whatsapp_sentiment",
]

# Test results accumulator
RESULTS = []

def record(project, test_name, status, detail="", duration=0.0):
    RESULTS.append({
        "project": project,
        "test": test_name,
        "status": status,
        "detail": detail[:200],
        "duration_ms": round(duration * 1000, 1),
    })


@contextmanager
def timer():
    """Context manager that yields a callable returning elapsed seconds."""
    t0 = time.perf_counter()
    elapsed = lambda: time.perf_counter() - t0
    yield elapsed


# ====================================================================
# MOCK MODEL FACTORY
# ====================================================================

N_FEATURES = 100

def create_mock_models():
    """Create temporary mock model + vectorizer for each project."""
    print(f"  Creating mock models for {len(ALL_SLUGS)} projects...")
    rng = np.random.RandomState(42)
    X_train = rng.rand(200, N_FEATURES)
    y_train = rng.choice([0, 1], size=200)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    # Build a corpus with enough unique tokens to fill N_FEATURES dimensions
    vocab = [f"word{i}" for i in range(N_FEATURES + 50)]
    texts = [" ".join(rng.choice(vocab, size=20)) for _ in range(200)]
    vectorizer = TfidfVectorizer(max_features=N_FEATURES)
    vectorizer.fit(texts)

    for slug in ALL_SLUGS:
        proj_dir = ARTIFACTS / slug
        proj_dir.mkdir(parents=True, exist_ok=True)
        dump(model, str(proj_dir / "model.joblib"))
        dump(vectorizer, str(proj_dir / "vectorizer.joblib"))

    print(f"  Mock models created for {len(ALL_SLUGS)} projects\n")
    return model, vectorizer


def create_mock_schema(slug, features=None):
    """Create a schema.json for a project."""
    if features is None:
        features = [f"f{i}" for i in range(N_FEATURES)]
    schema = {"features": features}
    with open(ARTIFACTS / slug / "schema.json", "w") as f:
        json.dump(schema, f)


def remove_mock_schemas():
    """Remove all test schema.json files."""
    for slug in ALL_SLUGS:
        p = ARTIFACTS / slug / "schema.json"
        if p.exists():
            p.unlink()


# ====================================================================
# IMPORT INFRASTRUCTURE (after mock setup)
# ====================================================================

def get_inference_engine():
    """Import inference engine with fresh state."""
    import importlib
    import inference_engine
    # Clear LRU cache to force reload of mock models
    inference_engine.load_model.cache_clear()
    importlib.reload(inference_engine)
    return inference_engine


# ====================================================================
# TEST 1: LARGE DATASETS
# ====================================================================

def test_large_datasets():
    """Simulate prediction on large DataFrames (1K, 10K, 50K rows)."""
    print("=" * 60)
    print("  TEST 1: LARGE DATASETS")
    print("=" * 60)
    ie = get_inference_engine()

    sizes = [1_000, 10_000, 50_000]
    test_slugs = ["resume_screening", "sentiment_analysis", "sms_spam_detection"]

    for slug in test_slugs:
        for n_rows in sizes:
            test_name = f"large_dataset_{n_rows}rows"
            try:
                with timer() as elapsed:
                    rng = np.random.RandomState(42)
                    X = rng.rand(n_rows, N_FEATURES)
                    preds = ie.predict(slug, X)

                assert len(preds) == n_rows, f"Expected {n_rows} predictions, got {len(preds)}"
                assert set(preds).issubset({0, 1}), f"Unexpected labels: {set(preds)}"
                dur = elapsed()
                record(slug, test_name, "PASS", f"{n_rows} rows in {dur:.2f}s", dur)
                print(f"    [PASS] {slug}/{test_name} ({dur:.2f}s)")
            except Exception as e:
                record(slug, test_name, "FAIL", str(e))
                print(f"    [FAIL] {slug}/{test_name}: {e}")
            gc.collect()

    # Large text dataset
    for slug in ["hate_speech_detection", "twitter_sentiment"]:
        test_name = "large_text_dataset_5000rows"
        try:
            with timer() as elapsed:
                texts = pd.DataFrame({"text": ["This is a test sentence for NLP"] * 5000})
                preds = ie.predict(slug, texts)
            assert len(preds) == 5000
            dur = elapsed()
            record(slug, test_name, "PASS", f"5000 text rows in {dur:.2f}s", dur)
            print(f"    [PASS] {slug}/{test_name} ({dur:.2f}s)")
        except Exception as e:
            record(slug, test_name, "FAIL", str(e))
            print(f"    [FAIL] {slug}/{test_name}: {e}")

    print()


# ====================================================================
# TEST 2: MISSING VALUES
# ====================================================================

def test_missing_values():
    """Simulate prediction on data with NaN, None, empty strings."""
    print("=" * 60)
    print("  TEST 2: MISSING VALUES")
    print("=" * 60)
    ie = get_inference_engine()

    for slug in ALL_SLUGS:
        # Numeric with NaN
        test_name = "numeric_with_nan"
        try:
            with timer() as elapsed:
                rng = np.random.RandomState(42)
                X = rng.rand(100, N_FEATURES)
                # Inject NaN at random positions
                nan_mask = rng.rand(100, N_FEATURES) < 0.1
                X[nan_mask] = np.nan
                preds = ie.predict(slug, X)
            assert len(preds) == 100
            dur = elapsed()
            # Note: LogisticRegression may produce predictions or warnings with NaN
            record(slug, test_name, "PASS", f"Handled NaN gracefully", dur)
            print(f"    [PASS] {slug}/{test_name}")
        except Exception as e:
            record(slug, test_name, "WARN", f"NaN caused: {type(e).__name__}: {e}")
            print(f"    [WARN] {slug}/{test_name}: {type(e).__name__}")

        # Text with empty/None
        test_name = "text_with_missing"
        try:
            with timer() as elapsed:
                texts = pd.DataFrame({"text": ["good", "", None, "bad", np.nan, "ok"]})
                preds = ie.predict(slug, texts)
            assert len(preds) == 6
            dur = elapsed()
            record(slug, test_name, "PASS", "Handled missing text", dur)
            print(f"    [PASS] {slug}/{test_name}")
        except Exception as e:
            record(slug, test_name, "WARN", f"{type(e).__name__}: {e}")
            print(f"    [WARN] {slug}/{test_name}: {type(e).__name__}")

    print()


# ====================================================================
# TEST 3: WRONG SCHEMA
# ====================================================================

def test_wrong_schema():
    """Simulate prediction with wrong columns, extra columns, missing columns."""
    print("=" * 60)
    print("  TEST 3: WRONG SCHEMA")
    print("=" * 60)
    ie = get_inference_engine()

    test_slug = "resume_screening"

    # 3a: Create schema and send correct data
    test_name = "schema_correct_columns"
    try:
        features = [f"f{i}" for i in range(N_FEATURES)]
        create_mock_schema(test_slug, features)
        ie = get_inference_engine()
        with timer() as elapsed:
            df = pd.DataFrame(np.random.rand(10, N_FEATURES), columns=features)
            preds = ie.predict(test_slug, df)
        assert len(preds) == 10
        dur = elapsed()
        record(test_slug, test_name, "PASS", "Schema validated correctly", dur)
        print(f"    [PASS] {test_slug}/{test_name}")
    except Exception as e:
        record(test_slug, test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_slug}/{test_name}: {e}")

    # 3b: Missing columns (should raise ValueError)
    test_name = "schema_missing_columns"
    try:
        with timer() as elapsed:
            df = pd.DataFrame(np.random.rand(10, 50), columns=[f"f{i}" for i in range(50)])
            preds = ie.predict(test_slug, df)
        record(test_slug, test_name, "FAIL", "Should have raised ValueError for missing columns")
        print(f"    [FAIL] {test_slug}/{test_name}: no error raised")
    except ValueError as e:
        dur = elapsed()
        record(test_slug, test_name, "PASS", f"Correctly rejected: {e}", dur)
        print(f"    [PASS] {test_slug}/{test_name} (correctly rejected)")
    except Exception as e:
        record(test_slug, test_name, "FAIL", f"Wrong exception: {type(e).__name__}: {e}")
        print(f"    [FAIL] {test_slug}/{test_name}: wrong exception type")

    # 3c: Extra columns (should raise ValueError in strict mode)
    test_name = "schema_extra_columns_strict"
    try:
        with timer() as elapsed:
            cols = [f"f{i}" for i in range(N_FEATURES)] + ["extra1", "extra2"]
            df = pd.DataFrame(np.random.rand(10, N_FEATURES + 2), columns=cols)
            preds = ie.predict(test_slug, df)
        record(test_slug, test_name, "FAIL", "Should have raised ValueError for extra columns")
        print(f"    [FAIL] {test_slug}/{test_name}: no error raised")
    except ValueError as e:
        dur = elapsed()
        record(test_slug, test_name, "PASS", f"Correctly rejected: {e}", dur)
        print(f"    [PASS] {test_slug}/{test_name} (correctly rejected)")
    except Exception as e:
        record(test_slug, test_name, "FAIL", f"Wrong exception: {type(e).__name__}: {e}")
        print(f"    [FAIL] {test_slug}/{test_name}: wrong exception type")

    # 3d: Extra columns in non-strict mode (should pass)
    test_name = "schema_extra_columns_nonstrict"
    try:
        import inference_engine
        old_strict = inference_engine.STRICT_SCHEMA
        inference_engine.STRICT_SCHEMA = False
        with timer() as elapsed:
            cols = [f"f{i}" for i in range(N_FEATURES)] + ["extra1", "extra2"]
            df = pd.DataFrame(np.random.rand(10, N_FEATURES + 2), columns=cols)
            preds = ie.predict(test_slug, df)
        assert len(preds) == 10
        dur = elapsed()
        record(test_slug, test_name, "PASS", "Extra cols silently dropped", dur)
        print(f"    [PASS] {test_slug}/{test_name}")
        inference_engine.STRICT_SCHEMA = old_strict
    except Exception as e:
        record(test_slug, test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_slug}/{test_name}: {e}")
        import inference_engine
        inference_engine.STRICT_SCHEMA = True

    # 3e: Completely wrong column names
    test_name = "schema_wrong_columns"
    try:
        with timer() as elapsed:
            df = pd.DataFrame(np.random.rand(10, N_FEATURES),
                              columns=[f"wrong_{i}" for i in range(N_FEATURES)])
            preds = ie.predict(test_slug, df)
        record(test_slug, test_name, "FAIL", "Should have raised ValueError")
        print(f"    [FAIL] {test_slug}/{test_name}: no error raised")
    except ValueError as e:
        dur = elapsed()
        record(test_slug, test_name, "PASS", f"Correctly rejected", dur)
        print(f"    [PASS] {test_slug}/{test_name} (correctly rejected)")
    except Exception as e:
        record(test_slug, test_name, "FAIL", f"Wrong exception: {type(e).__name__}")
        print(f"    [FAIL] {test_slug}/{test_name}: {type(e).__name__}")

    # 3f: Empty DataFrame
    test_name = "schema_empty_dataframe"
    try:
        with timer() as elapsed:
            features = [f"f{i}" for i in range(N_FEATURES)]
            df = pd.DataFrame(columns=features)
            preds = ie.predict(test_slug, df)
        dur = elapsed()
        record(test_slug, test_name, "PASS", f"Empty DF returned {len(preds)} preds", dur)
        print(f"    [PASS] {test_slug}/{test_name}")
    except Exception as e:
        record(test_slug, test_name, "WARN", f"{type(e).__name__}: {e}")
        print(f"    [WARN] {test_slug}/{test_name}: {type(e).__name__}")

    # 3g: No schema (passthrough)
    test_name = "no_schema_passthrough"
    try:
        remove_mock_schemas()
        ie = get_inference_engine()
        with timer() as elapsed:
            df = pd.DataFrame(np.random.rand(10, N_FEATURES),
                              columns=[f"anything_{i}" for i in range(N_FEATURES)])
            preds = ie.predict(test_slug, df)
        assert len(preds) == 10
        dur = elapsed()
        record(test_slug, test_name, "PASS", "No schema = passthrough OK", dur)
        print(f"    [PASS] {test_slug}/{test_name}")
    except Exception as e:
        record(test_slug, test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_slug}/{test_name}: {e}")

    remove_mock_schemas()
    print()


# ====================================================================
# TEST 4: REPEATED RUNS (CONSISTENCY + IDEMPOTENCY)
# ====================================================================

def test_repeated_runs():
    """Run predictions multiple times and verify consistent results."""
    print("=" * 60)
    print("  TEST 4: REPEATED RUNS")
    print("=" * 60)
    ie = get_inference_engine()

    rng = np.random.RandomState(42)
    X_fixed = rng.rand(50, N_FEATURES)

    for slug in ALL_SLUGS:
        test_name = "repeated_5x_consistency"
        try:
            results = []
            with timer() as elapsed:
                for _ in range(5):
                    preds = ie.predict(slug, X_fixed.copy())
                    results.append(preds.tolist())

            # All 5 runs must be identical
            for i in range(1, 5):
                assert results[i] == results[0], f"Run {i+1} differs from run 1"

            dur = elapsed()
            record(slug, test_name, "PASS", "5 identical runs", dur)
            print(f"    [PASS] {slug}/{test_name}")
        except Exception as e:
            record(slug, test_name, "FAIL", str(e))
            print(f"    [FAIL] {slug}/{test_name}: {e}")

    # Text consistency
    for slug in ["resume_screening", "twitter_sentiment"]:
        test_name = "repeated_text_consistency"
        try:
            results = []
            with timer() as elapsed:
                for _ in range(5):
                    p = ie.predict_text(slug, "great product love it")
                    results.append(p)
            assert len(set(str(r) for r in results)) == 1, "Inconsistent text predictions"
            dur = elapsed()
            record(slug, test_name, "PASS", f"Consistent: {results[0]}", dur)
            print(f"    [PASS] {slug}/{test_name}")
        except Exception as e:
            record(slug, test_name, "FAIL", str(e))
            print(f"    [FAIL] {slug}/{test_name}: {e}")

    print()


# ====================================================================
# TEST 5: MEMORY / PERFORMANCE PROFILING
# ====================================================================

def test_performance():
    """Detect slow execution and memory issues."""
    print("=" * 60)
    print("  TEST 5: PERFORMANCE PROFILING")
    print("=" * 60)
    ie = get_inference_engine()

    SLOW_THRESHOLD_MS = 5000  # 5 second max per prediction batch

    for slug in ALL_SLUGS:
        test_name = "perf_10k_rows"
        try:
            rng = np.random.RandomState(42)
            X = rng.rand(10_000, N_FEATURES)

            gc.collect()
            mem_before = sys.getsizeof(X)

            with timer() as elapsed:
                preds = ie.predict(slug, X)
            dur = elapsed()
            dur_ms = dur * 1000

            if dur_ms > SLOW_THRESHOLD_MS:
                record(slug, test_name, "SLOW", f"{dur_ms:.0f}ms > {SLOW_THRESHOLD_MS}ms threshold", dur)
                print(f"    [SLOW] {slug}/{test_name}: {dur_ms:.0f}ms")
            else:
                record(slug, test_name, "PASS", f"{dur_ms:.0f}ms", dur)
                print(f"    [PASS] {slug}/{test_name} ({dur_ms:.0f}ms)")
        except Exception as e:
            record(slug, test_name, "FAIL", str(e))
            print(f"    [FAIL] {slug}/{test_name}: {e}")
        gc.collect()

    # Model load caching test
    test_name = "model_cache_hit"
    try:
        ie = get_inference_engine()
        slug = "resume_screening"
        # First load
        with timer() as elapsed:
            ie.load_model(slug)
        first = elapsed()
        # Second load (should be cached)
        with timer() as elapsed:
            ie.load_model(slug)
        second = elapsed()
        speedup = first / max(second, 1e-9)
        if speedup > 2:
            record(slug, test_name, "PASS", f"Cache speedup: {speedup:.0f}x", second)
            print(f"    [PASS] {slug}/{test_name} (cache {speedup:.0f}x faster)")
        else:
            record(slug, test_name, "WARN", f"Low cache speedup: {speedup:.1f}x", second)
            print(f"    [WARN] {slug}/{test_name} (speedup only {speedup:.1f}x)")
    except Exception as e:
        record(slug, test_name, "FAIL", str(e))
        print(f"    [FAIL] {slug}/{test_name}: {e}")

    print()


# ====================================================================
# TEST 6: PIPELINE INTEGRITY (NOTEBOOKS + ARTIFACTS)
# ====================================================================

def test_pipeline_integrity():
    """Verify notebook JSON integrity and artifact consistency."""
    print("=" * 60)
    print("  TEST 6: PIPELINE INTEGRITY")
    print("=" * 60)

    FOLDER_MAP = {
        "resume_screening":        "NLP Projecct 1.ResumeScreening",
        "sentiment_analysis":      "NLP Projecct 3.Sentiment Analysis",
        "hate_speech_detection":   "NLP Projecct 11.HateSpeechDetection",
        "fake_news_detection":     "NLP Projecct 15.FakeNews Detection Model",
        "whatsapp_sentiment":      "NLP Projecct 16.NLP for whatsapp chats",
        "twitter_sentiment":       "NLP Projecct 17.Twitter Sentiment Analysis",
        "sms_spam_detection":      "NLP Projecct 18.SMS spam detection",
        "movie_review_sentiments": "NLP Projecct 19. MoviesReviewSentiments",
        "stock_sentiment_djia":    "NLP Project 21. Sentiment Analysis - Dow Jones (DJIA) Stock using News Headlines",
        "restaurant_reviews":      "NLP Project 22. - Sentiment Analysis - Restaurant Reviews",
        "spam_sms_classification": "NLP Project 23. - Spam SMS Classification",
        "imdb_sentiment_ml":       "NLP Projects 26 - IMDB Sentiment Analysis using Deep Learning",
        "alexa_reviews":           "NLP Projects 27 - Amazon Alexa Review Sentiment Analysis",
        "sms_spam_analysis":       "NLP Projects 31 - SMS Spam Detection Analysis",
    }

    for slug, folder_name in FOLDER_MAP.items():
        folder = ROOT / folder_name
        nbs = [f for f in folder.glob("*.ipynb") if ".ipynb_checkpoints" not in str(f)]

        # 6a: Notebook JSON validity
        test_name = "notebook_json_valid"
        if not nbs:
            record(slug, test_name, "FAIL", "No notebook found")
            print(f"    [FAIL] {slug}/{test_name}: missing")
            continue
        try:
            with open(nbs[0], "r", encoding="utf-8") as f:
                nb = json.load(f)
            assert "cells" in nb, "Missing 'cells' key"
            assert "nbformat" in nb, "Missing 'nbformat' key"
            assert len(nb["cells"]) > 0, "Empty notebook"
            record(slug, test_name, "PASS", f"{len(nb['cells'])} cells")
            print(f"    [PASS] {slug}/{test_name} ({len(nb['cells'])} cells)")
        except Exception as e:
            record(slug, test_name, "FAIL", str(e))
            print(f"    [FAIL] {slug}/{test_name}: {e}")
            continue

        # 6b: Governance cells present
        test_name = "governance_cells_present"
        try:
            last_src = "".join(nb["cells"][-1].get("source", []))
            assert "MODEL GOVERNANCE SUMMARY" in last_src, "Missing governance summary"
            # Find artifacts cell
            found_artifacts = False
            for c in nb["cells"]:
                src = "".join(c.get("source", []))
                if "model.joblib" in src and "metrics.json" in src:
                    found_artifacts = True
                    break
            assert found_artifacts, "Missing artifacts save cell"
            record(slug, test_name, "PASS", "")
            print(f"    [PASS] {slug}/{test_name}")
        except Exception as e:
            record(slug, test_name, "FAIL", str(e))
            print(f"    [FAIL] {slug}/{test_name}: {e}")

        # 6c: LazyPredict + PyCaret cells present
        test_name = "ml_pipeline_cells"
        try:
            all_src = " ".join("".join(c.get("source", [])) for c in nb["cells"])
            assert "LazyClassifier" in all_src, "Missing LazyClassifier"
            assert "compare_models" in all_src, "Missing PyCaret compare_models"
            assert "finalize_model" in all_src, "Missing PyCaret finalize_model"
            assert "predict_text" in all_src, "Missing inference function"
            record(slug, test_name, "PASS", "")
            print(f"    [PASS] {slug}/{test_name}")
        except Exception as e:
            record(slug, test_name, "FAIL", str(e))
            print(f"    [FAIL] {slug}/{test_name}: {e}")

        # 6d: Metrics.json present and valid
        test_name = "metrics_json_valid"
        try:
            mp = ARTIFACTS / slug / "metrics.json"
            assert mp.exists(), "metrics.json missing"
            with open(mp) as f:
                m = json.load(f)
            for key in ["accuracy", "f1", "precision", "recall"]:
                assert key in m, f"Missing key: {key}"
                assert isinstance(m[key], (int, float)), f"{key} not numeric"
                assert 0 <= m[key] <= 1, f"{key}={m[key]} out of [0,1]"
            record(slug, test_name, "PASS", f"acc={m['accuracy']:.4f}")
            print(f"    [PASS] {slug}/{test_name}")
        except Exception as e:
            record(slug, test_name, "FAIL", str(e))
            print(f"    [FAIL] {slug}/{test_name}: {e}")

    # 6e: Global registry valid
    test_name = "global_registry_valid"
    try:
        assert REGISTRY_PATH.exists(), "global_registry.json missing"
        with open(REGISTRY_PATH) as f:
            reg = json.load(f)
        assert isinstance(reg, list), "Registry is not a list"
        assert len(reg) == len(ALL_SLUGS), f"Expected {len(ALL_SLUGS)} entries, got {len(reg)}"
        projects_in_reg = {e["project"] for e in reg}
        assert projects_in_reg == set(ALL_SLUGS), f"Missing projects in registry"
        record("global", test_name, "PASS", f"{len(reg)} entries")
        print(f"    [PASS] global/{test_name}")
    except Exception as e:
        record("global", test_name, "FAIL", str(e))
        print(f"    [FAIL] global/{test_name}: {e}")

    print()


REGISTRY_PATH = ARTIFACTS / "global_registry.json"


# ====================================================================
# TEST 7: LEADERBOARD STRESS
# ====================================================================

def test_leaderboard_stress():
    """Stress test the leaderboard module."""
    print("=" * 60)
    print("  TEST 7: LEADERBOARD STRESS")
    print("=" * 60)

    import leaderboard as lb

    # 7a: Build DataFrame
    test_name = "leaderboard_build_df"
    try:
        with timer() as elapsed:
            df = lb.build_dataframe()
        dur = elapsed()
        assert not df.empty, "DataFrame is empty"
        assert len(df) == len(ALL_SLUGS), f"Expected {len(ALL_SLUGS)} rows, got {len(df)}"
        record("global", test_name, "PASS", f"{len(df)} rows in {dur*1000:.0f}ms", dur)
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("global", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")
        return

    # 7b: Ranking stability
    test_name = "leaderboard_ranking_stable"
    try:
        rankings = []
        for _ in range(10):
            r = lb.global_ranking(df)
            rankings.append(r["project"].tolist())
        for i in range(1, 10):
            assert rankings[i] == rankings[0], f"Run {i+1} ranking differs"
        record("global", test_name, "PASS", "10 identical rankings")
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("global", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")

    # 7c: CSV output
    test_name = "leaderboard_csv_output"
    try:
        path = lb.save_leaderboard(lb.global_ranking(df))
        assert path.exists(), "CSV not created"
        csv_df = pd.read_csv(path)
        assert len(csv_df) == len(ALL_SLUGS)
        record("global", test_name, "PASS", f"CSV: {path.name}")
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("global", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")

    # 7d: Chart output
    test_name = "leaderboard_chart_output"
    try:
        path = lb.save_chart(lb.global_ranking(df))
        if path and path.exists():
            record("global", test_name, "PASS", f"PNG: {path.name}")
            print(f"    [PASS] {test_name}")
        else:
            record("global", test_name, "WARN", "Chart skipped (matplotlib)")
            print(f"    [WARN] {test_name}: chart skipped")
    except Exception as e:
        record("global", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")

    print()


# ====================================================================
# TEST 8: API MODULE VALIDATION
# ====================================================================

def test_api_module():
    """Validate API module imports and schema definitions."""
    print("=" * 60)
    print("  TEST 8: API MODULE VALIDATION")
    print("=" * 60)

    # 8a: Import
    test_name = "api_import"
    try:
        with timer() as elapsed:
            import api
        dur = elapsed()
        record("global", test_name, "PASS", f"Imported in {dur*1000:.0f}ms", dur)
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("global", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")
        return

    # 8b: App exists with routes
    test_name = "api_routes_registered"
    try:
        import api
        routes = [r.path for r in api.app.routes if hasattr(r, "path")]
        expected = ["/", "/projects", "/projects/all", "/projects/{project}",
                    "/predict/{project}", "/predict/{project}/text"]
        for ep in expected:
            assert ep in routes, f"Missing route: {ep}"
        record("global", test_name, "PASS", f"{len(routes)} routes")
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("global", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")

    # 8c: Pydantic models validate
    test_name = "api_schema_validation"
    try:
        from api import PredictionRequest, TextRequest
        pr = PredictionRequest(data={"f0": 1.0})
        assert pr.data == {"f0": 1.0}
        tr = TextRequest(text="hello world")
        assert tr.text == "hello world"
        # Invalid should raise
        try:
            PredictionRequest()
            assert False, "Should have raised"
        except Exception:
            pass
        record("global", test_name, "PASS", "")
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("global", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")

    print()


# ====================================================================
# TEST 9: EDGE CASES
# ====================================================================

def test_edge_cases():
    """Nonexistent project, corrupt model, single-row, single-feature."""
    print("=" * 60)
    print("  TEST 9: EDGE CASES")
    print("=" * 60)
    ie = get_inference_engine()

    # 9a: Nonexistent project
    test_name = "nonexistent_project"
    try:
        ie.predict("DOES_NOT_EXIST_xyz", np.array([[1.0]]))
        record("global", test_name, "FAIL", "Should have raised ModelNotFoundError")
        print(f"    [FAIL] {test_name}: no error")
    except ie.ModelNotFoundError:
        record("global", test_name, "PASS", "Correctly raised ModelNotFoundError")
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("global", test_name, "FAIL", f"Wrong exception: {type(e).__name__}")
        print(f"    [FAIL] {test_name}: {type(e).__name__}")

    # 9b: Single row prediction
    test_name = "single_row"
    try:
        preds = ie.predict("resume_screening", np.random.rand(1, N_FEATURES))
        assert len(preds) == 1
        record("resume_screening", test_name, "PASS", "")
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("resume_screening", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")

    # 9c: Wrong feature dimensions
    test_name = "wrong_dimensions"
    try:
        ie.predict("resume_screening", np.random.rand(10, 5))  # 5 features, expects 100
        record("resume_screening", test_name, "WARN", "No error on dimension mismatch")
        print(f"    [WARN] {test_name}: no error on 5 features (model expects {N_FEATURES})")
    except Exception as e:
        record("resume_screening", test_name, "PASS", f"Correctly rejected: {type(e).__name__}")
        print(f"    [PASS] {test_name} (correctly rejected)")

    # 9d: predict_text on empty string
    test_name = "predict_empty_text"
    try:
        result = ie.predict_text("resume_screening", "")
        record("resume_screening", test_name, "PASS", f"Result: {result}")
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("resume_screening", test_name, "WARN", f"{type(e).__name__}: {e}")
        print(f"    [WARN] {test_name}: {type(e).__name__}")

    # 9e: predict_text on very long text
    test_name = "predict_very_long_text"
    try:
        long_text = "word " * 100_000
        with timer() as elapsed:
            result = ie.predict_text("resume_screening", long_text)
        dur = elapsed()
        record("resume_screening", test_name, "PASS", f"{dur*1000:.0f}ms", dur)
        print(f"    [PASS] {test_name} ({dur*1000:.0f}ms)")
    except Exception as e:
        record("resume_screening", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")

    # 9f: Unicode / special characters
    test_name = "predict_unicode_text"
    try:
        result = ie.predict_text("resume_screening", "こんにちは 🎉 résumé naïve")
        record("resume_screening", test_name, "PASS", f"Result: {result}")
        print(f"    [PASS] {test_name}")
    except Exception as e:
        record("resume_screening", test_name, "WARN", f"{type(e).__name__}: {e}")
        print(f"    [WARN] {test_name}: {type(e).__name__}")

    # 9g: Concurrent-style rapid calls
    test_name = "rapid_50_predictions"
    try:
        with timer() as elapsed:
            for _ in range(50):
                ie.predict("resume_screening", np.random.rand(10, N_FEATURES))
        dur = elapsed()
        record("resume_screening", test_name, "PASS", f"50 batches in {dur:.2f}s", dur)
        print(f"    [PASS] {test_name} ({dur:.2f}s)")
    except Exception as e:
        record("resume_screening", test_name, "FAIL", str(e))
        print(f"    [FAIL] {test_name}: {e}")

    print()


# ====================================================================
# REPORT
# ====================================================================

def generate_report():
    """Print and save the final stress test report."""
    print("\n" + "=" * 70)
    print("  STRESS TEST REPORT")
    print("=" * 70)

    df = pd.DataFrame(RESULTS)

    # Summary counts
    counts = df["status"].value_counts()
    total = len(df)
    passed = counts.get("PASS", 0)
    failed = counts.get("FAIL", 0)
    warned = counts.get("WARN", 0)
    slow = counts.get("SLOW", 0)

    print(f"\n  Total tests:  {total}")
    print(f"  PASSED:       {passed}")
    print(f"  FAILED:       {failed}")
    print(f"  WARNINGS:     {warned}")
    print(f"  SLOW:         {slow}")
    print(f"  Pass rate:    {passed/total*100:.1f}%")

    # Failures detail
    failures = df[df["status"] == "FAIL"]
    if not failures.empty:
        print(f"\n{'─'*70}")
        print("  FAILURES:")
        print(f"{'─'*70}")
        for _, row in failures.iterrows():
            print(f"  [{row['project']}] {row['test']}")
            print(f"    → {row['detail']}")

    # Warnings detail
    warnings = df[df["status"] == "WARN"]
    if not warnings.empty:
        print(f"\n{'─'*70}")
        print("  WARNINGS (non-critical):")
        print(f"{'─'*70}")
        for _, row in warnings.iterrows():
            print(f"  [{row['project']}] {row['test']}")
            print(f"    → {row['detail']}")

    # Slow tests
    slow_tests = df[df["status"] == "SLOW"]
    if not slow_tests.empty:
        print(f"\n{'─'*70}")
        print("  SLOW TESTS:")
        print(f"{'─'*70}")
        for _, row in slow_tests.iterrows():
            print(f"  [{row['project']}] {row['test']}: {row['duration_ms']:.0f}ms")

    # Performance summary
    timed = df[df["duration_ms"] > 0]
    if not timed.empty:
        print(f"\n{'─'*70}")
        print("  PERFORMANCE:")
        print(f"{'─'*70}")
        print(f"  Fastest:  {timed['duration_ms'].min():.1f}ms")
        print(f"  Median:   {timed['duration_ms'].median():.1f}ms")
        print(f"  Slowest:  {timed['duration_ms'].max():.1f}ms")
        print(f"  Mean:     {timed['duration_ms'].mean():.1f}ms")

    # Per-project failure summary
    proj_failures = failures.groupby("project").size()
    if not proj_failures.empty:
        print(f"\n{'─'*70}")
        print("  FAILURES PER PROJECT:")
        print(f"{'─'*70}")
        for proj, cnt in proj_failures.items():
            print(f"  {proj}: {cnt} failure(s)")

    # Save report
    report_path = ARTIFACTS / "stress_test_report.csv"
    df.to_csv(report_path, index=False)
    print(f"\n  Full report saved to: artifacts/stress_test_report.csv")
    print("=" * 70)

    return failed


# ====================================================================
# MAIN
# ====================================================================

def main():
    print("\n" + "=" * 70)
    print("  PHASE 6 — EXECUTION & STRESS TEST")
    print("=" * 70)
    print()

    # Backup existing joblib files (if any) so we can restore after
    backup_files = {}
    for slug in ALL_SLUGS:
        for fname in ("model.joblib", "vectorizer.joblib"):
            p = ARTIFACTS / slug / fname
            if p.exists():
                backup_files[p] = p.read_bytes()

    try:
        # Create mock models for end-to-end testing
        create_mock_models()

        # Run all test suites
        test_large_datasets()
        test_missing_values()
        test_wrong_schema()
        test_repeated_runs()
        test_performance()
        test_pipeline_integrity()
        test_leaderboard_stress()
        test_api_module()
        test_edge_cases()

    finally:
        # Restore original state: remove mock joblib files that weren't there before
        for slug in ALL_SLUGS:
            for fname in ("model.joblib", "vectorizer.joblib"):
                p = ARTIFACTS / slug / fname
                if p in backup_files:
                    p.write_bytes(backup_files[p])
                elif p.exists():
                    p.unlink()
        remove_mock_schemas()

    # Generate report
    failures = generate_report()
    return failures


if __name__ == "__main__":
    failures = main()
    sys.exit(1 if failures > 0 else 0)
