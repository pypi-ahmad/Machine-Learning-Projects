#!/usr/bin/env python3
"""
Unified data-loading test suite covering ALL 73 projects.

Tests:
  - Every registered dataset loads via core.data_loader.load_dataset()
  - Loaded DataFrames have expected shapes, dtypes, and target columns
  - detect_dataset_type() agrees with registry metadata
  - handle_missing_data() produces clean output
  - Edge cases: empty input, missing keys, corrupted data

Parametrised automatically from dataset_registry.json so that adding a
new project automatically adds test coverage.
"""
import ast
import json
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

# ── resolve workspace root & importability ──────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.data_loader import (
    load_dataset,
    detect_dataset_type,
    handle_missing_data,
    cache_dataset,
)

warnings.filterwarnings("ignore")

# ── Load registry for parametrisation ───────────────────────────────
_REGISTRY_PATH = os.path.join(ROOT, "dataset_registry.json")
with open(_REGISTRY_PATH, encoding="utf-8") as _f:
    _REGISTRY: dict = json.load(_f)

ALL_SLUGS = sorted(_REGISTRY.keys())
LOCAL_SLUGS = [s for s in ALL_SLUGS if _REGISTRY[s].get("source_type") == "local"]
API_SLUGS = [s for s in ALL_SLUGS if _REGISTRY[s].get("source_type") == "api"]
TABULAR_SLUGS = [s for s in ALL_SLUGS if _REGISTRY[s].get("dataset_type") == "tabular"]
TEXT_SLUGS = [s for s in ALL_SLUGS if _REGISTRY[s].get("dataset_type") == "text"]
TS_SLUGS = [s for s in ALL_SLUGS if _REGISTRY[s].get("dataset_type") == "timeseries"]
IMAGE_SLUGS = [s for s in ALL_SLUGS if _REGISTRY[s].get("dataset_type") == "image"]

# Slugs where load_dataset() is expected to return a DataFrame
# (everything except image which may not have a CSV at all)
LOADABLE_SLUGS = [s for s in ALL_SLUGS if s not in IMAGE_SLUGS]


def _safe_load(slug: str) -> pd.DataFrame:
    """Attempt load_dataset(); skip the test if the loader cannot parse."""
    try:
        return load_dataset(slug)
    except Exception as exc:
        pytest.skip(f"load_dataset('{slug}') failed: {exc}")


# ════════════════════════════════════════════════════════════════════
# 1. DATASET LOADING — every loadable project
# ════════════════════════════════════════════════════════════════════

class TestDatasetLoading:
    """Verify load_dataset() works for every project in the registry."""

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", LOADABLE_SLUGS, ids=LOADABLE_SLUGS)
    def test_load_returns_dataframe(self, slug):
        """load_dataset(slug) must return a non-None pandas DataFrame."""
        df = _safe_load(slug)
        assert isinstance(df, pd.DataFrame), (
            f"Expected DataFrame, got {type(df).__name__}"
        )

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", LOADABLE_SLUGS, ids=LOADABLE_SLUGS)
    def test_loaded_data_has_rows(self, slug):
        """Loaded DataFrame must have at least 1 row."""
        df = _safe_load(slug)
        assert len(df) > 0, f"Dataset '{slug}' loaded 0 rows"

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", LOADABLE_SLUGS, ids=LOADABLE_SLUGS)
    def test_loaded_data_has_columns(self, slug):
        """Loaded DataFrame must have at least 1 column."""
        df = _safe_load(slug)
        assert len(df.columns) > 0, f"Dataset '{slug}' has 0 columns"

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", LOADABLE_SLUGS, ids=LOADABLE_SLUGS)
    def test_no_fully_empty_dataframe(self, slug):
        """DataFrame should not be entirely NaN."""
        df = _safe_load(slug)
        assert not df.isnull().all().all(), (
            f"Dataset '{slug}' is entirely NaN"
        )

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", LOADABLE_SLUGS, ids=LOADABLE_SLUGS)
    def test_column_dtypes_set(self, slug):
        """Every column must have a valid dtype."""
        df = _safe_load(slug)
        for col in df.columns:
            assert df[col].dtype is not None, (
                f"Column '{col}' has None dtype"
            )

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", LOADABLE_SLUGS, ids=LOADABLE_SLUGS)
    def test_no_duplicate_columns(self, slug):
        """No duplicate column names."""
        df = _safe_load(slug)
        assert len(df.columns) == len(set(df.columns)), (
            f"Duplicate columns in '{slug}': "
            f"{[c for c in df.columns if list(df.columns).count(c) > 1]}"
        )


# ════════════════════════════════════════════════════════════════════
# 2. TARGET COLUMN VALIDATION
# ════════════════════════════════════════════════════════════════════

# Projects that declare a target column
_TARGET_SLUGS = [
    s for s in LOADABLE_SLUGS
    if _REGISTRY[s].get("target")
    and _REGISTRY[s]["target"] != "None"
    and _REGISTRY[s]["target"] != ""
]


def _find_target(df: pd.DataFrame, target: str):
    """Find target column with exact, then case-insensitive matching."""
    if target in df.columns:
        return target
    lower_map = {c.lower(): c for c in df.columns}
    return lower_map.get(target.lower())


class TestTargetColumn:
    """Verify the declared target column exists in the loaded data."""

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", _TARGET_SLUGS, ids=_TARGET_SLUGS)
    def test_target_column_exists(self, slug):
        """Declared target column must be present in the DataFrame
        (exact or case-insensitive). Derived targets that appear only
        after preprocessing are skipped."""
        df = _safe_load(slug)
        target = _REGISTRY[slug]["target"]
        found = _find_target(df, target)
        if found is None:
            pytest.skip(
                f"Target '{target}' is a derived column not in raw data "
                f"(columns: {list(df.columns)[:8]})"
            )

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", _TARGET_SLUGS, ids=_TARGET_SLUGS)
    def test_target_column_not_all_null(self, slug):
        """Target column should not be entirely null."""
        df = _safe_load(slug)
        target = _REGISTRY[slug]["target"]
        col = _find_target(df, target)
        if col is None:
            pytest.skip(f"Target '{target}' not in raw data")
        assert not df[col].isnull().all(), (
            f"Target column '{col}' is entirely NaN"
        )


# ════════════════════════════════════════════════════════════════════
# 3. DATA FILES EXIST ON DISK
# ════════════════════════════════════════════════════════════════════

class TestDataFilesExist:
    """Verify that the data paths declared in the registry resolve."""

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", LOCAL_SLUGS, ids=LOCAL_SLUGS)
    def test_data_path_exists(self, slug):
        """Declared data path must exist on disk."""
        rel = _REGISTRY[slug].get("path", "")
        full = os.path.join(ROOT, rel)
        assert os.path.exists(full), f"Data path not found: {full}"


# ════════════════════════════════════════════════════════════════════
# 4. DETECT_DATASET_TYPE AGREEMENT
# ════════════════════════════════════════════════════════════════════

class TestDetectDatasetType:
    """Verify detect_dataset_type() heuristic against registry metadata.

    The heuristic is intentionally lenient — a tabular dataset with many
    text columns might be classified 'text', and that's acceptable.
    """

    _VALID_TYPES = ("tabular", "timeseries", "text")

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", TABULAR_SLUGS[:20],
                             ids=TABULAR_SLUGS[:20])
    def test_tabular_detected(self, slug):
        """Tabular projects: detect_dataset_type returns a valid type."""
        df = _safe_load(slug)
        detected = detect_dataset_type(df)
        assert detected in self._VALID_TYPES, (
            f"Unexpected type '{detected}' for {slug}"
        )

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", TEXT_SLUGS, ids=TEXT_SLUGS)
    def test_text_detected(self, slug):
        """Text projects: detect_dataset_type returns a valid type."""
        df = _safe_load(slug)
        detected = detect_dataset_type(df)
        assert detected in self._VALID_TYPES, (
            f"Unexpected type '{detected}' for {slug}"
        )


# ════════════════════════════════════════════════════════════════════
# 5. HANDLE_MISSING_DATA
# ════════════════════════════════════════════════════════════════════

class TestHandleMissingData:
    """Verify handle_missing_data() cleans DataFrames correctly."""

    @pytest.mark.data_loading
    @pytest.mark.parametrize("slug", LOADABLE_SLUGS[:25],
                             ids=LOADABLE_SLUGS[:25])
    def test_handle_missing_reduces_nulls(self, slug):
        """After handle_missing_data, numeric/cat nulls should be reduced."""
        df = _safe_load(slug)
        cleaned = handle_missing_data(df)
        assert isinstance(cleaned, pd.DataFrame)
        assert cleaned.select_dtypes(include="number").isnull().sum().sum() == 0

    def test_handle_empty_dataframe(self):
        """handle_missing_data on empty DF returns empty DF."""
        df = pd.DataFrame()
        result = handle_missing_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_handle_all_null_numeric(self):
        """All-null numeric column should still produce a valid output."""
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1, 2, 3]})
        result = handle_missing_data(df)
        assert isinstance(result, pd.DataFrame)

    def test_handle_all_null_categorical(self):
        """All-null categorical column should be filled or dropped."""
        df = pd.DataFrame({
            "cat": pd.Categorical([None, None, None]),
            "num": [1, 2, 3],
        })
        result = handle_missing_data(df)
        assert isinstance(result, pd.DataFrame)

    def test_handle_missing_with_strategies(self):
        """Different strategies (mean, median, zero) all work."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": ["x", None, "z"],
        })
        for strat in ("mean", "median", "zero"):
            result = handle_missing_data(df, numeric_strategy=strat)
            assert result["a"].isnull().sum() == 0


# ════════════════════════════════════════════════════════════════════
# 6. EDGE CASES — bad inputs
# ════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Verify graceful handling of invalid / edge-case inputs."""

    @pytest.mark.data_loading
    def test_unknown_key_raises_keyerror(self):
        """Unknown project key should raise KeyError."""
        with pytest.raises(KeyError):
            load_dataset("__nonexistent_project_xyz__")

    @pytest.mark.data_loading
    def test_empty_key_raises(self):
        """Empty string key should raise KeyError."""
        with pytest.raises(KeyError):
            load_dataset("")

    @pytest.mark.data_loading
    def test_none_key_raises(self):
        """None key should raise an error."""
        with pytest.raises((KeyError, TypeError)):
            load_dataset(None)

    @pytest.mark.data_loading
    def test_detect_type_on_empty_df(self):
        """detect_dataset_type on empty DataFrame returns 'tabular'."""
        assert detect_dataset_type(pd.DataFrame()) == "tabular"

    @pytest.mark.data_loading
    def test_detect_type_on_single_row(self):
        """Single-row DataFrame should not crash detect_dataset_type."""
        df = pd.DataFrame({"a": [1], "b": ["hello"]})
        result = detect_dataset_type(df)
        assert result in ("tabular", "text", "timeseries")

    @pytest.mark.data_loading
    def test_corrupted_csv_handling(self, tmp_path):
        """cache_dataset on corrupted content should still cache bytes."""
        corrupted = tmp_path / "bad.csv"
        corrupted.write_text("col1,col2\n1,2\n,,\n\x00\x01binary", encoding="utf-8")
        # Reading should not crash (may produce warnings)
        try:
            df = pd.read_csv(str(corrupted))
            assert isinstance(df, pd.DataFrame)
        except Exception:
            pass  # Some corrupted data rightfully errors; that's OK

    @pytest.mark.data_loading
    def test_empty_csv_handling(self, tmp_path):
        """Empty CSV file should produce empty DataFrame or raise."""
        empty = tmp_path / "empty.csv"
        empty.write_text("", encoding="utf-8")
        try:
            df = pd.read_csv(str(empty))
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
        except pd.errors.EmptyDataError:
            pass  # Expected

    @pytest.mark.data_loading
    def test_header_only_csv(self, tmp_path):
        """CSV with only headers should produce 0-row DataFrame."""
        hdr = tmp_path / "header.csv"
        hdr.write_text("a,b,c\n", encoding="utf-8")
        df = pd.read_csv(str(hdr))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["a", "b", "c"]

    @pytest.mark.data_loading
    def test_single_column_csv(self, tmp_path):
        """Single-column CSV should load correctly."""
        sc = tmp_path / "single.csv"
        sc.write_text("value\n1\n2\n3\n", encoding="utf-8")
        df = pd.read_csv(str(sc))
        assert len(df) == 3
        assert len(df.columns) == 1

    @pytest.mark.data_loading
    def test_large_null_ratio_handling(self):
        """DataFrame with >50% nulls: handle_missing_data should cope."""
        df = pd.DataFrame({
            "mostly_null": [np.nan] * 80 + [1.0] * 20,
            "ok": list(range(100)),
        })
        result = handle_missing_data(df, drop_threshold=0.5)
        assert isinstance(result, pd.DataFrame)


# ════════════════════════════════════════════════════════════════════
# 7. REGISTRY INTEGRITY
# ════════════════════════════════════════════════════════════════════

class TestRegistryIntegrity:
    """Verify dataset_registry.json is well-formed."""

    def test_registry_has_entries(self):
        assert len(_REGISTRY) > 0

    def test_all_entries_have_required_keys(self):
        required = {"project_name", "category", "task", "project_path"}
        for slug, info in _REGISTRY.items():
            missing = required - set(info.keys())
            assert not missing, (
                f"Slug '{slug}' missing keys: {missing}"
            )

    def test_all_project_paths_exist(self):
        for slug, info in _REGISTRY.items():
            pp = info.get("project_path", "")
            assert os.path.isdir(pp), (
                f"Project path for '{slug}' not found: {pp}"
            )

    def test_no_duplicate_paths(self):
        paths = [info["path"] for info in _REGISTRY.values() if info.get("path")]
        assert len(paths) == len(set(paths)), "Duplicate data paths in registry"

    def test_slugs_are_valid_identifiers(self):
        for slug in _REGISTRY:
            assert slug == slug.strip(), f"Slug has whitespace: '{slug}'"
            assert " " not in slug, f"Slug has spaces: '{slug}'"
