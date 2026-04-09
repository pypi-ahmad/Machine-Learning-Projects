"""
Inference Engine — Load and run trained models from artifacts/.

Provides:
    list_projects()              → list of project slugs with trained models
    get_project_info(project)    → metrics + model metadata
    load_model(project)          → (model, vectorizer) tuple
    load_schema(project)         → feature schema dict (or None)
    validate_input(project, df)  → validated/reordered DataFrame
    predict(project, df)         → numpy array of predictions
    predict_text(project, text)  → single-text prediction (vectorize → predict)
"""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import load as joblib_load

from config import ROOT, ARTIFACTS

# When True, extra columns not in schema raise ValueError.
# When False, extra columns are silently dropped.
STRICT_SCHEMA = True


class ModelNotFoundError(Exception):
    """Raised when a project has no trained model."""
    pass


def list_projects() -> list[str]:
    """Return sorted list of project slugs that have a trained model."""
    if not ARTIFACTS.exists():
        return []
    return sorted(
        d.name
        for d in ARTIFACTS.iterdir()
        if d.is_dir() and (d / "model.joblib").exists()
    )


def list_all_projects() -> list[str]:
    """Return sorted list of ALL project slugs (even without model)."""
    if not ARTIFACTS.exists():
        return []
    return sorted(
        d.name
        for d in ARTIFACTS.iterdir()
        if d.is_dir() and (d / "metrics.json").exists()
    )


def get_project_info(project: str) -> dict:
    """Return metrics and model metadata for a project."""
    proj_dir = ARTIFACTS / project
    metrics_path = proj_dir / "metrics.json"
    if not metrics_path.exists():
        raise ModelNotFoundError(f"No metrics found for '{project}'")

    with open(metrics_path, encoding="utf-8") as f:
        info = json.load(f)

    info["project"] = project
    info["has_model"] = (proj_dir / "model.joblib").exists()
    info["has_vectorizer"] = (proj_dir / "vectorizer.joblib").exists()
    return info


def load_schema(project: str) -> dict | None:
    """
    Load the feature schema for a project.

    Returns:
        dict with at least a 'features' key (list of column names),
        or None if no schema file exists.
    """
    schema_path = ARTIFACTS / project / "schema.json"
    if not schema_path.exists():
        return None
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def validate_input(project: str, df):
    """
    Validate and reorder a DataFrame against the project's feature schema.

    - Missing required columns → ValueError
    - Extra columns + STRICT_SCHEMA → ValueError
    - Extra columns + not strict  → silently dropped
    - Columns reordered to match schema
    - No schema file → passthrough (returns df unchanged)

    Args:
        project: project slug
        df: pandas DataFrame to validate

    Returns:
        Validated and reordered DataFrame.
    """
    schema = load_schema(project)
    if schema is None:
        return df

    expected = schema.get("features", [])
    if not expected:
        return df

    # Check for missing columns
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for extra columns
    extra = [c for c in df.columns if c not in expected]
    if extra:
        if STRICT_SCHEMA:
            raise ValueError(f"Unexpected extra columns: {extra}")
        # non-strict: silently drop extras

    # Reorder to match schema
    df = df[expected]
    return df


@lru_cache(maxsize=32)
def load_model(project: str):
    """
    Load and cache a trained model + vectorizer.

    Returns:
        (model, vectorizer) — vectorizer may be None if not saved.

    Raises:
        ModelNotFoundError if model.joblib does not exist.
    """
    proj_dir = ARTIFACTS / project
    model_path = proj_dir / "model.joblib"
    vec_path = proj_dir / "vectorizer.joblib"

    if not model_path.exists():
        raise ModelNotFoundError(
            f"No trained model for '{project}'. "
            f"Run the notebook first to generate artifacts/{project}/model.joblib"
        )

    model = joblib_load(str(model_path))
    vectorizer = joblib_load(str(vec_path)) if vec_path.exists() else None
    return model, vectorizer


def predict(project: str, data) -> np.ndarray:
    """
    Run prediction on a project's trained model.

    Args:
        project: slug name (e.g. 'resume_screening')
        data: pandas DataFrame or numpy array of features.
              If a DataFrame with a single 'text' column is provided
              and the project has a vectorizer, text will be auto-vectorized.

    Returns:
        numpy array of predictions.
    """
    model, vectorizer = load_model(project)

    # Auto-vectorize text input
    if isinstance(data, pd.DataFrame):
        if "text" in data.columns and vectorizer is not None:
            features = vectorizer.transform(data["text"].astype(str))
        else:
            data = validate_input(project, data)
            features = data.values
    elif hasattr(data, "toarray"):
        features = data
    else:
        features = np.array(data)

    return model.predict(features)


def predict_text(project: str, text: str):
    """
    Single-text prediction.

    Vectorizes the text using the project's saved vectorizer,
    then runs prediction.

    Returns:
        The predicted label (scalar).
    """
    model, vectorizer = load_model(project)

    if vectorizer is None:
        raise ModelNotFoundError(
            f"Project '{project}' has no saved vectorizer. "
            f"Cannot predict from raw text."
        )

    vec = vectorizer.transform([text])
    return model.predict(vec)[0]
