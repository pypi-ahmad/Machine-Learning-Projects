"""
Model Serving API — FastAPI service exposing all trained NLP models.

Endpoints:
    GET  /                          → health check
    GET  /projects                  → list projects with trained models
    GET  /projects/all              → list ALL projects (incl. untrained)
    GET  /projects/{project}        → project metrics & metadata
    POST /predict/{project}         → batch prediction from feature dict
    POST /predict/{project}/text    → single raw-text prediction

Run:
    uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

from inference_engine import (
    list_projects,
    list_all_projects,
    get_project_info,
    predict,
    predict_text,
    ModelNotFoundError,
)

# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="NLP Model Serving API",
    description="Serves trained models from 14 NLP projects via a unified interface.",
    version="1.0.0",
)


# ── Schemas ──────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Feature dict for batch prediction. Keys are feature names, values are feature values."""
    data: dict = Field(
        ...,
        example={"f0": 0.1, "f1": 0.5, "f2": 0.3},
        description="Dictionary of feature names to values. "
                    "Include a 'text' key for automatic vectorization.",
    )


class TextRequest(BaseModel):
    """Single raw-text input for prediction."""
    text: str = Field(
        ...,
        example="This product is amazing!",
        description="Raw text to classify.",
    )


class HealthResponse(BaseModel):
    status: str
    projects_ready: int
    projects_total: int


class ProjectListResponse(BaseModel):
    projects: list[str]
    count: int


class PredictionResponse(BaseModel):
    project: str
    prediction: list


class TextPredictionResponse(BaseModel):
    project: str
    text: str
    prediction: str


# ── STEP 5: Health Check ────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
def root():
    """Health check — returns API status and project counts."""
    ready = list_projects()
    total = list_all_projects()
    return HealthResponse(
        status="ML API running",
        projects_ready=len(ready),
        projects_total=len(total),
    )


# ── STEP 3: List Projects ──────────────────────────────────────────

@app.get("/projects", response_model=ProjectListResponse)
def get_projects():
    """List all projects that have a trained model ready for inference."""
    projects = list_projects()
    return ProjectListResponse(projects=projects, count=len(projects))


@app.get("/projects/all", response_model=ProjectListResponse)
def get_all_projects():
    """List ALL projects (including those without trained models)."""
    projects = list_all_projects()
    return ProjectListResponse(projects=projects, count=len(projects))


@app.get("/projects/{project}")
def get_project(project: str):
    """Get metrics and metadata for a specific project."""
    try:
        return get_project_info(project)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── STEP 4: Predict Endpoint ───────────────────────────────────────

@app.post("/predict/{project}", response_model=PredictionResponse)
def run_prediction(project: str, request: PredictionRequest):
    """
    Run batch prediction using a project's trained model.

    Send feature values as a dict. Include a `text` key for
    automatic vectorization via the project's saved vectorizer.
    """
    try:
        df = pd.DataFrame([request.data])
        preds = predict(project, df)
        return PredictionResponse(
            project=project,
            prediction=preds.tolist(),
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")


@app.post("/predict/{project}/text", response_model=TextPredictionResponse)
def run_text_prediction(project: str, request: TextRequest):
    """
    Predict from raw text using the project's saved vectorizer + model.
    """
    try:
        result = predict_text(project, request.text)
        return TextPredictionResponse(
            project=project,
            text=request.text,
            prediction=str(result),
        )
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")
