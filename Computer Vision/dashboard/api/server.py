"""Lightweight FastAPI backend for the CV Projects Dashboard.

Wraps ``core.runner`` with REST endpoints for project discovery,
inference, and status checks.

Usage::

    python dashboard/api/server.py
    python dashboard/api/server.py --port 8042 --reload
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

log = logging.getLogger("dashboard.api")

app = FastAPI(
    title="CV Projects Dashboard API",
    version="1.0.0",
    docs_url="/api/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy project discovery ──────────────────────────────────

_discovered = False


def _ensure_discovered():
    global _discovered
    if _discovered:
        return
    from core.runner import discover_projects
    n = discover_projects()
    log.info("Discovered %d project modules", n)
    _discovered = True


# ── Cached project instances ────────────────────────────────

_loaded: dict[str, object] = {}


def _get_project(key: str):
    from core.registry import PROJECT_REGISTRY
    _ensure_discovered()
    if key not in PROJECT_REGISTRY:
        raise HTTPException(404, f"Unknown project: {key}")
    if key not in _loaded:
        proj = PROJECT_REGISTRY[key]()
        proj.load()
        _loaded[key] = proj
    return _loaded[key]


# ── Routes ──────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/projects")
def list_projects():
    """Return the pre-generated manifest."""
    manifest_path = REPO_ROOT / "dashboard" / "public" / "data" / "projects.json"
    if not manifest_path.exists():
        raise HTTPException(500, "projects.json not found. Run scan_projects.py first.")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data


@app.get("/api/projects/{key}")
def get_project(key: str):
    """Return metadata for a single project."""
    manifest_path = REPO_ROOT / "dashboard" / "public" / "data" / "projects.json"
    if not manifest_path.exists():
        raise HTTPException(500, "Manifest not found")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for p in data.get("projects", []):
        if p["key"] == key or key in p.get("aliases", []):
            return p
    raise HTTPException(404, f"Project not found: {key}")


@app.get("/api/registry")
def registry_status():
    """Show live registry state (requires project discovery)."""
    _ensure_discovered()
    from core.registry import PROJECT_REGISTRY
    return {
        "registered": sorted(PROJECT_REGISTRY.keys()),
        "count": len(PROJECT_REGISTRY),
    }


@app.post("/api/run/{key}")
async def run_project(key: str, file: UploadFile = File(...)):
    """Run inference on an uploaded image.

    Returns structured prediction results.
    """
    proj = _get_project(key)

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    try:
        result = proj.predict(img)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    return _serialize_result(result)


@app.post("/api/visualize/{key}")
async def visualize_project(key: str, file: UploadFile = File(...)):
    """Run inference + visualization. Returns annotated image as base64."""
    proj = _get_project(key)

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    try:
        output = proj.predict(img)
        if hasattr(proj, 'visualize'):
            vis = proj.visualize(img, output)
        else:
            vis = img
    except Exception as e:
        raise HTTPException(500, f"Visualization failed: {e}")

    _, buf = cv2.imencode('.jpg', vis)
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')

    return {
        "image": f"data:image/jpeg;base64,{b64}",
        "prediction": _serialize_result(output),
    }


def _serialize_result(result) -> dict:
    """Best-effort serialization of arbitrary prediction outputs."""
    if isinstance(result, dict):
        return {k: _serialize_val(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return {"results": [_serialize_val(v) for v in result]}
    return {"output": str(result)}


def _serialize_val(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _serialize_val(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize_val(i) for i in v]
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


@app.get("/api/system")
def system_info():
    """Return system info: GPU, Python, PyTorch versions."""
    info = {
        "gpu_available": False,
        "gpu_name": "",
        "gpu_memory_gb": 0,
        "cuda_version": "",
        "python_version": sys.version.split()[0],
        "torch_version": "",
        "projects_discovered": 0,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
            info["cuda_version"] = torch.version.cuda or ""
    except ImportError:
        pass

    _ensure_discovered()
    from core.registry import PROJECT_REGISTRY
    info["projects_discovered"] = len(PROJECT_REGISTRY)
    return info


@app.get("/api/datasets/{key}")
def dataset_status(key: str):
    """Check dataset status for a project."""
    config_dir = REPO_ROOT / "configs" / "datasets"
    data_dir = REPO_ROOT / "data"

    # Find YAML config
    yaml_path = config_dir / f"{key}.yaml"
    configured = yaml_path.exists()
    ds_type = ""
    ds_id = ""
    if configured:
        try:
            import yaml
            cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            ds_type = cfg.get("type", "")
            ds_id = cfg.get("id", cfg.get("dataset", ""))
        except Exception:
            pass

    # Check if data directory exists and has content
    project_data = data_dir / key
    ready = False
    size_mb = None
    if project_data.exists():
        files = list(project_data.rglob("*"))
        if any(f.is_file() for f in files):
            ready = True
            size_mb = round(sum(f.stat().st_size for f in files if f.is_file()) / 1e6, 1)

    return {
        "project": key,
        "configured": configured,
        "ready": ready,
        "type": ds_type,
        "id": ds_id,
        "size_mb": size_mb,
        "path": str(project_data) if ready else "",
    }


@app.post("/api/datasets/{key}/download")
def download_dataset(key: str):
    """Trigger dataset download for a project (blocking)."""
    config_dir = REPO_ROOT / "configs" / "datasets"
    yaml_path = config_dir / f"{key}.yaml"
    if not yaml_path.exists():
        raise HTTPException(404, f"No dataset config for project: {key}")

    try:
        from utils.datasets import DatasetResolver
        resolver = DatasetResolver()
        resolver.resolve(key)
        return {"status": "ok", "message": f"Dataset for {key} downloaded successfully"}
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")


@app.post("/api/train/{key}")
def start_training(key: str):
    """Placeholder for training launch — returns scaffold response.

    Full training integration requires async job management.
    """
    _ensure_discovered()
    from core.registry import PROJECT_REGISTRY
    if key not in PROJECT_REGISTRY:
        raise HTTPException(404, f"Unknown project: {key}")

    # Check if train.py exists
    manifest_path = REPO_ROOT / "dashboard" / "public" / "data" / "projects.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        proj = next((p for p in data["projects"] if p["key"] == key), None)
        if proj and not proj.get("hasTraining", False):
            raise HTTPException(400, f"Project {key} does not support training")

    return {
        "status": "not_implemented",
        "message": (
            f"Training for '{key}' is available via CLI: "
            f'cd "{key}/Source Code" && python train.py'
        ),
        "cli_command": f'cd "{key}/Source Code" && python train.py',
    }


@app.get("/api/train/{key}/status")
def train_status(key: str):
    """Return training status placeholder."""
    return {
        "project": key,
        "status": "idle",
        "epoch": 0,
        "total_epochs": 0,
        "metrics": {},
        "log": [],
    }


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="CV Projects Dashboard API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8042)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    log.info("Starting API on %s:%d", args.host, args.port)
    uvicorn.run(
        "dashboard.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
