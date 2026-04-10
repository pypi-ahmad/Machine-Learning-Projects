#!/usr/bin/env python3
"""Reproducible accuracy evaluation for all registered CV projects.

Evaluates each project's model (custom-trained via registry or pretrained
YOLO26 fallback) on its configured dataset, producing structured CSV + JSON
results with full provenance metadata.

Usage::

    python -m benchmarks.evaluate_accuracy
    python -m benchmarks.evaluate_accuracy --project face_mask_detection
    python -m benchmarks.evaluate_accuracy --limit 5
    python -m benchmarks.evaluate_accuracy --device cpu
    python -m benchmarks.evaluate_accuracy --write-registry
    python -m benchmarks.evaluate_accuracy --outdir benchmarks/results

Zero-skip policy: if dataset is missing but download is configured,
the evaluator will attempt automatic download before evaluating.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure repo root is on sys.path ──────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("benchmarks.evaluate_accuracy")

# ── Paths ─────────────────────────────────────────────────────────────────
CONFIGS_DIR = REPO_ROOT / "configs" / "datasets"
DEFAULT_OUTDIR = REPO_ROOT / "benchmarks" / "results"

# ── Project-type ↔ task mapping ───────────────────────────────────────────
_PROJECT_TYPE_TO_TASK: Dict[str, str] = {
    "detection": "detect",
    "classification": "cls",
    "segmentation": "seg",
    "pose": "pose",
    "tracking": "detect",
    "utility": "detect",
}


# =====================================================================
#  YAML loader (works with or without PyYAML)
# =====================================================================
def _load_yaml(path: Path) -> dict:
    """Load a YAML config file.  Tries PyYAML first, falls back to a
    minimal hand-rolled parser that covers our flat schema."""
    try:
        import yaml  # type: ignore[import-untyped]
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        pass
    return _load_yaml_fallback(path)


def _load_yaml_fallback(path: Path) -> dict:
    """Very simple key-value YAML parser for our known flat schema."""
    data: dict = {}
    current_section: Optional[dict] = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "  #" in line:
            line = line[: line.index("  #")].rstrip()

        indent = len(raw_line) - len(raw_line.lstrip())

        if indent == 0 and ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val in ("", "null"):
                if key in ("dataset", "metrics"):
                    current_section = {}
                    data[key] = current_section
                else:
                    data[key] = None
            elif val.lower() == "true":
                data[key] = True
            elif val.lower() == "false":
                data[key] = False
            else:
                data[key] = val
                current_section = None
        elif indent > 0 and current_section is not None and ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val in ("null", ""):
                current_section[key] = None
            elif val.lower() == "true":
                current_section[key] = True
            elif val.lower() == "false":
                current_section[key] = False
            else:
                current_section[key] = val

    return data


# =====================================================================
#  Provenance metadata
# =====================================================================
def _build_run_meta(args: argparse.Namespace) -> dict:
    """Collect reproducibility provenance for this evaluation run."""
    meta: Dict[str, Any] = {
        "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    # torch
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["cuda_device_name"] = torch.cuda.get_device_name(0)
        else:
            meta["cuda_device_name"] = None
    except ImportError:
        meta["torch_version"] = None
        meta["cuda_available"] = False
        meta["cuda_device_name"] = None

    # ultralytics
    try:
        import ultralytics
        meta["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        meta["ultralytics_version"] = None

    # git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=5,
        )
        meta["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        meta["git_commit"] = None

    # command args
    meta["command_args"] = vars(args)
    return meta


# =====================================================================
#  Metric extraction helpers
# =====================================================================
def _extract_ultralytics_metrics(results: Any, task: str) -> Dict[str, Any]:
    """Robustly extract metrics from an ultralytics val() results object."""
    metrics: Dict[str, Any] = {}
    try:
        rd = results.results_dict if hasattr(results, "results_dict") else {}

        # Collect all available numeric metrics
        for k, v in rd.items():
            try:
                metrics[k] = round(float(v), 6)
            except (TypeError, ValueError):
                pass

        # Also try .box, .seg, .pose attributes
        for prefix, attr_name in [("box", "box"), ("seg", "seg"), ("pose", "pose")]:
            attr = getattr(results, attr_name, None)
            if attr is None:
                continue
            for sub in ("map50", "map75", "map"):
                val = getattr(attr, sub, None)
                if val is not None:
                    try:
                        metrics[f"{prefix}/{sub}"] = round(float(val), 6)
                    except (TypeError, ValueError):
                        pass
            # map50-95
            maps = getattr(attr, "maps", None)
            if maps is not None:
                try:
                    import numpy as np
                    metrics[f"{prefix}/map50-95"] = round(float(np.mean(maps)), 6)
                except Exception:
                    pass
    except Exception:
        pass
    return metrics


def _pick_primary_metric(metrics: Dict[str, Any], task: str) -> tuple[str, Optional[float]]:
    """Choose the best primary metric name + value for the given task."""
    # Preference order by task
    preference: Dict[str, list[str]] = {
        "detect": [
            "metrics/mAP50-95(B)", "box/map50-95", "box/map",
            "metrics/mAP50(B)", "box/map50",
        ],
        "seg": [
            "metrics/mAP50-95(M)", "seg/map50-95", "seg/map",
            "metrics/mAP50-95(B)", "box/map50-95",
            "metrics/mAP50(M)", "seg/map50",
        ],
        "pose": [
            "metrics/mAP50-95(P)", "pose/map50-95", "pose/map",
            "metrics/mAP50-95(B)", "box/map50-95",
        ],
        "obb": [
            "metrics/mAP50-95(B)", "box/map50-95", "box/map",
        ],
        "cls": [
            "metrics/accuracy_top1", "acc_top1", "top1_acc",
            "metrics/accuracy_top5",
        ],
    }

    for key in preference.get(task, preference["detect"]):
        if key in metrics:
            return key, metrics[key]

    # Fallback: first numeric metric
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return k, v

    return "unknown", None


# =====================================================================
#  Evaluation backends
# =====================================================================
def _eval_ultralytics(
    weights: str,
    data_yaml: str,
    device: str,
    task: str,
) -> Dict[str, Any]:
    """Run ultralytics YOLO.val() and return metrics dict."""
    from ultralytics import YOLO

    model = YOLO(weights)
    use_half = device != "cpu"

    val_kwargs: Dict[str, Any] = dict(
        data=data_yaml,
        device=device,
        imgsz=640,
        half=use_half,
        plots=False,
        verbose=False,
    )

    results = model.val(**val_kwargs)
    return _extract_ultralytics_metrics(results, task)


def _eval_ultralytics_cls(
    weights: str,
    data_dir: str,
    device: str,
) -> Dict[str, Any]:
    """Run ultralytics YOLO classification val()."""
    from ultralytics import YOLO

    model = YOLO(weights)
    use_half = device != "cpu"

    results = model.val(
        data=data_dir,
        device=device,
        imgsz=224,
        half=use_half,
        plots=False,
        verbose=False,
    )
    return _extract_ultralytics_metrics(results, "cls")


def _eval_torchvision_cls(
    weights_path: str,
    val_dir: str,
    device_str: str,
) -> Dict[str, Any]:
    """Evaluate a torchvision model (ImageFolder val set) → acc_top1."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    num_classes = len(val_dataset.classes)

    # Load model — attempt to load state dict, fallback to pretrained
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if Path(weights_path).exists():
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)

    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    return {"acc_top1": round(acc, 6), "total_samples": total}


# =====================================================================
#  Core evaluation loop
# =====================================================================
def evaluate_project(
    project_key: str,
    task: str,
    device: str,
    no_download: bool = False,
) -> Dict[str, Any]:
    """Evaluate one project.  Returns a result dict with status/metrics."""
    from models.registry import resolve

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    base: Dict[str, Any] = {
        "timestamp_iso": now,
        "repo_name": REPO_ROOT.name,
        "project_key": project_key,
        "task": task,
    }

    # ── 1. Load dataset config ────────────────────────────────────────────
    cfg_path = CONFIGS_DIR / f"{project_key}.yaml"
    if not cfg_path.exists():
        return {**base, "status": "missing_dataset_config", "error": "No config YAML"}

    cfg = _load_yaml(cfg_path)
    ds = cfg.get("dataset", {}) or {}
    metrics_cfg = cfg.get("metrics", {}) or {}

    base["dataset_name"] = ds.get("name", "")
    base["dataset_kind"] = ds.get("kind", "")
    base["dataset_ref"] = ""

    # ── 2. Validate dataset paths (with auto-download) ────────────────────
    kind = ds.get("kind", "")
    data_yaml = ds.get("data_yaml")
    val_dir = ds.get("val")
    root_dir = ds.get("root")
    framework = cfg.get("framework", "auto")

    def _attempt_download() -> dict | None:
        """Try to download the dataset. Returns error-result dict or None on success."""
        if no_download:
            return {**base, "status": "dataset_missing_no_download",
                    "error": "Dataset not on disk and --no-download active"}
        try:
            from utils.data_downloader import download_dataset
        except ImportError:
            return None
        dl_cfg = ds.get("download", {})
        if not dl_cfg.get("enabled", False):
            return {**base, "status": "dataset_download_disabled",
                    "error": f"Dataset missing & download disabled. Edit configs/datasets/{project_key}.yaml"}
        dl_result = download_dataset(cfg)
        if not dl_result.get("ok", False):
            return {**base, "status": "dataset_download_failed",
                    "error": dl_result.get("error", "download failed")}
        return None  # success

    if kind == "ultralytics_yaml":
        if not data_yaml or data_yaml == "null":
            return {**base, "status": "missing_dataset_files", "error": "data_yaml not configured"}
        abs_yaml = REPO_ROOT / data_yaml
        if not abs_yaml.exists():
            base["dataset_ref"] = data_yaml
            err_result = _attempt_download()
            if err_result:
                return err_result
            # Re-check after download
            if not abs_yaml.exists():
                return {**base, "status": "dataset_download_failed",
                        "error": f"{data_yaml} still missing after download"}
        base["dataset_ref"] = data_yaml

    elif kind == "imagefolder":
        effective_val = None
        if val_dir and val_dir != "null":
            effective_val = REPO_ROOT / val_dir
            base["dataset_ref"] = val_dir
        elif root_dir and root_dir != "null":
            effective_val = REPO_ROOT / root_dir / "val"
            base["dataset_ref"] = f"{root_dir}/val"

        if not effective_val:
            return {**base, "status": "missing_dataset_files", "error": "val path not configured"}
        if not effective_val.exists():
            err_result = _attempt_download()
            if err_result:
                return err_result
            # Re-check after download
            if not effective_val.exists():
                return {**base, "status": "dataset_download_failed",
                        "error": f"{base['dataset_ref']} still missing after download"}

    elif kind == "custom":
        base["dataset_ref"] = root_dir or ""
        return {**base, "status": "unsupported_cls_backend", "error": "custom kind — evaluation not implemented"}
    else:
        return {**base, "status": "missing_dataset_config", "error": f"unknown kind '{kind}'"}

    # ── 3. Resolve model weights ──────────────────────────────────────────
    try:
        weights, version, used_pretrained = resolve(project_key, task)
    except Exception as exc:
        return {**base, "status": "weights_unavailable", "error": str(exc)}

    base["model_version"] = version
    base["model_path"] = weights
    base["used_pretrained_default"] = used_pretrained

    # ── 4. Run evaluation ─────────────────────────────────────────────────
    try:
        if task in ("detect", "seg", "pose", "obb") and kind == "ultralytics_yaml":
            raw_metrics = _eval_ultralytics(
                weights=weights,
                data_yaml=str(REPO_ROOT / data_yaml),
                device=device,
                task=task,
            )

        elif task == "cls":
            if framework == "torchvision" or (
                framework == "auto" and weights.endswith(".pth")
            ):
                # torchvision backend
                eff_val = val_dir if (val_dir and val_dir != "null") else f"{root_dir}/val"
                raw_metrics = _eval_torchvision_cls(
                    weights_path=weights,
                    val_dir=str(REPO_ROOT / eff_val),
                    device_str=device,
                )
            elif framework in ("ultralytics", "auto"):
                # ultralytics cls
                eff_data = root_dir if root_dir else str(Path(val_dir).parent) if val_dir else ""
                raw_metrics = _eval_ultralytics_cls(
                    weights=weights,
                    data_dir=str(REPO_ROOT / eff_data) if eff_data else "",
                    device=device,
                )
            else:
                return {**base, "status": "unsupported_cls_backend",
                        "error": f"framework '{framework}' not supported for cls"}
        else:
            return {**base, "status": "unsupported_cls_backend",
                    "error": f"no evaluation backend for task={task} kind={kind}"}

    except Exception as exc:
        tb = traceback.format_exc()
        log.warning("Eval failed for %s: %s", project_key, tb)
        short_err = str(exc)[:200]
        return {**base, "status": "eval_error", "error": short_err}

    # ── 5. Extract primary metric ─────────────────────────────────────────
    primary_name, primary_value = _pick_primary_metric(raw_metrics, task)

    base["metric_primary_name"] = primary_name
    base["metric_primary_value"] = primary_value
    base["metrics_json"] = json.dumps(raw_metrics, default=str)
    base["status"] = "ok"
    base["error"] = ""

    return base


# =====================================================================
#  Output writers
# =====================================================================
_CSV_FIELDS = [
    "timestamp_iso",
    "repo_name",
    "project_key",
    "task",
    "dataset_name",
    "dataset_kind",
    "dataset_ref",
    "model_version",
    "model_path",
    "used_pretrained_default",
    "metric_primary_name",
    "metric_primary_value",
    "metrics_json",
    "status",
    "error",
]


def _write_csv(results: List[Dict], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "accuracy_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return path


def _write_json(results: List[Dict], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "accuracy_results.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return path


def _write_meta(meta: dict, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "accuracy_run_meta.json"
    path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


# =====================================================================
#  Registry writeback
# =====================================================================
def _writeback_registry(results: List[Dict]) -> int:
    """Write evaluation metrics back to the model registry for evaluated
    projects that have a non-null version."""
    from models.registry import ModelRegistry

    reg = ModelRegistry()
    written = 0
    for r in results:
        if r.get("status") != "ok":
            continue
        version = r.get("model_version")
        if not version:
            continue
        try:
            metrics_raw = json.loads(r.get("metrics_json", "{}"))
        except (json.JSONDecodeError, TypeError):
            metrics_raw = {}

        reg.record_eval(
            project=r["project_key"],
            version=version,
            dataset={
                "name": r.get("dataset_name", ""),
                "kind": r.get("dataset_kind", ""),
                "ref": r.get("dataset_ref", ""),
            },
            metrics=metrics_raw,
            primary_name=r.get("metric_primary_name", ""),
            primary_value=r.get("metric_primary_value"),
        )
        written += 1
    return written


# =====================================================================
#  Main entry
# =====================================================================
def run_evaluations(
    project_filter: Optional[str] = None,
    limit: Optional[int] = None,
    device: str = "auto",
    write_registry: bool = False,
    outdir: Path = DEFAULT_OUTDIR,
    no_download: bool = False,
    args: Optional[argparse.Namespace] = None,
) -> List[Dict]:
    """Evaluate accuracy for all (or selected) projects."""
    from core import discover_projects, list_projects
    from core.registry import PROJECT_REGISTRY

    discover_projects()

    # Determine effective device
    if device == "auto":
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    all_keys = list_projects()
    if project_filter:
        all_keys = [k for k in all_keys if k == project_filter]

    if limit is not None and limit > 0:
        all_keys = all_keys[:limit]

    results: List[Dict] = []
    for key in all_keys:
        cls = PROJECT_REGISTRY.get(key)
        if cls is None:
            continue
        task = _PROJECT_TYPE_TO_TASK.get(cls.project_type, "detect")

        print(f"  [{task.upper():7s}] {key} ...", end=" ", flush=True)
        r = evaluate_project(key, task, device, no_download=no_download)
        results.append(r)

        status = r.get("status", "?")
        if status == "ok":
            pname = r.get("metric_primary_name", "?")
            pval = r.get("metric_primary_value")
            pval_str = f"{pval:.4f}" if pval is not None else "N/A"
            print(f"OK  {pname}={pval_str}")
        else:
            err = r.get("error", "")
            print(f"FAIL ({status}) {err[:60]}")

    # Write outputs
    csv_path = _write_csv(results, outdir)
    json_path = _write_json(results, outdir)
    meta = _build_run_meta(args or argparse.Namespace())
    meta_path = _write_meta(meta, outdir)

    print()
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Meta: {meta_path}")

    # Optional registry writeback
    if write_registry:
        n = _writeback_registry(results)
        print(f"  Registry: wrote eval data for {n} project(s)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model accuracy for all CV projects",
    )
    parser.add_argument(
        "--project", type=str, default=None,
        help="Evaluate only this project key",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of projects to evaluate",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "0", "1"],
        help="Device for evaluation (default: auto)",
    )
    parser.add_argument(
        "--write-registry", action="store_true",
        help="Write eval metrics back into models/metadata.json",
    )
    parser.add_argument(
        "--outdir", type=str, default=str(DEFAULT_OUTDIR),
        help="Output directory for results (default: benchmarks/results)",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Never attempt dataset downloads; report missing datasets as skipped",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Phase 3B-4 — Accuracy Evaluation Runner")
    print("=" * 65)
    print()

    results = run_evaluations(
        project_filter=args.project,
        no_download=args.no_download,
        limit=args.limit,
        device=args.device,
        write_registry=args.write_registry,
        outdir=Path(args.outdir),
        args=args,
    )

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]
    print()
    print("=" * 65)
    print(f"  ACCURACY EVAL SUMMARY: {len(ok)} evaluated, {len(failed)} failed, {len(results)} total")
    if ok:
        for r in ok:
            pval = r.get("metric_primary_value")
            pval_str = f"{pval:.4f}" if pval is not None else "N/A"
            print(f"    {r['project_key']:40s} {r['metric_primary_name']}={pval_str}")
    if failed:
        # Group failure reasons
        reasons: Dict[str, int] = {}
        for r in failed:
            s = r.get("status", "unknown")
            reasons[s] = reasons.get(s, 0) + 1
        for reason, count in sorted(reasons.items()):
            print(f"    Status: {reason} ({count})")
    print("=" * 65)


if __name__ == "__main__":
    main()
