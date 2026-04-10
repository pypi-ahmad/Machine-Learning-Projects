#!/usr/bin/env python3
"""Benchmark every registered modern.py project and write results to CSV.

Usage::

    python benchmarks/run_all.py                      # benchmark all registered projects
    python benchmarks/run_all.py --projects pedestrian_detection emotion_recognition
    python benchmarks/run_all.py --n-runs 20
    python benchmarks/run_all.py --list                # just list registered projects

Results are written to ``benchmarks/results.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# ── Ensure repo root is on sys.path ──────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from core import discover_projects, list_projects
from core.registry import PROJECT_REGISTRY
from core.runner import benchmark
from models.registry import ModelRegistry, resolve

RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"


def _make_dummy_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic BGR image for benchmarking."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def run_benchmarks(
    project_names: list[str] | None = None,
    n_runs: int = 10,
) -> list[dict]:
    """Benchmark selected (or all) projects and return a list of stat dicts."""
    if project_names is None:
        project_names = list_projects()

    img = _make_dummy_image()
    results: list[dict] = []
    registry = ModelRegistry()

    # Map project_type → resolve task key
    _TASK_MAP = {
        "detection": "detect",
        "classification": "cls",
        "segmentation": "seg",
        "pose": "pose",
        "tracking": "tracking",
        "utility": "detect",  # fallback for non-YOLO utilities
    }

    for name in project_names:
        if name not in PROJECT_REGISTRY:
            print(f"  [SKIP] {name} — not in registry")
            continue

        cls = PROJECT_REGISTRY[name]
        print(f"  [{cls.project_type.upper():14s}] {name} ...", end=" ", flush=True)

        try:
            task_key = _TASK_MAP.get(cls.project_type, "detect")
            weights, ver, fallback = resolve(name, task_key)
            stats = benchmark(name, img, n_runs=n_runs)
            stats["project"] = name
            stats["project_type"] = cls.project_type
            stats["model_version"] = ver or "pretrained"
            stats["model_path"] = weights
            stats["used_pretrained_default"] = fallback
            stats["status"] = "ok"
            print(f"{stats['fps']:.1f} FPS  (mean {stats['mean_latency_s']*1000:.1f} ms)")
        except Exception as exc:
            task_key = _TASK_MAP.get(cls.project_type, "detect")
            try:
                weights, ver, fallback = resolve(name, task_key)
            except Exception:
                weights, ver, fallback = "unknown", None, True
            stats = {
                "project": name,
                "project_type": cls.project_type,
                "model_version": ver or "pretrained",
                "model_path": weights,
                "used_pretrained_default": fallback,
                "status": f"error: {exc}",
                "load_time_s": 0,
                "mean_latency_s": 0,
                "std_latency_s": 0,
                "min_latency_s": 0,
                "max_latency_s": 0,
                "fps": 0,
                "n_runs": n_runs,
            }
            print(f"ERROR: {exc}")

        results.append(stats)

    return results


def write_csv(results: list[dict], path: Path = RESULTS_CSV) -> None:
    """Write benchmark results to a CSV file."""
    if not results:
        return

    fieldnames = [
        "project",
        "project_type",
        "model_version",
        "model_path",
        "used_pretrained_default",
        "status",
        "load_time_s",
        "mean_latency_s",
        "std_latency_s",
        "min_latency_s",
        "max_latency_s",
        "fps",
        "n_runs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all modern CV projects")
    parser.add_argument(
        "--projects",
        nargs="*",
        default=None,
        help="Specific project names to benchmark (default: all)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of inference runs per project (default: 10)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered projects and exit",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 3A — Unified Benchmark Runner")
    print("=" * 60)
    print()

    # Discover all modern.py wrappers
    count = discover_projects()
    print(f"  Discovered {count} modern.py modules  ({len(PROJECT_REGISTRY)} projects registered)")
    print()

    if args.list:
        for name in list_projects():
            cls = PROJECT_REGISTRY[name]
            print(f"  {name:40s}  [{cls.project_type}]")
        return

    results = run_benchmarks(project_names=args.projects, n_runs=args.n_runs)
    write_csv(results)

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    fail = [r for r in results if r["status"] != "ok"]
    print()
    print("=" * 60)
    print(f"  BENCHMARK SUMMARY: {len(ok)} OK, {len(fail)} failed out of {len(results)}")
    if ok:
        fps_vals = [r["fps"] for r in ok]
        print(f"  FPS range: {min(fps_vals):.1f} – {max(fps_vals):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
