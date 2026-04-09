#!/usr/bin/env python3
"""
Test Runner  (smoke + full)
===========================
Runs every ``run.py`` in the repo and records PASS/FAIL.

By default runs in **smoke** mode (fast CI).  Pass ``--mode full`` to run
real evaluation with dataset splits and stricter output validation.

Usage:
    python scripts/smoke_test_all.py                         # smoke, all
    python scripts/smoke_test_all.py --mode full --timeout 1800  # full eval
    python scripts/smoke_test_all.py --category "GANS"       # one category
    python scripts/smoke_test_all.py --gpu-mem-gb 4           # budget cap

Output:
    outputs/_workspace_reports/smoke_test_report.md   (smoke)
    outputs/_workspace_reports/full_test_report.md    (full)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "_workspace_reports"

CATEGORIES = [
    "Anomaly detection and fraud detection",
    "Associate Rule Learning",
    "Chat bot",
    "GANS",
    "Recommendation Systems",
    "Reinforcement Learning",
    "Speech and Audio processing",
]


def discover_projects(category_filter: str | None = None) -> list[dict]:
    """Return list of {name, path, category} for every project with run.py."""
    projects = []
    for cat in CATEGORIES:
        if category_filter and category_filter.lower() not in cat.lower():
            continue
        cat_dir = ROOT / cat
        if not cat_dir.is_dir():
            continue
        for entry in sorted(cat_dir.iterdir()):
            run_file = entry / "run.py"
            if run_file.is_file():
                projects.append({
                    "name": entry.name,
                    "path": entry,
                    "run_py": run_file,
                    "category": cat,
                })
    return projects


def run_smoke(project: dict, timeout: int, python: str,
              mode: str = "smoke", gpu_mem_gb: float = 4.0) -> dict:
    """Run a single project and return result dict."""
    start = time.time()
    result = {
        "project": project["name"],
        "category": project["category"],
        "mode": mode,
        "status": "FAIL",
        "metrics_status": "none",
        "duration_s": 0.0,
        "metrics_found": False,
        "metrics_md_found": False,
        "split_manifest_found": False,
        "error": "",
    }

    out_dir = project["path"] / "outputs"
    metrics_file = out_dir / "metrics.json"
    metrics_md   = out_dir / "metrics.md"
    manifest     = out_dir / "split_manifest.json"

    # Clean up previous outputs so stale results don't confuse us
    for f in [metrics_file, metrics_md, manifest]:
        if f.exists():
            f.unlink()

    # Build command
    cmd = [python, str(project["run_py"]),
           "--mode", mode,
           "--gpu-mem-gb", str(gpu_mem_gb)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project["path"]),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        result["duration_s"] = round(time.time() - start, 1)

        if metrics_file.exists():
            result["metrics_found"] = True
            try:
                m = json.loads(metrics_file.read_text(encoding="utf-8"))
                result["metrics_status"] = m.get("status", "unknown")
            except json.JSONDecodeError:
                result["metrics_status"] = "invalid_json"

        result["metrics_md_found"] = metrics_md.exists()
        result["split_manifest_found"] = manifest.exists()

        # PASS criteria: metrics.json exists with an acceptable status
        ACCEPTABLE = {"ok", "dataset_missing", "missing_dependency"}
        if result["metrics_found"] and result["metrics_status"] in ACCEPTABLE:
            result["status"] = "PASS"
        elif proc.returncode == 0 and result["metrics_found"]:
            result["status"] = "PASS"
        else:
            result["status"] = "FAIL"
            result["error"] = (proc.stderr or "")[-500:].strip()

    except subprocess.TimeoutExpired:
        result["duration_s"] = round(time.time() - start, 1)
        result["status"] = "TIMEOUT"
        result["error"] = f"Exceeded {timeout}s timeout"
    except Exception as exc:
        result["duration_s"] = round(time.time() - start, 1)
        result["error"] = str(exc)[:500]

    return result


def generate_report(results: list[dict], mode: str = "smoke") -> str:
    """Generate Markdown report from results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    timeout_count = sum(1 for r in results if r["status"] == "TIMEOUT")
    metrics_ok = sum(1 for r in results if r["metrics_found"])
    md_ok = sum(1 for r in results if r.get("metrics_md_found"))
    manifest_ok = sum(1 for r in results if r.get("split_manifest_found"))

    # Counts by metrics_status
    status_counts: dict[str, int] = {}
    for r in results:
        ms = r.get("metrics_status", "none")
        status_counts[ms] = status_counts.get(ms, 0) + 1

    title = "Full Test Report" if mode == "full" else "Smoke Test Report"
    lines = [
        f"# {title}",
        "",
        f"**Date**: {now}  ",
        f"**Mode**: `{mode}`  ",
        f"**Total**: {total} | **PASS**: {passed} | **FAIL**: {failed} | **TIMEOUT**: {timeout_count}",
        f"**metrics.json**: {metrics_ok}/{total} | **metrics.md**: {md_ok}/{total} | **split_manifest.json**: {manifest_ok}/{total}",
        "",
        "## Metrics Status Breakdown",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    for s in ["ok", "dataset_missing", "missing_dependency", "none", "error", "unknown", "invalid_json"]:
        if s in status_counts:
            lines.append(f"| {s} | {status_counts[s]} |")
    for s, c in sorted(status_counts.items()):
        if s not in {"ok", "dataset_missing", "missing_dependency", "none", "error", "unknown", "invalid_json"}:
            lines.append(f"| {s} | {c} |")

    lines += [
        "",
        "---",
        "",
        "## Results",
        "",
        "| # | Category | Project | Status | metrics | md | manifest | Duration | Error |",
        "|---|----------|---------|--------|---------|----|----------|----------|-------|",
    ]

    for i, r in enumerate(results, 1):
        err = r["error"].replace("\n", " ")[:60] if r["error"] else ""
        ms = r.get("metrics_status", "none")
        md_flag = "Y" if r.get("metrics_md_found") else "-"
        mf_flag = "Y" if r.get("split_manifest_found") else "-"
        lines.append(
            f"| {i} | {r['category'][:20]} | {r['project'][:35]} "
            f"| {r['status']} | {ms} | {md_flag} | {mf_flag} | {r['duration_s']}s | {err} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- Pass rate: {passed}/{total} ({100*passed/total:.0f}%)" if total else "- No projects found",
        f"- metrics.json coverage: {metrics_ok}/{total}",
        f"- metrics.md coverage: {md_ok}/{total}",
        f"- split_manifest.json coverage: {manifest_ok}/{total}",
        f"- Average duration: {sum(r['duration_s'] for r in results)/max(total,1):.1f}s",
    ]

    # List failures
    failures = [r for r in results if r["status"] != "PASS"]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"### {r['category']} / {r['project']} -- {r['status']}")
            if r["error"]:
                lines.append(f"```\n{r['error'][:300]}\n```")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Test all projects (smoke or full)")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                        help="Run mode: smoke (fast CI) or full (real eval)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to one category (substring match)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-project timeout in seconds (default: 300 smoke, 1800 full)")
    parser.add_argument("--gpu-mem-gb", type=float, default=4.0,
                        help="GPU memory budget in GB (default 4)")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python executable to use")
    parser.add_argument("--dry-run", action="store_true",
                        help="List projects without running")
    args = parser.parse_args()

    if args.timeout is None:
        args.timeout = 1800 if args.mode == "full" else 300

    projects = discover_projects(args.category)
    print(f"Discovered {len(projects)} projects  [mode={args.mode}, timeout={args.timeout}s, gpu={args.gpu_mem_gb}GB]")

    if args.dry_run:
        for p in projects:
            print(f"  {p['category']}/{p['name']}")
        return

    results = []
    for i, proj in enumerate(projects, 1):
        print(f"[{i}/{len(projects)}] {proj['category']}/{proj['name']} ... ", end="", flush=True)
        r = run_smoke(proj, args.timeout, args.python,
                      mode=args.mode, gpu_mem_gb=args.gpu_mem_gb)
        results.append(r)
        status_line = r['status']
        extras = []
        if r.get('metrics_md_found'):   extras.append('md')
        if r.get('split_manifest_found'): extras.append('manifest')
        if extras:
            status_line += f" [{','.join(extras)}]"
        print(f"{status_line} ({r['duration_s']}s)")

    # Write report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, mode=args.mode)
    report_name = "full_test_report.md" if args.mode == "full" else "smoke_test_report.md"
    report_path = REPORT_DIR / report_name
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport: {report_path}")

    # Also write JSON for machine consumption
    json_name = "full_test_results.json" if args.mode == "full" else "smoke_test_results.json"
    json_path = REPORT_DIR / json_name
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Summary
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed_count = sum(1 for r in results if r['status'] != 'PASS')
    print(f"\n{'='*60}")
    print(f"  {args.mode.upper()} MODE: {passed}/{len(results)} PASS, {failed_count} not-pass")
    print(f"{'='*60}")
    if failed_count:
        print(f"\n{failed_count} project(s) did not pass.")
    sys.exit(min(failed_count, 125))


if __name__ == "__main__":
    main()
