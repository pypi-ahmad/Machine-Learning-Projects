#!/usr/bin/env python3
"""
Run All Projects in Full Mode
==============================
Runs ``python run.py --mode full --profile <profile>`` in every project directory.

Statuses
--------
* **PASS**     – metrics.json exists with ``"status": "ok"``
* **EXPECTED** – known data-access issue (dataset_missing, auth_error, …)
* **TIMEOUT**  – ran past per-project timeout
* **OOM**      – CUDA out-of-memory or memory-related crash
* **FAIL**     – unhandled crash or no metrics.json

Usage:
    python scripts/run_all_full.py                         # all 50 projects
    python scripts/run_all_full.py --project 1             # single project
    python scripts/run_all_full.py --project 1 --project 5 # multiple projects
    python scripts/run_all_full.py --profile full_4gb      # explicit profile
    python scripts/run_all_full.py --timeout 7200          # 2-hour per-project timeout
    python scripts/run_all_full.py --skip-expected          # skip dataset-missing projects
    python scripts/run_all_full.py --resume                 # skip projects with existing ok metrics

Output:
    outputs/_workspace_reports/full_run_report.md
    outputs/_workspace_reports/full_run_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "outputs" / "_workspace_reports"

# Statuses in metrics.json that count as "expected" (not a code bug)
_EXPECTED_STATUSES = frozenset({
    "dataset_missing",
    "manual_action_required",
    "auth_error",
    "missing_dependency",
    "download_error",
})

# Projects known to require manual dataset setup (from smoke test baseline)
# These will be skipped when --skip-expected is used
_KNOWN_EXPECTED: set[int] = set()


def find_projects() -> list[Path]:
    """Return sorted list of project directories that have run.py."""
    projects = []
    for d in sorted(ROOT.iterdir()):
        if d.is_dir() and d.name.startswith("Deep Learning Projects") and (d / "run.py").exists():
            projects.append(d)
    return projects


def extract_num(folder_name: str) -> int:
    """Extract project number from folder name."""
    m = re.search(r"Projects?\s*(\d+)", folder_name)
    return int(m.group(1)) if m else 0


def _read_metrics(output_dir: Path) -> dict | None:
    """Read metrics.json if it exists."""
    mj = output_dir / "metrics.json"
    if mj.is_file():
        try:
            return json.loads(mj.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def classify(result: dict) -> str:
    """Classify a run result."""
    metrics = result.get("_metrics")

    # Timeout/OOM already classified
    if result["status"] in ("TIMEOUT", "OOM"):
        return result["status"]

    if metrics is None:
        return "FAIL"

    status = metrics.get("status", "")
    if status == "ok":
        return "PASS"
    if status in _EXPECTED_STATUSES:
        return "EXPECTED"
    if result["exit_code"] == 0:
        return "PASS" if status == "" else "EXPECTED"
    return "FAIL"


def _detect_oom(stderr: str) -> bool:
    """Check if stderr indicates CUDA OOM."""
    oom_patterns = [
        "CUDA out of memory",
        "OutOfMemoryError",
        "CUDA error: out of memory",
        "torch.cuda.OutOfMemoryError",
    ]
    return any(p.lower() in stderr.lower() for p in oom_patterns)


def run_full(project_dir: Path, profile: str, timeout: int,
             extra_args: list[str] | None = None) -> dict:
    """Run full-mode for one project, return result dict."""
    run_py = project_dir / "run.py"
    output_dir = project_dir / "outputs"
    result = {
        "project": project_dir.name,
        "num": extract_num(project_dir.name),
        "status": "UNKNOWN",
        "exit_code": -1,
        "duration_s": 0.0,
        "has_metrics_json": False,
        "has_metrics_md": False,
        "has_confusion_matrix": False,
        "has_training_curves": False,
        "has_split_manifest": False,
        "stdout_tail": "",
        "stderr_tail": "",
        "_metrics": None,
        "reason": "",
        "dataset_url": "",
        "suggested_fix": "",
    }

    cmd = [sys.executable, str(run_py), "--mode", "full", "--profile", profile]
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_dir),
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        result["exit_code"] = proc.returncode
        result["stdout_tail"] = proc.stdout[-1000:] if proc.stdout else ""
        result["stderr_tail"] = proc.stderr[-1000:] if proc.stderr else ""

        # Check for OOM in stderr
        if _detect_oom(proc.stderr or ""):
            result["status"] = "OOM"

    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["stderr_tail"] = f"Timed out after {timeout}s"
    except Exception as exc:
        result["status"] = "ERROR"
        result["stderr_tail"] = str(exc)[:500]

    result["duration_s"] = round(time.time() - start, 1)

    # Check output artifacts
    if output_dir.is_dir():
        result["has_metrics_json"] = (output_dir / "metrics.json").exists()
        result["has_metrics_md"] = (output_dir / "metrics.md").exists()
        result["has_confusion_matrix"] = (output_dir / "confusion_matrix.png").exists()
        result["has_training_curves"] = (output_dir / "training_curves.png").exists()
        result["has_split_manifest"] = (output_dir / "split_manifest.json").exists()

    # Read metrics.json
    if output_dir.is_dir():
        result["_metrics"] = _read_metrics(output_dir)
        if result["_metrics"]:
            result["reason"] = result["_metrics"].get("reason", "")
            result["dataset_url"] = result["_metrics"].get("dataset_url", "")
            result["suggested_fix"] = result["_metrics"].get("suggested_fix", "")

    # Classify if not already set
    if result["status"] == "UNKNOWN":
        result["status"] = classify(result)

    return result


def _already_ok(project_dir: Path) -> bool:
    """Check if project already has metrics.json with status=ok."""
    metrics = _read_metrics(project_dir / "outputs")
    return metrics is not None and metrics.get("status") == "ok"


def generate_report(results: list[dict], profile: str,
                    total_duration: float) -> str:
    """Build markdown report."""
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_expected = sum(1 for r in results if r["status"] == "EXPECTED")
    n_timeout = sum(1 for r in results if r["status"] == "TIMEOUT")
    n_oom = sum(1 for r in results if r["status"] == "OOM")
    n_fail = sum(1 for r in results if r["status"]
                 not in ("PASS", "EXPECTED", "TIMEOUT", "OOM", "SKIPPED"))
    n_skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    total = len(results)

    lines = [
        "# Full Run Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Profile**: `{profile}`",
        f"**Total Duration**: {total_duration:.0f}s "
        f"({total_duration / 60:.1f} min)",
        f"**Projects**: {total}  |  **Pass**: {n_pass}  "
        f"|  **Expected**: {n_expected}  |  **Timeout**: {n_timeout}  "
        f"|  **OOM**: {n_oom}  |  **Fail**: {n_fail}",
        "",
    ]

    if n_skipped:
        lines.append(f"**Skipped**: {n_skipped} (--skip-expected or --resume)")
        lines.append("")

    # Summary table
    lines.extend([
        "## Results",
        "",
        "| # | Project | Status | Time (s) | metrics | confusion | curves | manifest |",
        "|---|---------|--------|----------|:---:|:---:|:---:|:---:|",
    ])

    for r in sorted(results, key=lambda x: x["num"]):
        status_icon = {
            "PASS": "PASS",
            "EXPECTED": "EXPECTED",
            "FAIL": "**FAIL**",
            "TIMEOUT": "**TIMEOUT**",
            "OOM": "**OOM**",
            "SKIPPED": "SKIP",
        }.get(r["status"], r["status"])
        ck = lambda v: "Y" if v else "-"
        lines.append(
            f"| P{r['num']} | {r['project'][:45]} | {status_icon} "
            f"| {r['duration_s']} "
            f"| {ck(r['has_metrics_json'])} | {ck(r['has_confusion_matrix'])} "
            f"| {ck(r['has_training_curves'])} | {ck(r.get('has_split_manifest'))} |"
        )

    # Expected details
    expected = [r for r in results if r["status"] == "EXPECTED"]
    if expected:
        lines.extend(["", "## Expected (Dataset Access Issues)", ""])
        lines.append("| # | Reason | Dataset URL |")
        lines.append("|---|--------|-------------|")
        for r in sorted(expected, key=lambda x: x["num"]):
            lines.append(
                f"| P{r['num']} | {r['reason'][:80]} | {r['dataset_url']} |"
            )

    # OOM details
    ooms = [r for r in results if r["status"] == "OOM"]
    if ooms:
        lines.extend(["", "## OOM (GPU Memory Exhausted)", ""])
        lines.append("> These projects ran out of GPU memory even after batch-size backoff.")
        lines.append("> Try a lower img_size or smaller model, or increase VRAM budget.")
        lines.append("")
        lines.append("| # | Project | Time (s) |")
        lines.append("|---|---------|----------|")
        for r in sorted(ooms, key=lambda x: x["num"]):
            lines.append(f"| P{r['num']} | {r['project'][:50]} | {r['duration_s']} |")

    # Timeout details
    timeouts = [r for r in results if r["status"] == "TIMEOUT"]
    if timeouts:
        lines.extend(["", "## Timeouts", ""])
        lines.append("> Increase `--timeout` or use faster hardware.")
        lines.append("")
        lines.append("| # | Project | Time (s) |")
        lines.append("|---|---------|----------|")
        for r in sorted(timeouts, key=lambda x: x["num"]):
            lines.append(f"| P{r['num']} | {r['project'][:50]} | {r['duration_s']} |")

    # Failure details
    failures = [r for r in results
                if r["status"] not in ("PASS", "EXPECTED", "TIMEOUT", "OOM", "SKIPPED")]
    if failures:
        lines.extend(["", "## Failure Details", ""])
        for r in sorted(failures, key=lambda x: x["num"]):
            lines.append(f"### P{r['num']} — {r['status']}")
            lines.append("```")
            if r["stderr_tail"]:
                lines.append(r["stderr_tail"].strip()[-500:])
            elif r["stdout_tail"]:
                lines.append(r["stdout_tail"].strip()[-300:])
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run all projects in full mode with a profile")
    parser.add_argument("--project", type=int, action="append", default=None,
                        help="Run only these project numbers (repeatable)")
    parser.add_argument("--profile", type=str, default="full_4gb",
                        help="Profile name (default: full_4gb)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-project timeout in seconds (default: 3600 = 1h)")
    parser.add_argument("--skip-expected", action="store_true",
                        help="Skip projects known to have dataset issues")
    parser.add_argument("--resume", action="store_true",
                        help="Skip projects that already have status=ok in metrics.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="List projects without running them")
    parser.add_argument("extra", nargs="*",
                        help="Extra args passed to each run.py (e.g. --device cuda)")
    args = parser.parse_args()

    projects = find_projects()
    if args.project:
        project_set = set(args.project)
        projects = [p for p in projects if extract_num(p.name) in project_set]
        if not projects:
            print(f"  [ERROR] No matching projects found for: {args.project}")
            sys.exit(1)

    print("=" * 60)
    print(f"  Full Run — {len(projects)} project(s)")
    print(f"  Profile  : {args.profile}")
    print(f"  Timeout  : {args.timeout}s per project")
    print("=" * 60)

    if args.dry_run:
        for p in projects:
            num = extract_num(p.name)
            flag = ""
            if args.skip_expected and _read_metrics(p / "outputs"):
                m = _read_metrics(p / "outputs")
                if m and m.get("status") in _EXPECTED_STATUSES:
                    flag = " [SKIP: expected]"
            if args.resume and _already_ok(p):
                flag = " [SKIP: already ok]"
            print(f"  P{num:>2}  {p.name}{flag}")
        return

    results = []
    run_start = time.time()

    for i, project in enumerate(projects, 1):
        num = extract_num(project.name)
        short_name = project.name[:50]

        # Check skip conditions
        if args.skip_expected:
            m = _read_metrics(project / "outputs")
            if m and m.get("status") in _EXPECTED_STATUSES:
                print(f"\n[{i}/{len(projects)}] P{num}: {short_name}...")
                print(f"  -> SKIPPED (expected: {m.get('status')})")
                results.append({
                    "project": project.name, "num": num, "status": "SKIPPED",
                    "exit_code": -1, "duration_s": 0.0,
                    "has_metrics_json": True, "has_metrics_md": False,
                    "has_confusion_matrix": False, "has_training_curves": False,
                    "has_split_manifest": False,
                    "stdout_tail": "", "stderr_tail": "",
                    "_metrics": m, "reason": m.get("status", ""),
                    "dataset_url": "", "suggested_fix": "",
                })
                continue

        if args.resume and _already_ok(project):
            print(f"\n[{i}/{len(projects)}] P{num}: {short_name}...")
            print(f"  -> SKIPPED (already ok)")
            m = _read_metrics(project / "outputs")
            results.append({
                "project": project.name, "num": num, "status": "SKIPPED",
                "exit_code": 0, "duration_s": 0.0,
                "has_metrics_json": True, "has_metrics_md": True,
                "has_confusion_matrix": False, "has_training_curves": False,
                "has_split_manifest": False,
                "stdout_tail": "", "stderr_tail": "",
                "_metrics": m, "reason": "",
                "dataset_url": "", "suggested_fix": "",
            })
            continue

        # Run the project
        print(f"\n[{i}/{len(projects)}] P{num}: {short_name}...")
        result = run_full(project, args.profile, args.timeout,
                          extra_args=args.extra or None)
        results.append(result)

        icon = {
            "PASS": "OK", "EXPECTED": "EXPECTED", "FAIL": "FAIL",
            "TIMEOUT": "TIMEOUT", "OOM": "OOM", "ERROR": "ERR",
        }
        label = icon.get(result["status"], "??")
        extra = ""
        if result.get("reason"):
            extra = f"  ({result['reason'][:50]})"
        elapsed = time.time() - run_start
        print(f"  -> {label} ({result['duration_s']}s)  "
              f"[total elapsed: {elapsed:.0f}s]{extra}")

    total_duration = time.time() - run_start

    # Write report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = generate_report(results, args.profile, total_duration)
    report_path = REPORT_DIR / "full_run_report.md"
    report_path.write_text(report, encoding="utf-8")

    # JSON results
    json_results = [{k: v for k, v in r.items() if k != "_metrics"}
                    for r in results]
    json_path = REPORT_DIR / "full_run_results.json"
    json_path.write_text(json.dumps(json_results, indent=2), encoding="utf-8")

    # Summary
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_expected = sum(1 for r in results if r["status"] == "EXPECTED")
    n_timeout = sum(1 for r in results if r["status"] == "TIMEOUT")
    n_oom = sum(1 for r in results if r["status"] == "OOM")
    n_fail = sum(1 for r in results if r["status"]
                 not in ("PASS", "EXPECTED", "TIMEOUT", "OOM", "SKIPPED"))
    n_skipped = sum(1 for r in results if r["status"] == "SKIPPED")

    print(f"\n{'=' * 60}")
    parts = [f"{n_pass} ok", f"{n_expected} expected"]
    if n_timeout:
        parts.append(f"{n_timeout} timeout")
    if n_oom:
        parts.append(f"{n_oom} oom")
    parts.append(f"{n_fail} failed")
    if n_skipped:
        parts.append(f"{n_skipped} skipped")
    print(f"  Results  : {' / '.join(parts)}")
    print(f"  Duration : {total_duration:.0f}s ({total_duration / 60:.1f} min)")
    print(f"  Report   : {report_path}")
    print(f"{'=' * 60}")

    # Exit 0 if no genuine failures
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
