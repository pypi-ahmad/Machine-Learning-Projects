#!/usr/bin/env python3
"""Verify that benchmarks/evaluate_accuracy.py flags behave correctly.

Checks:
  1. --no-download flag is accepted
  2. --no-download prevents any dataset download attempts
  3. Result status is 'dataset_missing_no_download' (not a download error)
  4. --limit works alongside --no-download
"""

import subprocess
import sys
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")


def main() -> None:
    global PASS, FAIL
    print("=" * 65)
    print("  Evaluator Flag Checks")
    print("=" * 65)
    print()

    # 1. --no-download flag is accepted (exit code 0)
    r = subprocess.run(
        [sys.executable, "-m", "benchmarks.evaluate_accuracy",
         "--limit", "1", "--no-download"],
        capture_output=True, text=True, timeout=120, cwd=str(REPO),
    )
    check("--no-download accepted (exit code 0)", r.returncode == 0,
          r.stderr.strip()[:100] if r.returncode != 0 else "")

    # 2. No download attempt messages in output
    combined = r.stdout + r.stderr
    no_attempt = "Attempting download" not in combined
    check("--no-download prevents download attempts", no_attempt,
          "found 'Attempting download' in output" if not no_attempt else "")

    # 3. Result status is dataset_missing_no_download (read the JSON output)
    json_path = REPO / "benchmarks" / "results" / "accuracy_results.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        if data:
            statuses = {r.get("status") for r in data}
            expected_ok = statuses <= {"ok", "dataset_missing_no_download",
                                        "missing_dataset_config",
                                        "unsupported_cls_backend"}
            check("result status is acceptable (no download errors)",
                  expected_ok, f"statuses={statuses}")
        else:
            check("result status is acceptable", False, "empty results JSON")
    else:
        check("result status is acceptable", False, "results JSON missing")

    # 4. --limit works with --no-download
    r2 = subprocess.run(
        [sys.executable, "-m", "benchmarks.evaluate_accuracy",
         "--limit", "3", "--no-download"],
        capture_output=True, text=True, timeout=120, cwd=str(REPO),
    )
    check("--limit 3 --no-download exits cleanly", r2.returncode == 0,
          r2.stderr.strip()[:100] if r2.returncode != 0 else "")

    # Summary
    total = PASS + FAIL
    print()
    print("=" * 65)
    print(f"  EVALUATOR FLAG CHECKS: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 65)
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == "__main__":
    main()
