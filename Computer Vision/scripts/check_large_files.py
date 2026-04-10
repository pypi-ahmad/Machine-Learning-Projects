#!/usr/bin/env python3
"""Size guard — scans for files exceeding 50 MB threshold.

Usage::

    # Check all tracked files
    python scripts/check_large_files.py

    # Check only staged files (used by pre-commit hook)
    python scripts/check_large_files.py --staged

    # Quiet mode (exit code only)
    python scripts/check_large_files.py --staged --quiet
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 50 * 1024 * 1024  # 50 MB

REPO = Path(__file__).resolve().parents[1]


def get_staged_files() -> list[str]:
    """Return list of staged file paths (relative to repo root)."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=str(REPO),
            text=True,
        )
        return [f.strip() for f in out.splitlines() if f.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def get_all_tracked_files() -> list[str]:
    """Return list of all tracked file paths."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=str(REPO),
            text=True,
        )
        return [f.strip() for f in out.splitlines() if f.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: walk the directory
        files = []
        for root, dirs, fnames in os.walk(REPO):
            # Skip hidden dirs and common ignores
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {
                "__pycache__", "node_modules", ".venv", "venv",
            }]
            for fn in fnames:
                fp = Path(root) / fn
                files.append(str(fp.relative_to(REPO)))
        return files


def check_files(files: list[str], quiet: bool = False) -> list[tuple[str, int]]:
    """Check files for size violations. Returns list of (path, size_bytes) for violations."""
    violations = []
    for rel in files:
        fp = REPO / rel
        if not fp.exists():
            continue
        size = fp.stat().st_size
        if size > MAX_BYTES:
            violations.append((rel, size))
            if not quiet:
                mb = size / (1024 * 1024)
                print(f"  BLOCKED: {rel} ({mb:.1f} MB > 50 MB)")
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for files exceeding 50 MB")
    parser.add_argument("--staged", action="store_true", help="Check only staged files")
    parser.add_argument("--quiet", action="store_true", help="No output, exit code only")
    args = parser.parse_args()

    if args.staged:
        files = get_staged_files()
    else:
        files = get_all_tracked_files()

    if not args.quiet:
        mode = "staged" if args.staged else "all tracked"
        print(f"Size guard: checking {len(files)} {mode} files (limit: 50 MB)")

    violations = check_files(files, quiet=args.quiet)

    if violations:
        if not args.quiet:
            print(f"\n{len(violations)} file(s) exceed 50 MB limit.")
            print("Add to .gitignore or use Git LFS.")
        sys.exit(1)
    else:
        if not args.quiet:
            print("All files within size limit.")
        sys.exit(0)


if __name__ == "__main__":
    main()
