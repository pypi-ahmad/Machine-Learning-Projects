#!/usr/bin/env python3
"""Validate dataset configs — check that referenced paths exist.

Usage::

    python scripts/validate_datasets.py

Prints a summary table. Always exits 0 (informational only).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONFIGS_DIR = REPO_ROOT / "configs" / "datasets"


def _load_yaml_simple(path: Path) -> dict:
    """Minimal YAML loader — handles the flat schema we generate.

    Avoids requiring PyYAML for this simple validation script.
    Falls back to PyYAML if available.
    """
    try:
        import yaml  # type: ignore[import-untyped]
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        pass

    # Very simple key: value parser for our known flat schema
    data: dict = {}
    current_section: dict | None = None
    section_key: str = ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Remove inline comments
        if "  #" in line:
            line = line[: line.index("  #")].rstrip()

        indent = len(raw_line) - len(raw_line.lstrip())

        if indent == 0 and ":" in line:
            # Top-level key
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val == "" or val == "null":
                # Section header or null
                if key in ("dataset", "metrics"):
                    current_section = {}
                    section_key = key
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
            if val == "null" or val == "":
                current_section[key] = None
            elif val.lower() == "true":
                current_section[key] = True
            elif val.lower() == "false":
                current_section[key] = False
            else:
                current_section[key] = val

    return data


def validate() -> list[dict]:
    """Scan all dataset configs and check existence of referenced paths."""
    results: list[dict] = []

    if not CONFIGS_DIR.exists():
        print(f"  No configs directory found at {CONFIGS_DIR}")
        return results

    yamls = sorted(CONFIGS_DIR.glob("*.yaml"))
    if not yamls:
        print("  No .yaml configs found")
        return results

    for ypath in yamls:
        project_key = ypath.stem
        try:
            cfg = _load_yaml_simple(ypath)
        except Exception as exc:
            results.append({
                "project_key": project_key,
                "task": "?",
                "kind": "?",
                "configured": False,
                "exists": False,
                "notes": f"parse error: {exc}",
            })
            continue

        task = cfg.get("task", "?")
        ds = cfg.get("dataset", {}) or {}
        kind = ds.get("kind", "?")
        data_yaml = ds.get("data_yaml")
        val_dir = ds.get("val")
        root = ds.get("root")

        # Determine the key path(s) to check
        configured = False
        exists = False

        if kind == "ultralytics_yaml":
            if data_yaml and data_yaml != "null":
                configured = True
                check_path = REPO_ROOT / data_yaml
                exists = check_path.exists()
            else:
                notes = "data_yaml not set"
        elif kind == "imagefolder":
            if val_dir and val_dir != "null":
                configured = True
                check_path = REPO_ROOT / val_dir
                exists = check_path.exists()
            elif root and root != "null":
                configured = True
                check_path = REPO_ROOT / root / "val"
                exists = check_path.exists()
            else:
                notes = "val path not set"
        else:
            configured = bool(root and root != "null")
            if configured:
                exists = (REPO_ROOT / root).exists()

        notes_str = ds.get("notes", "") or ""
        if not configured:
            notes_str = notes_str or "placeholder config"

        # Download info
        dl = ds.get("download", {}) or {}
        download_enabled = bool(dl.get("enabled", False))
        has_sources = bool(dl.get("sources"))

        results.append({
            "project_key": project_key,
            "task": task,
            "kind": kind,
            "configured": configured,
            "exists": exists,
            "download_enabled": download_enabled,
            "has_sources": has_sources,
            "notes": notes_str,
        })

    return results


def main() -> None:
    print("=" * 80)
    print("  Dataset Config Validation")
    print("=" * 80)
    print()

    results = validate()
    if not results:
        print("  No dataset configs to validate.")
        return

    # Print table
    hdr = (
        f"  {'project_key':<42s} {'task':<7s} {'kind':<18s} "
        f"{'config?':<9s} {'exists?':<9s} {'dl_on?':<8s} {'srcs?':<7s} notes"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    n_configured = 0
    n_exists = 0
    n_dl_enabled = 0
    n_has_sources = 0
    for r in results:
        cfg_str = "YES" if r["configured"] else "no"
        ex_str = "YES" if r["exists"] else "no"
        dl_str = "YES" if r["download_enabled"] else "no"
        src_str = "YES" if r["has_sources"] else "no"
        if r["configured"]:
            n_configured += 1
        if r["exists"]:
            n_exists += 1
        if r["download_enabled"]:
            n_dl_enabled += 1
        if r["has_sources"]:
            n_has_sources += 1
        print(
            f"  {r['project_key']:<42s} {r['task']:<7s} {r['kind']:<18s} "
            f"{cfg_str:<9s} {ex_str:<9s} {dl_str:<8s} {src_str:<7s} {r.get('notes', '')}"
        )

    print()
    print(
        f"  SUMMARY: {len(results)} configs | {n_configured} configured | "
        f"{n_exists} on disk | {n_dl_enabled} download enabled | {n_has_sources} have sources"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
