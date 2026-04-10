#!/usr/bin/env python3
"""Dataset Download Helper — list sources, generate download commands, check status.

Usage:
    python scripts/download_datasets.py                # Show all datasets + status
    python scripts/download_datasets.py --project KEY  # Show one project
    python scripts/download_datasets.py --kaggle-only  # Only Kaggle-downloadable
    python scripts/download_datasets.py --missing-only  # Only datasets not yet on disk
    python scripts/download_datasets.py --json          # Machine-readable JSON output
"""

from __future__ import annotations
import argparse, json, os, pathlib, sys, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "configs" / "datasets"

# ── Minimal YAML parser (no PyYAML dependency) ──────────────────────────────

def _load_yaml_fallback(path: pathlib.Path) -> dict:
    """Minimal hand-rolled YAML parser for our flat-ish config schema."""
    result: dict = {}
    stack: list[tuple[int, dict]] = [(-1, result)]
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.rstrip("\n\r")
            if not stripped or stripped.lstrip().startswith("#"):
                continue
            indent = len(stripped) - len(stripped.lstrip())
            line = stripped.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # Pop stack to correct level
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1] if stack else result
            if val == "" or val == "":
                # Sub-dict
                child: dict = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                # Scalar
                if val == "null":
                    parent[key] = None
                elif val in ("true", "false"):
                    parent[key] = val == "true"
                elif val.startswith('"') and val.endswith('"'):
                    parent[key] = val[1:-1]
                elif val.startswith("'") and val.endswith("'"):
                    parent[key] = val[1:-1]
                else:
                    try:
                        parent[key] = int(val)
                    except ValueError:
                        try:
                            parent[key] = float(val)
                        except ValueError:
                            parent[key] = val
    return result


def load_yaml(path: pathlib.Path) -> dict:
    """Load YAML file, preferring PyYAML if available."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _load_yaml_fallback(path)


def load_all_configs() -> list[dict]:
    """Load all dataset configs, sorted by project_key."""
    configs = []
    for p in sorted(CFG_DIR.glob("*.yaml")):
        cfg = load_yaml(p)
        if "project_key" in cfg:
            configs.append(cfg)
    return configs


# ── Status checking ─────────────────────────────────────────────────────────

def check_dataset_status(cfg: dict) -> dict:
    """Check if a dataset exists on disk and return status info."""
    ds = cfg.get("dataset", {})
    src = cfg.get("source", {})
    task = cfg.get("task", "unknown")
    pk = cfg.get("project_key", "unknown")
    
    # Determine what path to check
    kind = ds.get("kind", "")
    root_path = ds.get("root")
    data_yaml = ds.get("data_yaml")
    val_path = ds.get("val")
    
    on_disk = False
    disk_detail = "not found"
    
    if root_path:
        full_root = ROOT / root_path
        if full_root.is_dir():
            # Count files
            n_files = sum(1 for _ in full_root.rglob("*") if _.is_file())
            if n_files > 0:
                on_disk = True
                disk_detail = f"{n_files} files in {root_path}"
            else:
                disk_detail = f"empty dir: {root_path}"
        else:
            disk_detail = f"missing: {root_path}"
    
    # Check for data.yaml specifically
    has_data_yaml = False
    if data_yaml:
        has_data_yaml = (ROOT / data_yaml).is_file()
    
    # Determine source type
    url = src.get("url") or ""
    source_type = "unknown"
    if not url:
        source_type = "none"
    elif "kaggle.com" in url:
        source_type = "kaggle"
    elif "github.com" in url:
        source_type = "github"
    elif "tensorflow.org" in url:
        source_type = "tfds"
    else:
        source_type = "web"
    
    # Generate kaggle CLI command if applicable
    kaggle_cmd = None
    if source_type == "kaggle":
        # Try to extract dataset slug or competition name
        m = re.search(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)", url)
        if m:
            kaggle_cmd = f"kaggle datasets download -d {m.group(1)}"
        else:
            m = re.search(r"kaggle\.com/c/([^/?#]+)", url)
            if m:
                kaggle_cmd = f"kaggle competitions download -c {m.group(1)}"
    
    return {
        "project_key": pk,
        "task": task,
        "dataset_name": ds.get("name", ""),
        "source_url": url or None,
        "alt_url": src.get("alt_url"),
        "license": src.get("license", ""),
        "source_type": source_type,
        "on_disk": on_disk,
        "disk_detail": disk_detail,
        "has_data_yaml": has_data_yaml,
        "kaggle_cmd": kaggle_cmd,
        "download_notes": src.get("download_notes", ""),
    }


# ── Display ─────────────────────────────────────────────────────────────────

TASK_COLORS = {
    "detect": "\033[93m",   # yellow
    "seg": "\033[96m",      # cyan
    "pose": "\033[95m",     # magenta
    "cls": "\033[92m",      # green
    "utility": "\033[90m",  # gray
}
RESET = "\033[0m"
GREEN_CHECK = "\033[92m✓\033[0m"
RED_X = "\033[91m✗\033[0m"


def print_table(statuses: list[dict]):
    """Print a human-readable table of dataset statuses."""
    # Header
    print()
    print(f"{'Project':<35} {'Task':<8} {'Source':<8} {'On Disk':<8} {'Dataset'}")
    print("─" * 100)
    
    by_task = {}
    for s in statuses:
        task = s["task"]
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(s)
    
    task_order = ["detect", "seg", "pose", "cls", "utility"]
    
    for task in task_order:
        if task not in by_task:
            continue
        color = TASK_COLORS.get(task, "")
        items = sorted(by_task[task], key=lambda x: x["project_key"])
        
        for s in items:
            check = GREEN_CHECK if s["on_disk"] else RED_X
            src_short = s["source_type"][:6]
            name = s["dataset_name"][:40]
            print(f"  {s['project_key']:<33} {color}{task:<8}{RESET} {src_short:<8} {check:<8} {name}")
    
    # Summary
    total = len(statuses)
    on_disk = sum(1 for s in statuses if s["on_disk"])
    with_url = sum(1 for s in statuses if s["source_url"])
    kaggle_count = sum(1 for s in statuses if s["source_type"] == "kaggle")
    
    print("─" * 100)
    print(f"  TOTAL: {total} projects | {with_url} with source URLs | "
          f"{kaggle_count} from Kaggle | {on_disk} on disk")
    print()


def print_detail(s: dict):
    """Print detailed info for a single project."""
    print(f"\n{'═' * 60}")
    print(f"  Project:  {s['project_key']}")
    print(f"  Task:     {s['task']}")
    print(f"  Dataset:  {s['dataset_name']}")
    print(f"  License:  {s['license']}")
    print(f"  On Disk:  {'YES' if s['on_disk'] else 'NO'} ({s['disk_detail']})")
    if s.get("has_data_yaml"):
        print(f"  data.yaml: present")
    print(f"{'─' * 60}")
    if s["source_url"]:
        print(f"  Source:   {s['source_url']}")
    if s["alt_url"]:
        print(f"  Alt:      {s['alt_url']}")
    if s["kaggle_cmd"]:
        print(f"  Kaggle:   {s['kaggle_cmd']}")
    if s["download_notes"]:
        print(f"  Notes:    {s['download_notes']}")
    print(f"{'═' * 60}\n")


def print_kaggle_commands(statuses: list[dict]):
    """Print all Kaggle download commands for easy copy-paste."""
    cmds = [(s["project_key"], s["kaggle_cmd"]) for s in statuses if s["kaggle_cmd"]]
    if not cmds:
        print("No Kaggle-downloadable datasets found.")
        return
    
    print(f"\n# Kaggle download commands ({len(cmds)} datasets)")
    print("# Requires: pip install kaggle && kaggle API key configured\n")
    for pk, cmd in sorted(cmds):
        target = f"data/{pk}"
        print(f"# {pk}")
        print(f"mkdir -p {target}")
        print(f"{cmd} -p {target} --unzip")
        print()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset download helper")
    parser.add_argument("--project", "-p", help="Show details for a specific project")
    parser.add_argument("--kaggle-only", action="store_true", help="Only show Kaggle-downloadable datasets")
    parser.add_argument("--missing-only", action="store_true", help="Only show datasets not on disk")
    parser.add_argument("--kaggle-cmds", action="store_true", help="Print Kaggle download commands")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    
    configs = load_all_configs()
    if not configs:
        print(f"No configs found in {CFG_DIR}")
        sys.exit(1)
    
    statuses = [check_dataset_status(c) for c in configs]
    
    # Filter by project
    if args.project:
        statuses = [s for s in statuses if s["project_key"] == args.project]
        if not statuses:
            print(f"Project '{args.project}' not found.")
            sys.exit(1)
    
    # Filter by source type
    if args.kaggle_only:
        statuses = [s for s in statuses if s["source_type"] == "kaggle"]
    
    # Filter by disk presence
    if args.missing_only:
        statuses = [s for s in statuses if not s["on_disk"]]
    
    # Output
    if args.json:
        print(json.dumps(statuses, indent=2, default=str))
    elif args.kaggle_cmds:
        print_kaggle_commands(statuses)
    elif args.project and len(statuses) == 1:
        print_detail(statuses[0])
    else:
        print_table(statuses)
        if not args.json:
            print("  Use --project KEY for details | --kaggle-cmds for download commands")
            print()


if __name__ == "__main__":
    main()
