#!/usr/bin/env python3
"""Central dataset downloader — attempt automatic download for any project.

Supports: zip, tar, kaggle, gdrive, git (shallow clone).

Usage from code::

    from utils.data_downloader import download_dataset, load_dataset_config
    cfg = load_dataset_config("face_mask_detection")
    result = download_dataset(cfg)
    # result = {"ok": True, "source": "primary", "error": None}

Usage from CLI::

    python -m utils.data_downloader face_mask_detection
    python -m utils.data_downloader --all
    python -m utils.data_downloader --all --force
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs" / "datasets"

log = logging.getLogger("utils.data_downloader")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


# ── YAML loader ──────────────────────────────────────────────────────────────

def _load_yaml_fallback(path: Path) -> dict:
    """Minimal YAML parser for our dataset config schema (supports nested 2-level + lists)."""
    result: dict = {}
    stack: list[tuple[int, dict | list]] = [(-1, result)]

    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(stripped) - len(stripped.lstrip())
        line = stripped.strip()

        # Remove inline comments
        if "  #" in line:
            line = line[: line.index("  #")].rstrip()

        # Pop stack to parent
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]

        # List item
        if line.startswith("- "):
            item_str = line[2:].strip()
            if isinstance(parent, list):
                if ":" in item_str and not item_str.startswith('"'):
                    # Dict item in list
                    child: dict = {}
                    k, _, v = item_str.partition(":")
                    child[k.strip()] = _parse_val(v.strip())
                    parent.append(child)
                    stack.append((indent + 2, child))
                else:
                    parent.append(_parse_val(item_str))
            continue

        if ":" not in line:
            continue

        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()

        if isinstance(parent, list):
            # Continuation of dict in list
            if stack[-1][1] and isinstance(stack[-1][1], dict):
                stack[-1][1][key] = _parse_val(val)
            continue

        if val == "" or val == "":
            # Check if next lines are list items
            child_obj: Any = {}
            parent[key] = child_obj
            stack.append((indent, child_obj))
        else:
            parent[key] = _parse_val(val)

    # Post-process: convert dicts that should be lists
    return result


def _parse_val(v: str) -> Any:
    if v == "null" or v == "~":
        return None
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        return _load_yaml_fallback(path)


def load_dataset_config(project_key: str) -> Optional[dict]:
    """Load the dataset config for a project, or return None."""
    cfg_path = CONFIGS_DIR / f"{project_key}.yaml"
    if not cfg_path.exists():
        return None
    return load_yaml(cfg_path)


def load_all_configs() -> List[dict]:
    """Load all dataset configs."""
    configs = []
    for p in sorted(CONFIGS_DIR.glob("*.yaml")):
        cfg = load_yaml(p)
        if cfg.get("project_key"):
            configs.append(cfg)
    return configs


# ── Download backends ────────────────────────────────────────────────────────

def _download_url(url: str, dest_path: Path, *, desc: str = "") -> Path:
    """Download a URL to a local file. Returns the path to downloaded file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading %s ...", desc or url)
    log.info("  URL: %s", url)

    req = urllib.request.Request(url, headers={"User-Agent": "CVProjects-Downloader/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  Progress: {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB ({pct}%)", end="", flush=True)

    if total:
        print()  # newline after progress

    size = dest_path.stat().st_size
    if size == 0:
        dest_path.unlink()
        raise RuntimeError(f"Downloaded file is empty: {url}")
    log.info("  Downloaded: %s (%.1f MB)", dest_path.name, size / (1024 * 1024))
    return dest_path


def _extract_zip(archive: Path, dest_dir: Path) -> None:
    """Extract a zip file to dest_dir."""
    log.info("  Extracting ZIP to %s ...", dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest_dir)
    log.info("  Extracted %d items", len(list(dest_dir.rglob("*"))))


def _extract_tar(archive: Path, dest_dir: Path) -> None:
    """Extract a tar(.gz/.bz2) file to dest_dir."""
    log.info("  Extracting TAR to %s ...", dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(dest_dir)
    log.info("  Extracted to %s", dest_dir)


def _download_zip(url: str, dest_dir: Path) -> None:
    """Download and extract a zip archive."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dataset.zip"
        _download_url(url, archive, desc="zip archive")
        _extract_zip(archive, dest_dir)


def _download_tar(url: str, dest_dir: Path) -> None:
    """Download and extract a tar archive."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dataset.tar.gz"
        _download_url(url, archive, desc="tar archive")
        _extract_tar(archive, dest_dir)


def _download_kaggle(url: str, dest_dir: Path) -> None:
    """Download from Kaggle using kaggle CLI."""
    # Check kaggle is available
    kaggle_path = shutil.which("kaggle")
    if not kaggle_path:
        raise RuntimeError(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Then configure API key: https://www.kaggle.com/docs/api\n"
            f"Manual download URL: {url}"
        )

    import re
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Detect dataset vs competition
    m = re.search(r"kaggle\.com/datasets/([^/?#]+/[^/?#]+)", url)
    if m:
        slug = m.group(1)
        cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir)]
    else:
        m = re.search(r"kaggle\.com/c(?:ompetitions)?/([^/?#]+)", url)
        if m:
            comp = m.group(1)
            cmd = ["kaggle", "competitions", "download", "-c", comp, "-p", str(dest_dir)]
        else:
            raise RuntimeError(f"Cannot parse Kaggle URL: {url}")

    log.info("  Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr.strip()}\nCommand: {' '.join(cmd)}")

    # Auto-extract any zip files that Kaggle downloaded
    for zf in dest_dir.glob("*.zip"):
        log.info("  Extracting %s", zf.name)
        _extract_zip(zf, dest_dir)
        zf.unlink()

    log.info("  Kaggle download complete to %s", dest_dir)


def _download_gdrive(url: str, dest_dir: Path) -> None:
    """Download from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown not installed. Install with: pip install gdown\n"
            f"Or download manually from: {url}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    output = str(dest_dir / "dataset_download")
    log.info("  Downloading from Google Drive ...")
    gdown.download(url, output, quiet=False, fuzzy=True)

    # Auto-extract if it's an archive
    out_path = Path(output)
    if out_path.exists():
        if zipfile.is_zipfile(out_path):
            _extract_zip(out_path, dest_dir)
            out_path.unlink()
        elif tarfile.is_tarfile(str(out_path)):
            _extract_tar(out_path, dest_dir)
            out_path.unlink()


def _download_git(url: str, dest_dir: Path) -> None:
    """Shallow clone a git repository."""
    git_path = shutil.which("git")
    if not git_path:
        raise RuntimeError(f"git not found. Install git and retry, or download manually from: {url}")

    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)

    cmd = ["git", "clone", "--depth", "1", url, str(dest_dir)]
    log.info("  Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    log.info("  Cloned to %s", dest_dir)


def _download_manual_page(url: str, dest_dir: Path) -> None:
    """Cannot auto-download — raise with actionable instructions."""
    raise RuntimeError(
        f"This dataset requires manual download.\n"
        f"  1. Visit: {url}\n"
        f"  2. Download the dataset files manually.\n"
        f"  3. Extract/place them into: {dest_dir}\n"
        f"  4. Re-run this command after placing the data."
    )


_DOWNLOAD_HANDLERS = {
    "zip": _download_zip,
    "tar": _download_tar,
    "kaggle": _download_kaggle,
    "gdrive": _download_gdrive,
    "git": _download_git,
    "manual_page": _download_manual_page,
}


# ── Main download logic ─────────────────────────────────────────────────────

def download_dataset(cfg: dict, *, force: bool = False) -> dict:
    """Attempt to download dataset for a project config.

    Returns:
        {"ok": bool, "source": str|None, "error": str|None}
    """
    pk = cfg.get("project_key", "unknown")
    ds = cfg.get("dataset", {})
    download = ds.get("download", {})

    if not download.get("enabled", False):
        return {"ok": False, "source": None, "error": "download.enabled is false in config"}

    sources = download.get("sources", [])
    if not sources:
        return {"ok": False, "source": None, "error": "no download sources configured"}

    root = REPO_ROOT / ds.get("root", f"data/{pk}")
    expected = download.get("expected", [])

    # Check if already downloaded (unless force)
    if not force and expected:
        all_exist = all((REPO_ROOT / e).exists() for e in expected)
        if all_exist:
            log.info("[%s] Dataset already present (expected paths exist)", pk)
            return {"ok": True, "source": "cached", "error": None}

    # Try each source in order
    last_error = None
    for src in sources:
        src_name = src.get("name", "unknown")
        src_url = src.get("url", "")
        src_type = src.get("type", "zip")
        dest = REPO_ROOT / src.get("dest", str(root))

        if not src_url:
            last_error = f"source '{src_name}' has no URL"
            continue

        handler = _DOWNLOAD_HANDLERS.get(src_type)
        if not handler:
            last_error = f"unsupported download type: {src_type}"
            continue

        try:
            log.info("[%s] Attempting download from '%s' (type=%s) ...", pk, src_name, src_type)
            handler(src_url, dest)

            # Verify expected paths
            if expected:
                missing = [e for e in expected if not (REPO_ROOT / e).exists()]
                if missing:
                    log.warning("[%s] Download succeeded but expected paths missing: %s", pk, missing)
                    # Don't fail — the data might be in a subfolder
                    log.info("[%s] Contents of %s: %s", pk, dest,
                             [str(p.relative_to(dest)) for p in dest.iterdir()] if dest.exists() else "empty")

            return {"ok": True, "source": src_name, "error": None}

        except Exception as exc:
            last_error = f"[{src_name}] {exc}"
            log.warning("[%s] Download from '%s' failed: %s", pk, src_name, exc)
            continue

    return {"ok": False, "source": None, "error": last_error}


def ensure_dataset(project_key: str, *, force: bool = False) -> dict:
    """High-level: load config, check paths, download if needed.

    Returns:
        {"ok": bool, "source": str|None, "error": str|None, "status": str}
    """
    cfg = load_dataset_config(project_key)
    if cfg is None:
        return {"ok": False, "source": None, "error": "no config file", "status": "missing_dataset_config"}

    ds = cfg.get("dataset", {})
    kind = ds.get("kind", "")
    task = cfg.get("task", "")

    # Check if data already exists
    if kind == "ultralytics_yaml":
        data_yaml = ds.get("data_yaml")
        if data_yaml and (REPO_ROOT / data_yaml).exists():
            return {"ok": True, "source": "cached", "error": None, "status": "ok"}
    elif kind == "imagefolder":
        val_path = ds.get("val")
        if val_path and (REPO_ROOT / val_path).exists():
            return {"ok": True, "source": "cached", "error": None, "status": "ok"}
    elif kind == "custom":
        root = ds.get("root")
        if root and (REPO_ROOT / root).exists():
            n = sum(1 for _ in (REPO_ROOT / root).rglob("*") if _.is_file())
            if n > 0:
                return {"ok": True, "source": "cached", "error": None, "status": "ok"}

    # Data not present — attempt download
    download_cfg = ds.get("download", {})
    if not download_cfg.get("enabled", False):
        return {
            "ok": False,
            "source": None,
            "error": f"Dataset not found and download disabled. See configs/datasets/{project_key}.yaml",
            "status": "dataset_download_disabled",
        }

    result = download_dataset(cfg, force=force)
    if result["ok"]:
        return {**result, "status": "ok"}
    else:
        return {
            "ok": False,
            "source": result.get("source"),
            "error": result.get("error"),
            "status": "dataset_download_failed",
        }


# ── Structure helpers ────────────────────────────────────────────────────────

def ensure_ultralytics_yaml(
    project_key: str,
    dest_root: Optional[str] = None,
    task: str = "detect",
    class_names: Optional[List[str]] = None,
) -> Path:
    """Create a minimal data.yaml for an Ultralytics YOLO project if missing.

    Returns the path to data.yaml (existing or newly created).
    """
    root = REPO_ROOT / (dest_root or f"data/{project_key}")
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        return yaml_path

    root.mkdir(parents=True, exist_ok=True)

    names = class_names or ["object"]
    nc = len(names)

    lines = [
        f"# Auto-generated data.yaml for {project_key}",
        f"path: {root.as_posix()}",
        f"train: images/train",
        f"val: images/val",
        f"nc: {nc}",
        f"names: {names}",
    ]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("[%s] Created data.yaml at %s", project_key, yaml_path)
    return yaml_path


def ensure_imagefolder_structure(
    project_key: str,
    dest_root: Optional[str] = None,
) -> Path:
    """Ensure data/<project>/train and data/<project>/val directories exist.

    Returns the root path.
    """
    root = REPO_ROOT / (dest_root or f"data/{project_key}")
    for split in ("train", "val"):
        (root / split).mkdir(parents=True, exist_ok=True)
    log.info("[%s] Ensured imagefolder structure at %s", project_key, root)
    return root


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset downloader for CV projects")
    parser.add_argument("project", nargs="?", help="Project key to download")
    parser.add_argument("--all", action="store_true", help="Download all configured datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    args = parser.parse_args()

    if not args.project and not args.all:
        parser.print_help()
        print("\nExamples:")
        print("  python -m utils.data_downloader face_mask_detection")
        print("  python -m utils.data_downloader --all")
        print("  python -m utils.data_downloader --all --dry-run")
        sys.exit(1)

    if args.all:
        configs = load_all_configs()
    else:
        cfg = load_dataset_config(args.project)
        if cfg is None:
            print(f"ERROR: No config found for '{args.project}'")
            sys.exit(1)
        configs = [cfg]

    results = []
    for cfg in configs:
        pk = cfg.get("project_key", "unknown")
        ds = cfg.get("dataset", {})
        dl = ds.get("download", {})

        if args.dry_run:
            enabled = dl.get("enabled", False)
            sources = dl.get("sources", [])
            src_names = [s.get("name", "?") for s in sources]
            print(f"  {pk:40s} enabled={enabled}  sources={src_names}")
            continue

        print(f"\n{'='*60}")
        print(f"  {pk}")
        print(f"{'='*60}")
        result = ensure_dataset(pk, force=args.force)
        result["project_key"] = pk
        results.append(result)
        status = result["status"]
        if status == "ok":
            print(f"  => OK (source: {result.get('source', '?')})")
        else:
            print(f"  => {status}: {result.get('error', '')}")

    if not args.dry_run:
        ok = sum(1 for r in results if r["status"] == "ok")
        failed = sum(1 for r in results if r["status"] not in ("ok",))
        print(f"\n{'='*60}")
        print(f"  DOWNLOAD SUMMARY: {ok} ok, {failed} failed, {len(results)} total")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
