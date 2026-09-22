"""Generate or verify a SHA-256 manifest for one directory tree."""

import argparse
import csv
import hashlib
import os
from pathlib import Path


MANIFEST_NAME = "integrity_manifest.csv"
HASH_ALGORITHM = "sha256"
CHUNK_SIZE = 1 << 20
MANIFEST_COLUMNS = ["path", "size", "hash"]


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    digest = hashlib.new(HASH_ALGORITHM)
    with path.open("rb") as file:
        chunk = file.read(CHUNK_SIZE)
        while chunk:
            digest.update(chunk)
            chunk = file.read(CHUNK_SIZE)
    return digest.hexdigest()


def files_in_root(root: Path, manifest_path: Path) -> list[Path]:
    """Collect regular files below root without including the manifest itself."""
    files = []
    manifest_resolved = manifest_path.resolve()
    for directory, _, names in os.walk(root):
        for name in names:
            candidate = Path(directory, name)
            if candidate.is_file() and candidate.resolve() != manifest_resolved:
                files.append(candidate)
    return sorted(files)


def manifest_target(root: Path, relative_path: str) -> Path | None:
    """Resolve a manifest entry only when it remains inside root."""
    relative = Path(relative_path)
    if not relative_path or relative.is_absolute() or ".." in relative.parts:
        return None
    root_resolved = root.resolve()
    target = (root_resolved / relative).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError:
        return None
    return target


def generate_manifest(root: Path, manifest_path: Path, overwrite: bool) -> Path:
    """Hash files and write a new CSV manifest after overwrite validation."""
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}. Re-run with --overwrite.")

    files = files_in_root(root, manifest_path)
    entries = []
    for file_path in files:
        entries.append(
            {
                "path": file_path.relative_to(root).as_posix(),
                "size": file_path.stat().st_size,
                "hash": hash_file(file_path),
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(entries)
    print(f"Manifest saved: {manifest_path} ({len(entries)} file(s), {HASH_ALGORITHM})")
    return manifest_path


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, list]:
    """Verify a manifest without modifying the tracked directory."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise ValueError("Manifest header must be: path,size,hash")
        expected = list(reader)

    results: dict[str, list] = {"ok": [], "modified": [], "missing": [], "new": [], "invalid": []}
    tracked = set()
    for entry in expected:
        relative_path = entry["path"]
        target = manifest_target(root, relative_path)
        if target is None:
            results["invalid"].append(relative_path)
            continue
        tracked.add(relative_path)
        if not target.is_file():
            results["missing"].append(relative_path)
            continue
        if hash_file(target) != entry["hash"] or str(target.stat().st_size) != entry["size"]:
            results["modified"].append(relative_path)
        else:
            results["ok"].append(relative_path)

    for file_path in files_in_root(root, manifest_path):
        relative_path = file_path.relative_to(root).as_posix()
        if relative_path not in tracked:
            results["new"].append(relative_path)
    return results


def print_results(results: dict[str, list]) -> bool:
    """Print a summary and return whether every tracked file is intact."""
    for label in ("ok", "modified", "missing", "new", "invalid"):
        print(f"{label.title():<8}: {len(results[label])}")
    for label in ("modified", "missing", "new", "invalid"):
        for item in results[label]:
            print(f"  {label}: {item}")
    intact = not results["modified"] and not results["missing"] and not results["invalid"]
    print("All tracked files intact." if intact else "Integrity issues detected.")
    return intact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("directory", type=Path, help="Existing directory to inspect.")
    parser.add_argument("--manifest", type=Path, help="Manifest path; defaults to integrity_manifest.csv in DIRECTORY.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing a manifest during generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.directory.resolve()
    if not root.is_dir():
        raise SystemExit(f"Directory does not exist: {args.directory}")
    manifest_path = (args.manifest or root / MANIFEST_NAME).resolve()
    if manifest_path.parent != root and not manifest_path.parent.is_dir():
        raise SystemExit(f"Manifest directory does not exist: {manifest_path.parent}")

    try:
        if args.action == "generate":
            generate_manifest(root, manifest_path, args.overwrite)
        else:
            if not print_results(verify_manifest(root, manifest_path)):
                raise SystemExit(1)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
