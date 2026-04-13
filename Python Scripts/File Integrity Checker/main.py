"""File Integrity Checker — CLI tool.

Generate a checksum manifest for a directory,
then verify files against it to detect tampering or corruption.

Usage:
    python main.py generate /path/to/dir
    python main.py verify   /path/to/dir
    python main.py          (interactive)
"""

import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path


MANIFEST_NAME = "integrity_manifest.csv"
HASH_ALGO     = "sha256"
CHUNK_SIZE    = 1 << 20   # 1 MB


def hash_file(path: Path) -> str:
    h = hashlib.new(HASH_ALGO)
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest(root: Path) -> Path:
    manifest_path = root / MANIFEST_NAME
    entries = []
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME)
    print(f"  Hashing {len(files)} file(s)…")
    for i, fpath in enumerate(files, 1):
        rel   = fpath.relative_to(root).as_posix()
        size  = fpath.stat().st_size
        chk   = hash_file(fpath)
        entries.append({"path": rel, "size": size, "hash": chk})
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", end="\r", flush=True)

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size", "hash"])
        writer.writeheader()
        writer.writerows(entries)

    print(f"\n  Manifest saved: {manifest_path}  ({len(entries)} files)")
    return manifest_path


def verify_manifest(root: Path) -> dict:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"  No manifest found at {manifest_path}")
        print("  Run 'generate' first.")
        return {}

    with open(manifest_path, newline="") as f:
        expected = {row["path"]: row for row in csv.DictReader(f)}

    results = {"ok": [], "modified": [], "missing": [], "new": []}
    checked_paths: set[str] = set()

    for rel, info in expected.items():
        fpath = root / rel
        checked_paths.add(rel)
        if not fpath.exists():
            results["missing"].append(rel)
        else:
            current_hash = hash_file(fpath)
            current_size = fpath.stat().st_size
            if current_hash != info["hash"]:
                results["modified"].append({
                    "path": rel,
                    "expected": info["hash"][:12] + "…",
                    "actual":   current_hash[:12] + "…",
                })
            else:
                results["ok"].append(rel)

    # New files not in manifest
    for fpath in root.rglob("*"):
        if fpath.is_file() and fpath.name != MANIFEST_NAME:
            rel = fpath.relative_to(root).as_posix()
            if rel not in checked_paths:
                results["new"].append(rel)

    return results


def print_results(results: dict):
    total = sum(len(v) for v in results.values())
    ok    = len(results.get("ok", []))
    mod   = len(results.get("modified", []))
    miss  = len(results.get("missing", []))
    new   = len(results.get("new", []))

    print(f"\n  ✓ OK:       {ok}")
    print(f"  ✗ Modified: {mod}")
    print(f"  ✗ Missing:  {miss}")
    print(f"  + New:      {new}")

    if results.get("modified"):
        print("\n  Modified files:")
        for e in results["modified"]:
            print(f"    {e['path']}")
            print(f"      expected: {e['expected']}  actual: {e['actual']}")

    if results.get("missing"):
        print("\n  Missing files:")
        for p in results["missing"]:
            print(f"    {p}")

    if results.get("new"):
        print("\n  New (untracked) files:")
        for p in results["new"]:
            print(f"    {p}")

    if mod == 0 and miss == 0:
        print("\n  ✓ All tracked files intact.")
    else:
        print(f"\n  ✗ INTEGRITY ISSUES DETECTED: {mod + miss} problem(s)")


def main():
    if len(sys.argv) >= 2:
        action = sys.argv[1].lower()
        root   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
        if action == "generate":
            generate_manifest(root)
        elif action == "verify":
            results = verify_manifest(root)
            if results:
                print_results(results)
        else:
            print(f"Unknown action: {action}. Use 'generate' or 'verify'.")
        return

    print("File Integrity Checker")
    print("────────────────────────────")
    print("  g → generate manifest")
    print("  v → verify files")
    print("  q → quit\n")

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("q", "quit"):
            break
        elif cmd in ("g", "generate"):
            path_str = input("  Directory [.]: ").strip() or "."
            root     = Path(path_str)
            if root.is_dir():
                generate_manifest(root)
            else:
                print(f"  Not a directory: {root}")
        elif cmd in ("v", "verify"):
            path_str = input("  Directory [.]: ").strip() or "."
            root     = Path(path_str)
            if root.is_dir():
                results = verify_manifest(root)
                if results:
                    print_results(results)
            else:
                print(f"  Not a directory: {root}")
        else:
            print("  Commands: g=generate  v=verify  q=quit")


if __name__ == "__main__":
    main()
