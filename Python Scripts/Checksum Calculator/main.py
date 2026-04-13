"""Checksum Calculator — CLI tool.

Compute MD5, SHA-1, SHA-256, SHA-512, and CRC32 checksums for files
or text strings.  Verify a file against a known checksum and batch
process entire directories.

Usage:
    python main.py
    python main.py somefile.zip
"""

import binascii
import hashlib
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

ALGORITHMS = ["md5", "sha1", "sha256", "sha512"]


def hash_file(path: Path, algorithm: str = "sha256",
              chunk: int = 65536) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def crc32_file(path: Path, chunk: int = 65536) -> str:
    crc = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            crc = binascii.crc32(buf, crc)
    return format(crc & 0xFFFFFFFF, "08X")


def hash_text(text: str, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def all_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for algo in ALGORITHMS:
        result[algo] = hash_file(path, algo)
    result["crc32"] = crc32_file(path)
    return result


def verify(path: Path, expected: str, algorithm: str = "sha256") -> bool:
    algo = algorithm.lower().replace("-", "")
    if algo == "crc32":
        actual = crc32_file(path)
    else:
        actual = hash_file(path, algo)
    return actual.lower() == expected.lower().strip()


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Checksum Calculator
-------------------
1. Hash a file (all algorithms)
2. Hash a file (single algorithm)
3. Hash text/string
4. Verify file against checksum
5. Batch hash directory
0. Quit
"""


def pick_algo() -> str:
    print("  Algorithms: md5 / sha1 / sha256 / sha512 / crc32")
    algo = input("  Algorithm (default sha256): ").strip().lower() or "sha256"
    return algo.replace("-", "")


def main() -> None:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_file():
            hashes = all_hashes(p)
            sz = p.stat().st_size
            print(f"\n{p.name}  ({human_size(sz)})")
            for algo, digest in hashes.items():
                print(f"  {algo.upper():<8} {digest}")
            return

    print("Checksum Calculator")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            path_str = input("  File path: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_file():
                print(f"  File not found: {path_str}")
                continue
            sz = p.stat().st_size
            print(f"\n  {p.name}  ({human_size(sz)})")
            hashes = all_hashes(p)
            for algo, digest in hashes.items():
                print(f"  {algo.upper():<8} {digest}")

        elif choice == "2":
            path_str = input("  File path: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_file():
                print(f"  File not found: {path_str}")
                continue
            algo = pick_algo()
            try:
                if algo == "crc32":
                    digest = crc32_file(p)
                else:
                    digest = hash_file(p, algo)
                print(f"\n  {algo.upper()}: {digest}")
            except ValueError as e:
                print(f"  Unknown algorithm: {e}")

        elif choice == "3":
            text = input("  Text: ")
            algo = pick_algo()
            try:
                if algo == "crc32":
                    crc = binascii.crc32(text.encode("utf-8")) & 0xFFFFFFFF
                    print(f"\n  CRC32: {crc:08X}")
                else:
                    digest = hash_text(text, algo)
                    print(f"\n  {algo.upper()}: {digest}")
            except ValueError as e:
                print(f"  Unknown algorithm: {e}")

        elif choice == "4":
            path_str = input("  File path: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_file():
                print(f"  File not found: {path_str}")
                continue
            expected = input("  Expected checksum: ").strip()
            algo     = pick_algo()
            ok = verify(p, expected, algo)
            if ok:
                print(f"\n  \033[32m✓ MATCH\033[0m — checksum verified.")
            else:
                actual = hash_file(p, algo) if algo != "crc32" else crc32_file(p)
                print(f"\n  \033[31m✗ MISMATCH\033[0m")
                print(f"  Expected: {expected}")
                print(f"  Actual  : {actual}")

        elif choice == "5":
            path_str = input("  Directory: ").strip().strip('"')
            p = Path(path_str)
            if not p.is_dir():
                print(f"  Not a directory: {path_str}")
                continue
            algo     = pick_algo()
            pattern  = input("  Pattern (default *): ").strip() or "*"
            rec      = input("  Recursive? (y/n, default n): ").strip().lower() == "y"
            glob     = p.rglob if rec else p.glob
            files    = [f for f in glob(pattern) if f.is_file()]
            if not files:
                print("  No files found.")
                continue
            print(f"\n  Hashing {len(files)} file(s) with {algo.upper()}...")
            results = []
            for f in files:
                try:
                    digest = crc32_file(f) if algo == "crc32" else hash_file(f, algo)
                    results.append((f, digest))
                except (PermissionError, OSError):
                    results.append((f, "ERROR"))
            print(f"\n  {'Checksum':<66}  File")
            for f, digest in results:
                rel = f.relative_to(p)
                print(f"  {digest:<66}  {rel}")

            save = input("\n  Save to checksums.txt? (y/n): ").strip().lower()
            if save == "y":
                out = p / "checksums.txt"
                with open(out, "w") as fout:
                    for f, digest in results:
                        fout.write(f"{digest}  {f.relative_to(p)}\n")
                print(f"  Saved to {out}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
