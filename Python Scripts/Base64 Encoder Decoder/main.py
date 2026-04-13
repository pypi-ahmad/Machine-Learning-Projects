"""Base64 Encoder/Decoder — CLI tool.

Encode and decode text or binary files using Base64 (standard, URL-safe,
and Base32/Base85).  Also detects and decodes Base64 strings found inside
larger blocks of text.

Usage:
    python main.py
"""

import base64
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def encode_text(text: str, variant: str = "standard", encoding: str = "utf-8") -> str:
    data = text.encode(encoding)
    if variant == "urlsafe":
        return base64.urlsafe_b64encode(data).decode("ascii")
    if variant == "base32":
        return base64.b32encode(data).decode("ascii")
    if variant == "base85":
        return base64.b85encode(data).decode("ascii")
    return base64.b64encode(data).decode("ascii")


def decode_text(b64: str, variant: str = "standard", encoding: str = "utf-8") -> str:
    b64_clean = b64.strip()
    try:
        if variant == "urlsafe":
            data = base64.urlsafe_b64decode(b64_clean + "==")
        elif variant == "base32":
            data = base64.b32decode(b64_clean.upper())
        elif variant == "base85":
            data = base64.b85decode(b64_clean)
        else:
            data = base64.b64decode(b64_clean + "==")
        return data.decode(encoding)
    except Exception as e:
        raise ValueError(str(e))


def encode_file(path: Path, variant: str = "standard") -> str:
    data = path.read_bytes()
    if variant == "urlsafe":
        return base64.urlsafe_b64encode(data).decode("ascii")
    return base64.b64encode(data).decode("ascii")


def decode_file(b64: str, out_path: Path, variant: str = "standard") -> int:
    b64_clean = b64.strip()
    if variant == "urlsafe":
        data = base64.urlsafe_b64decode(b64_clean + "==")
    else:
        data = base64.b64decode(b64_clean + "==")
    out_path.write_bytes(data)
    return len(data)


def find_base64_in_text(text: str) -> list[str]:
    """Find all candidate Base64 substrings (length >= 8) in text."""
    pattern = r"[A-Za-z0-9+/]{8,}={0,2}"
    return re.findall(pattern, text)


def is_valid_base64(s: str) -> bool:
    try:
        base64.b64decode(s + "==", validate=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

VARIANT_NAMES = {
    "1": ("standard",  "Base64 standard"),
    "2": ("urlsafe",   "Base64 URL-safe"),
    "3": ("base32",    "Base32"),
    "4": ("base85",    "Base85 (Ascii85)"),
}


def pick_variant() -> str:
    print("  Encoding variant:")
    for k, (_, name) in VARIANT_NAMES.items():
        print(f"    {k}. {name}")
    v = input("  Choice (default 1): ").strip() or "1"
    return VARIANT_NAMES.get(v, VARIANT_NAMES["1"])[0]


def read_multiline(label: str) -> str:
    print(f"  Enter {label} (type '###' on a new line to finish):")
    lines = []
    while True:
        line = input()
        if line == "###":
            break
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Base64 Encoder/Decoder
-----------------------
1. Encode text → Base64
2. Decode Base64 → text
3. Encode file → Base64
4. Decode Base64 → file
5. Find & decode Base64 in text
0. Quit
"""


def main() -> None:
    print("Base64 Encoder/Decoder")
    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            text    = read_multiline("plain text")
            variant = pick_variant()
            enc     = input("  Text encoding (default utf-8): ").strip() or "utf-8"
            try:
                result = encode_text(text, variant, enc)
                print(f"\n  Encoded ({len(result)} chars):")
                # print in 76-char lines
                for i in range(0, len(result), 76):
                    print(f"  {result[i:i+76]}")
            except LookupError:
                print(f"  Unknown encoding: {enc}")

        elif choice == "2":
            b64     = read_multiline("Base64 string")
            variant = pick_variant()
            enc     = input("  Output encoding (default utf-8): ").strip() or "utf-8"
            try:
                result = decode_text(b64, variant, enc)
                print(f"\n  Decoded ({len(result)} chars):")
                print(f"  {result[:500]}")
                if len(result) > 500:
                    print(f"  ... ({len(result) - 500} more chars)")
            except ValueError as e:
                print(f"  Decode error: {e}")

        elif choice == "3":
            path_str = input("  File to encode: ").strip().strip('"')
            p = Path(path_str)
            if not p.exists():
                print(f"  File not found: {path_str}")
                continue
            variant  = pick_variant()
            result   = encode_file(p, variant)
            out_path = p.with_suffix(p.suffix + ".b64")
            out_path.write_text(result, encoding="ascii")
            print(f"\n  Encoded {p.stat().st_size:,} bytes → {len(result):,} chars")
            print(f"  Saved to: {out_path}")

        elif choice == "4":
            b64_path = input("  Base64 file path: ").strip().strip('"')
            p = Path(b64_path)
            if not p.exists():
                print(f"  File not found: {b64_path}")
                continue
            b64     = p.read_text(encoding="ascii")
            variant = pick_variant()
            out_str = input("  Output file path: ").strip().strip('"')
            out_p   = Path(out_str) if out_str else p.with_suffix("")
            try:
                n = decode_file(b64, out_p, variant)
                print(f"\n  Decoded {n:,} bytes → {out_p}")
            except Exception as e:
                print(f"  Decode error: {e}")

        elif choice == "5":
            text       = read_multiline("text to scan")
            candidates = find_base64_in_text(text)
            valid      = [c for c in candidates if is_valid_base64(c)]
            if not valid:
                print("\n  No valid Base64 candidates found.")
                continue
            print(f"\n  Found {len(valid)} candidate(s):")
            for i, b64 in enumerate(valid[:10], 1):
                try:
                    decoded = base64.b64decode(b64 + "==").decode("utf-8", errors="replace")
                    print(f"  {i}. '{b64[:40]}...' → '{decoded[:60]}'")
                except Exception:
                    print(f"  {i}. '{b64[:40]}...' → (binary)")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
