"""Barcode Generator — CLI tool.

Generate barcodes as ASCII/text art and save as SVG/text files.
Supports Code 128, EAN-13, Code 39, and QR-like checksum display.

Requires: python-barcode (pip install python-barcode)
Falls back to ASCII representation if not installed.

Usage:
    python main.py
    python main.py --code "HELLO123" --type code128
    python main.py --ean "978030640615" --save barcode.svg
"""

import argparse
import sys
from pathlib import Path


def try_barcode_lib(code: str, barcode_type: str, save_path: str = None) -> bool:
    """Try to use python-barcode library. Returns True if successful."""
    try:
        import barcode
        from barcode.writer import SVGWriter, ImageWriter
    except ImportError:
        return False

    type_map = {
        "code128": "code128",
        "code39":  "code39",
        "ean13":   "ean13",
        "ean8":    "ean8",
        "upca":    "upca",
        "isbn13":  "isbn13",
    }
    bc_type = type_map.get(barcode_type.lower(), "code128")

    try:
        bc = barcode.get(bc_type, code, writer=SVGWriter())
        if save_path:
            fname = bc.save(str(Path(save_path).with_suffix("")))
            print(f"  Saved barcode to: {fname}")
        else:
            import io
            buf = io.StringIO()
            bc.write(buf)
            print(f"  Generated {bc_type.upper()} barcode for: {code}")
            print(f"  To save: python main.py --code '{code}' --type {bc_type} --save output.svg")
        return True
    except Exception as e:
        print(f"  python-barcode error: {e}")
        return False


# ── ASCII barcode renderer (Code 39 subset) ──────────────────────────────────

CODE39_CHARS = {
    "0": "1010011011", "1": "1101001011", "2": "1011001011", "3": "1101100101",
    "4": "1010011011", "5": "1101001101", "6": "1011001101", "7": "1010100011",  # noqa: E501
    "8": "1101010011", "9": "1011010011", "A": "1101001011", "B": "1011001011",
    "C": "1101100101", "D": "1010011011", "E": "1101001101", "F": "1011001101",
    "G": "1010100011", "H": "1101010011", "I": "1011010011", "J": "1010011101",
    "K": "1101001011", "L": "1011001011", "M": "1101100101", "N": "1010011011",
    "O": "1101001101", "P": "1011001101", "Q": "1010100011", "R": "1101010011",
    "S": "1011010011", "T": "1010011101", "U": "1100101011", "V": "1001101011",
    "W": "1100110101", "X": "1001011011", "Y": "1100101101", "Z": "1001101101",
    "-": "1001011011", ".": "1100101011", " ": "1001101011", "*": "1001011011",
    "$": "1001001001", "/": "1001001010", "+": "1001010010", "%": "1010010010",
}

# Minimal Code 128 B encoding (start B = 104, stop = 106)
CODE128_CHARS = {chr(i): i - 32 for i in range(32, 127)}


def code39_bits(text: str) -> str:
    text = text.upper()
    bits = "1001011011"   # start *
    for ch in text:
        if ch not in CODE39_CHARS:
            ch = " "
        bits += "0" + CODE39_CHARS[ch]
    bits += "0" + "1001011011"  # stop *
    return bits


def bits_to_ascii(bits: str, height: int = 5, bar_char: str = "█", space_char: str = " ") -> str:
    row = "".join(bar_char if b == "1" else space_char for b in bits)
    return "\n".join([row] * height)


def ean13_checksum(digits: str) -> int:
    """Compute EAN-13 check digit."""
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:12]))
    return (10 - total % 10) % 10


def generate_ascii(code: str, barcode_type: str = "code39") -> None:
    code = code.upper()
    print(f"\n  {barcode_type.upper()} barcode for: {code}")
    if barcode_type.lower() == "ean13":
        if len(code) == 12 and code.isdigit():
            chk = ean13_checksum(code)
            code += str(chk)
            print(f"  EAN-13 (with check digit): {code}")
        elif len(code) == 13 and code.isdigit():
            expected = ean13_checksum(code[:12])
            valid = int(code[12]) == expected
            print(f"  EAN-13 check: {'✅ valid' if valid else '❌ invalid'}")
        else:
            print("  EAN-13 needs 12 or 13 digits."); return
    print()
    bits = code39_bits(code)
    print(bits_to_ascii(bits, height=4))
    print()
    print(f"  {'  '.join(list(code))}")
    print(f"  Bit pattern length: {len(bits)}")


def save_text(code: str, path: str) -> None:
    bits   = code39_bits(code.upper())
    output = bits_to_ascii(bits, height=5)
    Path(path).write_text(output)
    print(f"  Saved ASCII barcode to: {path}")


def interactive():
    print("=== Barcode Generator ===")
    print("Commands: generate | ean13 | save | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break
        elif cmd in ("generate", "gen"):
            code = input("  Code (alphanumeric): ").strip()
            if not code: continue
            if not try_barcode_lib(code, "code128"):
                generate_ascii(code, "code39")
        elif cmd == "ean13":
            digits = input("  Enter 12 digits: ").strip()
            generate_ascii(digits, "ean13")
        elif cmd == "save":
            code = input("  Code: ").strip()
            path = input("  Output file [barcode.txt]: ").strip() or "barcode.txt"
            if path.endswith(".svg") and try_barcode_lib(code, "code128", path):
                pass
            else:
                save_text(code, path)
        else:
            print("  Commands: generate | ean13 | save | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Barcode Generator")
    parser.add_argument("--code",  metavar="STR",  help="Code to encode")
    parser.add_argument("--ean",   metavar="DIGITS", help="EAN-13 digits (12 without check)")
    parser.add_argument("--type",  default="code128",
                        choices=["code128","code39","ean13","ean8","isbn13"])
    parser.add_argument("--save",  metavar="FILE", help="Save to file (.svg or .txt)")
    args = parser.parse_args()

    if args.ean:
        generate_ascii(args.ean, "ean13")
    elif args.code:
        if args.save and args.save.endswith(".svg"):
            if not try_barcode_lib(args.code, args.type, args.save):
                generate_ascii(args.code, args.type)
        elif args.save:
            save_text(args.code, args.save)
        else:
            if not try_barcode_lib(args.code, args.type):
                generate_ascii(args.code, args.type)
    else:
        interactive()


if __name__ == "__main__":
    main()
