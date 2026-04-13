"""File Encryptor — CLI tool.

Encrypt and decrypt files using AES-256 (via the `cryptography`
library if available) or a pure-Python XOR cipher fallback.
Derives a key from a password using PBKDF2.

Usage:
    python main.py
"""

import hashlib
import os
import struct
from pathlib import Path

# Try to import cryptography for AES
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding, hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


# ---------------------------------------------------------------------------
# AES-256-CBC (requires cryptography library)
# ---------------------------------------------------------------------------

def _derive_key_iv(password: str, salt: bytes) -> tuple[bytes, bytes]:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=48,  # 32 key + 16 IV
        salt=salt,
        iterations=100_000,
        backend=default_backend(),
    )
    material = kdf.derive(password.encode("utf-8"))
    return material[:32], material[32:]


MAGIC = b"FENC1"  # file header magic


def encrypt_file_aes(src: Path, dst: Path, password: str) -> None:
    salt = os.urandom(16)
    key, iv = _derive_key_iv(password, salt)
    padder = padding.PKCS7(128).padder()

    plaintext = src.read_bytes()
    padded = padder.update(plaintext) + padder.finalize()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    enc    = cipher.encryptor()
    ciphertext = enc.update(padded) + enc.finalize()

    with open(dst, "wb") as f:
        f.write(MAGIC)
        f.write(salt)
        f.write(ciphertext)


def decrypt_file_aes(src: Path, dst: Path, password: str) -> None:
    data = src.read_bytes()
    if not data.startswith(MAGIC):
        raise ValueError("Not an encrypted file or wrong format.")
    salt       = data[len(MAGIC): len(MAGIC) + 16]
    ciphertext = data[len(MAGIC) + 16:]

    key, iv = _derive_key_iv(password, salt)
    cipher   = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    dec      = cipher.decryptor()
    padded   = dec.update(ciphertext) + dec.finalize()

    unpadder  = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded) + unpadder.finalize()
    dst.write_bytes(plaintext)


# ---------------------------------------------------------------------------
# XOR fallback (no dependencies)
# ---------------------------------------------------------------------------

XOR_MAGIC = b"FXOR1"


def _xor_key_stream(password: str, salt: bytes, length: int) -> bytes:
    """Generate a repeating key stream derived from password + salt."""
    seed = hashlib.sha256(password.encode("utf-8") + salt).digest()
    stream = bytearray()
    counter = 0
    while len(stream) < length:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        stream.extend(block)
        counter += 1
    return bytes(stream[:length])


def encrypt_file_xor(src: Path, dst: Path, password: str) -> None:
    salt      = os.urandom(16)
    plaintext = src.read_bytes()
    key       = _xor_key_stream(password, salt, len(plaintext))
    ct        = bytes(a ^ b for a, b in zip(plaintext, key))
    with open(dst, "wb") as f:
        f.write(XOR_MAGIC)
        f.write(salt)
        f.write(struct.pack(">I", len(plaintext)))
        f.write(ct)


def decrypt_file_xor(src: Path, dst: Path, password: str) -> None:
    data = src.read_bytes()
    if not data.startswith(XOR_MAGIC):
        raise ValueError("Not an XOR-encrypted file or wrong format.")
    offset = len(XOR_MAGIC)
    salt   = data[offset: offset + 16]
    offset += 16
    length = struct.unpack(">I", data[offset: offset + 4])[0]
    offset += 4
    ct = data[offset: offset + length]
    key = _xor_key_stream(password, salt, length)
    plaintext = bytes(a ^ b for a, b in zip(ct, key))
    dst.write_bytes(plaintext)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def encrypt_file(src: Path, dst: Path, password: str) -> str:
    if HAS_CRYPTO:
        encrypt_file_aes(src, dst, password)
        return "AES-256-CBC"
    encrypt_file_xor(src, dst, password)
    return "XOR-SHA256"


def decrypt_file(src: Path, dst: Path, password: str) -> None:
    data = src.read_bytes()
    if data.startswith(MAGIC):
        if not HAS_CRYPTO:
            raise RuntimeError("AES file detected but 'cryptography' library not installed.")
        decrypt_file_aes(src, dst, password)
    elif data.startswith(XOR_MAGIC):
        decrypt_file_xor(src, dst, password)
    else:
        raise ValueError("Unknown file format.")


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
File Encryptor
--------------
1. Encrypt file
2. Decrypt file
3. Batch encrypt (directory)
0. Quit
"""


def get_password(confirm: bool = False) -> str:
    import getpass
    while True:
        pw = getpass.getpass("  Password: ")
        if not pw:
            print("  Password cannot be empty.")
            continue
        if confirm:
            pw2 = getpass.getpass("  Confirm  : ")
            if pw != pw2:
                print("  Passwords do not match.")
                continue
        return pw


def main() -> None:
    algo = "AES-256-CBC" if HAS_CRYPTO else "XOR-SHA256 (install 'cryptography' for AES)"
    print(f"File Encryptor  [{algo}]")

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            src_s = input("  File to encrypt: ").strip().strip('"')
            src   = Path(src_s)
            if not src.is_file():
                print(f"  File not found: {src_s}")
                continue
            default_out = src.with_suffix(src.suffix + ".enc")
            out_s = input(f"  Output file (default {default_out.name}): ").strip().strip('"')
            dst   = Path(out_s) if out_s else default_out
            pw    = get_password(confirm=True)
            try:
                used_algo = encrypt_file(src, dst, pw)
                print(f"\n  Encrypted: {dst}  ({human_size(dst.stat().st_size)})  [{used_algo}]")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "2":
            src_s = input("  File to decrypt: ").strip().strip('"')
            src   = Path(src_s)
            if not src.is_file():
                print(f"  File not found: {src_s}")
                continue
            stem  = src.stem if src.suffix == ".enc" else src.name + ".dec"
            default_out = src.parent / stem
            out_s = input(f"  Output file (default {default_out.name}): ").strip().strip('"')
            dst   = Path(out_s) if out_s else default_out
            pw    = get_password()
            try:
                decrypt_file(src, dst, pw)
                print(f"\n  Decrypted: {dst}  ({human_size(dst.stat().st_size)})")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "3":
            dir_s = input("  Directory to encrypt: ").strip().strip('"')
            d     = Path(dir_s)
            if not d.is_dir():
                print(f"  Not a directory: {dir_s}")
                continue
            pat  = input("  Pattern (default *): ").strip() or "*"
            files = [f for f in d.glob(pat) if f.is_file() and not f.suffix == ".enc"]
            if not files:
                print("  No files found.")
                continue
            print(f"  Will encrypt {len(files)} file(s).")
            pw = get_password(confirm=True)
            ok = errors = 0
            for f in files:
                dst = f.with_suffix(f.suffix + ".enc")
                try:
                    encrypt_file(f, dst, pw)
                    ok += 1
                except Exception as e:
                    print(f"  Error ({f.name}): {e}")
                    errors += 1
            print(f"\n  Encrypted {ok} file(s). Errors: {errors}")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
