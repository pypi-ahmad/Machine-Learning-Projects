"""Text Encryptor — CLI tool.

Encrypt and decrypt text using multiple methods:
  • AES-256-CBC (via cryptography library if available)
  • XOR cipher with SHA-256 key stretching (pure Python fallback)
  • ROT-13 / ROT-N (for fun/demo)

Usage:
    python main.py
"""

import base64
import hashlib
import os
import struct
import sys


# ---------------------------------------------------------------------------
# AES-256-CBC (requires: pip install cryptography)
# ---------------------------------------------------------------------------

def _aes_encrypt(text: str, password: str) -> str:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.backends import default_backend

    salt = os.urandom(16)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, 32)
    iv   = os.urandom(16)

    padder  = padding.PKCS7(128).padder()
    padded  = padder.update(text.encode()) + padder.finalize()
    cipher  = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    ct      = cipher.encryptor().update(padded) + cipher.encryptor().finalize()
    enc     = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    enc_ctx = enc.encryptor()
    ct      = enc_ctx.update(padded) + enc_ctx.finalize()
    payload = b"AES1" + salt + iv + ct
    return base64.b64encode(payload).decode()


def _aes_decrypt(token: str, password: str) -> str:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.backends import default_backend

    payload = base64.b64decode(token)
    if not payload.startswith(b"AES1"):
        raise ValueError("Not an AES-encrypted token.")
    salt = payload[4:20]
    iv   = payload[20:36]
    ct   = payload[36:]
    key  = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000, 32)
    cipher  = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    pt      = cipher.decryptor().update(ct) + cipher.decryptor().finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return (unpadder.update(pt) + unpadder.finalize()).decode()


# ---------------------------------------------------------------------------
# XOR fallback (pure Python)
# ---------------------------------------------------------------------------

def _xor_keystream(data: bytes, password: str) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    ext = key * (len(data) // 32 + 1)
    return bytes(a ^ b for a, b in zip(data, ext[:len(data)]))


def _xor_encrypt(text: str, password: str) -> str:
    enc = _xor_keystream(text.encode("utf-8"), password)
    return base64.b64encode(b"XOR1" + enc).decode()


def _xor_decrypt(token: str, password: str) -> str:
    payload = base64.b64decode(token)
    if not payload.startswith(b"XOR1"):
        raise ValueError("Not an XOR-encrypted token.")
    return _xor_keystream(payload[4:], password).decode("utf-8")


# ---------------------------------------------------------------------------
# ROT-N
# ---------------------------------------------------------------------------

def rot_n(text: str, n: int = 13) -> str:
    result = []
    for c in text:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            result.append(chr((ord(c) - base + n) % 26 + base))
        else:
            result.append(c)
    return "".join(result)


# ---------------------------------------------------------------------------
# Unified encrypt / decrypt
# ---------------------------------------------------------------------------

def encrypt(text: str, password: str) -> str:
    try:
        return _aes_encrypt(text, password)
    except ImportError:
        return _xor_encrypt(text, password)


def decrypt(token: str, password: str) -> str:
    try:
        data = base64.b64decode(token)
        if data.startswith(b"AES1"):
            return _aes_decrypt(token, password)
        elif data.startswith(b"XOR1"):
            return _xor_decrypt(token, password)
        else:
            raise ValueError("Unknown token format.")
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MENU = """
Text Encryptor
──────────────────────────────
  1. Encrypt text
  2. Decrypt text
  3. ROT-13
  4. Custom ROT-N
  0. Quit
──────────────────────────────"""


def main():
    try:
        import cryptography
        method = "AES-256-CBC"
    except ImportError:
        method = "XOR+SHA-256 (install 'cryptography' for AES)"

    print(f"  Encryption method: {method}\n")

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            print("  Enter text (blank line to finish):")
            lines = []
            while True:
                line = input()
                if not line and lines:
                    break
                lines.append(line)
            text = "\n".join(lines)
            if not text:
                continue
            pwd = input("  Password: ")
            try:
                token = encrypt(text, pwd)
                print(f"\n  Encrypted:\n  {token}\n")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "2":
            token = input("  Paste encrypted token: ").strip()
            pwd   = input("  Password: ")
            try:
                plain = decrypt(token, pwd)
                print(f"\n  Decrypted:\n  {plain}\n")
            except Exception as e:
                print(f"  Error: {e}")

        elif choice == "3":
            text = input("  Text: ")
            print(f"  ROT-13: {rot_n(text, 13)}")

        elif choice == "4":
            try:
                n    = int(input("  Shift amount: ").strip())
                text = input("  Text: ")
                print(f"  ROT-{n}: {rot_n(text, n)}")
            except ValueError:
                print("  Invalid shift.")

        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
