"""Encrypt or decrypt UTF-8 text with a password-derived AES-256-GCM key."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Random import get_random_bytes


MAGIC = b"TXTENC01"
SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive an AES-256 key using scrypt and a per-file random salt."""
    return scrypt(password.encode("utf-8"), salt, key_len=32, N=2**15, r=8, p=1)


def encrypt_text(text: str, password: str) -> bytes:
    """Return a self-contained, authenticated encrypted text payload."""
    salt = get_random_bytes(SALT_SIZE)
    nonce = get_random_bytes(NONCE_SIZE)
    cipher = AES.new(derive_key(password, salt), AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(text.encode("utf-8"))
    return MAGIC + salt + nonce + tag + ciphertext


def decrypt_text(payload: bytes, password: str) -> str:
    """Authenticate and decrypt a payload created by :func:`encrypt_text`."""
    header_size = len(MAGIC) + SALT_SIZE + NONCE_SIZE + TAG_SIZE
    if len(payload) < header_size or not payload.startswith(MAGIC):
        raise ValueError("Input is not a supported encrypted text payload.")
    salt_start = len(MAGIC)
    nonce_start = salt_start + SALT_SIZE
    tag_start = nonce_start + NONCE_SIZE
    salt = payload[salt_start:nonce_start]
    nonce = payload[nonce_start:tag_start]
    tag = payload[tag_start:header_size]
    ciphertext = payload[header_size:]
    cipher = AES.new(derive_key(password, salt), AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode("utf-8")


def write_new(path: Path, content: bytes | str) -> None:
    """Write bytes or text without replacing an existing file."""
    if isinstance(content, bytes):
        with path.open("xb") as file:
            file.write(content)
    else:
        with path.open("x", encoding="utf-8") as file:
            file.write(content)


def confirmed_password() -> str:
    """Prompt twice to prevent accidental encryption with a mistyped password."""
    password = getpass.getpass("Password: ")
    if not password:
        raise ValueError("Password cannot be empty.")
    if password != getpass.getpass("Confirm password: "):
        raise ValueError("Passwords did not match.")
    return password


def main() -> None:
    """Run the encrypt or decrypt command without exposing passwords in arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encrypt = commands.add_parser("encrypt", help="encrypt text into a new binary file")
    encrypt.add_argument("--text", help="plaintext; omit to enter it interactively")
    encrypt.add_argument("--output", type=Path, required=True)
    decrypt = commands.add_parser("decrypt", help="decrypt a payload")
    decrypt.add_argument("input", type=Path)
    decrypt.add_argument("--output", type=Path, help="save plaintext instead of printing it")
    args = parser.parse_args()

    try:
        if args.command == "encrypt":
            text = args.text if args.text is not None else input("Text to encrypt: ")
            write_new(args.output, encrypt_text(text, confirmed_password()))
            print(f"Encrypted text saved to {args.output}")
            return
        password = getpass.getpass("Password: ")
        if not password:
            raise ValueError("Password cannot be empty.")
        plaintext = decrypt_text(args.input.read_bytes(), password)
        if args.output:
            write_new(args.output, plaintext)
            print(f"Decrypted text saved to {args.output}")
        else:
            print(plaintext)
    except (FileExistsError, OSError, UnicodeDecodeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
