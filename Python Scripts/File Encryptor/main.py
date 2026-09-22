"""Encrypt or decrypt one file without deleting or overwriting user data."""

import argparse
import base64
import getpass
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id


MAGIC = b"FENC2\0"
SALT_SIZE = 16
MIN_PASSWORD_LENGTH = 12


def derive_fernet(password: str, salt: bytes) -> Fernet:
    """Build a Fernet instance from a password and stored Argon2id salt."""
    kdf = Argon2id(
        salt=salt,
        length=32,
        iterations=3,
        lanes=4,
        memory_cost=2**16,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
    return Fernet(key)


def encrypt_bytes(plaintext: bytes, password: str) -> bytes:
    """Return a versioned, authenticated encrypted payload."""
    salt = os.urandom(SALT_SIZE)
    return MAGIC + salt + derive_fernet(password, salt).encrypt(plaintext)


def decrypt_bytes(payload: bytes, password: str) -> bytes:
    """Authenticate and decrypt a payload created by :func:`encrypt_bytes`."""
    if not payload.startswith(MAGIC) or len(payload) <= len(MAGIC) + SALT_SIZE:
        raise ValueError("This is not a supported encrypted file.")
    salt_start = len(MAGIC)
    salt = payload[salt_start : salt_start + SALT_SIZE]
    try:
        return derive_fernet(password, salt).decrypt(payload[salt_start + SALT_SIZE :])
    except InvalidToken as error:
        raise ValueError("Decryption failed. Check the password and file integrity.") from error


def default_output(source: Path, mode: str) -> Path:
    """Return the non-destructive default output path for an operation."""
    if mode == "encrypt":
        return source.with_name(f"{source.name}.enc")
    if source.suffix != ".enc":
        raise ValueError("Encrypted inputs must use the .enc extension or pass --output explicitly.")
    return source.with_suffix("")


def get_password(confirm: bool) -> str:
    """Prompt for a sufficiently long password without echoing it."""
    password = getpass.getpass("Password: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Use a password with at least {MIN_PASSWORD_LENGTH} characters.")
    if confirm and password != getpass.getpass("Confirm password: "):
        raise ValueError("Passwords do not match.")
    return password


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("encrypt", "decrypt"))
    parser.add_argument("source", type=Path, help="Existing regular file to process.")
    parser.add_argument("--output", "-o", type=Path, help="New destination path.")
    parser.add_argument("--dry-run", action="store_true", help="Check paths without reading or writing file contents.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.is_file():
        raise SystemExit(f"Source is not an existing regular file: {args.source}")

    try:
        output = (args.output or default_output(source, args.mode)).resolve()
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"Output directory does not exist: {output.parent}")

    print(f"Source: {source}")
    print(f"Output: {output}")
    if args.dry_run:
        print("Preview only. No file contents were read or written.")
        return

    password = get_password(confirm=args.mode == "encrypt")
    if args.mode == "encrypt":
        output.write_bytes(encrypt_bytes(source.read_bytes(), password))
    else:
        output.write_bytes(decrypt_bytes(source.read_bytes(), password))
    print(f"{args.mode.title()}ed file created: {output}")


if __name__ == "__main__":
    main()
