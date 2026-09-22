"""Encrypt or decrypt files with a password-derived AES-GCM key."""

import argparse
from getpass import getpass
import os
from pathlib import Path

from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import scrypt
from Cryptodome.Random import get_random_bytes

MAGIC = b'FENC1'
SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from a password and a per-file salt."""
    return scrypt(password.encode('utf-8'), salt, key_len=32, N=2**14, r=8, p=1)


def encrypt_file(source: Path, password: str) -> Path:
    """Create a new authenticated encrypted file beside source."""
    output = source.with_name(f'{source.name}.bin')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {output}')

    salt = get_random_bytes(SALT_SIZE)
    nonce = get_random_bytes(NONCE_SIZE)
    cipher = AES.new(derive_key(password, salt), AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(source.read_bytes())
    output.write_bytes(MAGIC + salt + nonce + tag + ciphertext)
    return output


def decrypt_file(source: Path, password: str) -> Path:
    """Restore a file created by encrypt_file without overwriting it."""
    if source.suffix != '.bin':
        raise ValueError(f'Expected a .bin file: {source}')

    payload = source.read_bytes()
    header_size = len(MAGIC) + SALT_SIZE + NONCE_SIZE + TAG_SIZE
    if len(payload) < header_size or not payload.startswith(MAGIC):
        raise ValueError(f'{source} is not an encrypted file created by this script.')

    output = source.with_name(f'{source.stem}.decrypted')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {output}')

    salt_start = len(MAGIC)
    nonce_start = salt_start + SALT_SIZE
    tag_start = nonce_start + NONCE_SIZE
    salt = payload[salt_start:nonce_start]
    nonce = payload[nonce_start:tag_start]
    tag = payload[tag_start:header_size]
    ciphertext = payload[header_size:]
    cipher = AES.new(derive_key(password, salt), AES.MODE_GCM, nonce=nonce)
    output.write_bytes(cipher.decrypt_and_verify(ciphertext, tag))
    return output


def process_path(path: Path, password: str, decrypt: bool) -> None:
    """Process a file or each matching file beneath a directory."""
    operation = decrypt_file if decrypt else encrypt_file
    if path.is_file():
        print(f'Created {operation(path, password)}')
        return
    if not path.is_dir():
        raise ValueError(f'Path is not a regular file or directory: {path}')

    processed = 0
    for root, _, files in os.walk(path):
        for filename in files:
            source = Path(root, filename)
            if decrypt != (source.suffix == '.bin'):
                continue
            print(f'Created {operation(source, password)}')
            processed += 1

    if not processed:
        print('No matching files found.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Encrypt or decrypt a file or directory.')
    parser.add_argument('path', type=Path, help='File or directory to process')
    parser.add_argument('--decrypt', action='store_true', help='Decrypt .bin files instead of encrypting')
    args = parser.parse_args()

    password = getpass('Password: ')
    if not password:
        parser.error('Password must not be empty')
    if not args.decrypt and password != getpass('Confirm password: '):
        parser.error('Passwords do not match')

    try:
        process_path(args.path, password, args.decrypt)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
