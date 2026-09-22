# Encrypt Files and Folders

## Overview

A Python script that encrypts or decrypts individual files and directory trees using password-derived AES-256-GCM. It always creates a new output file and leaves source files unchanged.

**Type:** CLI Utility

## Features

- Encrypts or decrypts a single file or matching files in a directory
- Uses authenticated AES-256-GCM encryption with a unique salt and nonce per file
- Derives keys from an interactively entered password with scrypt
- Outputs encrypted files with `.bin` extension appended to original filename
- Refuses to overwrite existing files
- Accepts a file or directory path as a command-line argument

## Dependencies

Managed in `pyproject.toml` and locked in `uv.lock`:

- `pycryptodomex`, managed by uv in `pyproject.toml`

Install with:

```bash
uv sync
```

## How It Works

1. The script accepts a file or directory path and prompts for a password.
2. Encryption derives an AES-256 key from the password and a per-file random salt.
3. The encrypted output stores a format marker, salt, nonce, authentication tag, and ciphertext in `<original_filename>.bin`.
4. `--decrypt` verifies the authentication tag and writes `<original_filename>.decrypted`.

## Project Structure

```
Create_a_script_to_encrypt_files_and_folder/
├── encrypt.py         # Main encryption script
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Locked dependency versions
└── README.md
```

## Setup & Installation

```bash
uv sync
```

## How to Run

**Encrypt a single file:**
```bash
uv run python encrypt.py path/to/file.txt
```

**Encrypt all files in a directory:**
```bash
uv run python encrypt.py path/to/directory
```

**Decrypt a file created by this version:**

```bash
uv run python encrypt.py --decrypt path/to/file.txt.bin
```

Encrypted files are saved as `<original_name>.bin`; decryption writes `<original_name>.decrypted`. Existing files are never overwritten.

## Testing

No formal test suite present.

## Security Notes

- **Original files are not deleted**: The original plaintext files remain on disk after encryption.
- **Password recovery is impossible**: Lost passwords cannot be recovered.
- **Legacy output is unrecoverable**: Files produced by the earlier CFB version omitted their IV and cannot be decrypted by this or any other tool.

## Limitations

- Directory processing creates one output file per matching input file and may need substantial disk space.
- The password is provided interactively; unattended operation is intentionally not supported.
