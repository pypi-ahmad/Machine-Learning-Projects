# File Encryptor

A command-line tool for encrypting or decrypting one local file at a time. It uses authenticated Fernet encryption and derives a key from a password with Argon2id.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

```powershell
uv sync --no-config
```

## Preview file paths

Check source and destination paths without reading or writing file contents:

```powershell
uv run --no-config python main.py encrypt report.pdf --dry-run
```

## Encrypt

```powershell
uv run --no-config python main.py encrypt report.pdf
```

The default output is `report.pdf.enc`. The original file is never deleted or modified. Enter a password of at least 12 characters, then confirm it when prompted.

## Decrypt

```powershell
uv run --no-config python main.py decrypt report.pdf.enc
```

The default output is `report.pdf`; the command refuses to overwrite it. Use `--output` to select a different new destination.

## Security and limitations

- The encrypted format includes a random salt and Fernet authentication, so incorrect passwords or altered ciphertext fail decryption.
- Fernet keeps the whole file in memory. Do not use this tool for very large files.
- The encrypted token exposes its creation time. Do not rely on it to hide file metadata.
- Lost passwords cannot be recovered. Keep backups of both the original and encrypted files until you have verified decryption.
- This tool does not securely erase originals, encrypt folders in bulk, transmit files, log passwords, or send email.

The design follows the [cryptography Fernet documentation](https://cryptography.io/en/stable/fernet/), which recommends deriving a Fernet key from a password with a KDF such as Argon2id.
