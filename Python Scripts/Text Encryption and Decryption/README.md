# Text Encryption and Decryption

Password-based UTF-8 text encryption using AES-256-GCM authentication and scrypt key derivation.

## Setup

Requirements: Python 3.13+.

```powershell
cd "Text Encryption and Decryption"
uv sync
```

## Usage

Encrypt text interactively:

```powershell
uv run python aes_encode.py encrypt --output message.txtenc
```

Decrypt and print plaintext:

```powershell
uv run python aes_encode.py decrypt message.txtenc
```

Decrypt into a new file instead:

```powershell
uv run python aes_encode.py decrypt message.txtenc --output message.txt
```

`encrypt --text "..."` is available for automation, but interactive entry avoids placing plaintext in shell history. Encryption prompts twice for the password; decryption prompts once.

## Security model

Each encrypted file stores a format marker, random salt, random nonce, authentication tag, and ciphertext. The password is never written, printed, or passed as a command-line argument. scrypt derives an independent 256-bit key from the password and per-file salt. AES-GCM detects an incorrect password or modified payload.

The tool refuses to overwrite existing files. Keep the password safe: it cannot be recovered, and losing it permanently prevents decryption.

## Project files

```text
Text Encryption and Decryption/
├── aes_encode.py
├── pyproject.toml
└── uv.lock
```
