# Checksum Calculator

## Overview

Checksum Calculator is an interactive command-line tool for generating and comparing file or text checksums. It supports MD5, SHA-1, SHA-256, SHA-512, and CRC32.

## Setup and run

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run python main.py
uv run python main.py path\to\file.zip
```

The interactive menu can calculate one checksum, verify an expected checksum, or process files in a directory.

## Security note

Use SHA-256 or SHA-512 for integrity checking when available. MD5, SHA-1, and CRC32 are included only for compatibility with existing checksums and are not suitable for security-sensitive verification.

## Verification

Run `uv run python -m py_compile main.py` to check syntax.
