# Hashing Passwords

> A CLI tool to hash passwords using SHA-256, SHA-512, or MD5.

## Overview

A command-line utility that takes a plaintext password and outputs its hash digest using a selectable algorithm. Built with Python's `hashlib` and `argparse` modules.

## Features

- Supports three hash algorithms: SHA-256 (default), SHA-512, and MD5
- Command-line argument parsing with help text
- Displays both the hash type and the hex digest

## Project Structure

```
Hashing_passwords/
├── hashing_passwords.py   # Main script
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses `hashlib` and `argparse` from the standard library)

## Installation

```bash
cd Hashing_passwords
```

No dependencies to install.

## Usage

```bash
# Default (SHA-256)
python hashing_passwords.py mypassword

# Specify hash type
python hashing_passwords.py mypassword -t sha512
python hashing_passwords.py mypassword -t md5
```

**Output:**
```
< hash-type : sha256 >
89e01536ac207279409d4de1e5253e01f4a1769e696db0d6062ca9b8f56767c8
```

## How It Works

1. Parses the password and optional `-t`/`--type` argument via `argparse`.
2. Uses `getattr(hashlib, hashtype)()` to dynamically select the hash function.
3. Encodes the password string to bytes and updates the hash object.
4. Prints the algorithm name and hex digest.

## Configuration

No configuration needed. Algorithm is selected via the `-t` flag.

## Limitations

- Hashes a single password per invocation (no batch mode).
- Does not use salting — output is a simple unsalted hash.
- Limited to three algorithm choices (sha256, sha512, md5).

## Security Notes

- This tool produces **unsalted hashes**, which are vulnerable to rainbow table attacks. For real password storage, use `bcrypt`, `scrypt`, or `argon2`.
- MD5 is considered cryptographically broken and should not be used for security purposes.

## License

Not specified.
