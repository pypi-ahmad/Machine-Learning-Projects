# AES Encrypt and Decrypt Text

> A command-line tool that encrypts and decrypts text using AES-128 in CFB mode with PyCryptodome.

## Overview

This script takes a plaintext string as a command-line argument, encrypts it using AES-128 in CFB mode with a random initialization vector (IV), immediately decrypts it to verify correctness, saves the ciphertext to a binary file, and prints the key, IV, ciphertext, and decrypted text to the console.

## Features

- **AES-128 encryption** in CFB (Cipher Feedback) mode
- Generates a **random IV** using `Crypto.Random` for each run
- Encrypts plaintext provided as a CLI argument
- Decrypts the ciphertext to verify round-trip correctness
- Saves encrypted data to `encrypted.bin`
- Displays key, IV, ciphertext (hex), and decrypted text on the console

## Project Structure

```
Encrypt_and_decrypt_text/
├── aes_encode.py
├── requirements.txt
├── output.png            # Sample output screenshot
└── README.md
```

## Requirements

- Python 3.x
- `pycryptodome==3.9.8`

## Installation

```bash
cd Encrypt_and_decrypt_text
pip install -r requirements.txt
```

## Usage

```bash
python aes_encode.py "hello world"
```

Sample output:

```
The key k is:  b'this is a 16 key'
iv is:  b'a1b2c3d4e5f67890'
The encrypted data is:  b'...'
The decrypted data is:  hello world
```

An `encrypted.bin` file is also generated containing the ciphertext (without the IV).

## How It Works

1. Reads the plaintext from `sys.argv[1]`.
2. Uses a 16-byte key (`b'this is a 16 key'`) for AES-128.
3. Generates a random 16-byte IV using `Random.new().read(AES.block_size)`.
4. Creates an AES cipher in CFB mode and encrypts the plaintext (encoded to bytes).
5. Prepends the IV to the ciphertext for transmission/storage: `iv + encrypted_data`.
6. For decryption, extracts the first 16 bytes as IV and decrypts the remaining bytes.
7. Writes only the ciphertext (without IV) to `encrypted.bin`.
8. Prints the key, IV (hex), ciphertext (hex), and decrypted plaintext.

## Configuration

- **AES key**: Hardcoded as `b'this is a 16 key'` (16 bytes = AES-128).
- **Mode**: AES.MODE_CFB (Cipher Feedback mode).
- **Output file**: Hardcoded as `encrypted.bin`.

## Security Notes

- **Hardcoded encryption key** — the key `b'this is a 16 key'` is embedded in the source code. In any real application, keys must be securely generated and stored.
- The `encrypted.bin` file contains ciphertext **without the IV**, so decryption from the file alone is not possible without also storing the IV.
- Not suitable for production encryption.
- The key is printed to the console in plaintext.

## Limitations

- Requires exactly one command-line argument — crashes with `IndexError` if no argument is provided.
- The encryption key is hardcoded and trivially discoverable.
- `encrypted.bin` lacks the IV needed for decryption, making the file alone insufficient.
- No standalone decryption mode — both encryption and decryption happen in a single run.
- No file-based encryption — only encrypts CLI string arguments.

## License

Not specified.
