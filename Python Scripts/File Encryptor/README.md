# Fi1e EncRypt0R

> An open-source AES file and folder encryption/decryption tool with a CLI menu interface.

## Overview

This is a command-line tool that provides AES-standard encryption and decryption for individual files and entire folders using the `pyAesCrypt` library. It generates random 30-character encryption keys, supports folder batch operations, and includes a contact/feedback feature via email.

## Features

- **Single file encryption**: Encrypts any file using AES with a randomly generated 30-character key
- **Single file decryption**: Decrypts `.aes` files with the correct key
- **Folder encryption**: Encrypts all files in a folder and moves them to an `encrypted_folder` directory
- **Folder decryption**: Decrypts all `.aes` files from `encrypted_folder` into a `dencrypted_folder` directory
- **Contact/feedback system**: Sends an email with the user's issue and attaches the key log file
- **Key logging**: Stores encryption keys with timestamps to `C:\Intel\temp_key.txt`
- **Interactive CLI menu** with 6 options

## Project Structure

```
Fi1e-EncRypt0R/
├── File_Encryptor.py    # Main encryption/decryption tool
├── File_Encryptor.exe   # Pre-compiled Windows executable
├── _config.yml          # Jekyll theme config (GitHub Pages)
└── LICENSE              # MIT License
```

## Requirements

- Python 3.x
- `pyAesCrypt` — AES encryption library
- `hashlib` (built-in)
- `shutil` (built-in)
- `smtplib` (built-in)

## Installation

```bash
cd "Fi1e-EncRypt0R"
pip install pyAesCrypt
```

## Usage

```bash
python File_Encryptor.py
```

### Menu Options

```
1. Encrypt       — Encrypt a single file
2. Decrypt       — Decrypt a single .aes file
3. Folder Encryption  — Encrypt all files in a folder
4. Folder Decryption  — Decrypt all files in encrypted_folder
5. Contact us    — Send feedback/issue via email
6. Exit          — Exit the program
```

### Example: Encrypt a File

```
Enter options: 1
Enter file path (without quotes): secret.txt
----------Encryption Sucessfull----------

Encryption Key: xYz+Ab3...  (30 characters)
```

### Example: Decrypt a File

```
Enter options: 2
Enter file path: secret.txt.aes
Enter ur decryption key: xYz+Ab3...
Decyrption Succesfull...
```

## How It Works

1. **Encryption**: Generates a random 30-character key from a pool of alphanumeric and special characters, encrypts the file using `pyAesCrypt.encryptFile()` with a 64KB buffer, then deletes the original file. The key and timestamp are logged to `C:\Intel\temp_key.txt`.
2. **Decryption**: Verifies the file has a `.aes` extension, prompts for the key, decrypts to a file prefixed with `Out_`, and removes the encrypted file.
3. **Folder Encryption**: Iterates over all files in a folder, encrypts each, and moves the `.aes` files to `encrypted_folder`.
4. **Folder Decryption**: Iterates over all `.aes` files in `encrypted_folder`, decrypts each, moves results to `dencrypted_folder`, then removes `encrypted_folder`.
5. **Contact**: Sends an email via Gmail SMTP with the user's issue and attaches the key log file.

## Configuration

| Item                          | Value / Location                        | Description                            |
|-------------------------------|-----------------------------------------|----------------------------------------|
| Key log file                  | `C:\Intel\temp_key.txt`                 | Where encryption keys are stored       |
| Buffer size                   | `64 * 1024` (64KB)                      | AES encryption buffer                  |
| Key length                    | 30 characters                           | Random key length                      |
| Email sender/receiver         | `xys@gmail.com` (hardcoded)             | Contact form email addresses           |
| Email password                | `password` (hardcoded)                  | Gmail SMTP password                    |

## Limitations

- **Windows-only** paths: Uses `C:\Intel\temp_key.txt` and backslash paths throughout
- Bare `except` clauses suppress all errors silently
- Original files are **deleted** after encryption — no undo if the key is lost
- Folder decryption always reads from `encrypted_folder` (hardcoded)
- The `hashlib` import is unused
- The decrypted file is prefixed with `Out_` which changes the filename
- `time.sleep(10)` pauses after showing the key, but the key may scroll off-screen
- Hidden option `000` opens the key log file — undocumented feature
- The contact feature has hardcoded placeholder email credentials

## Security Notes

- **Encryption keys are logged in plaintext** to `C:\Intel\temp_key.txt` — this undermines the security of the encryption
- **Email credentials are hardcoded** as `xys@gmail.com` / `password` in the contact feature
- The contact feature **attaches the key log file** to the email, sending all encryption keys over email
- The key generation uses `random.sample()` which is not cryptographically secure — `secrets` module should be used instead
- No secure deletion of original files (standard `os.remove`)

## License

MIT License — Copyright (c) 2019 rex_divakar
