# Encrypt Files and Folders

## Overview

A Python script that encrypts individual files or entire directory trees using AES-128 encryption in CFB mode via the PyCryptodome library. Encrypted output is saved as `.bin` files alongside the originals.

**Type:** CLI Utility

## Features

- Encrypts a single file or recursively encrypts all files in a directory
- Uses AES-128 encryption in CFB (Cipher Feedback) mode
- Generates a random initialization vector (IV) for each file encryption
- Outputs encrypted files with `.bin` extension appended to original filename
- Accepts file or directory path as a command-line argument
- Detects whether the argument is a file, directory, or special file (socket, FIFO, device)

## Dependencies

Listed in `requirements.txt`:

- `pycryptodome==3.9.8`

Install with:

```bash
pip install -r requirements.txt
```

## How It Works

1. The script accepts a path as a command-line argument (`sys.argv[1]`).
2. If the path is a directory, `encrypt_dir()` recursively walks the directory tree using `os.walk()` and encrypts each file.
3. If the path is a file, `encrypt_file()` is called directly.
4. `encrypt_file()`:
   - Reads the file's plaintext content.
   - Uses a hardcoded 16-byte AES key (`b'this is a 16 key'`).
   - Generates a random IV using `Random.new().read(AES.block_size)`.
   - Encrypts the plaintext using `AES.new(key, AES.MODE_CFB, iv)`.
   - Writes the ciphertext (excluding the IV prefix) to `<original_filename>.bin`.

## Project Structure

```
Create_a_script_to_encrypt_files_and_folder/
├── encrypt.py         # Main encryption script
├── requirements.txt   # Dependencies (pycryptodome==3.9.8)
└── README.md
```

## Setup & Installation

```bash
pip install pycryptodome
```

## How to Run

**Encrypt a single file:**
```bash
python encrypt.py path/to/file.txt
```

**Encrypt all files in a directory:**
```bash
python encrypt.py path/to/directory
```

Encrypted files will be saved as `<original_name>.bin` in the same location as the originals.

## Testing

No formal test suite present.

## Security Notes

- **Hardcoded encryption key**: The AES key (`b'this is a 16 key'`) is hardcoded in the source code. This is insecure for any production use — the key should be derived from a password or securely generated and stored.
- **IV discarded**: The IV is prepended to the ciphertext (`iv + mycipher.encrypt(...)`) but then only `ciphertext[16:]` (without the IV) is written to the output file. This means **decryption is impossible** without separately storing the IV.
- **No decryption script**: There is no corresponding decryption script provided.
- **Original files are not deleted**: The original plaintext files remain on disk after encryption.
- **Files opened in text mode**: `encrypt_file()` opens files with `open(path)` (text mode), which may fail on binary files.

## Limitations

- No decryption functionality — this is encryption-only.
- The IV is not saved, making decryption of the output files impossible as written.
- Files are read in text mode, so binary files (images, executables, etc.) may cause errors.
- The hardcoded key provides no real security.
- The recursive directory walker uses `"."` as the root in `os.walk`, not the provided `path` argument — this is a bug that would cause it to encrypt files in the current working directory instead of the specified directory.
