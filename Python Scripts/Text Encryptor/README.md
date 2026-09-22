# Text Encryptor

A terminal tool that encrypts UTF-8 text with password-derived AES-256-CBC and decrypts tokens created by the tool. It also includes ROT transformations as simple text demonstrations, not encryption.

## Run it

```powershell
uv sync
uv run python main.py
```

Choose `1` to encrypt text or `2` to decrypt a token. The tool reads text and passwords interactively and does not save them to disk.

## Security notes

- New encrypted tokens use AES-256-CBC with a random salt and IV, plus PBKDF2-HMAC-SHA256 password derivation.
- A wrong password or malformed token fails decryption.
- ROT transformations are reversible encodings, not secure encryption.
- Legacy `XOR1` tokens can still be decrypted for compatibility, but the tool never creates new XOR tokens. Re-encrypt any such data with AES.
- This learning project does not provide authenticated encryption. Do not use it as a substitute for a reviewed security system or for sensitive production data.

## Dependencies

- Python 3.14+
- cryptography
