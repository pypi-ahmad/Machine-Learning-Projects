# Vigenere Cipher Tool

An interactive command-line implementation of the Vigenere cipher. It encrypts and decrypts text while preserving case and non-letter characters. It also offers Index of Coincidence estimates and a simple frequency-analysis key guess for learning purposes.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

## Run

```powershell
cd "Python Scripts\Vigenere Cipher Tool"
uv run python .\main.py
```

Choose an option from the menu:

1. Encrypt text with a key.
2. Decrypt text when the key is known.
3. Estimate likely key lengths with Index of Coincidence analysis.
4. Guess a key from a selected length and show a tentative decryption.

## Notes

- Keys must contain letters; spaces in an entered key are ignored.
- The cryptanalysis options are educational heuristics. Their results depend on enough ciphertext and text that resembles English.
- Vigenere is not secure for modern encryption. Use a vetted cryptography library and an authenticated encryption mode for sensitive data.
