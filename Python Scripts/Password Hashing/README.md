# Password Hashing

Hash or verify passwords locally with Python's standard-library `hashlib.scrypt` implementation.

## Run

```powershell
uv sync --no-config
uv run --no-config python hashing_passwords.py
```

The script prompts for a password and confirmation without echoing either value. It prints a self-contained scrypt record containing the algorithm parameters, random salt, and derived key.

Verify a password against an existing record:

```powershell
uv run --no-config python hashing_passwords.py --verify "scrypt$..."
```

## Safety

- Passwords are never accepted as command-line arguments, so they are not exposed through shell history or process arguments.
- Each new record uses a fresh random 16-byte salt and scrypt parameters `N=16384`, `r=8`, and `p=1`.
- Verification accepts records only with this tool's fixed parameters, avoiding attacker-controlled work factors.
- Store the printed record in an access-controlled password database. Treat it as sensitive authentication data even though it is not plaintext.
- This is a learning utility, not a complete authentication system. Production systems also need secure account handling, rate limiting, recovery flows, and secret management.
