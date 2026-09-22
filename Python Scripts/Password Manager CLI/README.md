# Password Manager

A local Tkinter password manager that generates passwords and stores saved entries in an encrypted vault.

## Run

```powershell
uv sync --no-config
uv run --no-config python password.py
```

On the first launch, choose and confirm a master password. On later launches, enter the same password to open the vault. A lost master password cannot be recovered.

## Storage and safety

- New entries are encrypted into `vault.json` beside `password.py` using a key derived with scrypt from the master password and a random salt.
- Generated passwords use Python's `secrets` module and include lowercase, uppercase, numeric, and symbol characters.
- The **Copy** button places a password on the local clipboard; clear the clipboard when you no longer need it.
- Existing `info.txt` is treated as legacy plaintext data and is never read, changed, or imported by the updated app. Move any required entries manually, then handle the legacy file according to your retention policy.

## Limits

- The vault protects data at rest only. Passwords are decrypted in application memory while the app is open and are displayed in the local **View entries** window.
- The app has no synchronization, account recovery, automatic lock timer, or multi-user access control.
- Use a dedicated password manager for high-value production credentials and organizational secrets.
