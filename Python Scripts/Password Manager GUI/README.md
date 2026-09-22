# Password Manager GUI

A local Tkinter password manager that stores encrypted credentials in a project-local SQLite vault.

## Run

```powershell
uv sync --no-config
uv run --no-config python passwords.py
```

Set and confirm a master password when opening a new vault. On later launches, the same master password is required to unlock it. The master password is requested in a masked dialog and is never supplied as a command-line argument.

## Storage and safety

- New data is stored in `password_manager_vault.db` beside `passwords.py`.
- Credential passwords are encrypted with Fernet using a key derived from the master password with scrypt and a random per-vault salt.
- A verifier confirms the master password before entries are displayed or added.
- Any existing `passwordManager.db` is considered legacy plaintext data and is not read, changed, or imported by the updated application.

## Limits

- The app decrypts entries in memory while its local view window is open.
- There is no password generation, deletion, editing, account recovery, synchronization, automatic lock timer, or multi-user support.
- Keep a secure backup of the vault and master password. A forgotten master password cannot be recovered.
