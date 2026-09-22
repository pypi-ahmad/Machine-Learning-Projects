"""Local encrypted password manager with a Tkinter interface."""

import base64
import json
import secrets
import string
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


VAULT_PATH = Path(__file__).resolve().with_name("vault.json")
LEGACY_PATH = Path(__file__).resolve().with_name("info.txt")
SALT_BYTES = 16


def derive_key(master_password: str, salt: bytes) -> bytes:
    """Derive a Fernet key from a master password and random salt."""
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return base64.urlsafe_b64encode(kdf.derive(master_password.encode("utf-8")))


def encrypt_entries(entries: list[dict[str, str]], master_password: str, salt: bytes) -> str:
    """Return encrypted JSON entries as a URL-safe string."""
    payload = json.dumps(entries, ensure_ascii=False).encode("utf-8")
    return Fernet(derive_key(master_password, salt)).encrypt(payload).decode("ascii")


def decrypt_entries(token: str, master_password: str, salt: bytes) -> list[dict[str, str]]:
    """Decrypt and validate entries from a vault token."""
    payload = Fernet(derive_key(master_password, salt)).decrypt(token.encode("ascii"))
    entries = json.loads(payload.decode("utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Vault entries have an invalid format.")
    return entries


def load_vault(master_password: str) -> tuple[list[dict[str, str]], bytes]:
    """Load an encrypted vault, or create an in-memory empty vault."""
    if not VAULT_PATH.exists():
        return [], secrets.token_bytes(SALT_BYTES)
    data = json.loads(VAULT_PATH.read_text(encoding="utf-8"))
    salt = base64.urlsafe_b64decode(data["salt"].encode("ascii"))
    return decrypt_entries(data["entries"], master_password, salt), salt


def save_vault(entries: list[dict[str, str]], master_password: str, salt: bytes) -> None:
    """Write encrypted entries and their salt to the local vault file."""
    data = {
        "salt": base64.urlsafe_b64encode(salt).decode("ascii"),
        "entries": encrypt_entries(entries, master_password, salt),
    }
    VAULT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def generate_password(length: int) -> str:
    """Generate a password containing all common character classes."""
    if length < 4:
        raise ValueError("Password length must be at least 4.")
    groups = (string.ascii_lowercase, string.ascii_uppercase, string.digits, "!@#$%^&*()-_=+")
    password = list(map(secrets.choice, groups))
    alphabet = "".join(groups)
    password.extend(secrets.SystemRandom().choices(alphabet, k=length - len(password)))
    secrets.SystemRandom().shuffle(password)
    return "".join(password)


class PasswordManager(tk.Tk):
    def __init__(self, master_password: str, entries: list[dict[str, str]], salt: bytes) -> None:
        super().__init__()
        self.master_password = master_password
        self.entries = entries
        self.salt = salt
        self.title("Password Manager")
        self.resizable(False, False)

        self.website = tk.StringVar()
        self.username = tk.StringVar()
        self.password = tk.StringVar()
        self.length = tk.IntVar(value=20)
        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.grid()
        ttk.Label(frame, text="Website").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.website, width=38).grid(row=0, column=1, columnspan=2, pady=3)
        ttk.Label(frame, text="Username").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.username, width=38).grid(row=1, column=1, columnspan=2, pady=3)
        ttk.Label(frame, text="Password").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.password, width=38).grid(row=2, column=1, columnspan=2, pady=3)
        ttk.Label(frame, text="Length").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(frame, from_=4, to=64, textvariable=self.length, width=7).grid(row=3, column=1, sticky="w")
        ttk.Button(frame, text="Generate", command=self._generate).grid(row=3, column=2, padx=(8, 0))
        ttk.Button(frame, text="Copy", command=self._copy).grid(row=4, column=0, pady=(10, 0))
        ttk.Button(frame, text="Save", command=self._save).grid(row=4, column=1, pady=(10, 0))
        ttk.Button(frame, text="View entries", command=self._view).grid(row=4, column=2, pady=(10, 0))

        if LEGACY_PATH.exists() and LEGACY_PATH.stat().st_size:
            ttk.Label(frame, text="Legacy info.txt was not imported.").grid(row=5, column=0, columnspan=3, pady=(10, 0))

    def _generate(self) -> None:
        try:
            self.password.set(generate_password(self.length.get()))
        except ValueError as error:
            messagebox.showerror("Invalid length", str(error), parent=self)

    def _copy(self) -> None:
        if not self.password.get():
            messagebox.showwarning("No password", "Generate or enter a password first.", parent=self)
            return
        self.clipboard_clear()
        self.clipboard_append(self.password.get())
        self.update()
        messagebox.showinfo("Copied", "Password copied to the clipboard.", parent=self)

    def _save(self) -> None:
        website = self.website.get().strip()
        username = self.username.get().strip()
        password = self.password.get()
        if not website or not username or not password:
            messagebox.showwarning("Missing fields", "Website, username, and password are required.", parent=self)
            return
        self.entries.append({"website": website, "username": username, "password": password})
        save_vault(self.entries, self.master_password, self.salt)
        self.password.set("")
        messagebox.showinfo("Saved", "Entry saved to the encrypted vault.", parent=self)

    def _view(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Vault entries")
        text = tk.Text(dialog, width=70, height=18, wrap="word")
        text.pack(padx=12, pady=12)
        text.insert("1.0", "\n\n".join(map(self._format_entry, self.entries)) or "No entries yet.")
        text.config(state="disabled")

    @staticmethod
    def _format_entry(entry: dict[str, str]) -> str:
        return f"Website: {entry.get('website', '')}\nUsername: {entry.get('username', '')}\nPassword: {entry.get('password', '')}"


def prompt_for_master_password() -> tuple[str, list[dict[str, str]], bytes] | None:
    prompt = tk.Tk()
    prompt.withdraw()
    master_password = simpledialog.askstring("Password Manager", "Master password:", show="*", parent=prompt)
    if not master_password:
        prompt.destroy()
        return None
    if not VAULT_PATH.exists():
        confirmation = simpledialog.askstring("Password Manager", "Confirm master password:", show="*", parent=prompt)
        if not confirmation or not secrets.compare_digest(master_password, confirmation):
            messagebox.showerror("Password Manager", "Master passwords did not match.", parent=prompt)
            prompt.destroy()
            return None
    try:
        entries, salt = load_vault(master_password)
    except (InvalidToken, ValueError, KeyError, json.JSONDecodeError):
        messagebox.showerror("Password Manager", "Unable to open the encrypted vault.", parent=prompt)
        prompt.destroy()
        return None
    prompt.destroy()
    return master_password, entries, salt


if __name__ == "__main__":
    vault = prompt_for_master_password()
    if vault:
        app = PasswordManager(*vault)
        app.mainloop()
