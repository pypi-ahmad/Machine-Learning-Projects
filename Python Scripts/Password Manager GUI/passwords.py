"""Local encrypted SQLite password manager with a Tkinter interface."""

import base64
import secrets
import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


DATABASE_PATH = Path(__file__).resolve().with_name("password_manager_vault.db")
LEGACY_DATABASE_PATH = Path(__file__).resolve().with_name("passwordManager.db")
SALT_BYTES = 16


def derive_key(master_password: str, salt: bytes) -> bytes:
    """Derive a Fernet key from the supplied master password."""
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return base64.urlsafe_b64encode(kdf.derive(master_password.encode("utf-8")))


def open_database() -> sqlite3.Connection:
    """Open the maintained, project-local SQLite database and its schema."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("CREATE TABLE IF NOT EXISTS vault_meta (salt TEXT NOT NULL, verifier TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS passwords ("
        "id INTEGER PRIMARY KEY, website TEXT NOT NULL, username TEXT NOT NULL, password_token TEXT NOT NULL)"
    )
    connection.commit()
    return connection


def load_metadata(connection: sqlite3.Connection) -> tuple[bytes, str]:
    """Load or create the per-vault salt and password verifier."""
    row = connection.execute("SELECT salt, verifier FROM vault_meta LIMIT 1").fetchone()
    if row:
        return base64.urlsafe_b64decode(row[0].encode("ascii")), row[1]
    salt = secrets.token_bytes(SALT_BYTES)
    connection.execute(
        "INSERT INTO vault_meta (salt, verifier) VALUES (?, ?)",
        (base64.urlsafe_b64encode(salt).decode("ascii"), ""),
    )
    connection.commit()
    return salt, ""


class PasswordManager(tk.Tk):
    def __init__(self, connection: sqlite3.Connection, key: bytes) -> None:
        super().__init__()
        self.connection = connection
        self.cipher = Fernet(key)
        self.title("Password Manager")
        self.resizable(False, False)

        self.website = tk.StringVar()
        self.username = tk.StringVar()
        self.password = tk.StringVar()
        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=16)
        frame.grid()
        ttk.Label(frame, text="Website").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.website, width=42).grid(row=0, column=1, pady=3)
        ttk.Label(frame, text="Username").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.username, width=42).grid(row=1, column=1, pady=3)
        ttk.Label(frame, text="Password").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.password, width=42).grid(row=2, column=1, pady=3)
        ttk.Button(frame, text="Add password", command=self._save).grid(row=3, column=0, pady=(10, 0))
        ttk.Button(frame, text="View entries", command=self._view).grid(row=3, column=1, pady=(10, 0), sticky="e")
        if LEGACY_DATABASE_PATH.exists():
            ttk.Label(frame, text="Legacy passwordManager.db was not imported.").grid(row=4, column=0, columnspan=2, pady=(10, 0))

    def _save(self) -> None:
        website = self.website.get().strip()
        username = self.username.get().strip()
        password = self.password.get()
        if not website or not username or not password:
            messagebox.showwarning("Missing fields", "Website, username, and password are required.", parent=self)
            return
        token = self.cipher.encrypt(password.encode("utf-8")).decode("ascii")
        self.connection.execute(
            "INSERT INTO passwords (website, username, password_token) VALUES (?, ?, ?)",
            (website, username, token),
        )
        self.connection.commit()
        self.password.set("")
        messagebox.showinfo("Saved", "Entry saved to the encrypted vault.", parent=self)

    def _view(self) -> None:
        rows = self.connection.execute("SELECT website, username, password_token FROM passwords ORDER BY id").fetchall()
        dialog = tk.Toplevel(self)
        dialog.title("Vault entries")
        text = tk.Text(dialog, width=72, height=18, wrap="word")
        text.pack(padx=12, pady=12)
        text.insert("1.0", "\n\n".join(map(self._format_row, rows)) or "No entries yet.")
        text.config(state="disabled")

    def _format_row(self, row: tuple[str, str, str]) -> str:
        website, username, token = row
        try:
            password = self.cipher.decrypt(token.encode("ascii")).decode("utf-8")
        except InvalidToken:
            password = "[unable to decrypt]"
        return f"Website: {website}\nUsername: {username}\nPassword: {password}"


def prompt_for_master_password(connection: sqlite3.Connection, salt: bytes, verifier: str) -> bytes | None:
    prompt = tk.Tk()
    prompt.withdraw()
    master_password = simpledialog.askstring("Password Manager", "Master password:", show="*", parent=prompt)
    if not master_password:
        prompt.destroy()
        return None
    key = derive_key(master_password, salt)
    if verifier:
        try:
            if Fernet(key).decrypt(verifier.encode("ascii")) != b"password-manager-vault":
                raise InvalidToken
        except InvalidToken:
            messagebox.showerror("Password Manager", "Unable to open the encrypted vault.", parent=prompt)
            prompt.destroy()
            return None
    else:
        confirmation = simpledialog.askstring("Password Manager", "Confirm master password:", show="*", parent=prompt)
        if confirmation != master_password:
            messagebox.showerror("Password Manager", "Master passwords did not match.", parent=prompt)
            prompt.destroy()
            return None
        token = Fernet(key).encrypt(b"password-manager-vault").decode("ascii")
        connection.execute("UPDATE vault_meta SET verifier = ?", (token,))
        connection.commit()
    prompt.destroy()
    return key


if __name__ == "__main__":
    connection = open_database()
    salt, verifier = load_metadata(connection)
    key = prompt_for_master_password(connection, salt, verifier)
    if key:
        app = PasswordManager(connection, key)
        app.mainloop()
    connection.close()
