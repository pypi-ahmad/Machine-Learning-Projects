"""Hash and verify passwords with Python's standard-library scrypt support."""

import argparse
import base64
import getpass
import hashlib
import hmac
import secrets


N = 2**14
R = 8
P = 1
SALT_BYTES = 16
KEY_BYTES = 64


def encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value.encode("ascii"))


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Create a self-contained scrypt password record."""
    salt = salt or secrets.token_bytes(SALT_BYTES)
    derived_key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=N,
        r=R,
        p=P,
        dklen=KEY_BYTES,
    )
    return f"scrypt${N}${R}${P}${encode(salt)}${encode(derived_key)}"


def verify_password(password: str, record: str) -> bool:
    """Return whether a password matches a scrypt record made by this tool."""
    try:
        algorithm, n, r, p, encoded_salt, encoded_key = record.split("$")
        if algorithm != "scrypt" or (int(n), int(r), int(p)) != (N, R, P):
            return False
        derived_key = hashlib.scrypt(
            password.encode("utf-8"),
            salt=decode(encoded_salt),
            n=N,
            r=R,
            p=P,
            dklen=len(decode(encoded_key)),
        )
        return hmac.compare_digest(derived_key, decode(encoded_key))
    except (ValueError, TypeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash or verify a password with scrypt.")
    parser.add_argument("--verify", metavar="RECORD", help="Verify against a stored scrypt record")
    args = parser.parse_args()

    password = getpass.getpass("Password: ")
    if args.verify:
        print("Password matches." if verify_password(password, args.verify) else "Password does not match.")
        return

    confirmation = getpass.getpass("Confirm password: ")
    if not hmac.compare_digest(password, confirmation):
        parser.error("passwords did not match")
    print(hash_password(password))


if __name__ == "__main__":
    main()
