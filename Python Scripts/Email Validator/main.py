"""Validate email syntax and optionally check whether its domain resolves."""

import argparse
import re
import socket
from pathlib import Path


DISPOSABLE_DOMAINS = {
    "binkmail.com", "bob.email", "clrmail.com", "dispostable.com", "fakeinbox.com",
    "fakemail.net", "filzmail.com", "grr.la", "guerrillamail.com", "guerrillamailblock.com",
    "maildrop.cc", "mailinator.com", "mintemail.com", "mytemp.email", "sharklasers.com",
    "spam4.me", "spamgourmet.com", "spamgourmet.net", "tempinbox.com", "tempmail.com",
    "throwam.com", "trashmail.com", "yopmail.com",
}
ROLE_PREFIXES = {
    "abuse", "admin", "help", "hostmaster", "info", "no-reply", "noreply", "postmaster",
    "security", "support", "webmaster",
}
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def check_syntax(email: str) -> tuple[bool, str]:
    """Check the basic length and format constraints of an email address."""
    if not email:
        return False, "empty address"
    if len(email) > 254:
        return False, "address is longer than 254 characters"
    if email.count("@") != 1 or ".." in email or not EMAIL_PATTERN.fullmatch(email):
        return False, "invalid email format"
    local, _ = email.rsplit("@", 1)
    if len(local) > 64:
        return False, "local part is longer than 64 characters"
    return True, "valid syntax"


def check_domain(domain: str) -> tuple[bool, str]:
    """Check that DNS can resolve the domain; this is not an MX lookup."""
    try:
        socket.getaddrinfo(domain, None)
    except socket.gaierror:
        return False, "domain does not resolve"
    return True, "domain resolves"


def validate(email: str, check_dns: bool = True) -> dict[str, object]:
    """Return syntax, domain, and warning results without contacting SMTP servers."""
    address = email.strip().lower()
    result: dict[str, object] = {"email": address, "valid": False, "issues": [], "warnings": []}
    syntax_ok, syntax_message = check_syntax(address)
    if not syntax_ok:
        result["issues"] = [syntax_message]
        return result

    local, domain = address.rsplit("@", 1)
    warnings: list[str] = result["warnings"]  # type: ignore[assignment]
    if domain in DISPOSABLE_DOMAINS:
        warnings.append("disposable email domain")
    if local.split("+", 1)[0] in ROLE_PREFIXES:
        warnings.append("role-based address")

    if check_dns:
        domain_ok, domain_message = check_domain(domain)
        if not domain_ok:
            result["issues"] = [domain_message]
            return result

    result["valid"] = True
    return result


def format_result(result: dict[str, object]) -> str:
    """Render one concise, copyable validation result."""
    status = "VALID" if result["valid"] else "INVALID"
    details = list(result["issues"]) + list(result["warnings"])
    return " | ".join([str(result["email"]), status, *details])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address", nargs="?", help="One email address to validate.")
    parser.add_argument("--file", type=Path, help="UTF-8 text file with one email address per line.")
    parser.add_argument("--no-dns", action="store_true", help="Skip the network DNS-resolution check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.address) == bool(args.file):
        raise SystemExit("Provide exactly one email address or --file PATH.")

    emails = [args.address] if args.address else args.file.read_text(encoding="utf-8").splitlines()
    valid_count = 0
    for email in emails:
        if not email.strip():
            continue
        result = validate(email, check_dns=not args.no_dns)
        print(format_result(result))
        valid_count += int(bool(result["valid"]))
    print(f"Checked {len(emails)} address(es): {valid_count} valid, {len(emails) - valid_count} invalid.")


if __name__ == "__main__":
    main()
