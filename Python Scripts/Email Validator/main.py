"""Email Validator — CLI tool.

Validate email addresses with:
 • Regex syntax check
 • MX record lookup (DNS)
 • Common disposable domain detection
 • Bulk validation from file

Usage:
    python main.py
    python main.py user@example.com
    python main.py emails.txt
"""

import re
import socket
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Known disposable domains (sample list)
# ---------------------------------------------------------------------------
DISPOSABLE_DOMAINS = {
    "mailinator.com", "guerrillamail.com", "tempmail.com", "throwam.com",
    "trashmail.com", "yopmail.com", "sharklasers.com", "guerrillamailblock.com",
    "grr.la", "guerrillamail.info", "guerrillamail.biz", "guerrillamail.de",
    "guerrillamail.net", "guerrillamail.org", "spam4.me", "binkmail.com",
    "bob.email", "clrmail.com", "dispostable.com", "maildrop.cc",
    "mintemail.com", "mt2015.com", "mt2016.com", "mt2017.com",
    "spamgourmet.com", "spamgourmet.net", "tempinbox.com", "throwam.com",
    "fakeinbox.com", "fakemail.net", "filzmail.com", "mytemp.email",
}

ROLE_PREFIXES = {"admin", "webmaster", "postmaster", "hostmaster", "noreply",
                 "no-reply", "abuse", "security", "support", "info", "help"}

EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def check_syntax(email: str) -> tuple[bool, str]:
    email = email.strip()
    if not email:
        return False, "Empty string"
    if email.count("@") != 1:
        return False, "Must contain exactly one '@'"
    local, domain = email.rsplit("@", 1)
    if len(local) > 64:
        return False, "Local part too long (>64 chars)"
    if len(email) > 254:
        return False, "Email too long (>254 chars)"
    if ".." in email:
        return False, "Consecutive dots not allowed"
    if not EMAIL_REGEX.match(email):
        return False, "Invalid format"
    return True, "OK"


def check_mx(domain: str) -> tuple[bool, str]:
    try:
        socket.getaddrinfo(domain, None)
        return True, "Domain resolves"
    except socket.gaierror:
        return False, f"Domain '{domain}' not found"


def check_disposable(domain: str) -> bool:
    return domain.lower() in DISPOSABLE_DOMAINS


def validate(email: str, check_dns: bool = True) -> dict:
    email = email.strip().lower()
    result = {"email": email, "valid": False, "issues": [], "warnings": []}

    ok, msg = check_syntax(email)
    if not ok:
        result["issues"].append(f"Syntax: {msg}")
        return result

    local, domain = email.rsplit("@", 1)

    if check_disposable(domain):
        result["warnings"].append("Disposable email domain")

    if local.split("+")[0] in ROLE_PREFIXES:
        result["warnings"].append("Role-based address (may not reach a person)")

    if check_dns:
        mx_ok, mx_msg = check_mx(domain)
        if not mx_ok:
            result["issues"].append(f"DNS: {mx_msg}")
            return result

    result["valid"] = True
    return result


def format_result(r: dict) -> str:
    status = "✓ VALID" if r["valid"] else "✗ INVALID"
    lines  = [f"  {r['email']:40s} {status}"]
    for issue in r["issues"]:
        lines.append(f"    ✗ {issue}")
    for warn in r["warnings"]:
        lines.append(f"    ⚠ {warn}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        path = Path(arg)
        if path.exists():
            # Bulk file mode
            emails = [e.strip() for e in path.read_text().splitlines() if e.strip()]
            valid = invalid = 0
            for email in emails:
                r = validate(email)
                print(format_result(r))
                if r["valid"]: valid += 1
                else:          invalid += 1
            print(f"\n  Total: {len(emails)}  Valid: {valid}  Invalid: {invalid}")
        else:
            r = validate(arg)
            print(format_result(r))
        return

    print("Email Validator")
    print("──────────────────────────────")
    print("  Enter an email, a file path (for bulk), or 'q' to quit.\n")

    while True:
        raw = input("> ").strip()
        if raw.lower() in ("q", "quit"):
            print("Bye!")
            break
        if not raw:
            continue

        path = Path(raw)
        if path.exists():
            emails = [e.strip() for e in path.read_text().splitlines() if e.strip()]
            valid = invalid = 0
            for email in emails:
                r = validate(email)
                print(format_result(r))
                if r["valid"]: valid += 1
                else:          invalid += 1
            print(f"\n  Total: {len(emails)}  Valid: {valid}  Invalid: {invalid}\n")
        else:
            r = validate(raw)
            print(format_result(r))
            print()


if __name__ == "__main__":
    main()
