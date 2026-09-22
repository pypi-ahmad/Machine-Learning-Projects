"""Add one managed hostname redirect to a hosts file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

ENTRY_MARKER = "# website-blocker"


def default_hosts_file() -> Path:
    """Return the supported platform's default hosts file path."""
    if os.name == "nt":
        return Path(r"C:\Windows\System32\drivers\etc\hosts")
    if os.name == "posix":
        return Path("/etc/hosts")
    raise OSError("This platform's hosts file path is not configured.")


def normalize_domain(value: str) -> str:
    """Extract one hostname from a domain or URL."""
    candidate = value.strip()
    parsed = urlparse(candidate if "://" in candidate else f"//{candidate}")
    if not parsed.hostname:
        raise ValueError("Provide one valid domain or URL.")
    return parsed.hostname.lower()


def managed_entry(domain: str) -> str:
    """Return the exact hosts-file line managed by this tool."""
    return f"127.0.0.1 {domain} {ENTRY_MARKER}\n"


def parse_args() -> argparse.Namespace:
    """Parse the hostname and explicit write option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="Domain or URL to redirect to localhost")
    parser.add_argument(
        "--hosts-file",
        type=Path,
        default=default_hosts_file(),
        help="Hosts file to update (default: operating system hosts file)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the managed redirect to the selected hosts file",
    )
    return parser.parse_args()


def add_entry(content: str, domain: str) -> tuple[str, bool]:
    """Add one managed entry when it is not already present."""
    entry = managed_entry(domain)
    if entry in content:
        return content, False
    separator = "" if not content or content.endswith("\n") else "\n"
    return f"{content}{separator}{entry}", True


def main() -> None:
    """Preview or apply one hostname redirect."""
    args = parse_args()
    try:
        domain = normalize_domain(args.domain)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    if not args.apply:
        print(f"Would add {managed_entry(domain).strip()!r} to {args.hosts_file}.")
        print("Pass --apply only from an elevated shell to write the file.")
        return

    try:
        content = args.hosts_file.read_text(encoding="utf-8")
        updated_content, changed = add_entry(content, domain)
        if changed:
            args.hosts_file.write_text(updated_content, encoding="utf-8")
            print(f"Blocked {domain} in {args.hosts_file}.")
        else:
            print(f"{domain} is already blocked by this tool.")
    except OSError as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
