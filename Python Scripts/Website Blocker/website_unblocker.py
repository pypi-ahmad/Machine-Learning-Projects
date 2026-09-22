"""Remove one redirect that was added by website_blocker.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from website_blocker import default_hosts_file, managed_entry, normalize_domain


def parse_args() -> argparse.Namespace:
    """Parse the hostname and explicit write option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="Domain or URL to remove")
    parser.add_argument(
        "--hosts-file",
        type=Path,
        default=default_hosts_file(),
        help="Hosts file to update (default: operating system hosts file)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Remove the managed redirect from the selected hosts file",
    )
    return parser.parse_args()


def remove_entry(content: str, domain: str) -> tuple[str, bool]:
    """Remove only the exact entry managed by this project."""
    entry = managed_entry(domain)
    if entry not in content:
        return content, False
    return content.replace(entry, ""), True


def main() -> None:
    """Preview or apply one managed redirect removal."""
    args = parse_args()
    try:
        domain = normalize_domain(args.domain)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    if not args.apply:
        print(f"Would remove {managed_entry(domain).strip()!r} from {args.hosts_file}.")
        print("Pass --apply only from an elevated shell to write the file.")
        return

    try:
        content = args.hosts_file.read_text(encoding="utf-8")
        updated_content, changed = remove_entry(content, domain)
        if changed:
            args.hosts_file.write_text(updated_content, encoding="utf-8")
            print(f"Unblocked {domain} in {args.hosts_file}.")
        else:
            print(f"No matching managed entry found for {domain}.")
    except OSError as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
