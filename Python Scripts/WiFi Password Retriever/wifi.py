"""List saved Windows Wi-Fi profiles; reveal keys only when explicitly requested."""

from __future__ import annotations

import argparse
import os
import subprocess


def parse_args() -> argparse.Namespace:
    """Parse key-display and dry-run options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-keys",
        action="store_true",
        help="Display plaintext keys for saved profiles",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the intended netsh command without listing profiles or keys",
    )
    return parser.parse_args()


def run_netsh(*arguments: str) -> str:
    """Run a Windows netsh WLAN command and return decoded output."""
    if os.name != "nt":
        raise OSError("This tool is supported only on Windows.")
    completed = subprocess.run(
        ["netsh", "wlan", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def profiles_from_output(output: str) -> list[str]:
    """Extract saved profile names from `netsh wlan show profiles` output."""
    profiles: list[str] = []
    for line in output.splitlines():
        if "All User Profile" in line:
            profile = line.partition(":")[2].strip()
            if profile:
                profiles.append(profile)
    return profiles


def key_from_output(output: str) -> str:
    """Extract a displayed key from a profile response, when present."""
    for line in output.splitlines():
        if "Key Content" in line:
            return line.partition(":")[2].strip()
    return ""


def main() -> None:
    """List profiles or explicitly requested plaintext keys."""
    args = parse_args()
    if args.dry_run:
        action = "profiles and plaintext keys" if args.show_keys else "profile names"
        print(f"Would query netsh for saved Wi-Fi {action}.")
        return

    try:
        profiles = profiles_from_output(run_netsh("show", "profiles"))
        if not profiles:
            print("No saved Wi-Fi profiles found.")
            return
        for profile in profiles:
            if not args.show_keys:
                print(profile)
                continue
            key = key_from_output(run_netsh("show", "profile", profile, "key=clear"))
            print(f"{profile:<30} | {key}")
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
