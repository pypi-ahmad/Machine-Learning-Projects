"""Block selected domains through a Windows hosts file on a schedule.

Usage:
    uv run python web-blocker.py --once
    uv run python web-blocker.py --apply --once
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

HOSTS_PATH = Path(r"C:\Windows\System32\drivers\etc\hosts")
REDIRECT = "127.0.0.1"
DEFAULT_SITES = (
    "www.amazon.in",
    "www.youtube.com",
    "youtube.com",
    "www.facebook.com",
    "facebook.com",
)


def should_block(now: datetime, start_hour: int, end_hour: int) -> bool:
    """Return whether a schedule window contains the current hour."""
    if start_hour < end_hour:
        return start_hour <= now.hour < end_hour
    return now.hour >= start_hour or now.hour < end_hour


def managed_line(line: str, sites: tuple[str, ...]) -> bool:
    """Return whether a line is one of this tool's redirect entries."""
    fields = line.split()
    return len(fields) >= 2 and fields[0] == REDIRECT and fields[1].lower() in sites


def updated_hosts(content: str, sites: tuple[str, ...], block: bool) -> str:
    """Return hosts content with this tool's entries added or removed."""
    lines = list(filter(lambda line: not managed_line(line, sites), content.splitlines()))
    if block:
        lines.extend(map(lambda site: f"{REDIRECT} {site}", sites))
    return "\n".join(lines).rstrip() + "\n"


def apply_hosts(path: Path, sites: tuple[str, ...], block: bool) -> None:
    """Update the selected hosts file with this tool's redirect entries."""
    path.write_text(updated_hosts(path.read_text(encoding="utf-8"), sites, block), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Schedule Windows hosts-file redirects.")
    parser.add_argument("--apply", action="store_true", help="Allow writes to the hosts file")
    parser.add_argument("--once", action="store_true", help="Check the schedule once and exit")
    parser.add_argument("--start-hour", type=int, default=9, help="Blocking start hour (0-23)")
    parser.add_argument("--end-hour", type=int, default=18, help="Blocking end hour (0-23)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between checks")
    parser.add_argument("--hosts-file", type=Path, default=HOSTS_PATH, help="Hosts file to manage")
    args = parser.parse_args()

    if not 0 <= args.start_hour <= 23 or not 0 <= args.end_hour <= 23:
        parser.error("hours must be between 0 and 23")
    if args.start_hour == args.end_hour:
        parser.error("start and end hours must differ")
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    while True:
        block = should_block(datetime.now(), args.start_hour, args.end_hour)
        action = "block" if block else "unblock"
        if args.apply:
            try:
                apply_hosts(args.hosts_file, DEFAULT_SITES, block)
            except (OSError, UnicodeError) as error:
                raise SystemExit(f"Could not {action} sites in {args.hosts_file}: {error}") from error
            print(f"Applied: {action}ed configured sites in {args.hosts_file}")
        else:
            print(f"Dry run: would {action} configured sites in {args.hosts_file}. Use --apply to write.")

        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
