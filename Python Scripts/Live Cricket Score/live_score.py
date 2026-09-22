"""Show Windows notifications when Cricbuzz live scores change."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from win10toast import ToastNotifier


URL = "https://www.cricbuzz.com/cricket-match/live-scores"
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_INTERVAL_SECONDS = 60
ICON_PATH = Path(__file__).with_name("ipl.ico")


def fetch_scores_page() -> str:
    """Download the Cricbuzz live-scores page with a bounded request."""
    request = Request(URL, headers={"User-Agent": "Live-Cricket-Score/0.1"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_scores(html: str) -> list[tuple[str, str]]:
    """Extract match headers and score text from a live-scores page."""
    soup = BeautifulSoup(html, "html.parser")
    matches = soup.select("div.cb-lv-main")
    scores: list[tuple[str, str]] = []
    for match in matches:
        header = match.select_one("div.cb-schdl")
        score = match.select_one("div.cb-scr-wll-chvrn")
        if header is not None and score is not None:
            title = " ".join(header.get_text(" ", strip=True).split())
            value = " ".join(score.get_text(" ", strip=True).split())
            if title and value:
                scores.append((title, value))
    return scores


def changed_scores(scores: list[tuple[str, str]], previous: dict[str, str]) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Return new or changed scores plus the next known-score state."""
    current = dict(scores)
    return [(title, score) for title, score in scores if previous.get(title) != score], current


def notify(title: str, score: str) -> None:
    """Display one Windows toast notification."""
    ToastNotifier().show_toast(title, score, duration=10, icon_path=str(ICON_PATH))


def monitor(interval_seconds: int, once: bool, dry_run: bool) -> None:
    """Poll score data until interrupted or after one poll."""
    previous: dict[str, str] = {}
    while True:
        try:
            changes, previous = changed_scores(parse_scores(fetch_scores_page()), previous)
        except URLError as error:
            print(f"Unable to fetch live scores: {error.reason}")
        else:
            for title, score in changes:
                if dry_run:
                    print(f"{title}: {score}")
                else:
                    notify(title, score)
            if not changes:
                print("No new or changed live scores.")

        if once:
            return
        time.sleep(interval_seconds)


def main() -> None:
    """Parse command-line options and start monitoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS, help="poll interval in seconds")
    parser.add_argument("--once", action="store_true", help="fetch one update, then exit")
    parser.add_argument("--dry-run", action="store_true", help="print score changes instead of notifying")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least 1")
    monitor(args.interval, args.once, args.dry_run)


if __name__ == "__main__":
    main()
