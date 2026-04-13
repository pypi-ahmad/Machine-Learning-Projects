"""NASA APOD Viewer — CLI tool.

Fetch and display NASA's Astronomy Picture of the Day.
Uses NASA's public APOD API (demo key works for basic use).

Usage:
    python main.py
    python main.py --date 2024-01-01
    python main.py --random 5
    python main.py --api-key YOUR_KEY
"""

import argparse
import json
import os
import random as rnd
import sys
import urllib.request
import urllib.parse
from datetime import date, timedelta


DEMO_KEY = "DEMO_KEY"
BASE_URL = "https://api.nasa.gov/planetary/apod"


def fetch_apod(api_key: str, apod_date: str = None, random_count: int = None) -> list[dict]:
    params = {"api_key": api_key}
    if random_count:
        params["count"] = random_count
    elif apod_date:
        params["date"] = apod_date
    url = BASE_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            return data if isinstance(data, list) else [data]
    except Exception as e:
        raise ValueError(f"API error: {e}")


def display_apod(apod: dict) -> None:
    title      = apod.get("title", "—")
    apod_date  = apod.get("date", "—")
    media_type = apod.get("media_type", "image")
    url        = apod.get("url", "")
    hdurl      = apod.get("hdurl", "")
    copyright_ = apod.get("copyright", "").strip()
    explanation = apod.get("explanation", "")

    print(f"\n  {'═'*60}")
    print(f"  🌌  {title}")
    print(f"  📅  {apod_date}  [{media_type}]")
    if copyright_:
        print(f"  © {copyright_}")
    print(f"  {'─'*60}")
    if url:
        print(f"  🔗  URL: {url}")
    if hdurl and hdurl != url:
        print(f"  🖼️   HD:  {hdurl}")
    print()
    if explanation:
        # Word-wrap at 65 chars
        words = explanation.split()
        line  = "  "
        for w in words:
            if len(line) + len(w) + 1 > 67:
                print(line)
                line = f"  {w} "
            else:
                line += w + " "
        if line.strip(): print(line)
    print(f"  {'═'*60}\n")


def display_list(apods: list[dict]) -> None:
    for apod in apods:
        display_apod(apod)


def interactive(api_key: str):
    print("=== NASA APOD Viewer ===")
    print("Commands: today | date | random | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break
        elif cmd == "today":
            try:
                apods = fetch_apod(api_key)
                display_list(apods)
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "date":
            d = input("  Date (YYYY-MM-DD): ").strip()
            try:
                apods = fetch_apod(api_key, apod_date=d)
                display_list(apods)
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "random":
            n = int(input("  How many? [3]: ").strip() or 3)
            try:
                apods = fetch_apod(api_key, random_count=min(n, 10))
                display_list(apods)
            except ValueError as e: print(f"  Error: {e}")
        else:
            print("  Commands: today | date YYYY-MM-DD | random [n] | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="NASA APOD Viewer")
    parser.add_argument("--api-key", default=os.environ.get("NASA_API_KEY", DEMO_KEY),
                        help="NASA API key (or set NASA_API_KEY env var)")
    parser.add_argument("--date",    metavar="YYYY-MM-DD", help="Fetch specific date")
    parser.add_argument("--random",  type=int, metavar="N", help="Fetch N random APODs")
    args = parser.parse_args()

    if args.api_key == DEMO_KEY:
        print("  Using DEMO_KEY — rate limited. Set NASA_API_KEY env var for full access.")

    try:
        if args.date:
            apods = fetch_apod(args.api_key, apod_date=args.date)
            display_list(apods)
        elif args.random:
            apods = fetch_apod(args.api_key, random_count=args.random)
            display_list(apods)
        else:
            # Show today, then interactive
            apods = fetch_apod(args.api_key)
            display_list(apods)
            interactive(args.api_key)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
