"""Public Holidays Checker — CLI tool.

Fetch public holidays for any country and year using the Nager.Date API.
No API key required.

Usage:
    python main.py
    python main.py --country US --year 2025
    python main.py --country GB
    python main.py --list-countries
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import date


API = "https://date.nager.at/api/v3"


def fetch(url: str) -> dict | list:
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP {e.code}: {e.reason}")
    except Exception as e:
        raise ValueError(f"Network error: {e}")


def get_holidays(country: str, year: int) -> list[dict]:
    return fetch(f"{API}/PublicHolidays/{year}/{country.upper()}")


def get_available_countries() -> list[dict]:
    return fetch(f"{API}/AvailableCountries")


def display_holidays(holidays: list[dict], country: str, year: int) -> None:
    today   = date.today()
    print(f"\n  Public holidays for {country.upper()} in {year}")
    print(f"  {'─'*52}")
    print(f"  {'Date':<14}  {'Day':<10}  Name")
    print(f"  {'─'*52}")
    for h in holidays:
        d      = date.fromisoformat(h["date"])
        is_now = "◀ TODAY" if d == today else ("⬆ UPCOMING" if d > today else "")
        types  = h.get("types", [])
        name   = h["name"][:40]
        local  = h.get("localName", "")
        print(f"  {h['date']:<14}  {d.strftime('%A'):<10}  {name}")
        if local and local != name:
            print(f"  {' '*26}  ({local})")
        if is_now:
            print(f"  {' '*26}  {is_now}")
    print(f"\n  Total: {len(holidays)} holiday(s)\n")


def next_holiday(country: str) -> None:
    today = date.today()
    holidays = get_holidays(country, today.year)
    upcoming = [h for h in holidays if date.fromisoformat(h["date"]) >= today]
    if not upcoming and today.month >= 11:
        # Check next year
        holidays2 = get_holidays(country, today.year + 1)
        upcoming  = holidays2[:3]
    if upcoming:
        nxt = upcoming[0]
        d   = date.fromisoformat(nxt["date"])
        delta = (d - today).days
        print(f"\n  Next holiday in {country.upper()}:")
        print(f"  {nxt['date']}  {d.strftime('%A')}  {nxt['name']}")
        print(f"  {'Today!' if delta==0 else f'In {delta} day(s)'}\n")
    else:
        print("  No upcoming holidays found.")


def list_countries() -> None:
    countries = get_available_countries()
    print(f"\n  Available countries ({len(countries)}):")
    for i, c in enumerate(sorted(countries, key=lambda x: x["name"])):
        print(f"  {c['countryCode']:<5}  {c['name']}")
    print()


def interactive() -> None:
    print("=== Public Holidays Checker ===")
    print("Commands: holidays | next | countries | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break
        elif cmd == "countries":
            try: list_countries()
            except ValueError as e: print(f"  Error: {e}")
        elif cmd in ("holidays", "h"):
            country = input("  Country code (e.g. US, GB, DE): ").strip().upper()
            year    = input(f"  Year [{date.today().year}]: ").strip()
            year    = int(year) if year.isdigit() else date.today().year
            try:
                holidays = get_holidays(country, year)
                display_holidays(holidays, country, year)
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "next":
            country = input("  Country code: ").strip().upper()
            try: next_holiday(country)
            except ValueError as e: print(f"  Error: {e}")
        else:
            print("  Commands: holidays | next | countries | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Public Holidays Checker")
    parser.add_argument("--country",        metavar="CC",   help="Country code (e.g. US)")
    parser.add_argument("--year",           type=int,       default=date.today().year)
    parser.add_argument("--next",           action="store_true", help="Show next holiday")
    parser.add_argument("--list-countries", action="store_true", help="List available countries")
    args = parser.parse_args()

    try:
        if args.list_countries:
            list_countries()
        elif args.country:
            if args.next:
                next_holiday(args.country)
            else:
                holidays = get_holidays(args.country, args.year)
                display_holidays(holidays, args.country, args.year)
        else:
            interactive()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
