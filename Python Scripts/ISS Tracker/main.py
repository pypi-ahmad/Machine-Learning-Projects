"""ISS Tracker — CLI tool.

Track the International Space Station's real-time position,
astronauts aboard, and upcoming passes over a location.

Usage:
    python main.py
    python main.py --location "New York"
    python main.py --watch         # continuous tracking
"""

import argparse
import json
import math
import sys
import time
import urllib.request
import urllib.parse


def fetch(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception as e:
        raise ValueError(f"Network error: {e}")


def get_iss_position() -> dict:
    return fetch("http://api.open-notify.org/iss-now.json")


def get_astronauts() -> dict:
    return fetch("http://api.open-notify.org/astros.json")


def geocode_city(city: str) -> tuple[float, float]:
    """Geocode city using Open-Meteo geocoding API."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"
    data = fetch(url)
    results = data.get("results")
    if not results:
        raise ValueError(f"City '{city}' not found.")
    r = results[0]
    return r["latitude"], r["longitude"]


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    R   = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a   = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def display_position(data: dict, user_lat: float = None, user_lon: float = None) -> None:
    pos  = data["iss_position"]
    lat  = float(pos["latitude"])
    lon  = float(pos["longitude"])
    ts   = data.get("timestamp", 0)

    ns   = "N" if lat >= 0 else "S"
    ew   = "E" if lon >= 0 else "W"

    print(f"\n  🛰️  ISS Position  ({time.strftime('%H:%M:%S', time.localtime(ts))})")
    print(f"  Latitude:   {abs(lat):.4f}° {ns}")
    print(f"  Longitude:  {abs(lon):.4f}° {ew}")

    if user_lat is not None:
        dist = haversine(user_lat, user_lon, lat, lon)
        print(f"  Distance from you: {dist:,.0f} km")

    # Rough ocean / continent guess
    if -60 < lat < 75:
        if -180 < lon < -30:
            region = "Americas region"
        elif -30 < lon < 60:
            region = "Europe/Africa region"
        elif 60 < lon < 150:
            region = "Asia region"
        else:
            region = "Pacific region"
    elif lat >= 75:
        region = "Arctic"
    else:
        region = "Antarctic"
    print(f"  Approximate region: {region}")


def display_astronauts() -> None:
    data  = get_astronauts()
    crew  = data.get("people", [])
    print(f"\n  👨‍🚀  Crew in space ({len(crew)} people):")
    by_craft: dict[str, list] = {}
    for p in crew:
        by_craft.setdefault(p.get("craft","?"), []).append(p["name"])
    for craft, names in by_craft.items():
        print(f"  🚀  {craft}:")
        for name in names:
            print(f"       • {name}")
    print()


def watch_mode(interval: int, user_lat: float = None, user_lon: float = None) -> None:
    print("  Live ISS tracking — Ctrl+C to stop\n")
    try:
        while True:
            data = get_iss_position()
            print("\033[2J\033[H", end="")   # clear screen
            display_position(data, user_lat, user_lon)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Stopped.")


def interactive():
    print("=== ISS Tracker ===")
    print("Commands: position | crew | watch | quit\n")
    user_lat = user_lon = None

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break
        elif cmd == "position":
            try:
                data = get_iss_position()
                display_position(data, user_lat, user_lon)
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "crew":
            try: display_astronauts()
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "watch":
            if user_lat is None:
                city = input("  Your city (blank to skip): ").strip()
                if city:
                    try: user_lat, user_lon = geocode_city(city)
                    except ValueError as e: print(f"  {e}")
            interval = int(input("  Update interval (seconds) [5]: ").strip() or 5)
            watch_mode(interval, user_lat, user_lon)
        elif cmd.startswith("city"):
            city = cmd[4:].strip() or input("  City: ").strip()
            try:
                user_lat, user_lon = geocode_city(city)
                print(f"  Location set: {user_lat:.4f}, {user_lon:.4f}")
            except ValueError as e: print(f"  Error: {e}")
        else:
            print("  Commands: position | crew | watch | city <name> | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="ISS Tracker")
    parser.add_argument("--location", metavar="CITY",  help="Your city for distance display")
    parser.add_argument("--watch",    action="store_true", help="Continuous live tracking")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval (seconds)")
    parser.add_argument("--crew",     action="store_true", help="Show crew only")
    args = parser.parse_args()

    user_lat = user_lon = None
    if args.location:
        try:
            user_lat, user_lon = geocode_city(args.location)
        except ValueError as e:
            print(f"Warning: {e}", file=sys.stderr)

    try:
        if args.crew:
            display_astronauts()
        elif args.watch:
            watch_mode(args.interval, user_lat, user_lon)
        else:
            data = get_iss_position()
            display_position(data, user_lat, user_lon)
            display_astronauts()
            if not args.location:
                interactive()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
