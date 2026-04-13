"""Country Info Explorer — CLI tool.

Get detailed information about any country using the REST Countries API.
No API key required.

Usage:
    python main.py
    python main.py --country Germany
    python main.py --country US --code
    python main.py --region Europe
"""

import argparse
import json
import sys
import urllib.request
import urllib.parse


API = "https://restcountries.com/v3.1"


def fetch(url: str) -> list:
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read())
    except Exception as e:
        raise ValueError(f"Network/API error: {e}")


def search_by_name(name: str) -> list:
    return fetch(f"{API}/name/{urllib.parse.quote(name)}")


def search_by_code(code: str) -> list:
    return fetch(f"{API}/alpha/{code.upper()}")


def search_by_region(region: str) -> list:
    return fetch(f"{API}/region/{urllib.parse.quote(region)}")


def fmt_num(n) -> str:
    if n is None: return "—"
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:     return f"{n/1_000_000:.2f}M"
    if n >= 1_000:         return f"{n/1_000:.1f}K"
    return str(n)


def display_country(c: dict) -> None:
    name      = c.get("name", {}).get("common", "—")
    official  = c.get("name", {}).get("official", "—")
    capital   = ", ".join(c.get("capital", ["—"]))
    region    = c.get("region","—")
    subregion = c.get("subregion","—")
    pop       = c.get("population", 0)
    area      = c.get("area", 0)
    cca2      = c.get("cca2","—")
    cca3      = c.get("cca3","—")
    flag      = c.get("flag","")
    currencies = c.get("currencies",{})
    languages  = c.get("languages",{})
    borders    = c.get("borders",[])
    tlds       = c.get("tld",[])
    timezones  = c.get("timezones",[])
    calling    = c.get("idd",{})
    lat_lon    = c.get("latlng",[])

    cur_str = ", ".join(
        f"{v.get('name','?')} ({v.get('symbol','?')})" for v in currencies.values()
    ) or "—"
    lang_str = ", ".join(languages.values()) or "—"
    calling_str = (calling.get("root","") + "".join(calling.get("suffixes",[])[:1])) or "—"

    print(f"\n  {flag}  {name}")
    print(f"  Official: {official}")
    print(f"  {'─'*50}")
    print(f"  Capital:     {capital}")
    print(f"  Region:      {region} / {subregion}")
    print(f"  Population:  {fmt_num(pop)}")
    print(f"  Area:        {area:,.0f} km²" if area else "  Area: —")
    print(f"  Density:     {pop/area:.1f} /km²" if area and pop else "")
    print(f"  Currency:    {cur_str}")
    print(f"  Languages:   {lang_str}")
    print(f"  Codes:       {cca2} / {cca3}")
    print(f"  TLD:         {', '.join(tlds) or '—'}")
    print(f"  Calling:     +{calling_str}")
    if lat_lon:
        print(f"  Lat/Lon:     {lat_lon[0]:.2f}, {lat_lon[1]:.2f}")
    print(f"  Timezones:   {', '.join(timezones[:3])}" + ("…" if len(timezones)>3 else ""))
    if borders:
        print(f"  Borders:     {', '.join(borders)}")
    print()


def display_list(countries: list[dict], brief: bool = True) -> None:
    if brief and len(countries) > 1:
        print(f"\n  {len(countries)} country/countries found:")
        print(f"  {'─'*50}")
        for c in sorted(countries, key=lambda x: x.get("name",{}).get("common","")):
            name   = c.get("name",{}).get("common","—")
            region = c.get("region","—")
            pop    = fmt_num(c.get("population",0))
            cap    = ", ".join(c.get("capital",["—"]))
            cca2   = c.get("cca2","—")
            print(f"  {c.get('flag','')} {cca2:<4}  {name:<28}  {pop:>8}  {cap}")
        print()
    else:
        for c in countries:
            display_country(c)


def interactive():
    print("=== Country Info Explorer ===")
    print("Commands: name | code | region | quit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break
        elif cmd == "name":
            q = input("  Country name: ").strip()
            try:
                data = search_by_name(q)
                if len(data) == 1: display_country(data[0])
                else: display_list(data)
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "code":
            q = input("  2-letter or 3-letter code: ").strip()
            try:
                data = search_by_code(q)
                display_country(data[0])
            except ValueError as e: print(f"  Error: {e}")
        elif cmd == "region":
            r = input("  Region (Africa/Americas/Asia/Europe/Oceania): ").strip()
            try:
                data = search_by_region(r)
                display_list(data, brief=True)
            except ValueError as e: print(f"  Error: {e}")
        else:
            print("  Commands: name | code | region | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Country Info Explorer")
    parser.add_argument("--country", metavar="NAME",  help="Country name or search term")
    parser.add_argument("--code",    metavar="CODE",  help="ISO 2 or 3-letter code")
    parser.add_argument("--region",  metavar="R",     help="Region filter")
    args = parser.parse_args()

    try:
        if args.country:
            data = search_by_name(args.country)
            if len(data) == 1: display_country(data[0])
            else: display_list(data)
        elif args.code:
            data = search_by_code(args.code)
            display_country(data[0])
        elif args.region:
            data = search_by_region(args.region)
            display_list(data)
        else:
            interactive()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
