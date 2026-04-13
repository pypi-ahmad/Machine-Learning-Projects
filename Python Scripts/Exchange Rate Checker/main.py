"""Exchange Rate Checker — CLI tool.

Fetch live exchange rates using the free open.er-api.com.
Convert between currencies, compare rates, track favorites.

Usage:
    python main.py
    python main.py --from USD --to EUR --amount 100
    python main.py --base EUR
"""

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path


API_URL  = "https://open.er-api.com/v6/latest/{base}"
FAV_FILE = Path("exchange_favorites.json")

CURRENCY_NAMES = {
    "USD":"US Dollar","EUR":"Euro","GBP":"British Pound","JPY":"Japanese Yen",
    "AUD":"Australian Dollar","CAD":"Canadian Dollar","CHF":"Swiss Franc",
    "CNY":"Chinese Yuan","INR":"Indian Rupee","MXN":"Mexican Peso",
    "BRL":"Brazilian Real","KRW":"South Korean Won","SGD":"Singapore Dollar",
    "NOK":"Norwegian Krone","SEK":"Swedish Krona","DKK":"Danish Krone",
    "NZD":"New Zealand Dollar","HKD":"Hong Kong Dollar","TRY":"Turkish Lira",
    "ZAR":"South African Rand","RUB":"Russian Ruble","SAR":"Saudi Riyal",
    "AED":"UAE Dirham","PLN":"Polish Zloty","THB":"Thai Baht",
    "IDR":"Indonesian Rupiah","MYR":"Malaysian Ringgit","PHP":"Philippine Peso",
    "PKR":"Pakistani Rupee","EGP":"Egyptian Pound",
}


def fetch_rates(base: str) -> dict:
    url = API_URL.format(base=base.upper())
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise ValueError(f"Network error: {e}")
    if data.get("result") != "success":
        raise ValueError(f"API error: {data.get('error-type','unknown')}")
    return data


def convert(amount: float, from_cur: str, to_cur: str, rates: dict) -> float:
    r = rates.get("rates", {})
    if from_cur not in r or to_cur not in r:
        raise ValueError(f"Unknown currency.")
    # Rates are relative to base
    base_amount = amount / r[from_cur]
    return base_amount * r[to_cur]


def display_rates(data: dict, currencies: list[str] = None) -> None:
    base   = data["base_code"]
    rates  = data["rates"]
    update = data.get("time_last_update_utc","—")
    print(f"\n  Base: {base} ({CURRENCY_NAMES.get(base,'')})")
    print(f"  Updated: {update[:25]}")
    print(f"  {'─'*44}")
    print(f"  {'Code':<6}  {'Currency':<24}  {'Rate':>12}")
    print(f"  {'─'*44}")
    target = currencies or list(CURRENCY_NAMES.keys())
    for c in target:
        if c not in rates: continue
        name = CURRENCY_NAMES.get(c, c)
        print(f"  {c:<6}  {name:<24}  {rates[c]:>12.4f}")
    print()


def display_conversion(amount: float, from_cur: str, to_cur: str, result: float) -> None:
    fn = CURRENCY_NAMES.get(from_cur, from_cur)
    tn = CURRENCY_NAMES.get(to_cur,   to_cur)
    print(f"\n  {amount:,.2f} {from_cur} ({fn})")
    print(f"  = {result:,.4f} {to_cur} ({tn})\n")


def load_favorites() -> list:
    if FAV_FILE.exists():
        try: return json.loads(FAV_FILE.read_text())
        except: pass
    return [("USD","EUR"),("USD","GBP"),("USD","JPY")]


def save_favorites(favs: list) -> None:
    FAV_FILE.write_text(json.dumps(favs, indent=2))


def interactive():
    print("=== Exchange Rate Checker ===")
    print("Commands: rates | convert | compare | favorites | quit\n")
    cached_data: dict = {}

    while True:
        cmd = input("> ").strip().lower()
        if cmd in ("quit", "q", "exit"): break

        elif cmd == "rates":
            base = input("  Base currency [USD]: ").strip().upper() or "USD"
            try:
                data = fetch_rates(base)
                cached_data[base] = data
                display_rates(data)
            except ValueError as e: print(f"  Error: {e}")

        elif cmd == "convert":
            try:
                amount  = float(input("  Amount: ").strip())
                from_c  = input("  From currency: ").strip().upper()
                to_c    = input("  To currency: ").strip().upper()
                if from_c not in cached_data:
                    cached_data[from_c] = fetch_rates(from_c)
                result = convert(amount, from_c, to_c, cached_data[from_c])
                display_conversion(amount, from_c, to_c, result)
            except (ValueError, KeyError) as e: print(f"  Error: {e}")

        elif cmd == "compare":
            base = input("  Base currency [USD]: ").strip().upper() or "USD"
            targets = input("  Compare to (space-separated) [EUR GBP JPY]: ").strip().upper().split()
            if not targets: targets = ["EUR","GBP","JPY"]
            try:
                data = fetch_rates(base)
                display_rates(data, targets)
            except ValueError as e: print(f"  Error: {e}")

        elif cmd == "favorites":
            favs = load_favorites()
            base = input("  Base currency [USD]: ").strip().upper() or "USD"
            try:
                data = fetch_rates(base)
                print(f"\n  Favorite pairs (base {base}):")
                print(f"  {'─'*40}")
                for f, t in favs:
                    if t in data["rates"]:
                        rate = data["rates"][t]
                        print(f"  1 {f} = {rate:.4f} {t}")
            except ValueError as e: print(f"  Error: {e}")

        else:
            print("  Commands: rates | convert | compare | favorites | quit")
        print()


def main():
    parser = argparse.ArgumentParser(description="Exchange Rate Checker")
    parser.add_argument("--from",   dest="from_cur", metavar="CUR", help="From currency")
    parser.add_argument("--to",     dest="to_cur",   metavar="CUR", help="To currency")
    parser.add_argument("--amount", type=float, default=1.0)
    parser.add_argument("--base",   metavar="CUR", help="Show all rates with this base")
    args = parser.parse_args()

    try:
        if args.from_cur and args.to_cur:
            data   = fetch_rates(args.from_cur)
            result = convert(args.amount, args.from_cur, args.to_cur, data)
            display_conversion(args.amount, args.from_cur, args.to_cur, result)
        elif args.base:
            data = fetch_rates(args.base)
            display_rates(data)
        else:
            interactive()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
