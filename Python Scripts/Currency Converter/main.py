"""Currency Converter — CLI tool.

Convert between currencies using the exchangerate-api.com free tier.
Falls back to a built-in static rate table when offline.

Usage:
    python main.py
    python main.py 100 USD EUR
"""

import json
import sys
import urllib.request
from datetime import datetime

# Static fallback rates relative to USD (updated periodically)
FALLBACK_RATES = {
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "CAD": 1.36,
    "AUD": 1.53, "CHF": 0.89, "CNY": 7.24, "INR": 83.1, "MXN": 17.2,
    "BRL": 4.97, "KRW": 1325.0, "SGD": 1.34, "HKD": 7.82, "NOK": 10.6,
    "SEK": 10.4, "DKK": 6.89, "NZD": 1.63, "ZAR": 18.6, "RUB": 90.0,
    "TRY": 32.0, "AED": 3.67, "SAR": 3.75, "PLN": 4.02, "THB": 35.1,
}

API_URL = "https://open.er-api.com/v6/latest/USD"


def fetch_rates() -> dict[str, float]:
    try:
        with urllib.request.urlopen(API_URL, timeout=5) as resp:
            data = json.loads(resp.read())
            if data.get("result") == "success":
                return data["rates"]
    except Exception:
        pass
    return FALLBACK_RATES


def convert(amount: float, from_cur: str, to_cur: str, rates: dict) -> float:
    from_cur = from_cur.upper()
    to_cur   = to_cur.upper()
    if from_cur not in rates or to_cur not in rates:
        raise ValueError(f"Unknown currency: {from_cur if from_cur not in rates else to_cur}")
    usd = amount / rates[from_cur]
    return usd * rates[to_cur]


def print_table(amount: float, from_cur: str, rates: dict):
    targets = ["USD","EUR","GBP","JPY","CAD","AUD","CHF","CNY","INR","BRL"]
    print(f"\n  {amount:,.2f} {from_cur} =")
    print("  " + "─" * 28)
    for cur in targets:
        if cur == from_cur.upper():
            continue
        try:
            result = convert(amount, from_cur, cur, rates)
            print(f"  {result:>12,.4f}  {cur}")
        except ValueError:
            pass


def main():
    # Quick mode: python main.py 100 USD EUR
    if len(sys.argv) == 4:
        try:
            amount   = float(sys.argv[1])
            from_cur = sys.argv[2].upper()
            to_cur   = sys.argv[3].upper()
            rates    = fetch_rates()
            result   = convert(amount, from_cur, to_cur, rates)
            print(f"{amount:,.2f} {from_cur} = {result:,.4f} {to_cur}")
        except Exception as e:
            print(f"Error: {e}")
        return

    print("Currency Converter")
    print("Fetching rates…", end=" ", flush=True)
    rates = fetch_rates()
    source = "live" if rates is not FALLBACK_RATES else "offline fallback"
    print(f"({source})\n")

    currencies = sorted(rates.keys())

    while True:
        print("Options:  [c] convert  [l] list currencies  [t] table  [q] quit")
        choice = input("Choice: ").strip().lower()

        if choice == "q":
            break

        elif choice == "l":
            cols = 8
            for i, c in enumerate(currencies):
                print(f"  {c}", end="\n" if (i+1) % cols == 0 else "")
            print()

        elif choice in ("c", ""):
            try:
                amount   = float(input("Amount: ").strip())
                from_cur = input("From currency (e.g. USD): ").strip().upper()
                to_cur   = input("To currency   (e.g. EUR): ").strip().upper()
                result   = convert(amount, from_cur, to_cur, rates)
                print(f"\n  {amount:,.2f} {from_cur} = {result:,.4f} {to_cur}\n")
            except (ValueError, KeyError) as e:
                print(f"  Error: {e}\n")

        elif choice == "t":
            try:
                amount   = float(input("Amount: ").strip())
                from_cur = input("From currency (e.g. USD): ").strip().upper()
                print_table(amount, from_cur, rates)
                print()
            except ValueError as e:
                print(f"  Error: {e}\n")

        else:
            print("  Invalid choice.\n")


if __name__ == "__main__":
    main()
