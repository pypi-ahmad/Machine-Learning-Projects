"""Calculate completed age and elapsed-day totals from a birth date."""

from __future__ import annotations

import argparse
from datetime import date, timedelta


def calculate_age(birth_date: date, today: date | None = None) -> tuple[int, int, int, int]:
    """Return completed years, months, days, and total elapsed days."""
    current = today or date.today()
    if birth_date > current:
        raise ValueError("Birth date cannot be in the future.")
    years = current.year - birth_date.year
    months = current.month - birth_date.month
    days = current.day - birth_date.day
    if days < 0:
        months -= 1
        previous_month = current.replace(day=1) - timedelta(days=1)
        days += previous_month.day
    if months < 0:
        years -= 1
        months += 12
    return years, months, days, (current - birth_date).days


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("birth_date", help="birth date in YYYY-MM-DD format")
    parser.add_argument("--name", default="You")
    args = parser.parse_args()
    try:
        birth_date = date.fromisoformat(args.birth_date)
        years, months, days, total_days = calculate_age(birth_date)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"{args.name}'s age: {years} years, {months} months, {days} days ({total_days:,} total days).")


if __name__ == "__main__":
    main()
