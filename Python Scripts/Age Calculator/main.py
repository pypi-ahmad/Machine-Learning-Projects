"""Calculate age and birthday statistics from a birth date."""

from __future__ import annotations

import argparse
import calendar
from datetime import date, datetime, timedelta


DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y")


def parse_date(value: str) -> date:
    """Parse one documented birth-date format."""
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), date_format).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {value!r}. Use YYYY-MM-DD.")


def birthday_in_year(birth_date: date, year: int) -> date:
    """Return the birthday in a year, using February 28 for non-leap years."""
    day = min(birth_date.day, calendar.monthrange(year, birth_date.month)[1])
    return date(year, birth_date.month, day)


def calculate_age(birth_date: date, today: date | None = None) -> dict[str, int | str | date]:
    """Return whole-year age, elapsed time statistics, and the next birthday."""
    current_date = today or date.today()
    if birth_date > current_date:
        raise ValueError("Birth date is in the future.")

    years = current_date.year - birth_date.year
    months = current_date.month - birth_date.month
    days = current_date.day - birth_date.day
    if days < 0:
        months -= 1
        days += (current_date.replace(day=1) - timedelta(days=1)).day
    if months < 0:
        years -= 1
        months += 12

    next_birthday = birthday_in_year(birth_date, current_date.year)
    if next_birthday <= current_date:
        next_birthday = birthday_in_year(birth_date, current_date.year + 1)
    total_days = (current_date - birth_date).days
    total_hours = total_days * 24
    total_minutes = total_hours * 60
    return {
        "years": years,
        "months": months,
        "days": days,
        "total_months": years * 12 + months,
        "total_weeks": total_days // 7,
        "total_days": total_days,
        "total_hours": total_hours,
        "total_minutes": total_minutes,
        "total_seconds": total_minutes * 60,
        "days_to_birthday": (next_birthday - current_date).days,
        "next_birthday": next_birthday,
        "weekday_born": birth_date.strftime("%A"),
    }


def display(birth_date: date) -> None:
    """Print the calculated statistics in a terminal-friendly layout."""
    result = calculate_age(birth_date)
    print(f"Date of birth: {birth_date} ({result['weekday_born']})")
    print(f"Age: {result['years']} years, {result['months']} months, {result['days']} days")
    print(f"Total lived: {result['total_months']:,} months, {result['total_weeks']:,} weeks, {result['total_days']:,} days")
    print(f"             {result['total_hours']:,} hours, {result['total_minutes']:,} minutes, {result['total_seconds']:,} seconds")
    print(f"Next birthday: {result['next_birthday']} ({result['days_to_birthday']} days away)")


def main() -> None:
    """Parse a birth date and print its current age statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("birth_date", type=parse_date, help="YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY, or MM/DD/YYYY")
    args = parser.parse_args()
    try:
        display(args.birth_date)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
