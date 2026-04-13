"""Age Calculator — CLI tool.

Calculate exact age, days until next birthday, day of birth,
and life statistics from a birth date.

Usage:
    python main.py
    python main.py 1990-05-15
"""

import sys
from datetime import date, datetime


def parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognised date format: '{s}'. Use YYYY-MM-DD.")


def calc_age(dob: date, today: date | None = None) -> dict:
    today = today or date.today()
    if dob > today:
        raise ValueError("Birth date is in the future.")

    years  = today.year  - dob.year
    months = today.month - dob.month
    days   = today.day   - dob.day

    if days < 0:
        months -= 1
        prev_month = (today.replace(day=1) - __import__("datetime").timedelta(days=1))
        days += prev_month.day

    if months < 0:
        years  -= 1
        months += 12

    # Days until next birthday
    next_bday = dob.replace(year=today.year)
    if next_bday <= today:
        next_bday = dob.replace(year=today.year + 1)
    days_to_bday = (next_bday - today).days

    total_days    = (today - dob).days
    total_weeks   = total_days // 7
    total_months  = years * 12 + months
    total_hours   = total_days * 24
    total_minutes = total_hours * 60
    total_seconds = total_minutes * 60

    weekday_born = dob.strftime("%A")

    return {
        "years": years, "months": months, "days": days,
        "total_days": total_days, "total_weeks": total_weeks,
        "total_months": total_months, "total_hours": total_hours,
        "total_minutes": total_minutes, "total_seconds": total_seconds,
        "days_to_birthday": days_to_bday, "next_birthday": next_bday,
        "weekday_born": weekday_born,
    }


def display(dob: date):
    today = date.today()
    try:
        r = calc_age(dob, today)
    except ValueError as e:
        print(f"  Error: {e}")
        return

    print(f"\n  Date of birth : {dob}  ({r['weekday_born']})")
    print(f"  Today         : {today}")
    print(f"\n  Age           : {r['years']} years, {r['months']} months, {r['days']} days")
    print(f"\n  Total time lived:")
    print(f"    {r['total_months']:>12,}  months")
    print(f"    {r['total_weeks']:>12,}  weeks")
    print(f"    {r['total_days']:>12,}  days")
    print(f"    {r['total_hours']:>12,}  hours")
    print(f"    {r['total_minutes']:>12,}  minutes")
    print(f"    {r['total_seconds']:>12,}  seconds")
    print(f"\n  Next birthday : {r['next_birthday']}  ({r['days_to_birthday']} days away)")
    print()


def main():
    if len(sys.argv) > 1:
        try:
            dob = parse_date(sys.argv[1])
            display(dob)
        except ValueError as e:
            print(f"Error: {e}")
        return

    print("Age Calculator")
    print("──────────────")
    while True:
        raw = input("Enter birth date (YYYY-MM-DD) or 'q' to quit: ").strip()
        if raw.lower() == "q":
            break
        try:
            dob = parse_date(raw)
            display(dob)
        except ValueError as e:
            print(f"  Error: {e}\n")


if __name__ == "__main__":
    main()
