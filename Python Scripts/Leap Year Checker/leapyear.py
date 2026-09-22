"""Check whether a Gregorian calendar year is a leap year."""

import argparse


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a year is a Gregorian leap year.")
    parser.add_argument("year", type=int, help="Year to check.")
    args = parser.parse_args()
    print(f"{args.year} is {'a' if is_leap_year(args.year) else 'not a'} leap year.")


if __name__ == "__main__":
    main()
