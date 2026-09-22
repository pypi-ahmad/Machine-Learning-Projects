"""Send confirmed birthday emails from a local Excel contact list."""

from __future__ import annotations

import argparse
import getpass
import smtplib
from datetime import date
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"Birthday", "Email", "Dialogue", "LastWishedYear"}


def pending_wishes(data: pd.DataFrame, today: date) -> pd.DataFrame:
    """Return contacts whose birthday is today and who were not wished this year."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    birthdays = pd.to_datetime(data["Birthday"], dayfirst=True, errors="coerce")
    years = data["LastWishedYear"].fillna("").astype(str)
    return data[(birthdays.dt.day == today.day) & (birthdays.dt.month == today.month) & ~years.str.contains(str(today.year), regex=False)].copy()


def send_email(sender: str, password: str, recipient: str, message: str) -> None:
    """Send one plain-text Gmail message through STARTTLS."""
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, f"Subject: Happy Birthday\n\n{message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data.xlsx"))
    parser.add_argument("--send", action="store_true", help="send emails; default is dry run")
    args = parser.parse_args()
    data = pd.read_excel(args.data)
    wishes = pending_wishes(data, date.today())
    if wishes.empty:
        print("No birthday wishes are due today.")
        return
    print(f"{len(wishes)} birthday wish(es) due.")
    if not args.send:
        print("Dry run: no email was sent. Re-run with --send to deliver.")
        return
    sender = input("Gmail address: ").strip()
    password = getpass.getpass("Gmail app password: ")
    for index, wish in wishes.iterrows():
        send_email(sender, password, wish["Email"], wish["Dialogue"])
        previous = str(data.at[index, "LastWishedYear"] or "").strip(", ")
        data.at[index, "LastWishedYear"] = ", ".join(filter(None, [previous, str(date.today().year)]))
    data.to_excel(args.data, index=False)
    print("Birthday wishes sent and spreadsheet updated.")


if __name__ == "__main__":
    main()
