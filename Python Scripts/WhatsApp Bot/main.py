"""Schedule one WhatsApp Web message after explicit confirmation."""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta

import pywhatkit


def parse_args() -> argparse.Namespace:
    """Parse the recipient, message, and delivery delay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phone", help="Recipient number in international format, for example +15551234567")
    parser.add_argument("message", help="Message text to schedule")
    parser.add_argument(
        "--in-minutes",
        type=int,
        default=3,
        help="Minutes from now to schedule the message (minimum: 2; default: 3)",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Open WhatsApp Web and schedule the message",
    )
    return parser.parse_args()


def validate_phone(phone: str) -> str:
    """Validate and return an international phone number."""
    number = phone.strip()
    if not re.fullmatch(r"\+[1-9]\d{6,14}", number):
        raise ValueError("Phone number must be in international format, for example +15551234567.")
    return number


def scheduled_time(delay_minutes: int, now: datetime | None = None) -> datetime:
    """Return a minute-aligned time that is safely in the future."""
    if delay_minutes < 2:
        raise ValueError("Delay must be at least two minutes.")
    current_time = now or datetime.now()
    return current_time.replace(second=0, microsecond=0) + timedelta(minutes=delay_minutes + 1)


def schedule_message(phone: str, message: str, delivery_time: datetime) -> None:
    """Pass one validated message to pywhatkit's WhatsApp scheduler."""
    pywhatkit.sendwhatmsg(
        phone,
        message,
        delivery_time.hour,
        delivery_time.minute,
        wait_time=15,
        tab_close=True,
        close_time=3,
    )


def main() -> None:
    """Preview or schedule one WhatsApp message."""
    args = parse_args()
    try:
        phone = validate_phone(args.phone)
        delivery_time = scheduled_time(args.in_minutes)
    except ValueError as error:
        raise SystemExit(f"Error: {error}") from error

    if not args.send:
        print(f"Would schedule a message to {phone} at {delivery_time:%Y-%m-%d %H:%M}.")
        print("Pass --send to open WhatsApp Web and schedule it.")
        return

    try:
        schedule_message(phone, args.message, delivery_time)
    except Exception as error:
        raise SystemExit(f"Scheduling failed: {error}") from error
    print(f"Message scheduled for {delivery_time:%Y-%m-%d %H:%M}.")


if __name__ == "__main__":
    main()
