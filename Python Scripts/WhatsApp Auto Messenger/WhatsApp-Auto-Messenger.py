"""Schedule a WhatsApp Web message with an explicit send confirmation."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta

import pywhatkit


@dataclass(frozen=True)
class MessagePlan:
    """The recipient, content, and local delivery time selected by the user."""

    recipient: str
    message: str
    scheduled_for: datetime


def phone_number(value: str) -> str:
    """Normalize and validate an E.164-style recipient number."""
    normalized = re.sub(r"[\s()\-]", "", value)
    if not re.fullmatch(r"\+[1-9]\d{7,14}", normalized):
        raise argparse.ArgumentTypeError("recipient must be an E.164 number, such as +15551234567")
    return normalized


def clock_time(value: str) -> time:
    """Parse a local 24-hour HH:MM time."""
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise argparse.ArgumentTypeError("--at must use 24-hour HH:MM format") from error


def next_occurrence(at: time, now: datetime) -> datetime:
    """Return the next local occurrence of a requested clock time."""
    scheduled_for = now.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
    if scheduled_for <= now:
        scheduled_for += timedelta(days=1)
    return scheduled_for


def send_message(plan: MessagePlan, wait_time: int, sender=pywhatkit.sendwhatmsg) -> None:
    """Delegate the confirmed schedule to pywhatkit's WhatsApp Web sender."""
    sender(
        plan.recipient,
        plan.message,
        plan.scheduled_for.hour,
        plan.scheduled_for.minute,
        wait_time=wait_time,
        tab_close=True,
        close_time=3,
    )


def main() -> None:
    """Validate a plan, preview it, and send only with --send."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipient", type=phone_number, help="recipient in E.164 format")
    parser.add_argument("message", help="message text")
    parser.add_argument("--at", type=clock_time, required=True, help="local 24-hour send time, HH:MM")
    parser.add_argument("--wait", type=int, default=15, help="WhatsApp Web load wait in seconds (default: 15)")
    parser.add_argument("--send", action="store_true", help="open WhatsApp Web and schedule the message")
    args = parser.parse_args()
    if not args.message.strip():
        parser.error("message cannot be empty")
    if args.wait < 1:
        parser.error("--wait must be at least 1")

    now = datetime.now()
    plan = MessagePlan(args.recipient, args.message, next_occurrence(args.at, now))
    if plan.scheduled_for - now < timedelta(minutes=2):
        parser.error("choose a send time at least two minutes from now")
    print(f"Recipient: {plan.recipient}\nScheduled: {plan.scheduled_for:%Y-%m-%d %H:%M}\nMessage: {plan.message}")
    if not args.send:
        print("Dry run only. Add --send to open WhatsApp Web and schedule this message.")
        return

    try:
        send_message(plan, args.wait)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print("WhatsApp Web scheduling was requested.")


if __name__ == "__main__":
    main()
