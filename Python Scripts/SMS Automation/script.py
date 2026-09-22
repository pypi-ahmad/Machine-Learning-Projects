"""Send SMS messages through Twilio after explicit confirmation.

Usage:
    uv run python script.py --from-number +15551234567 --to +15557654321 --message "Hello"
"""

import argparse
import os

from twilio.rest import Client


def parse_arguments() -> argparse.Namespace:
    """Parse SMS details without accepting credentials on the command line."""
    parser = argparse.ArgumentParser(description="Send SMS messages through Twilio.")
    parser.add_argument("--from-number", required=True, help="Twilio sender number in E.164 format")
    parser.add_argument("--to", required=True, nargs="+", help="One or more E.164 recipient numbers")
    parser.add_argument("--message", required=True, help="SMS body")
    parser.add_argument("--send", action="store_true", help="Send messages instead of showing a dry run")
    args = parser.parse_args()
    if not args.from_number.startswith("+"):
        parser.error("--from-number must use E.164 format, beginning with +")
    if not all(map(lambda number: number.startswith("+"), args.to)):
        parser.error("every --to number must use E.164 format, beginning with +")
    if not args.message.strip():
        parser.error("--message cannot be empty")
    return args


def twilio_client() -> Client:
    """Create a Twilio client from process environment credentials."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        raise SystemExit(
            "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN are required for --send. "
            "Relaunch the host if they were recently configured."
        )
    return Client(account_sid, auth_token)


def main() -> None:
    args = parse_arguments()
    recipient_count = len(args.to)
    if not args.send:
        print(f"Dry run: would send one SMS to {recipient_count} recipient(s). Use --send to dispatch.")
        return

    client = twilio_client()
    for recipient in args.to:
        client.messages.create(from_=args.from_number, body=args.message, to=recipient)
    print(f"Sent one SMS to {recipient_count} recipient(s).")


if __name__ == "__main__":
    main()
