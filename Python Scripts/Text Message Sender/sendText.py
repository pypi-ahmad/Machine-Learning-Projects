"""Send one SMS through Twilio after an explicit confirmation flag.

Usage:
    uv run python sendText.py --to +15551234567 --body "Hello" --dry-run
    uv run python sendText.py --to +15551234567 --body "Hello" --send
"""

import argparse
import os

from twilio.rest import Client


def get_client() -> Client:
    """Create a Twilio client from required process environment variables."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    missing = []
    if not account_sid:
        missing.append("TWILIO_ACCOUNT_SID")
    if not auth_token:
        missing.append("TWILIO_AUTH_TOKEN")
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")
    return Client(account_sid, auth_token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one SMS through Twilio.")
    parser.add_argument("--to", required=True, help="Recipient phone number in E.164 format.")
    parser.add_argument("--body", required=True, help="Message body.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate inputs without sending an SMS.")
    mode.add_argument("--send", action="store_true", help="Send the SMS using configured Twilio credentials.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(f"Dry run: would send {len(args.body)} character(s) to {args.to}.")
        return

    from_number = os.getenv("TWILIO_FROM_NUMBER")
    if not from_number:
        raise RuntimeError("Missing required environment variable: TWILIO_FROM_NUMBER")

    message = get_client().messages.create(to=args.to, from_=from_number, body=args.body)
    print(f"Message queued: {message.sid}")


if __name__ == "__main__":
    main()
