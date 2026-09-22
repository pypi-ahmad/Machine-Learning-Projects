"""Preview a CSV mailing before explicitly sending it through Gmail SMTP."""

import argparse
import csv
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path


SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
DEFAULT_SUBJECT = "Welcome to Python"
DEFAULT_BODY = """Python is an interpreted, high-level, general-purpose programming language.

Created by Guido van Rossum and first released in 1991, Python emphasizes
readability through its use of significant whitespace.
"""


def load_recipients(csv_path: Path) -> list[str]:
    """Return non-empty recipient addresses from the first CSV column."""
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))

    recipients = []
    for row in rows:
        if row and row[0].strip():
            recipients.append(row[0].strip())
    if not recipients:
        raise ValueError("The recipient CSV does not contain any email addresses.")
    return recipients


def build_message(sender: str, recipients: list[str], subject: str, body: str) -> EmailMessage:
    """Create one BCC message so recipients do not see one another's addresses."""
    message = EmailMessage()
    message["From"] = sender
    message["To"] = sender
    message["Bcc"] = ", ".join(recipients)
    message["Subject"] = subject
    message.set_content(body)
    return message


def send_message(message: EmailMessage, sender: str, app_password: str) -> None:
    """Send a prepared message with STARTTLS encryption."""
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as client:
        client.starttls(context=ssl.create_default_context())
        client.login(sender, app_password)
        client.send_message(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipients", type=Path, default=Path("emails.csv"))
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--body-file", type=Path, help="UTF-8 text file for the email body.")
    parser.add_argument("--send", action="store_true", help="Ask for final confirmation and send.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipients = load_recipients(args.recipients)
    body = args.body_file.read_text(encoding="utf-8") if args.body_file else DEFAULT_BODY
    sender = os.environ.get("GMAIL_SMTP_ADDRESS", "preview@example.invalid")
    message = build_message(sender, recipients, args.subject, body)

    print(f"Prepared one BCC message for {len(recipients)} recipient(s).")
    print(f"Subject: {args.subject}")
    print(f"Recipients file: {args.recipients}")
    if not args.send:
        print("Preview only. Re-run with --send to request final confirmation.")
        return

    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not app_password or sender == "preview@example.invalid":
        raise RuntimeError("Set GMAIL_SMTP_ADDRESS and GMAIL_APP_PASSWORD before sending.")
    if input("Type SEND to deliver this message: ") != "SEND":
        print("Cancelled. No email was sent.")
        return

    send_message(message, sender, app_password)
    print(f"Sent one BCC message to {len(recipients)} recipient(s).")


if __name__ == "__main__":
    main()
