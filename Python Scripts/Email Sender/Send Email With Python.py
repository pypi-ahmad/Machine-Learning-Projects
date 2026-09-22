"""Preview or send personalized plain-text Gmail SMTP messages from a CSV file."""

import argparse
import csv
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from string import Template

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
ADDRESS_VARIABLE = "GMAIL_SMTP_ADDRESS"
PASSWORD_VARIABLE = "GMAIL_APP_PASSWORD"
REQUIRED_COLUMNS = {"name", "email", "math", "eng", "sci"}


def read_template(path: Path) -> Template:
    """Read a UTF-8 message template."""
    return Template(path.read_text(encoding="utf-8"))


def read_recipients(path: Path) -> list[dict[str, str]]:
    """Read recipient rows with the required grade-mail columns."""
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or not REQUIRED_COLUMNS.issubset(rows[0]):
        required = ", ".join(sorted(REQUIRED_COLUMNS))
        raise ValueError(f"CSV must include these headers: {required}")
    if not all(row["name"].strip() and row["email"].strip() for row in rows):
        raise ValueError("Every row must include a name and email.")
    return rows


def build_message(template: Template, sender: str, recipient: dict[str, str], subject: str) -> EmailMessage:
    """Render one recipient's template and construct an email message."""
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient["email"].strip()
    message["Subject"] = subject
    message.set_content(template.substitute(
        PERSON_NAME=recipient["name"],
        MATH=recipient["math"],
        ENG=recipient["eng"],
        SCI=recipient["sci"],
    ))
    return message


def send_messages(messages: list[EmailMessage], sender: str, app_password: str) -> None:
    """Send prepared messages through Gmail SMTP using STARTTLS."""
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(sender, app_password)
        for message in messages:
            server.send_message(message)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse template, recipient CSV, subject, and explicit send request."""
    parser = argparse.ArgumentParser(description="Preview or send personalized grade emails.")
    parser.add_argument("--template", type=Path, default=Path("template.txt"))
    parser.add_argument("--recipients", type=Path, default=Path("details.csv"))
    parser.add_argument("--subject", default="Mid-term grades")
    parser.add_argument("--send", action="store_true", help="request SMTP delivery after typed confirmation")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Preview emails by default and send only after explicit confirmation."""
    settings = parse_arguments(arguments)
    try:
        template = read_template(settings.template)
        recipients = read_recipients(settings.recipients)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        print(f"Could not prepare messages: {error}")
        return 1

    sender = os.environ.get(ADDRESS_VARIABLE)
    if not sender:
        print(f"Set {ADDRESS_VARIABLE} before running this script.")
        return 1
    try:
        messages = [build_message(template, sender, recipient, settings.subject) for recipient in recipients]
    except (KeyError, ValueError) as error:
        print(f"Could not render messages: {error}")
        return 1

    for message in messages:
        print(f"{message['To']} <- {message['Subject']}")
    if not settings.send:
        print(f"Preview only: {len(messages)} message(s). Add --send to request delivery.")
        return 0
    if input(f"Type SEND to email {len(messages)} recipient(s): ").strip() != "SEND":
        print("Cancelled.")
        return 0

    app_password = os.environ.get(PASSWORD_VARIABLE)
    if not app_password:
        print(f"Set {PASSWORD_VARIABLE} before sending.")
        return 1
    try:
        send_messages(messages, sender, app_password)
    except (OSError, smtplib.SMTPException) as error:
        print(f"Messages were not sent: {error}")
        return 1
    print(f"Sent {len(messages)} message(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
