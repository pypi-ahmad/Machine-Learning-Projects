# Send Email from CSV

This small command-line tool prepares one email for the addresses in a CSV file. It previews the work by default and requires both `--send` and a typed confirmation before it contacts Gmail.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)
- A Gmail account with two-step verification and an app password for SMTP sending. See [Google's app-password guidance](https://support.google.com/accounts/answer/2461835).

## Setup

```powershell
uv sync --no-config
```

Set the sender credentials only in the current PowerShell session. Do not put them in a file or commit them.

```powershell
$env:GMAIL_SMTP_ADDRESS = "you@gmail.com"
$env:GMAIL_APP_PASSWORD = "your-app-password"
```

Create a UTF-8 CSV such as `emails.csv`, with one address in the first column of each non-empty row. The tool does not create or store recipient data.

## Preview

Preview the message and recipient count without connecting to SMTP:

```powershell
uv run --no-config python Sending_mail.py --recipients emails.csv --subject "Welcome to Python"
```

Use a text file for the message body when the built-in example is not suitable:

```powershell
uv run --no-config python Sending_mail.py --recipients emails.csv --body-file message.txt
```

## Send

After reviewing the preview, add `--send`. The program will ask you to type `SEND` exactly before it opens the SMTP connection.

```powershell
uv run --no-config python Sending_mail.py --recipients emails.csv --subject "Welcome to Python" --send
```

The email is sent once with all recipient addresses in BCC, so recipients cannot see one another's addresses.

## Safety notes

- No email is sent without the two-step `--send` and typed-confirmation flow.
- Credentials come from environment variables and are never written to disk by the script.
- Validate the recipient list and message content before sending; delivery errors are returned by Gmail at runtime.
