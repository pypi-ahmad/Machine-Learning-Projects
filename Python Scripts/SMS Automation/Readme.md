# SMS Automation

A Twilio command-line sender for one message and one or more recipient numbers.
The command is a dry run unless `--send` is explicitly supplied.

## Credentials

Set these process environment variables before sending:

```text
TWILIO_ACCOUNT_SID
TWILIO_AUTH_TOKEN
```

Do not place Twilio credentials in source files, command-line arguments, or
committed `.env` files. If these variables were added recently on Windows,
relaunch the terminal or coding host before using `--send`.

## Dry run

```powershell
uv sync
uv run python script.py --from-number +15551234567 --to +15557654321 --message "Hello"
```

This validates the command shape and reports the recipient count without
contacting Twilio or sending a message.

## Send

```powershell
uv run python script.py --from-number +15551234567 --to +15557654321 +15559876543 --message "Hello" --send
```

Numbers must use E.164 format and begin with `+`. Sending can incur charges,
and trial accounts may require verified recipients. Confirm the message,
recipients, account restrictions, and cost before using `--send`.

## Dependencies

uv manages the official Twilio SDK in `pyproject.toml`; exact resolved versions
are recorded in `uv.lock`.
