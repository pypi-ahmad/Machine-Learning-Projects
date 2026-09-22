# Text Message Sender

A small Twilio CLI for sending one SMS only after an explicit `--send` flag. It never stores credentials, phone numbers, or messages in project files.

## Setup

```powershell
uv sync
```

Set these environment variables in your operating system before sending:

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM_NUMBER` — a Twilio-enabled sender number in E.164 format

Restart the terminal or coding host after adding environment variables so it inherits them. Do not put credentials in source code or a committed `.env` file.

## Use

First confirm the recipient and message without sending anything:

```powershell
uv run python sendText.py --to +15551234567 --body "Hello" --dry-run
```

To send an SMS, add the explicit confirmation flag:

```powershell
uv run python sendText.py --to +15551234567 --body "Hello" --send
```

The recipient and sender must comply with your Twilio account configuration, regional rules, consent requirements, and any applicable messaging policies.

## Dependencies

- Python 3.14+
- twilio
