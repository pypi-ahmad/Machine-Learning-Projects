# SMS Automation

> A Python script to send SMS messages to multiple recipients using the Twilio API.

## Overview

This script uses the Twilio REST API to send a custom SMS message to one or more phone numbers. All credentials and message details are entered interactively at runtime.

## Features

- Send SMS to multiple recipients (comma-separated phone numbers)
- Interactive prompts for Twilio credentials, sender number, message body, and recipient numbers
- Uses the official Twilio Python SDK (`twilio.rest.Client`)

## Project Structure

```
SMS Automation/
└── script.py
```

## Requirements

- Python 3.x
- `twilio`
- A Twilio account with:
  - Account SID
  - Auth Token
  - A Twilio phone number (or verified sender number)
  - Verified recipient numbers (required for trial accounts)

## Installation

```bash
cd "SMS Automation"
pip install twilio
```

## Usage

```bash
python script.py
```

**Interactive prompts:**

```
Enter your ACCOUNT SID: ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Enter your AUTH TOKEN: your_auth_token_here
Enter number from which you want to send the SMS: +1234567890
Enter the massage: Hello from Python!
Enter comma separated numbers to which you want to send the SMS: +1111111111,+2222222222
```

- Phone numbers should include country codes (e.g., `+1` for US)
- Separate multiple recipient numbers with commas (no spaces)

## How It Works

1. Prompts the user for Twilio Account SID, Auth Token, sender number, message body, and comma-separated recipient numbers
2. Splits the recipient input on commas into a list
3. Creates a `twilio.rest.Client` with the provided credentials
4. Iterates over the recipient list and calls `client.messages.create()` for each number

## Configuration

All configuration is provided at runtime via interactive input:

- **Account SID**: Your Twilio account identifier
- **Auth Token**: Your Twilio authentication token
- **From number**: The Twilio phone number to send from
- **To numbers**: Comma-separated list of recipient phone numbers

## Limitations

- No error handling — if Twilio credentials are invalid or a number is malformed, the script crashes with an unhandled exception
- No confirmation or delivery status feedback after sending
- Phone numbers are not validated before sending
- Prompt says "massage" instead of "message" (typo in source code)
- No support for reading credentials from environment variables or config files

## Security Notes

- Twilio Account SID and Auth Token are entered in plaintext via `input()` — they may be visible in terminal history
- Credentials should ideally be stored in environment variables rather than entered interactively
- Trial Twilio accounts can only send to pre-verified phone numbers

## License

Not specified.
