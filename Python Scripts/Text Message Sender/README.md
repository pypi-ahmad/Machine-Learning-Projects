# Send Texts

> Sends an SMS message using the Twilio API.

## Overview

A Python script that sends a text message ("Hello from Python!") using the Twilio REST API. Uses placeholder credentials and phone numbers.

## Features

- Sends an SMS message via the Twilio API
- Prints the message SID upon successful delivery

## Project Structure

```
Send-Texts/
├── sendText.py       # Main script to send SMS via Twilio
├── pythonFrom.txt    # Educational text about Python imports (not used by the script)
└── README.md
```

## Requirements

- Python 3.x
- `twilio`
- A Twilio account with Account SID, Auth Token, and a Twilio phone number

## Installation

```bash
cd "Send-Texts"
pip install twilio
```

## Usage

1. Edit `sendText.py` and replace the placeholder values:
   - `account_sid`: Your Twilio Account SID
   - `auth_token`: Your Twilio Auth Token
   - `to`: The recipient's phone number
   - `from_`: Your Twilio phone number
2. Run the script:

```bash
python sendText.py
```

The message SID is printed on success.

## How It Works

1. Imports the Twilio `Client` from `twilio.rest`.
2. Initializes the client with `account_sid` and `auth_token`.
3. Calls `client.messages.create()` with `to`, `from_`, and `body` parameters.
4. Prints the returned `message.sid` to confirm the message was queued.

## Configuration

All values are hardcoded in `sendText.py` and must be edited:

- `account_sid`: Set to `"0000"` (placeholder)
- `auth_token`: Set to `"0000"` (placeholder)
- `to`: Set to `"0000"` (placeholder)
- `from_`: Set to `"0000"` (placeholder)
- `body`: Hardcoded as `"Hello from Python!"`

## Limitations

- All credentials and phone numbers are hardcoded as `"0000"` placeholders.
- Only sends a single hardcoded message — no CLI arguments or dynamic input.
- No error handling for invalid credentials, network failures, or Twilio API errors.
- The `pythonFrom.txt` file is unrelated educational text about Python imports and is not used by the script.

## Security Notes

- **Hardcoded credentials**: The Account SID and Auth Token are stored directly in the source code. These should be loaded from environment variables or a secure configuration file, never committed to version control.
- Twilio Auth Tokens grant full API access; exposure could allow unauthorized messaging and billing.

## License

Not specified.
