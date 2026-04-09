# WhatsApp Auto Messenger

> Send scheduled WhatsApp messages using `pywhatkit` with an interactive prompt.

## Overview

This script prompts the user for a recipient's phone number, a message, and a scheduled time, then uses the `pywhatkit` library to send the WhatsApp message at the specified time via WhatsApp Web.

## Features

- Interactive command-line prompts for phone number, message, and schedule time
- Scheduled message delivery at a user-specified hour and minute
- Supports both individual and group messaging (group messaging code included as a comment)

## Project Structure

```
WhatsApp-Auto-Messenger/
├── README.md
├── requirements.txt
└── WhatsApp-Auto-Messenger.py
```

## Requirements

- Python 3.x
- `pywhatkit` — WhatsApp automation library
- A web browser with an active WhatsApp Web session (logged in via QR code)

## Installation

```bash
cd WhatsApp-Auto-Messenger
pip install -r requirements.txt
```

## Usage

```bash
python WhatsApp-Auto-Messenger.py
```

The script will prompt for:
1. **Receiver Phone Number** — Must include country code (e.g., `+1234567890`)
2. **Message** — The text message to send
3. **Hour** — Scheduled hour in 24-hour format
4. **Minutes** — Scheduled minutes

### Group messaging (commented out in code)

```python
pywhatkit.sendwhatmsg_to_group(GroupID, message, time_hour, time_min, wait_time)
```

The Group ID is found in the group's invite link.

## How It Works

1. Collects user input via `input()` for phone number, message, hour, and minute.
2. Calls `pywhatkit.sendwhatmsg(phoneno, message, Time_hrs, Time_min)`.
3. `pywhatkit` waits until the scheduled time, opens WhatsApp Web in the default browser, finds the contact, pastes the message, and sends it.

## Configuration

No configuration files. All parameters are provided interactively at runtime.

## Limitations

- The schedule time must be in the future; no validation is performed on the input.
- Requires the default browser to be logged into WhatsApp Web before the scheduled time.
- The browser window opens visibly — not a background operation.
- No error handling for invalid phone numbers or connection issues.
- No confirmation prompt before sending.
- `pywhatkit` depends on `pyautogui` for keyboard simulation, which may not work on all systems.

## Security Notes

No sensitive data or credentials in the code.

## License

Not specified.
