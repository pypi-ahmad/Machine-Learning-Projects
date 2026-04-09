# WhatsApp Bot

> A WhatsApp message scheduler using `pywhatkit` with time offset from the current hour.

## Overview

This script collects a mobile number, message, and a time offset (hours and minutes from now), then uses `pywhatkit` to schedule and send a WhatsApp message via WhatsApp Web.

## Features

- Interactive CLI prompts for recipient number, message, and delivery time
- Time offset relative to the current hour using `datetime`
- Sends WhatsApp messages via `pywhatkit.sendwhatmsg()`

## Project Structure

```
whatsapp_Bot/
├── main.py
└── Readme.md
```

## Requirements

- Python 3.x
- `pywhatkit` — WhatsApp automation library
- `datetime` (stdlib) — Current time retrieval
- A web browser with an active WhatsApp Web session

## Installation

```bash
cd whatsapp_Bot
pip install pywhatkit
```

## Usage

```bash
python main.py
```

The script will prompt for:
1. **Receiver Mobile No** — Must include country code (e.g., `+1234567890`)
2. **Message** — The text to send
3. **Hour offset** — Number of hours to add to the current hour
4. **Minute** — The minute value for the scheduled time

### Example

If the current time is 14:00 and you enter hour offset `1` and minute `30`, the message will be scheduled for 15:30.

## How It Works

1. Gets the current hour using `datetime.now().strftime("%H")`.
2. Adds the user-specified hour offset to the current hour.
3. Calls `pywhatkit.sendwhatmsg(mobile, message, hour, minute)` to schedule the message.
4. `pywhatkit` opens WhatsApp Web in the default browser at the scheduled time and sends the message.

## Configuration

No configuration files. All parameters are provided interactively at runtime.

## Limitations

- The hour calculation can produce values > 23 (e.g., current hour 22 + offset 3 = 25), which `pywhatkit` won't handle correctly.
- No input validation for phone number format, hour, or minute values.
- The "Enter hour" prompt is ambiguous — it's an offset, not an absolute hour.
- No error handling for invalid inputs or connection issues.
- Requires the browser to be logged into WhatsApp Web before the scheduled time.

## Security Notes

No sensitive data or credentials in the code.

## License

Not specified.
