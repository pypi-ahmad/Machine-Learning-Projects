# WhatsApp auto messenger

`WhatsApp-Auto-Messenger.py` prepares a WhatsApp Web message schedule through
`pywhatkit`. It validates the recipient and schedule, then previews the plan.
It does not open WhatsApp Web unless you explicitly add `--send`.

## Requirements

- Python 3.13 or later
- A browser with an active WhatsApp Web session

## Install

```powershell
cd "Python Scripts/WhatsApp Auto Messenger"
uv sync
```

## Preview a message

Use an E.164 recipient number and a local 24-hour delivery time. If the time
has already passed today, the script schedules the next day.

```powershell
uv run python WhatsApp-Auto-Messenger.py +15551234567 "Meeting starts soon" --at 14:30
```

The default command is a dry run. It prints the recipient, schedule, and text,
then exits without opening a browser.

## Send a confirmed schedule

Add `--send` only after reviewing the dry-run output:

```powershell
uv run python WhatsApp-Auto-Messenger.py +15551234567 "Meeting starts soon" --at 14:30 --send
```

The selected time must be at least two minutes in the future. `--wait` controls
how long pywhatkit waits for WhatsApp Web to load; it defaults to 15 seconds.

## Limits

`pywhatkit` controls the browser and WhatsApp Web interface. A missing login,
network issue, changed web interface, or browser automation restriction can
prevent delivery. The tool does not confirm that WhatsApp accepted the message.

Run `uv run python WhatsApp-Auto-Messenger.py --help` for the command
reference.
