# WhatsApp Bot

Schedule one WhatsApp Web message with `pywhatkit`. The command previews the schedule by default and only opens WhatsApp Web when `--send` is supplied.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- A browser session that can use WhatsApp Web

## Install

```powershell
cd "Python Scripts\WhatsApp Bot"
uv sync
```

## Preview a schedule

```powershell
uv run python .\main.py +15551234567 "Hello" --in-minutes 5
```

The number must be in international format and the delay must be at least two minutes.

## Schedule a message

Pass `--send` only when ready for `pywhatkit` to open WhatsApp Web:

```powershell
uv run python .\main.py +15551234567 "Hello" --in-minutes 5 --send
```

The tool adds a one-minute safety margin to the selected delay because `pywhatkit` schedules at minute precision. It closes the WhatsApp Web tab after attempting to send.

## Notes

- The browser must be able to use WhatsApp Web at the scheduled time.
- Scheduling depends on local clock accuracy, browser availability, and WhatsApp Web behavior.
- Send messages only with the recipient's consent and in accordance with WhatsApp's terms.
