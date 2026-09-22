# WhatsApp Automation

Send one WhatsApp Web message through Selenium after an explicit confirmation flag. The default command only previews the recipient and message; it does not open a browser or send anything.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- Google Chrome or another Selenium-supported browser
- A WhatsApp Web login

## Install

```powershell
cd "Python Scripts\WhatsApp Automation"
uv sync
```

Selenium manages a compatible driver when a supported browser is available, so no hardcoded ChromeDriver path is required.

## Preview a message

```powershell
uv run python .\whatsappAutomation.py "Contact name" "Hello from the automation"
```

## Send a message

Pass `--send` only when ready to open WhatsApp Web and send the message:

```powershell
uv run python .\whatsappAutomation.py "Contact name" "Hello from the automation" --send
```

Scan the WhatsApp Web QR code if prompted. The command waits up to 60 seconds for login and the contact. Adjust that window when needed:

```powershell
uv run python .\whatsappAutomation.py "Contact name" "Hello" --send --timeout 120
```

The browser closes after the send attempt unless `--keep-open` is supplied.

## Notes

- Contact names must match the saved WhatsApp contact name and cannot contain quote characters.
- WhatsApp Web's page structure and automation restrictions may change, so a selector can stop working.
- Send messages only with the recipient's consent and in accordance with WhatsApp's terms.
