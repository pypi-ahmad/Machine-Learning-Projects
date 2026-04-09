# WhatsApp Automation

> Automates sending WhatsApp messages via Selenium WebDriver and WhatsApp Web.

## Overview

This script uses Selenium to open WhatsApp Web in Chrome, finds a specified contact by name, types a message into the chat input, and sends it. The user has 15 seconds to scan the WhatsApp Web QR code before the script proceeds.

## Features

- Sends WhatsApp messages programmatically via WhatsApp Web
- Finds contacts by their exact saved name
- Interactive CLI for recipient name and message input
- Uses Selenium WebDriver for browser automation

## Project Structure

```
Whatsapp-Automation/
├── README.md
└── whatsappAutomation.py
```

## Requirements

- Python 3.x
- `selenium` — Browser automation
- Google Chrome browser
- [ChromeDriver](https://chromedriver.chromium.org/downloads) matching your Chrome version

## Installation

```bash
cd Whatsapp-Automation
pip install selenium
```

Download ChromeDriver and place it at the path specified in the script (or update the path).

## Usage

1. Update the ChromeDriver path in `whatsappAutomation.py`:
   ```python
   chrome_driver_binary = "C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe"
   ```
2. Run:
   ```bash
   python whatsappAutomation.py
   ```
3. Enter the contact name (exactly as saved in WhatsApp) and the message when prompted.
4. Scan the QR code within 15 seconds when WhatsApp Web opens.
5. The message will be sent automatically.

## How It Works

1. The `whatsapp(to, message)` function launches Chrome via Selenium and navigates to `https://web.whatsapp.com/`.
2. Waits 15 seconds (`sleep(15)`) for the user to scan the QR code.
3. Finds the contact by XPath: `//span[@title='CONTACT_NAME']` and clicks on it.
4. Locates the message input box via XPath: `//*[@id="main"]/footer/div[1]/div[2]/div/div[2]`.
5. Types the message using `send_keys()`.
6. Finds and clicks the send button via XPath: `//*[@id="main"]/footer/div[1]/div[3]/button`.
7. Waits 10 seconds, then prints "Message Sent!!".

## Configuration

| Setting | Location | Value |
|---|---|---|
| ChromeDriver path | `whatsappAutomation.py` | `"C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe"` |
| QR scan timeout | `whatsappAutomation.py` | `sleep(15)` — 15 seconds |

## Limitations

- ChromeDriver path is hardcoded; must be manually updated.
- Fixed 15-second wait for QR scanning — no dynamic wait/check.
- Uses deprecated Selenium methods (`find_element_by_xpath`, `find_elements_by_xpath`); newer Selenium versions require `find_element(By.XPATH, ...)`.
- WhatsApp Web's DOM structure changes frequently, which may break the hardcoded XPath selectors.
- Bare `except` clause in send logic catches all exceptions with only a print statement.
- The browser remains open after sending.
- Contact name must match exactly (case-sensitive) as saved in WhatsApp.
- Only sends a single message per run.

## Security Notes

No sensitive data or credentials in the code.

## License

Not specified.
