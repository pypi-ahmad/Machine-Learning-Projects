# Auto Birthday Wisher

A Python script that automatically sends birthday-wish emails to contacts whose birthdays match today's date, reading data from an Excel spreadsheet and sending via Gmail SMTP.

## Overview

- Reads contact information from `data.xlsx`, checks each person's birthday against today's date, sends a personalized email via Gmail, and tracks sent wishes to avoid duplicates
- **Project type:** CLI / Automation Bot

## Features

- Reads friend/contact data (name, birthday, email, custom message) from an Excel file (`data.xlsx`)
- Compares each contact's birthday (day and month) against today's date
- Sends a personalized email with a custom "Dialogue" (message) column from the spreadsheet
- Tracks the year each wish was sent in a `LastWishedYear` column to prevent duplicate wishes in the same year
- Updates `data.xlsx` after sending, appending the current year to `LastWishedYear`
- Prompts for Gmail credentials at runtime via `input()`

## Dependencies

| Package | Source | Install |
|---------|--------|---------|
| `pandas` | PyPI (inferred from import) | `pip install pandas` |
| `openpyxl` | PyPI (inferred — required by pandas for `.xlsx` I/O) | `pip install openpyxl` |
| `datetime` | Python standard library | — |
| `smtplib` | Python standard library | — |
| `os` | Python standard library | — |

## How It Works

1. The script prompts the user to enter their Gmail address and Gmail password via `input()`.
2. `data.xlsx` is loaded into a Pandas DataFrame using `pd.read_excel()`.
3. Today's date is formatted as `dd-mm` and the current year as `YYYY`.
4. The script iterates over each row in the DataFrame:
   - Parses the `Birthday` column (expected format `dd-mm-YYYY`) into a `datetime` object, then formats it as `dd-mm`.
   - If the birthday matches today **and** the current year is not already present in the `LastWishedYear` column, `sendEmail()` is called.
5. `sendEmail()` connects to `smtp.gmail.com` on port 587 with STARTTLS, logs in with the provided credentials, sends the email with the subject "Happy Birthday" and the custom message, then quits the SMTP session.
6. After processing all rows, `LastWishedYear` is updated for each wished contact by appending the current year (comma-separated).
7. The updated DataFrame is written back to `data.xlsx`.

## Project Structure

```
Auto Birthday Wisher/
├── Auto B'Day Wisher.py   # Main script
├── data.xlsx              # Contact & birthday data spreadsheet
├── emailReceived.jpg      # Screenshot showing a received email (demo)
└── README.md
```

### Expected `data.xlsx` Columns

| Column | Format | Description |
|--------|--------|-------------|
| `Name` | String | Contact's name |
| `Birthday` | `dd-mm-YYYY` | Date of birth |
| `Email` | String | Recipient email address |
| `Dialogue` | String | Custom birthday message body |
| `LastWishedYear` | String | Comma-separated years when wishes were already sent |

## Setup & Installation

```bash
pip install pandas openpyxl
```

## How to Run

```bash
cd "Auto Birthday Wisher"
python "Auto B'Day Wisher.py"
```

You will be prompted to enter your Gmail address and password. The script will check `data.xlsx` and send emails to any contacts whose birthday matches today.

## Configuration

| Item | Description |
|------|-------------|
| `data.xlsx` | Must be in the same directory. Columns: `Name`, `Birthday` (dd-mm-YYYY), `Email`, `Dialogue`, `LastWishedYear` |
| Gmail credentials | Entered interactively via `input()` at runtime |
| **App Password** | If 2FA is enabled on the Gmail account, use a Google App Password instead of your regular password |

## Testing

No formal test suite present.

## Limitations

- Gmail password is entered in plaintext via `input()` — visible on screen during entry.
- The SMTP server (`smtp.gmail.com`) and port (`587`) are hardcoded — only Gmail is supported.
- Birthday format must be exactly `dd-mm-YYYY`; no validation or alternate format support.
- No error handling around SMTP operations or file I/O — failures will crash the script.
- The script must be scheduled externally (e.g., cron, Windows Task Scheduler) to run daily for automated wishing.
- `os.chdir(current_path)` is called but `current_path` is already `os.getcwd()`, making it a no-op.
- Google has deprecated "Less Secure App Access"; an App Password or OAuth 2.0 flow is now required.

## Security Notes

- **Gmail password** is entered in plaintext and passed directly to `smtplib.SMTP.login()`. Consider using environment variables, a `.env` file, or a secrets manager instead.
- Prefer **App Passwords** (with 2-Step Verification enabled) over your main account password.
- `data.xlsx` contains personal data (names and email addresses) — treat it as sensitive and do not commit it to public repositories.

