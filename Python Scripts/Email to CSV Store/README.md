# Store Emails in CSV

> Python script that connects to a Gmail IMAP server and exports inbox emails (date, sender, subject, body) to a CSV file.

## Overview

This script connects to Gmail's IMAP server using SSL, authenticates with credentials from a local text file, fetches the N most recent emails from the inbox, extracts header information and body text (including HTML-to-text conversion), and writes the results to a CSV file.

## Features

- Connects to Gmail IMAP server with SSL
- Reads credentials from a local `credentials.txt` file
- Fetches the N most recent emails from the inbox (configurable via hardcoded variable)
- Extracts email date, sender, subject, and body text
- Handles both multipart and single-part email messages
- Converts HTML email bodies to plain text using BeautifulSoup
- Outputs results to a CSV file with columns: Date, From, Subject, Text mail
- Includes logging for warnings and exceptions

## Project Structure

```
Store_emails_in_csv/
├── store_emails.py
├── credentials.txt
├── mails.csv
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- `beautifulsoup4` (specified in requirements.txt)
- `lxml` (specified in requirements.txt)

## Installation

```bash
cd Store_emails_in_csv
pip install -r requirements.txt
```

## Usage

1. **Set up credentials:** Edit `credentials.txt` with your Gmail address on line 1 and your app password on line 2:

   ```
   yourEmailID
   yourPassword
   ```

2. **Configure Gmail:** Enable IMAP access in your Gmail settings. If using 2FA, generate an [App Password](https://support.google.com/accounts/answer/185833).

3. **Run the script:**

   ```bash
   python store_emails.py
   ```

4. **Output:** The fetched emails will be saved to `mails.csv`.

## How It Works

1. Reads Gmail credentials from `credentials.txt`
2. Establishes an SSL connection to `imap.gmail.com:993`
3. Logs in and selects the INBOX
4. Iterates over the N most recent emails (newest first), fetching each via RFC822
5. Parses each email using Python's `email` library with `policy.default`
6. For multipart messages, walks through all parts and extracts text content
7. Converts HTML bodies to plain text using `BeautifulSoup` with the `lxml` parser
8. Writes extracted fields (date, from, subject, body) to `mails.csv`

## Configuration

| Item | Location | Description |
|---|---|---|
| Email credentials | `credentials.txt` | Line 1: email address, Line 2: password |
| Number of emails to fetch | `store_emails.py` line ~107 | Hardcoded `N = 2` |
| IMAP host/port | `store_emails.py` lines 19-20 | Hardcoded to `imap.gmail.com:993` |
| Output file | `store_emails.py` line 15 | Hardcoded to `mails.csv` |
| Mailbox | `store_emails.py` line 34 | Hardcoded to `"INBOX"` |

## Limitations

- The number of emails to fetch (`N = 2`) is hardcoded — must edit source code to change
- Only fetches from the INBOX folder
- Only supports Gmail's IMAP server (hardcoded host)
- No command-line arguments — all configuration requires editing source files
- The `credentials.txt` file stores credentials in plaintext
- No retry logic for network failures
- Logging level is set to `WARNING`, so normal operation is silent

## Security Notes

- **Plaintext credentials:** `credentials.txt` stores the email address and password in plaintext. Consider using environment variables or a secrets manager instead.
- **Credential file included:** The sample `credentials.txt` is committed to the repository with placeholder values. Ensure real credentials are never committed.

## License

Not specified.
