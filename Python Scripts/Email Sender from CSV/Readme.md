# Send Email from CSV

> Sends bulk emails to a list of recipients read from a CSV file, with SMTP credentials stored in a separate text file.

## Overview

A Python script that reads email addresses from `emails.csv` and SMTP login credentials from `credentials.txt`, then sends a pre-defined email to each recipient via Gmail's SMTP server. The email has a hardcoded subject ("Welcome to Python") and a body describing Python.

## Features

- Reads recipient email addresses from a CSV file (`emails.csv`)
- Reads SMTP credentials from a separate text file (`credentials.txt`)
- Sends emails via Gmail SMTP (`smtp.gmail.com:587`) with TLS encryption
- Uses Python's `email.message.EmailMessage` for message construction
- Modular code with separate functions for credential loading, login, and sending

## Project Structure

```
Send_email_from_csv/
├── Sending_mail.py    # Main email sending script
├── credentials.txt    # SMTP login credentials (email and password, one per line)
├── emails.csv         # List of recipient email addresses (one per line)
└── Readme.md
```

## Requirements

- Python 3.x
- `smtplib` (Python standard library)
- `csv` (Python standard library)
- `email` (Python standard library)

No external dependencies required.

## Installation

```bash
cd "Send_email_from_csv"
```

## Usage

1. Edit `credentials.txt` with your Gmail address (line 1) and password/app password (line 2).
2. Edit `emails.csv` with one recipient email address per line.
3. Run the script:

```bash
python Sending_mail.py
```

**Console output:**
```
login
Send To recipient1@example.com
Send To recipient2@example.com
sent
```

## How It Works

1. **`get_credentials(filepath)`**: Reads the first two lines of `credentials.txt` — line 1 is the email address, line 2 is the password. (Note: the `filepath` parameter is ignored; `"credentials.txt"` is hardcoded inside the function.)
2. **`login(email_address, email_pass, s)`**: Calls `ehlo()`, `starttls()`, `ehlo()` again, and `login()` on the SMTP connection.
3. **`send_mail()`**: Creates an SMTP connection to `smtp.gmail.com:587`, logs in, constructs an `EmailMessage` with a hardcoded subject and body, reads `emails.csv`, and sends the message to each address using `s.send_message()`.
4. The script runs `send_mail()` when executed directly.

## Configuration

- **SMTP server**: Hardcoded to `smtp.gmail.com` port `587`.
- **Subject**: Hardcoded as `"Welcome to Python"`.
- **Body**: Hardcoded description of the Python programming language.
- **Credentials file**: `credentials.txt` — line 1: email address, line 2: password.
- **Recipients file**: `emails.csv` — one email address per line.

## Limitations

- The `get_credentials()` function ignores its `filepath` parameter and always reads `"credentials.txt"` (hardcoded inside the function body).
- The `s.send_message()` call passes `email_address` (a string) as the first argument instead of the `EmailMessage` object, which will likely cause a runtime error.
- Credentials are read with `readline()` which may include trailing newline characters, potentially causing login failures.
- The email subject and body are hardcoded — no way to customize without editing the script.
- The CSV reader uses space as a delimiter and `|` as a quote character, which is non-standard and may not parse all email formats correctly.
- The included `emails.csv` contains what appears to be a real email address repeated four times.
- No error handling for SMTP failures, missing files, or malformed email addresses.

## Security Notes

- **Plaintext credentials**: `credentials.txt` stores the email and password in plaintext. This file should be added to `.gitignore` and never committed to version control.
- The included `credentials.txt` contains placeholder text (`YourEmail` / `Yourpass`), but the `emails.csv` file contains what appears to be a real email address.
- Gmail requires an "App Password" for SMTP access when 2-Factor Authentication is enabled; regular passwords may not work.
- Uses TLS (`starttls()`) for encrypted SMTP communication.

## License

Not specified.
