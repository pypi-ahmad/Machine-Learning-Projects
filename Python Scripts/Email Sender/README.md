# Send Email With Python

> A script that reads a message template and recipient details from files, then sends personalized emails via Gmail's SMTP server.

## Overview

This script reads an email template from `template.txt`, substitutes personalized fields (name, grades) using Python's `string.Template`, reads recipient information from `details.csv`, and sends individual emails through Gmail's SMTP server. **Note:** The code has significant indentation/syntax errors and will not run as-is.

## Features

- Reads email body from an external template file (`template.txt`) using `string.Template`
- Substitutes personalized fields (person name, math/eng/sci grades) per recipient
- Reads recipient details from `details.csv` (CSV format)
- Sends emails via Gmail SMTP (`smtp.gmail.com:587`) with TLS encryption
- Uses `MIMEMultipart` and `MIMEText` for email construction

## Project Structure

```
Send Email With Python/
├── Send Email With Python.py   # Main email sending script
└── README.md
```

## Requirements

- Python 3.x
- `smtplib` (Python standard library)
- `csv` (Python standard library)
- `email` (Python standard library)
- `string` (Python standard library)

The script also requires two external files (not included):
- `template.txt` — email body template with `$PERSON_NAME`, `$MATH`, `$ENG`, `$SCI` placeholders
- `details.csv` — CSV file with recipient details (name, email, math, eng, sci columns)

## Installation

```bash
cd "Send Email With Python"
```

No external dependencies required, but you must create `template.txt` and `details.csv`.

## Usage

```bash
python "Send Email With Python.py"
```

**Required files:**

1. **`template.txt`**: Email body template using `$PERSON_NAME`, `$MATH`, `$ENG`, `$SCI` as substitution variables.
2. **`details.csv`**: CSV file where each row contains: name, email, math grade, English grade, science grade.

## How It Works

1. `read_template()` reads `template.txt` and returns a `string.Template` object.
2. `main()` connects to Gmail's SMTP server on port 587, starts TLS, and logs in.
3. Reads `details.csv`, skipping the header row.
4. For each row, substitutes template variables with the student's name and grades.
5. Constructs a `MIMEMultipart` email with subject "Mid term grades" and sends it.
6. Closes the SMTP session.

## Configuration

- **SMTP server**: Hardcoded to `smtp.gmail.com` port `587`.
- **Subject line**: Hardcoded as `"Mid term grades"`.
- **Sender credentials**: Hardcoded in the script (masked with asterisks in the source code). Must be replaced with real Gmail credentials.
- **Template file**: `template.txt` (must exist in the same directory).
- **Data file**: `details.csv` (must exist in the same directory).

## Limitations

- **The code has broken indentation and will not execute.** Python's indentation-sensitive syntax means the current code raises `IndentationError`.
- The `message_template.substitute()` call references `row[0]`, `row[2]`, etc., but the loop variable is named `lines` — this is a `NameError` bug.
- Email/password are hardcoded (masked with asterisks) rather than read from environment variables or a config file.
- No error handling for SMTP failures, missing files, or invalid CSV data.
- Required files (`template.txt`, `details.csv`) are not included in the project.
- Gmail requires an "App Password" or OAuth2 for SMTP access when 2FA is enabled.

## Security Notes

- **Hardcoded credentials**: The script contains placeholders for a Gmail address and password. These must never be committed with real values.
- Gmail credentials should be stored in environment variables or a separate credentials file excluded from version control.
- Uses TLS (`starttls()`) for encrypted communication with the SMTP server.

## License

Not specified.
