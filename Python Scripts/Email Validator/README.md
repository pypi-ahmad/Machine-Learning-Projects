# Email Validator

> A Python script that validates email addresses through three checks: syntax verification, DNS MX record lookup, and SMTP server response.

## Overview

This script performs a three-stage validation of an email address: first checking the syntax with a regex pattern, then verifying the domain's MX DNS record exists, and finally connecting to the mail server via SMTP to check if the specific mailbox exists (via RCPT TO response code).

## Features

- **Check 1 — Syntax**: Validates email format using a regex pattern
- **Check 2 — DNS**: Resolves the domain's MX record to verify the mail server exists
- **Check 3 — SMTP**: Connects to the mail server and performs HELO, MAIL FROM, and RCPT TO commands to verify the mailbox exists (250 = valid)
- Step-by-step console output showing pass/fail for each check
- Graceful handling when SMTP access is restricted

## Project Structure

```
Email-Validator/
├── email_verification.py
└── README.md
```

## Requirements

- Python 3.x
- `dnspython` (imported as `dns.resolver`)
- `smtplib` (standard library)
- `socket` (standard library)
- `re` (standard library)

## Installation

```bash
cd Email-Validator
pip install dnspython
```

## Usage

```bash
python email_verification.py
```

When prompted, enter an email address:

```
Enter your Email id : user@example.com
```

Sample output for a valid email:

```
Check 1 (Syntax) Passed
Check 2 (DNS - mail.example.com.) Passed
Check 3 (SMTP response) Passed
user@example.com is a VALID email address!
```

## How It Works

1. **`check_syntax(email)`** — Matches the email against the regex `^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$`. Exits if the pattern doesn't match.
2. **`check_dns(email, domain)`** — Uses `dns.resolver.resolve(domain, 'MX')` to fetch MX records. Returns the first MX record's exchange hostname. Exits if resolution fails.
3. **`check_response(email, domain, mxRecord)`** — Opens an SMTP connection to the MX server, sends HELO, MAIL FROM, and RCPT TO commands. If the server responds with status code 250, the mailbox exists. Catches `socket.error` for servers that block external SMTP connections.

## Configuration

No configuration files. The email address is provided interactively at runtime.

## Limitations

- The regex pattern only supports lowercase letters and 2-3 character TLDs — rejects valid emails with uppercase letters, plus signs, hyphens in the local part, or longer TLDs (e.g., `.info`, `.museum`).
- Uses a bare `except:` in `check_dns()` — catches all exceptions including `KeyboardInterrupt`.
- Calls `exit()` on check failure instead of raising exceptions or returning a result.
- Many mail servers block or rate-limit external SMTP RCPT TO verification, making Check 3 unreliable.
- The SMTP MAIL FROM uses the email being checked as the sender, which may be rejected.
- No timeout configured for DNS or SMTP operations.
- Only checks the first MX record, ignoring backup mail servers.

## License

Not specified.
