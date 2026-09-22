# Email Validator

This command-line tool checks email-address syntax and can optionally ask DNS whether the address's domain resolves. It never connects to an SMTP server and cannot prove that a mailbox exists or will accept mail.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)

## Setup

```powershell
uv sync --no-config
```

There are no third-party dependencies.

## Validate one address

```powershell
uv run --no-config python main.py user@example.com
```

The default check validates the format and performs a DNS-resolution check for the domain. DNS requires network access and may fail because of local network policy or temporary DNS conditions.

## Validate a file

Provide a UTF-8 text file containing one address per line:

```powershell
uv run --no-config python main.py --file emails.txt
```

## Offline syntax-only check

Skip DNS when you only need local validation:

```powershell
uv run --no-config python main.py user@example.com --no-dns
```

## What the result means

- `VALID` means the syntax passed and, unless `--no-dns` was used, the domain resolved in DNS.
- A disposable-domain or role-address message is a warning, not a delivery failure.
- The tool does not query MX records, connect to SMTP, or test an individual mailbox. A valid result is not a guarantee that an email can be delivered.
