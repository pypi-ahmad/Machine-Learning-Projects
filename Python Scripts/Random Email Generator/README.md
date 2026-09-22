# Random Email Generator

Generate random email-like placeholders for examples or test data.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python random_email_generator.py 5
```

The generator uses the Python standard library and prints addresses to the terminal. It does not verify ownership, deliverability, or availability, and it makes no network requests. Do not use generated addresses for sending email without confirming the recipient.
