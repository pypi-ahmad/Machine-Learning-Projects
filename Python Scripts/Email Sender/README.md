# Email Sender

A preview-first CLI that renders personalized plain-text Gmail SMTP messages from a UTF-8 template and recipient CSV. It is intended for careful, authorized batch delivery such as grade notices.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Gmail SMTP credentials available only in the process environment

## Input files

`template.txt` uses these placeholders:

```text
Hello $PERSON_NAME,

Math: $MATH
English: $ENG
Science: $SCI
```

`details.csv` must be UTF-8 with a header containing:

```csv
name,email,math,eng,sci
```

## Configure credentials

Set credentials only for the active PowerShell session. Do not place them in source files, templates, CSV files, or Git.

```powershell
$env:GMAIL_SMTP_ADDRESS = "sender@gmail.com"
$env:GMAIL_APP_PASSWORD = "app-password"
```

Google documents app-password use for compatible account setups and notes that app passwords can be revoked. [Google Account Help](https://support.google.com/accounts/answer/2461835), [App password help](https://support.google.com/accounts/answer/185833)

## Preview and send

```powershell
uv sync
uv run python "Send Email With Python.py"
```

The default command lists recipients and subjects without connecting to Gmail. To request delivery:

```powershell
uv run python "Send Email With Python.py" --send
```

You must then type `SEND` exactly. The script sends through `smtp.gmail.com:587` using STARTTLS.

## Safety and privacy

- Grade data and recipient addresses are sensitive. Verify authorization, data accuracy, and recipient list before delivery.
- The script neither writes credentials nor logs message bodies.
- SMTP acceptance is not a guarantee of final delivery to the recipient inbox.

## Verification

```powershell
uv run python -m py_compile "Send Email With Python.py"
uv lock --check
```
