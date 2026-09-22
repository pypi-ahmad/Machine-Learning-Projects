# Email GUI

A Tkinter desktop composer that sends one plain-text email through Gmail SMTP after an explicit confirmation.

## Requirements

- Python 3.13+ with Tkinter available
- [uv](https://docs.astral.sh/uv/)
- A Gmail account configured for SMTP authentication

## Run

From this directory:

```powershell
uv sync
uv run python script.py
```

Enter a sender Gmail address, Gmail app password, recipient, subject, and message. Select **Send message**, then confirm the recipient before the app contacts Gmail.

## Security notes

- The password field is masked and the app does not write credentials to disk or log them.
- For accounts with 2-Step Verification, Google documents using an app password for apps that cannot complete the normal sign-in flow. [Google Account Help](https://support.google.com/accounts/answer/2461835)
- App passwords can be revoked by Google, including after a Google Account password change. [Google Account Help](https://support.google.com/accounts/answer/185833)
- Sending email is an external action. Verify the recipient and message before confirming.

## Behavior

- Connects to `smtp.gmail.com:587` with STARTTLS.
- Sends a structured plain-text message with From, To, and Subject headers.
- Clears the app password, subject, and message after a successful send; the sender and recipient stay available for the next message.
- Reports SMTP and connection errors in the interface.

## Verification

```powershell
uv run python -m py_compile script.py
uv lock --check
```
