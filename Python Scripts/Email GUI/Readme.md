# Email GUI

> A Tkinter-based desktop application for sending emails via Gmail's SMTP server.

## Overview

This application provides a graphical user interface built with Python's Tkinter library that allows users to send emails through Gmail's SMTP server. Users enter the sender's email, password, recipient's email, and message body, then click a button to send the email.

## Features

- Tkinter GUI with a 500×500 window and light blue background
- Input fields for sender's email, sender's password, recipient's email, and message body
- Sends email via Gmail's SMTP server (`smtp.gmail.com`, port 587) with TLS encryption
- Input fields are cleared automatically after a message is sent
- Console output confirms successful login and message delivery

## Project Structure

```
Email GUI/
├── script.py    # Main application with GUI and email sending logic
└── Readme.md
```

## Requirements

- Python 3.x
- `tkinter` (included with standard Python installations)
- `smtplib` (included with standard Python installations)
- A Gmail account with SMTP access enabled

> **Note:** No `requirements.txt` is provided. All dependencies are part of the Python standard library.

## Installation

```bash
cd "Email GUI"
```

No additional package installation is required.

## Usage

```bash
python script.py
```

1. Enter the sender's Gmail address in the "Sender's Email" field
2. Enter the sender's Gmail password (or App Password) in the "Sender's Password" field
3. Enter the recipient's email address in the "Recipient Email" field
4. Enter the message text in the "Message" field
5. Click **"Send Message"** to send the email
6. The console will display "Login successful" followed by "Message sent"

## How It Works

1. **GUI Setup**: Creates a `Tk` window (500×500 pixels) with a yellow heading bar, light blue background, and four labeled input fields using `Label`, `Entry`, and `StringVar` widgets. A "Send Message" button is placed at the bottom.
2. **`send_message()`**: Retrieves the text from all four input fields. Creates an SMTP connection to `smtp.gmail.com` on port 587, initiates TLS encryption with `server.starttls()`, logs in with the provided credentials via `server.login()`, and sends the email using `server.sendmail()`. After sending, all input fields are cleared.
3. **`mainloop()`**: Starts the Tkinter event loop to keep the window responsive.

## Configuration

- **SMTP server**: Hardcoded to `smtp.gmail.com` on port `587` — modify in `send_message()` to use a different email provider
- **Window size**: Set to `500x500` via `gui.geometry("500x500")`
- **Background color**: Set to `"light blue"` via `gui.configure()`

## Limitations

- Password is entered as plain text (not masked with `show="*"` on the Entry widget)
- No error handling for failed login, invalid credentials, or network issues
- Only supports Gmail's SMTP server out of the box
- Message body is limited to a single-line `Entry` widget (not a multi-line `Text` widget)
- Gmail requires an "App Password" if 2-Factor Authentication is enabled; direct password login may be blocked by Google's security policies
- No email subject field — the `sendmail()` call sends the body as the raw message without headers
- No input validation (empty fields are not checked)
- The SMTP connection is not explicitly closed after sending

## Security Notes

- **Passwords are entered in plain text** in the GUI (visible on screen)
- Credentials are transmitted over TLS-encrypted SMTP, but are held in memory as plain strings
- Gmail may block sign-in attempts from "less secure apps" unless App Passwords are configured
- No credential storage — the password must be entered each time

## License

Not specified.
