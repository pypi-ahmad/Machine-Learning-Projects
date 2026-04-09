# Get WiFi Password

> Retrieve saved WiFi network names and their stored passwords on Windows.

## Overview

A Python script that uses the Windows `netsh` command-line utility to enumerate all saved WiFi profiles and extract their plaintext passwords (key content). Results are printed in a formatted two-column table.

## Features

- Lists all saved WiFi profiles on the system
- Extracts the stored password (Key Content) for each profile
- Displays results in a formatted table: `Profile Name | Password`
- Handles profiles with no stored password gracefully (shows empty string)

## Project Structure

```
Get_wifi_password/
├── wifi.py
└── README.md
```

## Requirements

- Python 3.x
- Windows OS (uses `netsh wlan` commands)
- No external dependencies (uses only `subprocess` from the standard library)

## Installation

```bash
cd "Get_wifi_password"
```

No package installation needed.

## Usage

```bash
python wifi.py
```

Example output:
```
HomeNetwork                   |  MyPassword123
OfficeWiFi                    |  SecurePass!
CoffeeShop                    |
```

## How It Works

1. Runs `netsh wlan show profiles` to get all saved WiFi profile names.
2. Parses the output, splitting on `"All User Profile"` lines to extract profile names.
3. For each profile, runs `netsh wlan show profile <name> key=clear` to reveal the stored key.
4. Parses the `"Key Content"` line to extract the password.
5. Prints each profile name and password in a left-aligned formatted table.

## Configuration

No configuration needed.

## Limitations

- **Windows-only:** Relies on `netsh wlan` which is a Windows-specific command.
- **Requires elevated privileges:** Some systems may require running as Administrator to view passwords.
- **Encoding assumptions:** Uses UTF-8 decoding; may fail on systems with different console encodings.
- **Fragile parsing:** Splits on `":"` which could break if profile names contain colons.
- The trailing `[1:-1]` slice removes a space and trailing `\r` — brittle string manipulation.

## Security Notes

- This script displays plaintext WiFi passwords stored on the system. Use responsibly.
- Ensure the terminal output is not logged or shared inadvertently.

## License

Not specified.
