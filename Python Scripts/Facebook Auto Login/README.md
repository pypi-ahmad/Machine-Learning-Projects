# Facebook Autologin

> A Selenium-based Python script that automates the Facebook login process using Chrome WebDriver.

## Overview

This script uses Selenium WebDriver to open Facebook in a Chrome browser, fill in login credentials, and click the login button automatically.

## Features

- Opens Facebook in a Chrome browser via Selenium
- Automatically fills in email/username and password fields
- Clicks the login button programmatically

## Project Structure

```
Facebook-Autologin/
├── facebookAuto.py    # Main script for automated Facebook login
├── chromedriver.exe   # Chrome WebDriver executable (bundled)
└── requirements.txt   # Python dependencies
```

## Requirements

- Python 3.x
- `selenium==3.141.0`
- Chrome browser installed
- Compatible `chromedriver.exe` (bundled in the project)

## Installation

```bash
cd "Facebook-Autologin"
pip install -r requirements.txt
```

## Usage

1. Edit `facebookAuto.py` and replace the placeholder credentials:
   ```python
   driver.find_element_by_id("email").send_keys("username@email.com")  # your email
   driver.find_element_by_id("pass").send_keys("password")             # your password
   ```

2. Run the script:
   ```bash
   python facebookAuto.py
   ```

## How It Works

1. Initializes a Chrome WebDriver using the local `chromedriver.exe`.
2. Navigates to `https://www.facebook.com`.
3. Locates the email field by ID (`email`) and sends the configured email/username.
4. Locates the password field by ID (`pass`) and sends the configured password.
5. Locates the login button by name (`login`) and clicks it.

## Configuration

Edit the following hardcoded values in `facebookAuto.py`:

| Line | Value                  | Description              |
|------|------------------------|--------------------------|
| 6    | `"username@email.com"` | Your Facebook email      |
| 8    | `"password"`           | Your Facebook password   |

## Limitations

- Uses deprecated Selenium methods (`find_element_by_id`, `find_element_by_name`) — these were removed in Selenium 4.x
- The bundled `chromedriver.exe` may not match your installed Chrome version
- No error handling for failed logins, missing elements, or network issues
- Hardcoded credentials in source code
- No support for two-factor authentication
- WebDriver path is hardcoded as `./chromedriver.exe`

## Security Notes

- **Credentials are hardcoded in plaintext** in the source file — never commit real credentials to version control
- Uses an unencrypted local ChromeDriver binary

## License

Not specified.
