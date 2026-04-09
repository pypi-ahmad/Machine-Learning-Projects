# Google Classroom Bot

> Automatically join Google Meet classes on schedule using Selenium browser automation.

## Overview

A Selenium-based bot that reads a weekly class schedule from a CSV file, monitors the current time, and automatically logs into Google Classroom at the appropriate times to join Google Meet sessions. It uses Firefox with a pre-configured profile to handle camera/microphone permissions.

## Features

- Automated Google account login via Selenium (supports multiple login page layouts)
- Reads class schedule from `schedule.csv` with configurable time slots
- Polls current time every 30 seconds to detect class start times
- Navigates to Google Classroom, finds the course by name, and extracts the Meet link
- Automatically clicks "Join now" in Google Meet
- Stays in the meeting for 1 hour, then quits and waits for the next class
- Tracks position in the daily schedule (3 time slots) across sessions

## Project Structure

```
Google-Classroom-Bot/
├── gmeet_bot.py      # Main bot script with ClassAutomation class
├── config.ini        # Google account credentials (username/password)
├── schedule.csv      # Weekly class schedule (Day × Time Slot → Course Name)
├── requirements.txt  # Python dependencies
└── README.md
```

## Requirements

- Python 3.x
- `selenium==3.141.0`
- `urllib3==1.26.5`
- Firefox browser
- `geckodriver` (download from [Mozilla geckodriver releases](https://github.com/mozilla/geckodriver/releases))
- A pre-configured Firefox profile with camera and microphone blocked for Google Meet

## Installation

```bash
cd "Google-Classroom-Bot"
pip install -r requirements.txt
```

Download `geckodriver` and place it in the project folder or add it to your system PATH.

## Usage

1. Configure credentials in `config.ini`:
   ```ini
   [AUTH]
   USERNAME=your.email@gmail.com
   PASSWORD=your_password
   ```

2. Fill in `schedule.csv` with your course names:
   ```csv
   Day,09:20,11:40,14:25
   Mon,CourseName1,,
   Tue,,CourseName2,
   Wed,,,CourseName3
   Thu,CourseName1,,CourseName3
   Fri,,,
   ```

3. Update the Firefox profile path in `gmeet_bot.py`:
   ```python
   profile = webdriver.FirefoxProfile('/path/to/the/created/profile')
   ```

4. Run the bot:
   ```bash
   python gmeet_bot.py
   ```

## How It Works

1. **`__init__`** — Initializes the class counter, calls `findCount()` to determine the next class, then enters an infinite loop checking time every 30 seconds.
2. **`initClass()`** — Looks up the current course from the schedule CSV, logs in, navigates to Classroom, clicks the course, extracts the Meet link, joins the meeting, waits 1 hour, then quits the browser.
3. **`findClass()`** — Reads `schedule.csv` using `csv.DictReader`, matches the current day (`%a` format: Mon, Tue, etc.), and returns the course name for the current time slot.
4. **`findCount()`** — Determines which of the 3 time slots (09:20, 11:40, 14:25) is next based on current time.
5. **`login()`** — Opens Firefox with the configured profile, navigates to Google Accounts, enters username and password with fallback selectors for different Google login page versions.

## Configuration

| File | Setting | Description |
|------|---------|-------------|
| `config.ini` | `USERNAME` | Google account email |
| `config.ini` | `PASSWORD` | Google account password |
| `schedule.csv` | Day × Time | Course names mapped to days and time slots |
| `gmeet_bot.py` | `classTime` | List of class start times: `["09:20", "11:40", "14:25"]` |
| `gmeet_bot.py` | `FirefoxProfile` path | Path to pre-configured Firefox profile |

## Limitations

- **Hardcoded time slots:** Only supports 3 class times per day (`09:20`, `11:40`, `14:25`).
- **Fixed meeting duration:** Always stays for exactly 1 hour (`sleep(60 * 60)`).
- **Selenium 3 with deprecated methods:** Uses `find_element_by_xpath`, `find_element_by_name`, etc. which are removed in Selenium 4.
- **Bare `except` blocks** for login flow — errors are silently caught.
- **Firefox-only:** Hardcoded to use Firefox with `geckodriver`.
- **No headless mode** — requires a visible browser window.
- **Meet link extraction** depends on finding `https://meet.google.com/lookup/` as link text, which may vary.
- **`sleep()`-based timing** is fragile; page load times may vary.

## Security Notes

- **Plaintext credentials:** Google username and password are stored in plain text in `config.ini`. Use environment variables or a secrets manager instead.
- **Automating Google login** may trigger account security alerts or CAPTCHAs. Google may block or lock the account.

## License

Not specified.
