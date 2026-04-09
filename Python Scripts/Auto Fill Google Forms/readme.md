# Auto-Fill Google Forms

A Selenium-based automation script that reads contact data from a CSV file and automatically fills out and submits a Google Form for each record using Firefox WebDriver.

## Overview

- Reads name, email, and phone number records from `input.csv` and uses Selenium to fill and submit a Google Form for each entry automatically
- **Project type:** Bot / Web Automation (Selenium)

## Features

- Reads records from `input.csv` using Python's built-in `csv.DictReader`
- Opens a Google Form in Firefox via Selenium WebDriver (geckodriver)
- Iterates through each CSV row, filling in the Name, Email ID, and Phone Number fields via XPath selectors
- Automatically clicks the Submit button for each entry
- Navigates back to the form after each submission to process the next record
- 3-second delays between actions to allow page loads

## Dependencies

| Package | Source | Install |
|---------|--------|---------|
| `selenium` | PyPI (inferred from import) | `pip install selenium` |
| `csv` | Python standard library | — |
| `time` | Python standard library | — |

Additionally, **geckodriver** (Firefox WebDriver) must be installed separately.

## How It Works

1. The script initializes a Firefox WebDriver instance with a hardcoded path to `geckodriver.exe`.
2. It navigates to a specific Google Form URL.
3. `input.csv` is read using `csv.DictReader`; values from the `name`, `email`, and `phone_number` columns are loaded into separate Python lists.
4. A `while` loop iterates through each index:
   - Fills the Name field using `find_element_by_xpath()` with a hardcoded absolute XPath.
   - Fills the Email ID field using another hardcoded XPath.
   - Fills the Phone Number field using another hardcoded XPath.
   - Waits 3 seconds (`time.sleep(3)`), then clicks the Submit button (also via hardcoded XPath).
   - Waits 3 seconds, calls `browser.back()` to return to the form, waits another 3 seconds.
5. After all records are submitted, the browser is closed with `browser.quit()`.

## Project Structure

```
Auto-Fill-Google-Forms/
├── test.py       # Main automation script
├── input.csv     # Sample CSV data (name, email, phone_number)
└── readme.md
```

### Expected `input.csv` Format

```csv
name,email,phone_number
Constance,Constance@gmail.com,555-1234
Homer,Homer@gmail.com,555-1235
```

## Setup & Installation

```bash
pip install selenium
```

Download **geckodriver** from [https://github.com/mozilla/geckodriver/releases](https://github.com/mozilla/geckodriver/releases) and place it at a known path on your system.

## How to Run

1. Edit `test.py` to set the correct `executable_path` for geckodriver on your system (line 18).
2. Update the Google Form URL (line 20) if targeting a different form.
3. Update the XPath selectors (lines 6–8, 11) if the form's field structure differs.
4. Populate `input.csv` with your data.
5. Run:

```bash
cd Auto-Fill-Google-Forms
python test.py
```

## Configuration

| Item | Location | Description |
|------|----------|-------------|
| geckodriver path | `test.py` line 18 | Hardcoded to `C:\geckodriver-v0.28.0-win64\geckodriver.exe` — must be updated for your system |
| Google Form URL | `test.py` line 20 | Hardcoded to a specific Google Form — change to target a different form |
| Name field XPath | `test.py` line 6 | Absolute XPath for the Name input field |
| Email field XPath | `test.py` line 7 | Absolute XPath for the Email input field |
| Phone field XPath | `test.py` line 8 | Absolute XPath for the Phone Number input field |
| Submit button XPath | `test.py` line 11 | Absolute XPath for the Submit button |

## Testing

No formal test suite present.

## Limitations

- **Hardcoded geckodriver path** — `C:\geckodriver-v0.28.0-win64\geckodriver.exe` will not work on other machines without editing.
- **Hardcoded Google Form URL** — the URL is specific to one form and will not work for other forms.
- **Hardcoded absolute XPaths** — extremely brittle; any change to the form's DOM structure will break the script.
- Uses the deprecated `find_element_by_xpath()` API (removed in Selenium 4+); should use `find_element(By.XPATH, ...)`.
- Fixed 3-second `time.sleep()` delays instead of explicit WebDriver waits — unreliable on slow connections.
- No error handling — if a field is not found or the form structure changes, the script crashes.
- The browser variable is named `browswer` (typo for `browser`) throughout the script.
- All CSV data is loaded into memory lists before processing rather than streaming row by row.

## Security Notes

- The CSV file (`input.csv`) may contain personal data (names, emails, phone numbers). Handle it accordingly and do not commit sensitive data to public repositories.
- Selenium automation of Google Forms may violate Google's Terms of Service. Use responsibly and only with forms you own or have permission to automate.
