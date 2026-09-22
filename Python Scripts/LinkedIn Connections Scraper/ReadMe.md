# LinkedIn Connections Scrapper

## uv setup and safety

```powershell
uv sync --no-config
uv run --no-config python script.py --help
```

This legacy Selenium workflow logs into LinkedIn and collects connection data. Run it only for an account you are authorized to access, keep credentials out of command history where possible, and review LinkedIn's current rules before use. The browser login and scraping flow are not run during local verification.

> A Selenium-based CLI tool that scrapes LinkedIn connection details and exports them to a CSV file.

## Overview

This script automates Chrome to log into LinkedIn, open the connections page, load the list through scrolling, and extract each connection's name, headline, and profile link. An optional skills mode visits each profile to extract listed skills. Results are saved to a CSV file.

## Features

- Automated LinkedIn login via Selenium WebDriver
- Two operation modes:
  - **Basic mode:** Scrapes names, headlines, and profile links from the connections list
  - **Skills mode (`-s`):** Additionally visits each profile to extract their skill set
- Handles Ajax-loaded content by scrolling up and down to trigger lazy loading
- Regex-based extraction of names and headlines from connection cards
- Exports results to `scrap.csv` using Pandas
- CLI interface with `optparse` for email, password, and skills flag arguments

## Project Structure

```
Linkedin_Connections_Scrapper/
└── script.py
```

## Requirements

- Python 3.x
- `selenium`
- `pandas`
- ChromeDriver executable (`chromedriver.exe`) placed in the script's directory

## Installation

```bash
cd "Linkedin_Connections_Scrapper"
pip install selenium pandas
```

Download [ChromeDriver](https://chromedriver.chromium.org/downloads) matching your Chrome browser version and place `chromedriver.exe` in the same directory as `script.py`.

## Usage

**Basic mode** (scrapes names, headlines, and profile links):

```bash
python script.py -e <your_email> -p <your_password>
```

**Skills scraping mode** (also scrapes each connection's skills):

```bash
python script.py -e <your_email> -p <your_password> -s
```

### CLI Options

| Flag | Long Form | Description |
|------|-----------|-------------|
| `-e` | `--email` | LinkedIn login email |
| `-p` | `--password` | LinkedIn login password |
| `-s` | `--skills` | Enable skills scraping mode |
| `-h` | `--help` | Show help message |

## How it works

1. **`login(email, password)`**: Opens LinkedIn in Chrome, fills in credentials, clicks submit, and verifies login by checking the page title equals "LinkedIn".

2. **`scrap_basic(driver)`**: Navigates to the connections page, scrolls up and down repeatedly (with 3-second waits) to load all connections via Ajax. Extracts connection card details using class `mn-connection-card__details` and regex patterns for names (`\n(.+)\n`) and headlines (`occupation\n(.+)\n`). Extracts profile links by filtering `<a>` tags for URLs containing `linkedin.com/in`.

3. **`scrap_skills(driver, links)`**: Visits each profile link, scrolls to load all content, clicks the "skill_details" button, and extracts skill names from elements with class prefix `pv-skill-category-entity__name-text`. Skills are joined with ` -- ` as separator.

4. **`save_to_csv(names, headlines, links, skills)`**: Creates a Pandas DataFrame with columns Name, Headline, Link, and Skills, then saves to `scrap.csv`.

## Configuration

- **Scroll wait time:** `time_to_wait = 3` seconds in both `scrap_basic()` and `scrap_skills()` — adjust based on your internet speed.
- **ChromeDriver path:** Hardcoded as `chromedriver.exe` (must be in the same directory).
- **Output file:** `scrap.csv` in the current working directory.

## Limitations

- Uses deprecated Selenium methods (`find_element_by_name`, `find_element_by_class_name`, etc.) — requires Selenium 3.x or needs updating for Selenium 4.x.
- The scroll-based Ajax loading approach is fragile and dependent on internet speed.
- Skills scraping mode is very time-consuming due to visiting each profile individually.
- Bare `except` blocks in name/headline extraction silently skip errors.
- ChromeDriver path is hardcoded as a Windows executable (`chromedriver.exe`).
- LinkedIn may block automated access, require CAPTCHA, or rate-limit requests.
- Uses `pd.DataFrame.append()` which is deprecated in newer Pandas versions.

## Security Notes

- **Credentials passed via CLI:** Email and password are passed as command-line arguments, which may be visible in process listings and shell history. Consider using environment variables instead.
- **Scraping ToS:** Automated scraping of LinkedIn may violate their Terms of Service.

## License

Not specified.
