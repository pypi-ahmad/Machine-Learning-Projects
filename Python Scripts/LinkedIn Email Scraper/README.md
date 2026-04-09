# LinkedIn Email Scraper

> A Selenium-based scraper that extracts email addresses from comments on a LinkedIn post and saves them to a CSV file.

## Overview

This script automates a Chrome browser to log into LinkedIn, navigate to a specified post, expand the comments section, and extract email addresses found in comment links. Extracted emails are validated using the `email_validator` library and saved to a CSV file with the commenter's name.

## Features

- Automated LinkedIn login via Selenium
- Navigates to a specified LinkedIn post URL
- Expands the comments section by clicking "load more" (up to 3 iterations)
- Extracts email addresses from `<a>` tags within comments
- Validates extracted emails using the `email_validator` library
- Maps commenter names to their email addresses
- Exports results to a CSV file (`emails.csv`)

## Project Structure

```
Linkedin-Email-Scraper/
├── emails.csv
├── main.py
└── requirements.txt
```

## Requirements

- Python 3.x
- `selenium`
- `email-validator`
- ChromeDriver installed and available in PATH

## Installation

```bash
cd "Linkedin-Email-Scraper"
pip install -r requirements.txt
```

You also need to download [ChromeDriver](https://chromedriver.chromium.org/downloads) matching your Chrome browser version and ensure it is in your system's PATH.

## Usage

1. Open `main.py` and replace the placeholder values:
   - Set `url` to the LinkedIn post URL you want to scrape (line with `'[INSERT URL TO LINKEDIN POST]'`).
   - Set `userEmail` to your LinkedIn email (line with `'[INSERT YOUR EMAIL ADDRESS FOR LINKEDIN ACCOUNT]'`).
   - Set `userPassword` to your LinkedIn password (line with `'[INSERT YOUR PASSWORD FOR LINKEDIN ACCOUNT'`).

2. Run the script:
   ```bash
   python main.py
   ```

3. The scraped names and emails will be written to `emails.csv`.

## How It Works

1. **`LinkedInEmailScraper(userEmail, userPassword)`**:
   - Opens Chrome via Selenium and navigates to the LinkedIn post URL.
   - Finds the comment button, follows the login link, and submits credentials.
   - Clicks "show more comments" up to 3 times to expand the comment section.
   - Iterates through all `<article>` elements in the comments section.
   - Extracts the commenter's name (`.hoverable-link-text` class) and email (from `<a>` tag `innerHTML` inside `<p>` tags).
   - Validates each email with `email_validator.validate_email()`.
   - Returns a dictionary of `{name: email}`.

2. **`DictToCSV(input_dict)`**:
   - Writes the name-email dictionary to `emails.csv` with columns `name` and `email`.

## Configuration

- **Post URL:** Hardcoded placeholder `'[INSERT URL TO LINKEDIN POST]'` in the `LinkedInEmailScraper` function.
- **Credentials:** Hardcoded placeholders in the `__main__` block.
- **Comment expansion iterations:** The `range(3)` loop controls how many times "load more comments" is clicked.
- **CSV output path:** Hardcoded as `'./LinkedIn Email Scraper/emails.csv'` in `DictToCSV()`.

## Limitations

- Uses deprecated Selenium methods (`find_element_by_xpath`, `find_element_by_css_selector`, etc.) — these were removed in Selenium 4.x; the script requires Selenium 3.x or updating to the new `find_element(By.*)` syntax.
- The comment expansion loop only clicks "load more" 3 times, so posts with many comments won't be fully scraped.
- Only extracts emails found inside `<a>` tags within comments — plain-text emails are not captured.
- Bare `except` blocks silently swallow all errors during scraping.
- The CSV output path `'./LinkedIn Email Scraper/emails.csv'` assumes a specific working directory structure.
- LinkedIn may block automated access or require CAPTCHA verification.

## Security Notes

- **Plaintext credentials:** LinkedIn email and password are stored as plaintext strings in the source code. Use environment variables or a secrets manager instead.
- **Scraping ToS:** Automated scraping of LinkedIn may violate their Terms of Service.

## License

Not specified.
