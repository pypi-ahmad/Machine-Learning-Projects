# Currency Exchange Rates

## Overview

A CLI-based currency exchange rate viewer that scrapes live exchange rates from [x-rates.com](https://www.x-rates.com/) using BeautifulSoup. The user selects a source currency and enters an amount, then the script displays conversion rates to all available currencies.

**Type:** CLI / Web Scraper

## Features

- Scrapes available currencies from the x-rates.com homepage
- Displays a numbered list of currencies for user selection
- Accepts a user-specified amount for conversion
- Fetches and displays a full exchange rate table for the chosen currency and amount
- Outputs conversion amounts formatted to 3 decimal places

## Dependencies

- `beautifulsoup4` — HTML parsing
- `requests` — HTTP requests

### Python Standard Library

- None beyond builtins

## How It Works

1. Fetches the x-rates.com homepage and parses all `<option>` elements to build a list of available currencies (excluding the last 11 options).
2. Displays the currency list to the user with numbered indices.
3. The user selects a currency by position number and enters an amount.
4. Constructs a URL to the x-rates.com table page with the selected currency and amount as query parameters.
5. Parses the resulting HTML table (class `tablesorter`) and extracts each row's currency name and converted amount.
6. Prints all exchange rates to the console.

## Project Structure

```
Currency-Exchange-Rates/
└── exchange_rates.py   # Main script
```

## Setup & Installation

1. Ensure Python 3.x is installed.
2. Install dependencies:
   ```bash
   pip install beautifulsoup4 requests
   ```

## How to Run

```bash
python exchange_rates.py
```

Follow the interactive prompts:
1. Select a currency by entering its position number.
2. Enter the amount to convert (use a dot for decimals, not a comma).

## Configuration

No configuration required.

## Testing

No formal test suite present.

## Limitations

- Requires an active internet connection to scrape x-rates.com.
- Depends on the HTML structure of x-rates.com; changes to the site may break the scraper.
- The script uses `\033c` ANSI escape codes to clear the terminal, which may not work on all terminals (e.g., Windows CMD without ANSI support).
- No error handling for invalid user input or network failures.
- Decimal amounts must use a dot (`.`) separator, not a comma.
