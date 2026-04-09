# Currency Converter

## Overview

A CLI-based currency converter that fetches live exchange rates from the [Fixer.io](https://fixer.io/) API and performs real-time currency conversions. Supports 168 world currencies including cryptocurrencies (BTC) and precious metals (XAU, XAG).

**Type:** CLI Application

## Features

- Fetches live exchange rates from the Fixer.io API
- Supports conversion between 168 currencies
- Interactive command-line interface with `SHOW` command to list all available currencies
- Displays currency codes with full names and associated countries
- Quit option (`Q`) to exit the program
- Basic error handling for incorrect currency codes (`KeyError`)

## Dependencies

- `requests` — HTTP requests to the Fixer.io API
- `json` (Python standard library)
- `sys` (Python standard library)
- `pprint` (Python standard library)

## How It Works

1. On startup, the script fetches the latest exchange rates from `data.fixer.io/api/latest` using a hardcoded API access key.
2. The JSON response is parsed and the `rates` dictionary is extracted, containing currency codes mapped to their EUR-based exchange rates.
3. A comprehensive list of 168 currency strings (code, name, and associated countries) is defined in the script.
4. The user enters a conversion query in the format: `<amount> <FROM_CURRENCY> <TO_CURRENCY>` (e.g., `100 USD EUR`).
5. The conversion is calculated as: `amount × rates[TO] / rates[FROM]`, since all rates are relative to EUR.
6. Typing `SHOW` displays the full currency list; typing `Q` exits the program.

## Project Structure

```
Currency_converter/
├── cc.py          # Main application script
├── output.png     # Screenshot of the application
└── README.md
```

## Setup & Installation

1. Ensure Python 3.x is installed.
2. Install dependencies:
   ```bash
   pip install requests
   ```

## How to Run

```bash
python cc.py
```

Follow the prompt to enter: `<amount> <FROM_CODE> <TO_CODE>` (e.g., `100 USD GBP`).

## Configuration

- **API Key:** The Fixer.io API access key is hardcoded in the script on the `url` variable line. Replace it with your own key from [fixer.io](https://fixer.io/) if needed.

## Testing

No formal test suite present.

## Limitations

- The Fixer.io API key is hardcoded in the source code.
- The free tier of Fixer.io only supports EUR as a base currency; cross-currency conversion is derived mathematically.
- The function is not truly recursive-safe — `function1()` calls itself on `SHOW` and on `KeyError`, which could theoretically cause a stack overflow after many consecutive errors.
- Only single conversion per run (the function does not loop for multiple queries).
- No input validation beyond `KeyError` handling; malformed input (wrong number of space-separated tokens) will raise an unhandled exception.

## Security Notes

- The Fixer.io API access key (`33ec7c73f8a4eb6b9b5b5f95118b2275`) is exposed in the source code. This should be moved to an environment variable or configuration file for production use.


