# USSD Banking Simulator

> A terminal-based simulation of Unstructured Supplementary Service Data (USSD) mobile banking operations.

## Overview

A command-line Python application that simulates a USSD-style mobile banking system. Users dial a USSD code (`*919#`) to log in and access banking services including account opening, upgrade/migration, balance checking, transfers, and fund transfers between banks.

## Features

- **USSD login:** Authenticate by entering the code `*919#` (3 attempts allowed)
- **Open Account:** Collects first name, last name, and sex; generates a basic BVN
- **Upgrade/Migrate:** Simulates account upgrade or migration options
- **Balance:** Requires a 4-digit PIN with digit and length validation
- **Transfer:** Self-transfer or transfer to others via mobile number
- **Funds:** Select from a list of 8 Nigerian banks and enter an account number
- **Navigation:** `#` returns to the options menu; `*` exits in some contexts

## Project Structure

```
Unstructured Supplemenrary  Service Data/
├── ussdtim.py
├── Screenshot_20200910-134857.png
└── README.md
```

## Requirements

- Python 3.x
- No external dependencies (uses only `time` and `sys` from the standard library)

## Installation

```bash
cd "Unstructured Supplemenrary  Service Data"
```

No package installation required.

## Usage

```bash
python ussdtim.py
```

1. Wait for the welcome message (8-second delay).
2. Enter the USSD code `*919#` to log in (3 attempts before lockout).
3. Select an option from the menu (1–5):
   - `1` — Open Account
   - `2` — Upgrade/Migrate
   - `3` — Balance
   - `4` — Transfer
   - `5` — Funds

## How It Works

1. **`log_in()`:** Prompts for USSD code in a loop (max 3 attempts). Correct code `*919#` proceeds to `options_menu()`.
2. **`options_menu()`:** Displays 5 options and dispatches to the corresponding function using a dictionary lookup.
3. **`open_acct()`:** Collects user details, generates a simple BVN (string `"01234"`), and displays the info.
4. **`balance()`:** Validates a 4-digit PIN using `len()` and `isdigit()` checks.
5. **`transf()`:** Offers self-transfer or transfer-to-others with mobile number input.
6. **`funds()`:** Displays a list of 8 banks, takes bank selection and account number.
7. **`exit()`:** Asks if the user wants another transaction; `N` exits, `#` returns to menu, anything else re-enters login.
8. **`BVN_checker()`:** Generates a trivial BVN by joining `["0", "1", "2", "3", "4"]`.

### Supported Banks

1. Access Bank
2. Fidelity Bank
3. Guarantee Trust Bank
4. Heritage Bank
5. Polaris Bank
6. Stanbic IBTC
7. Unity Bank
8. Wema Bank

## Configuration

No configuration files. All values are hardcoded in the script.

## Limitations

- The BVN generation is not real — it always produces `"01234"`.
- No actual banking operations are performed; all actions are simulated with print statements and `time.sleep()`.
- The built-in `exit()` function is overridden by a custom `exit()` function, which shadows Python's default.
- Inconsistent indentation (mix of tabs and spaces) may cause issues in some Python environments.
- Several `time.sleep()` calls add unnecessary delays (e.g., 8 seconds at startup, 5–15 seconds for operations).
- Error handling uses bare `except` clauses and `sys.exit()`.
- The `balance()` function calls itself recursively on invalid input instead of using a loop.
- No actual PIN storage or account data — all operations are stateless.

## License

Not specified.
