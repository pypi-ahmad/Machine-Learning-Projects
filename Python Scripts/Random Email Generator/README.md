# Random Email Generator

> A Python script that generates a specified number of random email addresses with random usernames, domains, and extensions.

## Overview

This script prompts the user for a count, then generates that many random email addresses composed of random lowercase alphanumeric usernames (1–20 characters), randomly selected domains, and randomly selected extensions. Progress is displayed via a progress bar.

## Features

- Generates any number of random email addresses
- Random username length between 1 and 20 characters (lowercase letters and digits)
- Randomly selects from 8 email domains: gmail, yahoo, comcast, verizon, charter, hotmail, outlook, frontier
- Randomly selects from 4 extensions: com, net, org, gov
- Progress bar display during generation
- Input validation for the email count (must be an integer, with recursive retry)

## Project Structure

```
Random-Email-Generator/
├── random_email_generator.py    # Main script
└── README.md
```

## Requirements

- Python 3.x
- `progressbar`

## Installation

```bash
cd "Random-Email-Generator"
pip install progressbar
```

## Usage

```bash
python random_email_generator.py
```

```
How many email addresses?: 5
Creating email addresses...
Progress:
100% |########################################|
Creation completed.
x7k2m@outlook.gov
abc@gmail.com
...
```

## How It Works

1. **`getcount()`**: Prompts for the number of emails. If the input isn't a valid integer, recursively re-prompts.
2. **`makeEmail()`**: Builds a single email by:
   - Choosing a random extension from `['com', 'net', 'org', 'gov']`
   - Choosing a random domain from `['gmail', 'yahoo', 'comcast', 'verizon', 'charter', 'hotmail', 'outlook', 'frontier']`
   - Generating a random username of 1–20 characters from `string.ascii_lowercase + string.digits`
   - Concatenating as `username@domain.extension`
3. **Main loop**: Uses `progressbar.ProgressBar` to display progress while generating emails in a while loop.
4. **Output**: All generated emails are printed to the console after completion.

## Configuration

- `extensions` list — email extensions (hardcoded in `makeEmail()`)
- `domains` list — email domains (hardcoded in `makeEmail()`)
- Username character set: `string.ascii_lowercase + string.digits`
- Username length range: `random.randint(1, 20)`

## Limitations

- Emails are only printed to console; no file export option
- The progress bar and while loop have a redundant counter structure (both iterate to `howmany`)
- No option to specify custom domains, extensions, or username patterns
- No deduplication — duplicate emails are possible
- `getcount()` uses recursion for retry, which could hit the recursion limit on repeated invalid inputs
- `random.randint(0, len(extensions) - 1)` could be simplified with `random.choice()`

## Security Notes

No security concerns identified.

## License

Not specified.
