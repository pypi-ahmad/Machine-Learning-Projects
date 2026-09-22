# LinkedIn Email Scraper

This local Selenium utility exports email addresses that people visibly share in comments on one LinkedIn post. Use it only for posts and data you are authorized to access, and follow LinkedIn's terms and applicable privacy law.

## Setup

Requirements: Python 3.13+ and a locally installed Chrome browser.

```powershell
cd "LinkedIn Email Scraper"
uv sync
```

Selenium Manager obtains a compatible ChromeDriver automatically. Do not download or configure ChromeDriver manually.

## Usage

```powershell
uv run python main.py "https://www.linkedin.com/posts/example" --output emails.csv
```

The script prompts for the LinkedIn account email and password. They are used only in memory for that browser session and are never written to source code or the CSV file.

Options:

- `--output PATH`: CSV destination. Default: `emails.csv` in the current directory.
- `--max-load-more N`: Number of earlier-comment batches to request. Default: `3`; use `0` to inspect only comments initially visible.

## What it does

1. Opens LinkedIn's login page with Selenium.
2. Opens the supplied post after sign-in.
3. Expands up to the requested number of earlier-comment batches.
4. Finds valid email-shaped text visibly present in comments and writes name/email pairs to CSV.

The scraper does not bypass logins, CAPTCHAs, rate limits, or access controls. If LinkedIn changes its page structure, requires extra verification, or does not expose comments to the account, the run can fail or export no records.

## Project files

```text
LinkedIn Email Scraper/
├── main.py
├── pyproject.toml
└── uv.lock
```

## Privacy

The CSV can contain personal data. Store it securely, use it only for the purpose you are authorized for, and delete it when no longer needed.
