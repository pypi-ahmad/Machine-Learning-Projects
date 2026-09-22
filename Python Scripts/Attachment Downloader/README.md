# Attachment Downloader

A CLI tool that searches a Gmail inbox with a query string and downloads attachments from matching email threads through EZGmail.

## Overview

- Searches Gmail threads with attachments and downloads confirmed results to a selected local folder
- **Project type:** CLI / Utility

## Features

- Command-line or interactive Gmail search query
- Automatically adds `has:attachment` when the query does not already contain it
- Lists the subject lines of all matching email threads before downloading
- Requires a `y`/`yes` confirmation before downloading, unless `--yes` is supplied
- Handles both single-message and multi-message threads, downloading attachments from every message
- Downloads to `downloads/` by default, or a directory supplied with `--output-dir`
- Does not overwrite existing files unless `--overwrite` is supplied

## Dependencies

| Package | Source | Install |
|---------|--------|---------|
| `EZGmail` | PyPI | managed by `uv` |

### Authentication Prerequisites

EZGmail requires Google Gmail API OAuth credentials:

1. Obtain `credentials.json` from the [Google Cloud Console](https://console.cloud.google.com/) (Gmail API, OAuth 2.0, Desktop app type).
2. Place `credentials.json` in the project directory.
3. On first run, a browser window opens for authorization; a `token.json` file is generated for subsequent runs.

## How it works

1. The user supplies a search query (e.g., `from:boss subject:report`).
2. The script ensures the query includes `has:attachment` and calls `ezgmail.search()`.
3. If no results are found, a message is printed and the script exits.
4. If results are found, the subject line of each thread's first message is printed.
5. The user is prompted to confirm the download unless `--yes` is supplied.
6. If confirmed, every message in each matching thread calls `downloadAllAttachments()`.
7. Attachment files are saved to `downloads/` by default.

## Project Structure

```
Attachment_Downloader/
├── attachment.py   # Main script
└── README.md
```

## Setup & Installation

```powershell
cd "Python Scripts/Attachment Downloader"
uv sync
```

Place your `credentials.json` in the project directory before first run.

## How to Run

```powershell
uv run python attachment.py "from:boss subject:report"
uv run python attachment.py "from:boss" --output-dir C:\Downloads\Reports
```

Run `uv run python attachment.py` without a query for the interactive prompt.
Use `--yes` only after reviewing the query and `--overwrite` only when replacing
existing downloaded files is intended.

## Configuration

| Item | Description |
|------|-------------|
| `credentials.json` | Google OAuth 2.0 credentials file - must be in the project directory |
| `token.json` | Auto-generated after first successful OAuth authorization |
| `downloads/` | Default local attachment output folder |

## Testing

No formal test suite present.

## Limitations

- A broad query can still download many files; narrow the Gmail query before confirming.
- Authentication opens a Google OAuth browser flow on the first real use.
- This project does not make a live Gmail connection during local verification.

## Security Notes

- **`token.json`** grants access to your Gmail account. Keep it secure and do not commit it.
- **`credentials.json`** contains your OAuth client secret. Do not share or commit it.
- Project-local `.gitignore` excludes both OAuth files and the default download folder.
