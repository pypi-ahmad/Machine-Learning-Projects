# Attachment Downloader

A CLI tool that searches your Gmail inbox using a query string and downloads all attachments from matching email threads via the EZGmail library.

## Overview

- Prompts the user for a search query, finds matching Gmail threads with attachments, and downloads all attachment files to the current working directory
- **Project type:** CLI / Utility

## Features

- Interactive prompt for a Gmail search query at runtime
- Automatically appends `has:attachment` to the query to filter for threads with attachments
- Lists the subject lines of all matching email threads before downloading
- Asks for user confirmation (`Yes`/`No`) before proceeding with the download
- Handles both single-message and multi-message threads, downloading attachments from every message
- Downloads all attachments to the current working directory

## Dependencies

| Package | Source | Install |
|---------|--------|---------|
| `ezgmail` | PyPI (inferred from import) | `pip install EZGmail` |

### Authentication Prerequisites

EZGmail requires Google Gmail API OAuth credentials:

1. Obtain `credentials.json` from the [Google Cloud Console](https://console.cloud.google.com/) (Gmail API, OAuth 2.0, Desktop app type).
2. Place `credentials.json` in the working directory.
3. On first run, a browser window opens for authorization; a `token.json` file is generated for subsequent runs.

## How It Works

1. The user enters a search query (e.g., `from:boss subject:report`).
2. The script appends `+ has:attachment` and calls `ezgmail.search()`.
3. If no results are found, a message is printed and the script exits.
4. If results are found, the subject line of each thread's first message is printed.
5. The user is prompted to confirm the download with `Yes` or `No`.
6. If confirmed, `attachmentdownload()` iterates through each `GmailThread`:
   - If a thread has more than one message, it calls `downloadAllAttachments()` on each message individually.
   - If a thread has a single message, it calls `downloadAllAttachments()` on that message.
7. Attachment files are saved to the current working directory.

## Project Structure

```
Attachment_Downloader/
├── attachment.py   # Main script
└── README.md
```

## Setup & Installation

```bash
pip install EZGmail
```

Place your `credentials.json` in the project directory before first run.

## How to Run

```bash
cd Attachment_Downloader
python attachment.py
```

Follow the interactive prompts to enter a search query and confirm the download.

## Configuration

| Item | Description |
|------|-------------|
| `credentials.json` | Google OAuth 2.0 credentials file — must be in the working directory |
| `token.json` | Auto-generated after first successful OAuth authorization |

## Testing

No formal test suite present.

## Limitations

- **Bare `except` clauses** — errors during download are caught generically (`except:` with no specific exception type), making debugging difficult.
- Attachments are saved to the current working directory with no option to specify a custom target folder.
- No duplicate-file handling; re-running may overwrite previously downloaded files with the same name.
- No pagination or limit on search results — a broad query could trigger a very large download.
- The confirmation prompt only accepts the exact string `"Yes"` (case-sensitive); any other input (e.g., `yes`, `y`) exits the program.
- The `newquery` concatenation uses `" + has:attachment"` which includes a literal `+` sign — this works as a Gmail search operator but is unconventional.

## Security Notes

- **`token.json`** grants access to your Gmail account. Keep it secure and do not commit it to version control.
- **`credentials.json`** contains your OAuth client secret. Do not share or commit this file.
- Add both files to `.gitignore`.
