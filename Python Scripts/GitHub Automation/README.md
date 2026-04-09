# Github-Automation

> Automate the Git workflow by monitoring file changes, calculating diffs, and performing git add/commit/push automatically.

## Overview

A multi-module Python tool that watches a project directory for file changes in real time. When changes are detected, it calculates a diff, displays the changed lines, prompts for a commit message, and executes git add, commit, and push operations. It includes an installer script that copies the tool into any target project directory.

## Features

- Auto-detects git repository info (remote URL, branch) from the local `.git/config`
- Monitors all nested files for changes in real-time (polling loop)
- Calculates and displays line-by-line diffs for changed files
- Automatically runs `git add`, `git commit` (with user-provided message), and `git push`
- Supports rejecting a commit with `-r` flag during the commit prompt
- Respects `.gitignore` — reads and excludes ignored directories
- Maintains a JSON-based change log (`tmp.json`) to track uncommitted changes across sessions
- Installer script (`install.py`) copies all tool files into any target project's `auto-scripts/` subdirectory
- Colorized terminal output with ASCII art banner
- Initializes new repos (git init, README, first commit, set remote/branch) if no existing repo is found

## Project Structure

```
Github-Automation/
├── main.py           # Entry point — displays banner, reads repo info, starts monitoring
├── filechange.py     # File change detection loop, triggers git operations
├── diffcalc.py       # Calculates and displays line diffs using difflib.ndiff
├── gitcommands.py    # Wrappers for git CLI commands (init, add, commit, push, etc.)
├── repoInfo.py       # Reads remote URL and branch from .git/config
├── ignore.py         # Parses .gitignore to get excluded paths
├── logger.py         # JSON-based change logging (tmp.json)
├── colors.py         # ANSI color code constants for terminal output
├── utils.py          # Utility functions (nested file listing, file reading, init workflow)
├── install.py        # Installer — copies tool files to a target directory
├── samplefile.txt    # Sample file for testing
├── requirements.txt  # pyfiglet
└── README.md
```

## Requirements

- Python 3.x
- `pyfiglet` (for ASCII art banner)
- Git installed and available in PATH

## Installation

```bash
cd "Github-Automation"
pip install pyfiglet
```

### Installing into a project

```bash
python install.py
```

When prompted, enter the target project directory path. The script will:
1. Copy all `.py` files and `tmp.json` into `<project_dir>/auto-scripts/`
2. Create or update `.gitignore` to exclude `auto-scripts`, `.idea`, `__pycache__`, `.git`

## Usage

From the target project directory:

```bash
python ./auto-scripts/main.py
```

The tool will:
1. Display a "G - AUTO" ASCII art banner
2. Read repository info from `.git/config` (or prompt for URL and branch if not found)
3. Check `tmp.json` for any uncommitted changes from previous sessions
4. Begin monitoring all files for changes
5. When a change is detected, display the diff and prompt for a commit message
6. Execute git add, commit, and push

### Rejecting a Commit

When prompted for a commit message, enter `-r` to reject the commit for that file.

## How It Works

1. **`main.py`** — Entry point. Uses `pyfiglet` for banner, calls `repoInfo.checkinfoInDir()` to get URL/branch, then starts the change listener.
2. **`repoInfo.py`** — Checks for `.git/config`. If found, extracts remote URL and branch via `git config` and `git rev-parse`. Otherwise prompts the user.
3. **`filechange.py`** — Reads all files initially, then polls continuously comparing current file contents to the initial snapshot. On change, identifies modified files and passes them to diff calculation and git commands.
4. **`diffcalc.py`** — Uses `difflib.ndiff` to compute line-level diffs; displays added lines with `+` markers.
5. **`gitcommands.py`** — Wraps `subprocess.call` for `git init`, `git add`, `git commit -m`, `git push`, etc. Cross-platform README creation (touch/type nul).
6. **`ignore.py`** — Reads `.gitignore` and returns a list of paths that exist on disk.
7. **`logger.py`** — Stores change records (file path + diff) in `tmp.json`. On startup, replays uncommitted changes. After successful push, removes entries.
8. **`utils.py`** — `getNestedFiles()` walks the directory tree excluding ignored dirs. `commitAndUpdate()` orchestrates add → commit → push → log update.
9. **`install.py`** — Lists all `.py` files (except itself) plus `tmp.json`, copies them into the target's `auto-scripts/` folder, and manages `.gitignore`.

## Configuration

- **Ignored directories:** Parsed from `.gitignore` at runtime. Additional directories can be added to `.gitignore` in the target project.
- **Class times / schedules:** Not applicable.
- **`install.py`** adds `auto-scripts`, `.idea`, `__pycache__`, `.git` to the target's `.gitignore`.

## Limitations

- **Polling-based detection:** Continuously reads all files in a `while True` loop with no sleep interval, leading to high CPU usage.
- **`subprocess.call`** with string arguments (not lists) — may behave differently across platforms.
- **Windows-biased:** `gitcommands.py` runs `call('cls', shell=True)` to clear screen; won't work on Linux/macOS.
- **No branch protection** — directly pushes to the configured branch.
- **Fragile file comparison:** Compares entire file contents as lists of lines; may produce incorrect results with simultaneous multi-file changes.
- **`tmp.json` path** in `logger.py` is hardcoded to `os.getcwd()/auto-scripts/tmp.json`.

## Security Notes

No credentials are stored in the code. Git authentication relies on the user's existing git configuration.

## License

Not specified.
