# GitHub Repo Automation

> Automate GitHub repository creation and local folder setup from the command line.

## Overview

A Python script that creates a new local directory and a corresponding GitHub repository in a single command. Uses the `PyGithub` library to authenticate and create the remote repository, then creates a local folder at a configured path.

## Features

- Creates a new GitHub repository under the authenticated user's account
- Creates a matching local directory at a specified path
- Takes the repository/folder name as a command-line argument

## Project Structure

```
GitHub Repo Automation/
├── GitHub.py
├── License
└── README.md
```

## Requirements

- Python 3.x
- `PyGithub`

## Installation

```bash
cd "GitHub Repo Automation"
pip install PyGithub
```

## Usage

```bash
python GitHub.py <repository_name>
```

Example:
```bash
python GitHub.py my-new-project
```

This will:
1. Create a local folder at the configured `path` + `my-new-project`
2. Create a GitHub repository named `my-new-project` under your account

## How It Works

1. Reads `sys.argv[1]` as the folder/repository name.
2. Creates a local directory using `os.makedirs(path + folderName)`.
3. Authenticates to GitHub using `Github(username, password)`.
4. Calls `user.create_repo(folderName)` to create the remote repository.
5. Prints a success message.

## Configuration

Before running, edit `GitHub.py` to set:

```python
path = ""      # Local path where the folder will be created (e.g., "C:/Projects/")
username = ""  # Your GitHub username
password = ""  # Your GitHub password or personal access token
```

## Limitations

- **Syntax error in source code:** The `path` variable assignment (`path = "" add the path of the folder`) is missing a `#` comment marker and will cause a `SyntaxError` at runtime.
- No local git initialization — only creates the folder and remote repo, but doesn't run `git init` or link them.
- Uses username/password authentication which is deprecated by GitHub (personal access tokens should be used instead).
- No input validation; crashes if no command-line argument is provided.
- No error handling for existing directories or duplicate repository names.

## Security Notes

- **Hardcoded credentials:** Username and password are stored in plaintext in the source code. Use environment variables or a config file instead.
- GitHub has deprecated password-based authentication for API access; use a personal access token.

## License

MIT License — Copyright (c) 2020 Arbaz Khan (see `License` file).
