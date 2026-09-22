# Python Code Runner

Run local Python snippets or `.py` files from an interactive terminal menu.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
uv run --no-config python main.py .\script.py
```

The interactive menu can run inline code, manage saved snippets, and run a selected Python file. Saved snippets are stored in `code_snippets.json` beside `main.py` after they are changed.

## Safety

This is not a sandbox. Snippets and files run as your Windows user and can read, modify, or delete accessible files, use the network, and start subprocesses. Run only code you understand and trust. The timeout only stops the child process after the configured time; it does not restrict capabilities.
