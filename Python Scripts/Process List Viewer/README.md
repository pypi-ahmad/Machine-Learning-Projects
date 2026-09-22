# Process List Viewer

An interactive terminal utility for listing and filtering running processes, viewing CPU and memory usage, and optionally terminating a process by PID.

## Setup

```powershell
uv sync --no-config
```

## Run

```powershell
uv run --no-config python main.py
```

Use the menu to list processes, filter by name or user, or show top CPU and memory consumers.

## Safety

Option 6 sends a termination request to the selected PID after confirmation. Ending the wrong process can lose unsaved work or destabilize Windows. Verify the PID and process name before confirming, and avoid terminating system or security processes.

The tool uses `psutil` for richer process information. It may show limited information for protected processes because Windows access controls still apply. It runs locally and makes no network requests.
