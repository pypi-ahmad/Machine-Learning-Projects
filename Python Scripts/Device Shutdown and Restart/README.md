# Device Shutdown and Restart

A local CLI that previews the platform shutdown or restart command, then runs it only when you explicitly add `--execute`.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Permission to run the operating system's power command

## Preview a command

From this directory:

```powershell
uv sync
uv run python PowerOptions.py shutdown
uv run python PowerOptions.py restart
```

These commands only display the resulting OS command. If no action is supplied, the script accepts the original interactive `s` (shutdown) or `r` (restart) selection and still remains in preview mode.

## Execute a command

```powershell
uv run python PowerOptions.py shutdown --execute
uv run python PowerOptions.py restart --execute
```

`--execute` immediately asks the operating system to perform the action. Save your work first. The Windows restart command does not use the former forced-close flag, but application behavior remains platform-dependent.

## Platform behavior

- Windows: uses `shutdown /s /t 0` or `shutdown /r /t 0`.
- Linux and macOS: uses `shutdown -h now` or `shutdown -r now`; elevated permission may be required.
- Other systems: the script reports that the platform is unsupported.

## Project files

```text
PowerOptions.py  # Preview-first CLI entry point
pyproject.toml   # uv project definition
uv.lock          # Resolved Python environment
```

## Verification

```powershell
uv run python PowerOptions.py shutdown
uv run python -m py_compile PowerOptions.py
uv lock --check
```
