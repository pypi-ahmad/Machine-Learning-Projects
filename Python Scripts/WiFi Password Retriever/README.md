# WiFi Password Retriever

List saved Windows Wi-Fi profile names. Plaintext keys are shown only when explicitly requested.

## Requirements

- Windows
- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

The project uses the standard-library `subprocess` module and the Windows `netsh` utility.

## List saved profiles

```powershell
cd "Python Scripts\WiFi Password Retriever"
uv run python .\wifi.py
```

## Display stored keys

Only use this on profiles you own or are authorized to administer. Keys are printed as plaintext:

```powershell
uv run python .\wifi.py --show-keys
```

Preview the intended action without executing `netsh`:

```powershell
uv run python .\wifi.py --show-keys --dry-run
```

## Notes

- The default command lists profile names only.
- `--show-keys` may require an elevated terminal and can expose sensitive credentials in terminal history or logs.
- Parsing depends on the English `netsh` labels; localized Windows installations may need an adjusted parser.
