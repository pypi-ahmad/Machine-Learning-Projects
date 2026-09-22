# Work Setup Automation

Open selected work URLs and an optional local Windows application. The command previews all targets by default and launches nothing until `--open` is supplied.

## Requirements

- Windows for launching a local application
- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

The project uses only the Python standard library.

## Preview a browser setup

```powershell
cd "Python Scripts\Work Setup Automation"
uv run python .\workstation.py github.com pypi.org
```

## Open browser tabs

```powershell
uv run python .\workstation.py github.com pypi.org --open
```

## Open a local application and browser tabs

```powershell
uv run python .\workstation.py github.com --app "C:\Path\To\editor.exe" --open
```

URLs without a scheme default to HTTPS. The default browser handles URLs; no Chrome path is configured in source code.

## Notes

- The script never configures Windows Startup automatically.
- Use `--open` deliberately because it launches applications and browser tabs.
- Verify each local application path before using it.
