# Website Blocker

Add or remove one clearly marked localhost redirect in a hosts file. The commands are previews by default and change a file only with `--apply`.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- Administrator privileges on Windows or root privileges on Linux when applying changes to the system hosts file

The project has no third-party runtime dependencies.

## Preview a block

```powershell
cd "Python Scripts\Website Blocker"
uv run python .\website_blocker.py example.com
```

The preview does not read or write the system hosts file. To apply the redirect from an elevated PowerShell session:

```powershell
uv run python .\website_blocker.py example.com --apply
```

The managed entry is:

```text
127.0.0.1 example.com # website-blocker
```

## Preview or apply removal

```powershell
uv run python .\website_unblocker.py example.com
uv run python .\website_unblocker.py example.com --apply
```

Only an exact entry that includes the `# website-blocker` marker is removed. Other hosts-file entries are left unchanged.

## Test against another file

Use `--hosts-file` to preview or apply an operation against a non-system file:

```powershell
uv run python .\website_blocker.py example.com --hosts-file .\sample-hosts
```

## Notes

- Enter a domain such as `example.com` or a full URL; the hostname is extracted automatically.
- Hosts-file redirects are local to the device and do not replace network-level controls.
- Changing the system hosts file can affect other software. Review the preview and use `--apply` deliberately.
