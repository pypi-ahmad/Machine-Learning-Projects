# Site Blocker

A Windows hosts-file scheduler that redirects a fixed list of domains to
`127.0.0.1` during configured hours. It is dry-run by default.

## Check the current action

```powershell
uv run python web-blocker.py --once
```

This prints whether the current schedule would block or unblock the configured
domains. It does not change any file.

## Apply a single update

Run PowerShell as Administrator, then explicitly opt in to modifying the
Windows hosts file:

```powershell
uv run python web-blocker.py --apply --once
```

To keep checking every minute between 09:00 and 18:00, omit `--once`:

```powershell
uv run python web-blocker.py --apply --start-hour 9 --end-hour 18 --interval 60
```

Use `Ctrl+C` to stop the continuous process. A schedule that crosses midnight,
such as `--start-hour 22 --end-hour 6`, is supported.

## Safety and limitations

- `--apply` is required before any hosts-file write.
- The tool removes only redirect lines it manages: `127.0.0.1` followed by one
  of its configured domains.
- It affects all applications that use the system hosts resolver, not just one
  browser.
- Administrator rights are normally required for the default Windows hosts file.
- DNS-over-HTTPS, other DNS resolvers, cached connections, or direct IP access
  can bypass hosts-file blocking.
- This is not an access-control or parental-control system.

## Dependencies

The project uses only the Python standard library. uv records the Python
requirement and provides the reproducible environment.
