# Website Load Time Checker

Measure the elapsed time to open an HTTP(S) URL and read its response body. This is a simple request timing tool, not a browser performance measurement.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

The project uses only the Python standard library.

## Run

```powershell
cd "Python Scripts\Website Load Time Checker"
uv run python .\time_to_load_website.py https://example.com
```

Hostnames without a scheme default to HTTPS:

```powershell
uv run python .\time_to_load_website.py example.com --timeout 5
```

Validate the request plan without connecting:

```powershell
uv run python .\time_to_load_website.py example.com --dry-run
```

## Notes

- The reported duration includes connection setup, request handling, and reading the response body.
- The default timeout is 10 seconds.
- Timing varies with network conditions, remote-server behavior, redirects, and response size.
- Run checks only against URLs you are authorized to contact.
