# URL Shortener

A small command-line client for the TinyURL API. It accepts one or more absolute HTTP(S) URLs and prints the returned short URLs.

## Run it

```powershell
uv sync
uv run python app.py https://example.com --dry-run
```

Use `--dry-run` first to validate URLs and show the API request without making a network call. To create short links, omit the flag:

```powershell
uv run python app.py https://example.com https://www.python.org
```

## Notes

Creating a short URL sends the original URL to TinyURL. The service controls link availability, destination handling, and terms of use. Do not shorten URLs that contain passwords, tokens, private identifiers, or other secrets.

## Dependencies

- Python 3.14+
- No third-party packages
