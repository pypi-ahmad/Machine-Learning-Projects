# PageSpeed API

A small Python client for Google PageSpeed Insights API v5. It returns a structured response with loading-experience and Lighthouse fields, and can save the raw API response as JSON.

## Setup

```powershell
uv sync --no-config
```

The client uses `GOOGLE_API_KEY` from the process environment when it is available. The key is optional but can provide a higher quota. It can also be supplied directly to `PageSpeed(api_key=...)`; do not commit keys to source or configuration files.

## Example

```python
from pagespeed import PageSpeed

client = PageSpeed(timeout=30)
response = client.analyse("https://www.example.com", strategy="mobile")

print(response.finalUrl)
print(response.loadingExperience)
client.save(response, "pagespeed-result.json")
```

Run a script that uses the client through uv:

```powershell
uv run --no-config python your_script.py
```

## Behavior and limits

- Calling `analyse()` sends a request to Google's PageSpeed Insights service, which fetches the supplied URL remotely.
- Requests use a 30-second timeout by default. The API may apply quotas or reject malformed or inaccessible target URLs.
- `save()` writes the unmodified JSON response to the supplied file. If a directory is passed, it writes `json_data.json` inside that directory.
- The client supports one PageSpeed strategy and one Lighthouse category per request. It does not retry requests or perform batch analysis.
