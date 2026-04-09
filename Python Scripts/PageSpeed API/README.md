# Pagespeed-API

> A Python client for the Google PageSpeed Insights API v5, with a structured response object for easy data extraction.

## Overview

This project provides a reusable `PageSpeed` class that wraps the Google PageSpeed Insights API v5. It sends HTTP requests to analyse a URL's performance and returns a structured `PageSpeedResponse` object with properties for accessing loading experience metrics, Lighthouse results, and more. Results can also be saved as JSON.

## Features

- Analyse any URL for `desktop` or `mobile` strategy
- Support for optional Google API key authentication
- Structured response object with properties for:
  - Loading experience (overall category)
  - Origin loading experience (overall and detailed metrics)
  - Lighthouse results (requested/final URL, version, user agent)
- Save full API response as formatted JSON file
- Input validation for strategy parameter
- HTTP error handling via `raise_for_status()`

## Project Structure

```
Pagespeed-API/
├── pagespeed.py    # Main PageSpeed client class
├── responses.py    # Response and PageSpeedResponse classes
└── test.py         # Example usage/test script
```

## Requirements

- Python 3.x
- `requests`
- `json` (standard library)

## Installation

```bash
cd "Pagespeed-API"
pip install requests
```

## Usage

### Basic Usage

```python
from pagespeed import PageSpeed

ps = PageSpeed()
response = ps.analyse('https://www.example.com', strategy='mobile')

print(response.url)
print(response.loadingExperience)
print(response.originLoadingExperience)
print(response.loadingExperienceDetailed)
print(response.originLoadingExperienceDetailed)
print(response.finalUrl)
print(response.requestedUrl)
print(response.version)
print(response.userAgent)
```

### With API Key

```python
ps = PageSpeed(api_key='YOUR_API_KEY')
response = ps.analyse('https://www.example.com', strategy='desktop', category='performance')
```

### Save Results to JSON

```python
ps.save(response, path='./')  # Saves as "json_data.json"
```

### Run Test Script

```bash
python test.py
```

## How It Works

1. **`PageSpeed` class** (`pagespeed.py`): Constructs a GET request to the Google PageSpeed Insights API v5 endpoint (`https://www.googleapis.com/pagespeedonline/v5/runPagespeed`) with query parameters for URL, strategy, and category. Optionally includes an API key.
2. **`Response` base class** (`responses.py`): Wraps the raw `requests` response, calls `raise_for_status()` for HTTP error handling, and parses JSON content.
3. **`PageSpeedResponse` class** (`responses.py`): Extends `Response` with `@property` accessors that extract specific fields from the JSON response (loading experience, Lighthouse data, etc.).
4. **`save()` method**: Writes the full JSON response to a file with indented formatting.

## Configuration

- **API Key**: Pass to `PageSpeed(api_key='...')` constructor (optional; the API works without a key but with rate limits).
- **Strategy**: `'desktop'` or `'mobile'` (passed to `analyse()`).
- **Category**: Lighthouse category string, defaults to `'performance'`.
- **Save path**: Defaults to `'./'`, output filename is always `json_data.json`.

## Limitations

- Only one Lighthouse category can be specified per request (the API supports multiple).
- The `save()` method always writes to `json_data.json` — no custom filename support.
- No retry logic or rate-limit handling.
- `test.py` uses a hardcoded URL (`https://www.example.com`).
- No async support for batch URL analysis.

## Security Notes

- API keys are passed as URL query parameters (visible in logs/network traffic). Consider environment variables for key storage.

## License

Not specified.
