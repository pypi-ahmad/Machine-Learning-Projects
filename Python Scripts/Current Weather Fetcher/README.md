# Current Weather Fetcher

An interactive CLI that retrieves current weather conditions for a city through the [OpenWeather API](https://openweathermap.org/api).

## Requirements

- Python 3.13+
- An OpenWeather API key
- [uv](https://docs.astral.sh/uv/)

## Configuration

Set the API key in the environment for the current PowerShell session. Do not place it in source code or commit it to the repository.

```powershell
$env:OPENWEATHER_API_KEY = "your-key"
```

If you configured the variable after launching your coding host, restart that host before running the script.

## Run

From this directory:

```powershell
uv sync
uv run python fetch_current_weather.py
```

Enter a city name when prompted. The script prints the temperature in Kelvin, pressure in hPa, humidity as a percentage, and the provider's weather description.

## Behavior and limits

- Requests use HTTPS and a 15-second timeout.
- Missing credentials, an empty city, network/API failures, unknown cities, and unexpected responses produce clear messages.
- Current conditions can change quickly and may not match local observations. This is an informational utility, not a weather or safety advisory.

## Project files

```text
fetch_current_weather.py  # CLI entry point
pyproject.toml            # uv dependency definition
uv.lock                   # Resolved dependency versions
```

## Verification

```powershell
uv run python -m py_compile fetch_current_weather.py
uv lock --check
```
