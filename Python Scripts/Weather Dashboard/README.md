# Weather Dashboard

A Streamlit dashboard for current conditions and a seven-day forecast from Open-Meteo. It resolves a city with Open-Meteo's geocoding endpoint and does not require an API key.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)

## Install and run

```powershell
cd "Python Scripts\Weather Dashboard"
uv sync
uv run streamlit run .\main.py
```

Enter a city, choose Celsius or Fahrenheit, and select **Get weather**.

## Behavior

- City searches and forecast requests run only after form submission.
- Geocoding results are cached for one hour; weather results are cached for 30 minutes.
- Requests use Open-Meteo's public APIs and require network access.
- Forecasts are provided as returned by Open-Meteo and should not be used for safety-critical decisions.

## Headless verification

The dashboard can be checked with Streamlit's in-process `AppTest`; this does not start a server or browser. The initial screen performs no network request.
