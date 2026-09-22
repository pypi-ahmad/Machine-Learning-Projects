# Weather App

A small Tkinter desktop app that shows current temperature, pressure, humidity, and conditions for a city through OpenWeatherMap.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenWeatherMap API key in `OPENWEATHER_API_KEY`

## Install

```powershell
cd "Python Scripts\Weather App"
uv sync
```

Make the API key available to the process before launching the app. Do not put it in source code or a committed `.env` file.

```powershell
$env:OPENWEATHER_API_KEY = "your-api-key"
```

## Run

```powershell
uv run python .\weatherapp.py
```

Enter a city name and select **Get weather**. The application requests metric data and displays temperature in Celsius.

Preview the configuration without opening a window or making a weather request:

```powershell
uv run python .\weatherapp.py --dry-run
```

## Notes

- The key is read only when you request weather; it is never written to disk or displayed.
- The app uses HTTPS and a 15-second request timeout.
- Weather availability and accuracy depend on OpenWeatherMap and network access.
