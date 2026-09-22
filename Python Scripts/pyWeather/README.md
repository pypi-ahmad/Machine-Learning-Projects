# pyWeather

Fetch current conditions for one city from OpenWeatherMap.

## Setup

```powershell
uv sync --no-config
```

Set `OPENWEATHER_API_KEY` in your Windows environment, then relaunch the terminal or coding host so it inherits the variable. The key is never stored in this project.

## Run

```powershell
uv run --no-config python weather.py London
uv run --no-config python weather.py London --units imperial
```

Omit the city to enter it interactively. Units default to metric; use `standard` for Kelvin or `imperial` for Fahrenheit.

## Notes

The tool calls OpenWeatherMap's live current-weather API and requires internet access plus a valid API key. Live conditions can change; check official weather warnings before making safety-critical decisions.
