# Weather App

> A Tkinter-based GUI application that fetches and displays current weather data for any city using the OpenWeatherMap API.

## Overview

This application provides a graphical interface where users can enter a city name and retrieve the current temperature, atmospheric pressure, humidity, and weather description from the OpenWeatherMap API.

## Features

- GUI built with Tkinter
- Fetches real-time weather data from OpenWeatherMap API
- Displays temperature (in Kelvin), atmospheric pressure (hPa), humidity (%), and weather description
- Error dialog for invalid city names (HTTP 404 response)
- Clear button to reset all fields

## Project Structure

```
WEATHER_APP/
├── README.md
└── weatherapp.py
```

## Requirements

- Python 3.x
- `requests` — HTTP requests to OpenWeatherMap API
- `tkinter` (stdlib) — GUI framework
- An [OpenWeatherMap API key](https://openweathermap.org/api)

## Installation

```bash
cd WEATHER_APP
pip install requests
```

## Usage

1. Open `weatherapp.py` and replace `"api_key"` with your actual OpenWeatherMap API key:
   ```python
   api_key = "YOUR_API_KEY_HERE"
   ```
2. Run:
   ```bash
   python weatherapp.py
   ```
3. Enter a city name in the input field and click **Submit**.
4. Click **Clear** to reset all fields.

## How It Works

1. The `tell_weather()` function reads the city name from the input field.
2. Constructs the API URL: `http://api.openweathermap.org/data/2.5/weather?appid=API_KEY&q=CITY`
3. Sends a GET request via `requests.get()` and parses the JSON response.
4. If the response code is not `"404"`, extracts `temp`, `pressure`, `humidity` from `x["main"]` and `description` from `x["weather"][0]`.
5. Populates the corresponding Entry fields in the GUI.
6. If the city is not found, shows an error dialog via `messagebox.showerror()`.

## Configuration

- **API Key**: Hardcoded as `api_key = "api_key"` in the `tell_weather()` function. Must be replaced with a valid OpenWeatherMap API key.
- **API Endpoint**: Uses the free-tier endpoint `http://api.openweathermap.org/data/2.5/weather`.

## Limitations

- Temperature is displayed in **Kelvin** (no conversion to Celsius or Fahrenheit).
- The API key is stored as plaintext in the source code.
- The `requests` and `json` modules are imported inside the `tell_weather()` function rather than at the top of the file.
- Existing field content is not cleared before inserting new data; submitting multiple times appends results.
- The URL construction has spaces around `=` signs (`"appid =" + api_key + "&q =" + city_name`), which may cause issues with some API versions.
- No loading indicator while the API request is in progress.
- Window size is fixed at 425x175 pixels.

## Security Notes

- The API key placeholder `"api_key"` must be replaced before use. Avoid committing real API keys to version control; consider using environment variables.

## License

Not specified.
