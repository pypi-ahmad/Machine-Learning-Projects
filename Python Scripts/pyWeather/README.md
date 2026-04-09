# pyWeather

> A command-line weather forecast script that fetches current weather data for any city using the OpenWeatherMap API.

## Overview

This script prompts the user for a city name, queries the OpenWeatherMap API, and displays current weather information including temperature, pressure, humidity, wind speed, wind direction, cloudiness, and a weather description.

## Features

- Fetches real-time weather data for any city via the OpenWeatherMap API
- Displays temperature (in Kelvin), atmospheric pressure, humidity, wind speed, wind direction, cloudiness percentage, and weather description
- Validates city name and reports "City Not Found" for invalid entries

## Project Structure

```
pyWeather/
├── weather.py    # Main script
└── README.md
```

## Requirements

- Python 3.x
- `requests`
- An OpenWeatherMap API key (free tier available at [openweathermap.org](https://openweathermap.org/))

## Installation

```bash
cd "pyWeather"
pip install requests
```

## Usage

1. Edit `weather.py` and replace `'Your API key goes here'` with your actual OpenWeatherMap API key
2. Run the script:

```bash
python weather.py
```

3. Enter a city name when prompted:

```
Enter city name: London
Temperature (in Kelvin) = 283.15
 Atmospheric Pressure (in hPa) = 1012
 Humidity (in percentage) = 81
 Wind Speed (in m/s) = 4.1
 Wind Direction (in degrees) = 250
 Cloudliness (in percentage) = 75
 Weather Description = broken clouds
```

## How It Works

1. Constructs an API URL by combining the base URL `http://api.openweathermap.org/data/2.5/weather?` with the API key and city name.
2. Sends a GET request using `requests.get()`.
3. Parses the JSON response and checks if the city was found (`cod != '404'`).
4. Extracts data from `main` (temperature, pressure, humidity), `weather` (description), `wind` (speed, direction), and `clouds` (cloudiness) objects.

## Configuration

- `api_key` — must be set to a valid OpenWeatherMap API key (line 5 of `weather.py`)
- `base_url` — the OpenWeatherMap API endpoint (hardcoded)

## Limitations

- API key is hardcoded in the source file as a placeholder string
- Temperature is displayed in Kelvin only (no Celsius/Fahrenheit conversion)
- No error handling for network failures or malformed JSON responses
- Only queries current weather (no forecasts)
- City name input has no validation beyond the API's 404 response

## Security Notes

- **API key is stored in plaintext** in the source file. Consider using environment variables instead.

## License

Not specified.
