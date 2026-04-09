# Fetch Current Weather

> A command-line Python script that fetches and displays current weather data for any city using the OpenWeatherMap API.

## Overview

This script prompts the user for a city name, queries the OpenWeatherMap API, and displays the current temperature, atmospheric pressure, humidity, and weather description.

## Features

- Fetches real-time weather data from the OpenWeatherMap API
- Displays temperature (in Kelvin), atmospheric pressure (hPa), humidity (%), and weather description
- Handles invalid city names with a "City Not Found" message
- Interactive city name input via command line

## Project Structure

```
Fetch_current_weather/
└── fetch_current_weather.py   # Main script
```

## Requirements

- Python 3.x
- `requests`
- An OpenWeatherMap API key (free tier available)

## Installation

```bash
cd "Fetch_current_weather"
pip install requests
```

## Usage

1. Get a free API key from [OpenWeatherMap](https://home.openweathermap.org/api_keys).
2. Edit `fetch_current_weather.py` and replace `"Your_API_Key"` with your actual key.
3. Run:

```bash
python fetch_current_weather.py
```

Example output:

```
Enter city name : London
 Temperature (in kelvin unit) = 283.15
 atmospheric pressure (in hPa unit) = 1012
 humidity (in percentage) = 72
 description = overcast clouds
```

## How It Works

1. Constructs the API URL: `http://api.openweathermap.org/data/2.5/weather?appid={key}&q={city}`
2. Sends a GET request using `requests.get()`.
3. Parses the JSON response.
4. If the city is found (status code ≠ 404), extracts `main.temp`, `main.pressure`, `main.humidity`, and `weather[0].description`.
5. If the city is not found, prints "City Not Found".

## Configuration

| Variable   | Location                     | Description                     |
|------------|------------------------------|---------------------------------|
| `api_key`  | `fetch_current_weather.py`   | Your OpenWeatherMap API key     |

## Limitations

- Temperature is displayed in **Kelvin** by default — no conversion to Celsius or Fahrenheit
- No error handling for network failures or invalid API keys
- Uses HTTP instead of HTTPS for the API endpoint
- No `requirements.txt` file included
- The API key is hardcoded as a placeholder string
- Only displays basic weather data (no wind, visibility, etc.)

## Security Notes

- The API key is stored as a **plaintext placeholder** in the source code — replace with your own key and consider using environment variables instead

## License

Not specified.
