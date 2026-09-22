"""Fetch and display the current weather for an entered city."""

import os
import sys

import requests

API_URL = "https://api.openweathermap.org/data/2.5/weather"
API_KEY_VARIABLE = "OPENWEATHER_API_KEY"
REQUEST_TIMEOUT_SECONDS = 15


def fetch_weather(city_name, api_key):
    """Return OpenWeather's response data for one city."""
    response = requests.get(
        API_URL,
        params={"appid": api_key, "q": city_name},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def main():
    """Read a city name and print the supplied weather fields."""
    api_key = os.environ.get(API_KEY_VARIABLE)
    if not api_key:
        print(f"Set the {API_KEY_VARIABLE} environment variable before running this script.")
        return 1

    city_name = input("Enter city name: ").strip()
    if not city_name:
        print("Enter a city name.")
        return 1

    try:
        weather = fetch_weather(city_name, api_key)
    except requests.RequestException as error:
        print(f"Could not fetch weather data: {error}")
        return 1

    if str(weather.get("cod")) == "404":
        print("City not found.")
        return 1

    try:
        details = weather["main"]
        description = weather["weather"][0]["description"]
        print(
            f"Temperature (Kelvin): {details['temp']}\n"
            f"Pressure (hPa): {details['pressure']}\n"
            f"Humidity (%): {details['humidity']}\n"
            f"Description: {description}"
        )
    except (KeyError, IndexError, TypeError):
        print("The weather service returned an unexpected response.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
