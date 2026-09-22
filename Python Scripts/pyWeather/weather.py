"""Fetch current weather for one city from OpenWeatherMap."""

import argparse
import os

import requests


API_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather(city: str, api_key: str, units: str) -> dict:
    """Request current conditions without exposing the API key."""
    response = requests.get(
        API_URL,
        params={"appid": api_key, "q": city, "units": units},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def display_weather(data: dict, units: str) -> None:
    """Print the current conditions returned by OpenWeatherMap."""
    main = data["main"]
    wind = data.get("wind", {})
    clouds = data.get("clouds", {})
    description = data.get("weather", [{}])[0].get("description", "Unknown")
    temperature_unit = {"metric": "C", "imperial": "F"}.get(units, "K")
    print(f"Temperature: {main['temp']} {temperature_unit}")
    print(f"Pressure: {main['pressure']} hPa")
    print(f"Humidity: {main['humidity']}%")
    print(f"Wind speed: {wind.get('speed', 'Unknown')} m/s")
    print(f"Wind direction: {wind.get('deg', 'Unknown')} degrees")
    print(f"Cloudiness: {clouds.get('all', 'Unknown')}%")
    print(f"Weather: {description}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch current weather from OpenWeatherMap.")
    parser.add_argument("city", nargs="?", help="City name to look up")
    parser.add_argument("--units", choices=["standard", "metric", "imperial"], default="metric")
    args = parser.parse_args()
    city = args.city or input("Enter city name: ").strip()
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not city:
        parser.error("a city is required")
    if not api_key:
        parser.exit(1, "Required environment variable OPENWEATHER_API_KEY is unavailable.\n")

    try:
        display_weather(get_weather(city, api_key, args.units), args.units)
    except requests.RequestException as error:
        parser.exit(1, f"Weather request failed: {error}\n")
    except (KeyError, IndexError, TypeError) as error:
        parser.exit(1, f"Unexpected weather response: {error}\n")


if __name__ == "__main__":
    main()
