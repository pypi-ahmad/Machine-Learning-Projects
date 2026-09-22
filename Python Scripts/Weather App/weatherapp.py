"""Show current city weather from OpenWeatherMap in a Tkinter window."""

from __future__ import annotations

import argparse
import os
import tkinter as tk
from tkinter import messagebox, ttk

import requests

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


def parse_args() -> argparse.Namespace:
    """Parse the safe dry-run option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the required configuration without opening the GUI or requesting weather",
    )
    return parser.parse_args()


def openweather_api_key() -> str:
    """Return the OpenWeatherMap key without exposing it."""
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is required. Set it in the process environment first."
        )
    return api_key


def fetch_weather(city: str, api_key: str) -> dict[str, str]:
    """Fetch normalized metric weather details for a city."""
    response = requests.get(
        WEATHER_URL,
        params={"appid": api_key, "q": city, "units": "metric"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("cod")) != "200":
        raise RuntimeError(payload.get("message", "Weather data was unavailable."))
    try:
        conditions = payload["weather"][0]
        measurements = payload["main"]
        return {
            "temperature": f"{measurements['temp']:.1f} C",
            "pressure": f"{measurements['pressure']} hPa",
            "humidity": f"{measurements['humidity']} %",
            "description": str(conditions["description"]).capitalize(),
        }
    except (IndexError, KeyError, TypeError) as error:
        raise RuntimeError("OpenWeatherMap returned incomplete weather data.") from error


def build_app() -> tk.Tk:
    """Build and return the weather application window."""
    root = tk.Tk()
    root.title("Weather App")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=16)
    frame.grid()
    city = tk.StringVar()
    temperature = tk.StringVar()
    pressure = tk.StringVar()
    humidity = tk.StringVar()
    description = tk.StringVar()

    ttk.Label(frame, text="City").grid(row=0, column=0, sticky="w", pady=3)
    city_entry = ttk.Entry(frame, textvariable=city, width=34)
    city_entry.grid(row=0, column=1, columnspan=2, sticky="ew", pady=3)
    ttk.Label(frame, text="Temperature").grid(row=1, column=0, sticky="w", pady=3)
    ttk.Label(frame, textvariable=temperature).grid(row=1, column=1, columnspan=2, sticky="w")
    ttk.Label(frame, text="Pressure").grid(row=2, column=0, sticky="w", pady=3)
    ttk.Label(frame, textvariable=pressure).grid(row=2, column=1, columnspan=2, sticky="w")
    ttk.Label(frame, text="Humidity").grid(row=3, column=0, sticky="w", pady=3)
    ttk.Label(frame, textvariable=humidity).grid(row=3, column=1, columnspan=2, sticky="w")
    ttk.Label(frame, text="Description").grid(row=4, column=0, sticky="w", pady=3)
    ttk.Label(frame, textvariable=description).grid(row=4, column=1, columnspan=2, sticky="w")

    def clear() -> None:
        city.set("")
        temperature.set("")
        pressure.set("")
        humidity.set("")
        description.set("")
        city_entry.focus_set()

    def submit() -> None:
        city_name = city.get().strip()
        if not city_name:
            messagebox.showerror("Weather App", "Enter a city name.")
            return
        try:
            weather = fetch_weather(city_name, openweather_api_key())
        except (requests.RequestException, RuntimeError) as error:
            messagebox.showerror("Weather App", str(error))
            return
        temperature.set(weather["temperature"])
        pressure.set(weather["pressure"])
        humidity.set(weather["humidity"])
        description.set(weather["description"])

    ttk.Button(frame, text="Get weather", command=submit).grid(row=5, column=1, pady=(12, 0))
    ttk.Button(frame, text="Clear", command=clear).grid(row=5, column=2, pady=(12, 0))
    city_entry.focus_set()
    return root


def main() -> None:
    """Run the weather app or the safe configuration preview."""
    args = parse_args()
    if args.dry_run:
        print("Would open the weather GUI and read OPENWEATHER_API_KEY on submission.")
        return
    build_app().mainloop()


if __name__ == "__main__":
    main()
