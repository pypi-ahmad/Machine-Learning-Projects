"""Current conditions and a seven-day forecast from Open-Meteo."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}
WMO_EMOJI = {
    0: "sunny", 1: "partly_cloudy_day", 2: "partly_cloudy_day", 3: "cloud",
    45: "foggy", 48: "foggy", 51: "rainy", 53: "rainy", 55: "rainy",
    61: "rainy", 63: "rainy", 65: "rainy", 71: "weather_snowy", 73: "weather_snowy",
    75: "weather_snowy", 77: "weather_snowy", 80: "rainy", 81: "rainy",
    82: "thunderstorm", 85: "weather_snowy", 86: "weather_snowy", 95: "thunderstorm",
    96: "thunderstorm", 99: "thunderstorm",
}


@st.cache_data(ttl="1h", max_entries=100)
def geocode(city: str) -> dict[str, float | str] | None:
    """Resolve a city to the first Open-Meteo geocoding result."""
    query = urllib.parse.urlencode({"name": city, "count": 1})
    url = f"https://geocoding-api.open-meteo.com/v1/search?{query}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            results = json.loads(response.read()).get("results")
    except (urllib.error.URLError, ValueError):
        return None
    if not results:
        return None
    result = results[0]
    return {
        "lat": result["latitude"],
        "lon": result["longitude"],
        "name": result["name"],
        "country": result.get("country", ""),
    }


@st.cache_data(ttl="30m", max_entries=100)
def fetch_weather(latitude: float, longitude: float) -> dict | None:
    """Fetch current conditions and a seven-day forecast from Open-Meteo."""
    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code,"
                "apparent_temperature,precipitation"
            ),
            "daily": (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,wind_speed_10m_max"
            ),
            "timezone": "auto",
            "forecast_days": 7,
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{query}"
    try:
        with urllib.request.urlopen(url, timeout=6) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, ValueError):
        return None


def convert_temperature(celsius: float, unit: str) -> float:
    """Convert Celsius to the selected display unit."""
    return celsius if unit == "C" else round(celsius * 9 / 5 + 32, 1)


def forecast_frame(data: dict, unit: str) -> pd.DataFrame:
    """Build the forecast table for display."""
    daily = data["daily"]
    high = pd.Series(daily["temperature_2m_max"])
    low = pd.Series(daily["temperature_2m_min"])
    if unit == "F":
        high = (high * 9 / 5 + 32).round(1)
        low = (low * 9 / 5 + 32).round(1)
    return pd.DataFrame(
        {
            "Date": daily["time"],
            "Condition": pd.Series(daily["weather_code"]).map(WMO_CODES).fillna("Unknown"),
            f"High ({unit})": high,
            f"Low ({unit})": low,
            "Rain (mm)": daily["precipitation_sum"],
            "Wind (km/h)": daily["wind_speed_10m_max"],
        }
    )


st.set_page_config(page_title="Weather dashboard", page_icon=":material/cloud:", layout="wide")
st.title("Weather dashboard")
st.caption("Current conditions and a seven-day forecast from Open-Meteo. No API key required.")

st.session_state.setdefault("weather_data", None)
st.session_state.setdefault("weather_location", None)

with st.form("weather_search"):
    city = st.text_input("City", placeholder="For example, London", key="weather_city")
    unit = st.segmented_control(
        "Temperature unit",
        ["C", "F"],
        default="C",
        format_func=lambda value: f"°{value}",
        key="weather_unit",
    )
    submitted = st.form_submit_button("Get weather", icon=":material/search:")

if submitted:
    st.session_state.weather_data = None
    st.session_state.weather_location = None
    if not city.strip():
        st.warning("Enter a city name.")
    else:
        with st.spinner("Fetching weather..."):
            location = geocode(city.strip())
            weather_data = (
                fetch_weather(location["lat"], location["lon"]) if location else None
            )
        if not location:
            st.error(f"City '{city.strip()}' was not found.")
        elif not weather_data:
            st.error("Weather data could not be fetched. Please try again.")
        else:
            st.session_state.weather_location = location
            st.session_state.weather_data = weather_data

location = st.session_state.weather_location
weather_data = st.session_state.weather_data
if location and weather_data:
    current = weather_data["current"]
    weather_code = current["weather_code"]
    unit_label = f"°{unit}"
    st.subheader(
        f":material/{WMO_EMOJI.get(weather_code, 'thermostat')}: "
        f"{location['name']}, {location['country']}"
    )
    st.caption(f"Updated: {current['time']}")
    with st.container(horizontal=True):
        st.metric(
            f"Temperature ({unit_label})",
            f"{convert_temperature(current['temperature_2m'], unit)}{unit_label}",
            border=True,
        )
        st.metric(
            "Feels like",
            f"{convert_temperature(current['apparent_temperature'], unit)}{unit_label}",
            border=True,
        )
        st.metric("Humidity", f"{current['relative_humidity_2m']}%", border=True)
        st.metric("Wind", f"{current['wind_speed_10m']} km/h", border=True)

    st.info(
        f"{WMO_CODES.get(weather_code, 'Unknown conditions')} · "
        f"Precipitation: {current['precipitation']} mm"
    )
    forecast = forecast_frame(weather_data, unit)
    with st.container(border=True):
        st.subheader("Seven-day forecast")
        st.dataframe(forecast, hide_index=True)

    chart_data = forecast.set_index("Date")[[f"High ({unit})", f"Low ({unit})"]]
    with st.container(border=True):
        st.subheader("Temperature range")
        st.line_chart(chart_data)
