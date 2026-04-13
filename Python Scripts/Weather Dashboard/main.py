"""Weather Dashboard — Streamlit app.

Current weather and 5-day forecast via Open-Meteo (no API key needed).
Geocoding via Open-Meteo geocoding API.

Usage:
    streamlit run main.py
"""

import json
import urllib.request
import urllib.parse
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Weather Dashboard", layout="wide")
st.title("🌤️ Weather Dashboard")

WMO_CODES = {
    0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",
    45:"Fog",48:"Icy fog",51:"Light drizzle",53:"Moderate drizzle",55:"Dense drizzle",
    61:"Slight rain",63:"Moderate rain",65:"Heavy rain",
    71:"Slight snow",73:"Moderate snow",75:"Heavy snow",77:"Snow grains",
    80:"Slight showers",81:"Moderate showers",82:"Violent showers",
    85:"Slight snow showers",86:"Heavy snow showers",
    95:"Thunderstorm",96:"Thunderstorm + hail",99:"Thunderstorm + heavy hail",
}
WMO_EMOJI = {
    0:"☀️",1:"🌤️",2:"⛅",3:"☁️",45:"🌫️",48:"🌫️",
    51:"🌦️",53:"🌦️",55:"🌧️",61:"🌧️",63:"🌧️",65:"🌧️",
    71:"🌨️",73:"🌨️",75:"❄️",77:"❄️",80:"🌦️",81:"🌧️",82:"⛈️",
    85:"🌨️",86:"❄️",95:"⛈️",96:"⛈️",99:"⛈️",
}


@st.cache_data(ttl=3600)
def geocode(city: str) -> dict | None:
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            results = data.get("results")
            if results:
                r = results[0]
                return {"lat": r["latitude"], "lon": r["longitude"],
                        "name": r["name"], "country": r.get("country", "")}
    except Exception:
        pass
    return None


@st.cache_data(ttl=1800)
def fetch_weather(lat: float, lon: float) -> dict | None:
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        f"weather_code,apparent_temperature,precipitation"
        f"&daily=weather_code,temperature_2m_max,temperature_2m_min,"
        f"precipitation_sum,wind_speed_10m_max"
        f"&timezone=auto&forecast_days=7"
    )
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ── UI ─────────────────────────────────────────────────────────────────────
city = st.text_input("🔍 Enter city name", "London")
unit = st.radio("Temperature unit", ["°C", "°F"], horizontal=True)

if st.button("Get Weather") or city:
    geo = geocode(city)
    if not geo:
        st.error(f"City '{city}' not found.")
        st.stop()

    data = fetch_weather(geo["lat"], geo["lon"])
    if not data:
        st.error("Could not fetch weather data.")
        st.stop()

    def to_unit(c): return c if unit == "°C" else round(c * 9/5 + 32, 1)
    u_label = unit

    cur = data["current"]
    code = cur["weather_code"]
    emoji = WMO_EMOJI.get(code, "🌡️")
    desc  = WMO_CODES.get(code, "Unknown")

    st.subheader(f"{emoji} {geo['name']}, {geo['country']}")
    st.caption(f"Updated: {cur['time']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Temperature ({u_label})", f"{to_unit(cur['temperature_2m'])}{u_label}")
    c2.metric(f"Feels Like",  f"{to_unit(cur['apparent_temperature'])}{u_label}")
    c3.metric("Humidity",     f"{cur['relative_humidity_2m']}%")
    c4.metric("Wind",         f"{cur['wind_speed_10m']} km/h")

    st.info(f"**{desc}**  ·  Precipitation: {cur['precipitation']} mm")

    st.subheader("7-Day Forecast")
    daily = data["daily"]
    rows  = []
    for i, date_str in enumerate(daily["time"]):
        c = daily["weather_code"][i]
        rows.append({
            "Date":    date_str,
            "":        WMO_EMOJI.get(c, ""),
            "Condition": WMO_CODES.get(c, ""),
            f"High ({u_label})": to_unit(daily["temperature_2m_max"][i]),
            f"Low ({u_label})":  to_unit(daily["temperature_2m_min"][i]),
            "Rain (mm)":  daily["precipitation_sum"][i],
            "Wind (km/h)": daily["wind_speed_10m_max"][i],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Temperature chart
    chart_df = pd.DataFrame({
        "High": [to_unit(t) for t in daily["temperature_2m_max"]],
        "Low":  [to_unit(t) for t in daily["temperature_2m_min"]],
    }, index=daily["time"])
    st.subheader("Temperature Range")
    st.line_chart(chart_df)
