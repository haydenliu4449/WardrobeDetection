"""Today's forecast from Open-Meteo"""

import logging

import requests

log = logging.getLogger(__name__)

URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes, grouped down to something short enough for the mirror
CODES = [
    ((0,), "clear"),
    ((1, 2), "partly cloudy"),
    ((3,), "overcast"),
    ((45, 48), "fog"),
    ((51, 53, 55, 56, 57), "drizzle"),
    ((61, 63, 65, 66, 67, 80, 81, 82), "rain"),
    ((71, 73, 75, 77, 85, 86), "snow"),
    ((95, 96, 99), "thunderstorms"),
]


def describe(code):
    for codes, text in CODES:
        if code in codes:
            return text
    return ""


def today(lat, lon, units="fahrenheit"):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max,"
                 "precipitation_probability_max,weather_code",
        "temperature_unit": units,
        "timezone": "auto",
        "forecast_days": 1,
    }
    r = requests.get(URL, params=params, timeout=15)
    r.raise_for_status()
    d = r.json()["daily"]
    return {
        "high": round(d["temperature_2m_max"][0]),
        "low": round(d["temperature_2m_min"][0]),
        "feels_like": round(d["apparent_temperature_max"][0]),
        "rain_chance": d["precipitation_probability_max"][0] or 0,
        "summary": describe(d["weather_code"][0]),
        "units": units,
    }


def fallback(units="fahrenheit"):
    # mild day, used when the forecast can't be fetched
    high, low = (65, 52) if units == "fahrenheit" else (18, 11)
    return {"high": high, "low": low, "feels_like": high, "rain_chance": 0,
            "summary": "", "units": units, "offline": True}


def get(cfg):
    loc = cfg["location"]
    try:
        return today(loc["latitude"], loc["longitude"], loc.get("units", "fahrenheit"))
    except (requests.RequestException, KeyError, IndexError) as e:
        log.warning("weather lookup failed (%s), using a default", e)
        return fallback(loc.get("units", "fahrenheit"))


if __name__ == "__main__":
    import config
    print(get(config.load()))
