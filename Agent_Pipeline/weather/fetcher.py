import requests

COUNTRY_COORDS = {
    'AT': {'lat': 47.51, 'lon': 14.55},
    'BE': {'lat': 50.50, 'lon': 4.47},
    'BG': {'lat': 42.73, 'lon': 25.48},
    'CH': {'lat': 46.81, 'lon': 8.22},
    'CY': {'lat': 35.12, 'lon': 33.42},
    'CZ': {'lat': 49.81, 'lon': 15.47},
    'DE': {'lat': 51.16, 'lon': 10.45},
    'DK': {'lat': 56.26, 'lon': 9.50},
    'EE': {'lat': 58.59, 'lon': 25.01},
    'EL': {'lat': 39.07, 'lon': 21.82},
    'ES': {'lat': 40.46, 'lon': -3.74},
    'FI': {'lat': 61.92, 'lon': 25.74},
    'FR': {'lat': 46.22, 'lon': 2.21},
    'HR': {'lat': 45.10, 'lon': 15.20},
    'HU': {'lat': 47.16, 'lon': 19.50},
    'IE': {'lat': 53.41, 'lon': -8.24},
    'IT': {'lat': 41.87, 'lon': 12.56},
    'LT': {'lat': 55.16, 'lon': 23.88},
    'LU': {'lat': 49.81, 'lon': 6.12},
    'LV': {'lat': 56.87, 'lon': 24.60},
    'NL': {'lat': 52.13, 'lon': 5.29},
    'NO': {'lat': 60.47, 'lon': 8.46},
    'PL': {'lat': 51.91, 'lon': 19.14},
    'PT': {'lat': 39.39, 'lon': -8.22},
    'RO': {'lat': 45.94, 'lon': 24.96},
    'SE': {'lat': 60.12, 'lon': 18.64},
    'SI': {'lat': 46.15, 'lon': 14.99},
    'SK': {'lat': 48.66, 'lon': 19.69},
    'UK': {'lat': 55.37, 'lon': -3.43},
}

MAX_FORECAST_DAYS = 15  # Open-Meteo max is 16 days total; offset 0=today, max offset=15


def fetch_weather_forecast(country_code: str, day_offset: int = 1) -> list:
    """
    Fetch 24-hour weather for a specific forecast day.

    day_offset: days from today (1 = tomorrow, 2 = day after, ... 15 = max)
    Returns a list of 24 dicts with irradiance, temperature, wind_speed.
    """
    if not 1 <= day_offset <= MAX_FORECAST_DAYS:
        raise ValueError(f"day_offset must be between 1 and {MAX_FORECAST_DAYS}, got {day_offset}.")

    coords = COUNTRY_COORDS.get(country_code)
    if coords is None:
        raise RuntimeError(f"Unknown country code '{country_code}'.")

    lat, lon = coords["lat"], coords["lon"]
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=shortwave_radiation,temperature_2m,wind_speed_10m"
        f"&wind_speed_unit=ms&forecast_days=16&timezone=auto"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Open-Meteo timed out for '{country_code}'.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Open-Meteo request failed for '{country_code}': {exc}")

    try:
        hourly = data["hourly"]
        irr = hourly["shortwave_radiation"]
        tmp = hourly["temperature_2m"]
        wnd = hourly["wind_speed_10m"]
    except KeyError as exc:
        raise RuntimeError(f"Unexpected Open-Meteo response structure: missing key {exc}")

    start = day_offset * 24  # offset 1 → hours 24-47 (tomorrow)
    end = start + 24
    if len(irr) < end:
        raise RuntimeError(f"Open-Meteo returned fewer hours than expected for day_offset={day_offset}.")

    return [
        {
            "irradiance": float(irr[h]) if irr[h] is not None else 0.0,
            "temperature": float(tmp[h]) if tmp[h] is not None else 0.0,
            "wind_speed": float(wnd[h]) if wnd[h] is not None else 0.0,
        }
        for h in range(start, end)
    ]
