import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent_model_loader import predict_capacity_factor
from weather.fetcher import fetch_weather_forecast, MAX_FORECAST_DAYS
from state import GridAdvisorState


def forecast_node(state: GridAdvisorState) -> dict:
    country = state["country"]
    today = datetime.date.today()

    # Use forecast_date from state if provided, otherwise default to tomorrow
    if state.get("forecast_date"):
        target = datetime.date.fromisoformat(state["forecast_date"])
        day_offset = (target - today).days
        if not 1 <= day_offset <= MAX_FORECAST_DAYS:
            raise ValueError(
                f"forecast_date must be 1–{MAX_FORECAST_DAYS} days from today (Open-Meteo limit). Got {day_offset}."
            )
    else:
        day_offset = 1
        target = today + datetime.timedelta(days=1)

    forecast_date = str(target)
    month = target.month

    weather = fetch_weather_forecast(country, day_offset=day_offset)

    hourly_profile = [
        predict_capacity_factor(
            country, h, month,
            weather[h]["irradiance"],
            weather[h]["temperature"],
            weather[h]["wind_speed"],
        )
        for h in range(24)
    ]

    cf_value = max(hourly_profile)
    peak_hour = hourly_profile.index(cf_value)
    peak_weather = weather[peak_hour]

    monthly_profile = [
        predict_capacity_factor(
            country, peak_hour, m,
            peak_weather["irradiance"],
            peak_weather["temperature"],
            peak_weather["wind_speed"],
        )
        for m in range(1, 13)
    ]

    return {
        "cf_value": cf_value,
        "hourly_profile": hourly_profile,
        "monthly_profile": monthly_profile,
        "weather_forecast": weather,
        "forecast_date": forecast_date,
    }
