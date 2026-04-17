import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_loader import predict_capacity_factor
from weather.fetcher import fetch_weather_forecast
from state import GridAdvisorState


def forecast_node(state: GridAdvisorState) -> dict:
    country = state["country"]
    weather = fetch_weather_forecast(country)

    today = datetime.date.today()
    month = today.month
    forecast_date = str(today + datetime.timedelta(days=1))

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
