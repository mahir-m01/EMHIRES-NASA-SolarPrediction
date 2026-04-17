from typing import TypedDict, Optional


class GridAdvisorState(TypedDict):
    country: str
    capacity_kw: float
    model_name: str
    forecast_date: str
    weather_forecast: list
    cf_value: float
    hourly_profile: list
    monthly_profile: list
    risk_summary: dict
    risk_flags: list
    retrieved_chunks: list
    final_recommendations: dict
    error: Optional[str]
