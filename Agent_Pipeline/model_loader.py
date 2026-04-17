import os
import pickle
import io
import numpy as np
import pandas as pd

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "solar_model_rfr.pkl")

_COUNTRIES = [
    "AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "EL",
    "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV",
    "NL", "NO", "PL", "PT", "RO", "SE", "SI", "SK", "UK",
]
_COUNTRY_ALIASES = {"GB": "UK", "GR": "EL"}

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    try:
        with open(_MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
        return _model_cache
    except Exception:
        try:
            import joblib
            _model_cache = joblib.load(_MODEL_PATH)
            return _model_cache
        except Exception as e:
            raise RuntimeError(f"Failed to load RFR model from {_MODEL_PATH}: {e}")


def predict_capacity_factor(country, hour, month, irradiance, temperature, wind_speed):
    model = load_model()
    country = _COUNTRY_ALIASES.get(country, country)
    features = {
        "Hour": hour, "Month": month,
        "Irradiance": irradiance, "Temperature": temperature, "Wind_Speed": wind_speed,
    }
    for c in _COUNTRIES:
        features[f"Country_{c}"] = 1 if c == country else 0
    df = pd.DataFrame([features])
    result = model.predict(df)[0]
    return float(max(0.0, min(1.0, result)))
