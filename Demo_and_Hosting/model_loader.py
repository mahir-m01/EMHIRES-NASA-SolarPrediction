"""
Model Loader for Solar Forecasting
Supports Linear Regression and Random Forest models
"""

import pandas as pd
import numpy as np
import pickle
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "Final_Pipeline")

AVAILABLE_MODELS = {
    "Linear Regression": os.path.join(MODEL_DIR, "solar_model.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "solar_model_rfr.pkl"),
}


def load_trained_model(model_path=None):
    """Load a trained model from disk."""
    if model_path is None:
        model_path = AVAILABLE_MODELS["Linear Regression"]

    try:
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception:
            pass

        try:
            import joblib
            model = joblib.load(model_path)
            return model
        except Exception:
            pass

        print(f"Failed to load model from {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def prepare_features(country, hour, month, irradiance, temperature, wind_speed):
    """Prepare feature vector for prediction."""

    countries = [
        "AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "EL",
        "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV",
        "NL", "NO", "PL", "PT", "RO", "SE", "SI", "SK", "UK",
    ]

    country_map = {"GB": "UK", "GR": "EL"}
    mapped_country = country_map.get(country, country)

    features = {
        "Hour": hour,
        "Month": month,
        "Irradiance": irradiance,
        "Temperature": temperature,
        "Wind_Speed": wind_speed,
    }

    for c in countries:
        features[f"Country_{c}"] = 1 if c == mapped_country else 0

    return pd.DataFrame([features])


def predict_capacity_factor(model, country, hour, month, irradiance, temperature, wind_speed):
    """Predict capacity factor using the trained model."""
    if model is None:
        return estimate_capacity_factor(hour, month, irradiance, temperature, wind_speed)

    try:
        features = prepare_features(country, hour, month, irradiance, temperature, wind_speed)
        prediction = model.predict(features)[0]
        return max(0.0, min(1.0, prediction))
    except Exception as e:
        print(f"Prediction error: {e}")
        return estimate_capacity_factor(hour, month, irradiance, temperature, wind_speed)


def estimate_capacity_factor(hour, month, irradiance, temperature, wind_speed):
    """Physics based capacity factor estimation (fallback)."""
    base_cf = (irradiance / 1000.0) * 0.85
    temp_factor = 1 - (abs(temperature - 25) / 150)

    if 6 <= hour <= 18:
        hour_factor = max(0, np.sin((hour - 6) * np.pi / 12))
    else:
        hour_factor = 0

    month_factor = 0.5 + 0.5 * np.sin((month - 3) * np.pi / 6)
    cf = base_cf * temp_factor * hour_factor * month_factor
    return max(0.0, min(1.0, cf))
