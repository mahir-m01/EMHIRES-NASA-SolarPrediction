"""
Model Loader for Solar Forecasting
Supports Linear Regression and Random Forest models.
Reads models directly from the zipped archive in ../Models/ so that
the Models folder stays completely untouched.
"""

import pandas as pd
import numpy as np
import pickle
import zipfile
import io
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_ZIP = os.path.join(_BASE_DIR, "..", "Models", "solar_models.zip")

# Filenames inside the zip archive
AVAILABLE_MODELS = {
    "Linear Regression": "solar_model_lr.pkl",
    "Random Forest": "solar_model_rfr.pkl",
}


def load_trained_model(model_name=None):
    """
    Load a trained model directly from the Models/solar_models.zip archive.
    No files are extracted to disk — everything is read into memory.

    Parameters:
    -----------
    model_name : str or None
        Key from AVAILABLE_MODELS (e.g. "Linear Regression").
        If None, defaults to "Linear Regression".

    Returns:
    --------
    model : sklearn model or None
    """
    if model_name is None:
        model_name = "Linear Regression"

    # Resolve the pkl filename inside the zip
    pkl_filename = AVAILABLE_MODELS.get(model_name, model_name)

    zip_path = os.path.normpath(_MODELS_ZIP)

    if not os.path.exists(zip_path):
        print(f"Models archive not found at {zip_path}")
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            if pkl_filename not in zf.namelist():
                print(f"'{pkl_filename}' not found inside {zip_path}")
                print(f"  Available: {zf.namelist()}")
                return None

            data = zf.read(pkl_filename)

            # Try pickle first
            try:
                model = pickle.loads(data)
                return model
            except Exception:
                pass

            # Fallback to joblib via BytesIO
            try:
                import joblib
                model = joblib.load(io.BytesIO(data))
                return model
            except Exception:
                pass

        print(f"Failed to deserialise '{pkl_filename}'")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def prepare_features(country, hour, month, irradiance, temperature, wind_speed):
    """
    Prepare feature vector for prediction

    Parameters:
    -----------
    country : str
        Country code (e.g., 'ES', 'DE')
    hour : int
        Hour of day (0-23)
    month : int
        Month (1-12)
    irradiance : float
        Solar irradiance (W/m²)
    temperature : float
        Temperature (°C)
    wind_speed : float
        Wind speed (m/s)

    Returns:
    --------
    pd.DataFrame
        Feature vector ready for model.predict()
    """

    # All countries in the trained model (must match model's feature_names_in_)
    # Note: UK=United Kingdom (not GB), EL=Greece (not GR), CY=Cyprus
    countries = [
        "AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "EL",
        "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV",
        "NL", "NO", "PL", "PT", "RO", "SE", "SI", "SK", "UK",
    ]

    # Map common aliases to model's expected codes
    country_map = {"GB": "UK", "GR": "EL"}
    mapped_country = country_map.get(country, country)

    # Create base features
    features = {
        "Hour": hour,
        "Month": month,
        "Irradiance": irradiance,
        "Temperature": temperature,
        "Wind_Speed": wind_speed,
    }

    # One-hot encode country
    for c in countries:
        features[f"Country_{c}"] = 1 if c == mapped_country else 0

    # Convert to DataFrame
    df = pd.DataFrame([features])

    return df


def predict_capacity_factor(
    model, country, hour, month, irradiance, temperature, wind_speed
):
    """
    Predict capacity factor using the trained model

    Parameters:
    -----------
    model : sklearn model
        Trained Linear Regression model
    country : str
        Country code
    hour : int
        Hour of day
    month : int
        Month
    irradiance : float
        Solar irradiance
    temperature : float
        Temperature
    wind_speed : float
        Wind speed

    Returns:
    --------
    float
        Predicted capacity factor (0.0 to 1.0)
    """

    if model is None:
        # Fallback to physics-based estimation
        return estimate_capacity_factor(
            hour, month, irradiance, temperature, wind_speed
        )

    try:
        features = prepare_features(
            country, hour, month, irradiance, temperature, wind_speed
        )
        prediction = model.predict(features)[0]

        # Clip to valid range
        return max(0.0, min(1.0, prediction))

    except Exception as e:
        print(f"Prediction error: {e}")
        return estimate_capacity_factor(
            hour, month, irradiance, temperature, wind_speed
        )


def estimate_capacity_factor(hour, month, irradiance, temperature, wind_speed):
    """
    Physics-based capacity factor estimation (fallback)

    Parameters:
    -----------
    hour : int
        Hour of day
    month : int
        Month
    irradiance : float
        Solar irradiance
    temperature : float
        Temperature
    wind_speed : float
        Wind speed

    Returns:
    --------
    float
        Estimated capacity factor
    """

    # Base efficiency from irradiance
    base_cf = (irradiance / 1000.0) * 0.85

    # Temperature derating (optimal at 25°C)
    temp_factor = 1 - (abs(temperature - 25) / 150)

    # Hour factor (solar angle)
    if 6 <= hour <= 18:
        hour_factor = max(0, np.sin((hour - 6) * np.pi / 12))
    else:
        hour_factor = 0

    # Seasonal factor
    month_factor = 0.5 + 0.5 * np.sin((month - 3) * np.pi / 6)

    # Combined capacity factor
    cf = base_cf * temp_factor * hour_factor * month_factor

    return max(0.0, min(1.0, cf))
