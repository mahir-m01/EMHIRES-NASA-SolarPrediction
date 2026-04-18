"""SolarIntel — Energy Generation Analytics"""

import os
import streamlit as st
from dotenv import load_dotenv
import sys

# Ensure current directory is in sys path so tabs module can load utils
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

load_dotenv()
from model_loader import load_trained_model, AVAILABLE_MODELS
from utils import COUNTRIES
import tabs.prediction_dashboard as prediction_dashboard
import tabs.country_comparison as country_comparison
import tabs.grid_advisor as grid_advisor

# ---- page config ----
st.set_page_config(page_title="SolarIntel", page_icon="☀️", layout="wide")

# ---- load model ----
@st.cache_resource
def get_model(model_name):
    return load_trained_model(model_name)

# ---- sidebar ----
st.sidebar.header("☀️ SolarIntel")
st.sidebar.caption("Milestone 1 & 2 — ML Forecasting + Agentic AI")
st.sidebar.markdown("---")

# Model selection
st.sidebar.header("Model")
model_choice = st.sidebar.selectbox(
    "Algorithm", list(AVAILABLE_MODELS.keys()), index=1,
    help="Random Forest captures non-linear patterns."
)
model = get_model(model_choice)

st.sidebar.markdown("---")

st.sidebar.header("Input Parameters")
country_code = st.sidebar.selectbox(
    "Region", list(COUNTRIES.keys()),
    index=list(COUNTRIES.keys()).index("ES"),
    format_func=lambda x: f"{x} — {COUNTRIES[x]}"
)

month = st.sidebar.slider("Month", 1, 12, 6)
hour = st.sidebar.slider("Hour (UTC)", 0, 23, 12)

st.sidebar.markdown("---")
st.sidebar.header("Weather Conditions")
irradiance = st.sidebar.slider("Solar Irradiance (W/m²)", 0, 1000, 600, step=10,
                                help="Global Horizontal Irradiance")
temperature = st.sidebar.slider("Temperature (°C)", -20, 50, 25,
                                 help="Surface air temperature at 2m height")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0, step=0.5,
                                help="Wind speed at 10m height")

st.sidebar.markdown("---")
st.sidebar.header("System")
installed_capacity = st.sidebar.number_input("Installed Capacity (kW)", value=100.0, min_value=1.0, step=10.0)

st.sidebar.markdown("---")
if model is not None:
    st.sidebar.success(f"{model_choice} loaded")
else:
    st.sidebar.warning("Model not found — using physics fallback")


# ---- main area ----
st.title("☀️ SolarIntel")
st.caption(f"Solar Energy Generation Forecasting | {model_choice}")

tab_forecast, tab_compare, tab_advisor = st.tabs(
    ["Prediction Dashboard", "Country Comparison", "Grid Advisor"]
)

# TAB 1: PREDICTION DASHBOARD
with tab_forecast:
    prediction_dashboard.render(
        model, country_code, hour, month, irradiance, temperature, wind_speed, installed_capacity
    )

# TAB 2: COUNTRY COMPARISON
with tab_compare:
    country_comparison.render(
        model, country_code, hour, month, irradiance, temperature, wind_speed, installed_capacity
    )

# TAB 3: GRID ADVISOR
with tab_advisor:
    grid_advisor.render(installed_capacity)

# End of app
