# SolarIntel Demo

Streamlit web application for the EMHIRES NASA Solar Energy Prediction project. Provides an interactive dashboard to explore model predictions, compare countries, and walk through the ML pipeline.

## What it does

1. **Prediction Dashboard**: Select a country, time, and weather conditions to get a capacity factor prediction. Shows a 24 hour generation profile, monthly breakdown, radar chart of factor contributions, and an annual heatmap.

2. **Pipeline Explorer**: Step by step walkthrough of the ML pipeline, from data collection to model evaluation. Each step includes a description and an illustrative chart.

3. **Country Comparison**: Compare solar generation potential across multiple EU countries under the same weather conditions. Includes ranked bars, overlaid 24 hour profiles, monthly trends, and a country by month heatmap.

4. **Project Info**: Datasets, tech stack, pipeline modules, and team.

## Setup

```bash
cd Demo_and_Hosting
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## Models

The app loads trained models from the `Final_Pipeline` folder:

| Model | File | Size |
|-------|------|------|
| Linear Regression | solar_model.pkl | ~1.5 KB |
| Random Forest | solar_model_rfr.pkl | ~34 MB |

Both are trained on 3.8 million hourly observations across 29 European countries (2001 to 2015).

## Files

| File | Purpose |
|------|---------|
| app.py | Streamlit application (all tabs, charts, layout) |
| model_loader.py | Loads pkl models and prepares features for prediction |
| requirements.txt | Python dependencies |

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web interface |
| pandas | Data handling |
| numpy | Numerical operations |
| plotly | Charts |
| scikit-learn | Model loading and prediction |
