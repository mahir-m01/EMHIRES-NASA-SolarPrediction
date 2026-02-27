# Solar Energy Forecasting - Streamlit Demo

Interactive Streamlit-based GUI for visualising solar energy forecasting results using the trained Linear Regression and Random Forest Regressor models.

## Features

- Real-time solar energy predictions with interactive controls
- 24-hour generation profile with sunrise/sunset markers
- Feature importance analysis
- Seasonal pattern visualisation
- Model performance metrics (MAE, RMSE, R2)
- Educational walkthrough of the ML pipeline

## Installation

```bash
cd Demo_and_Hosting
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Files

- `app.py` - Main Streamlit application with all tabs and visualisations
- `model_loader.py` - Utility module for loading trained models and generating predictions
- `requirements.txt` - Python dependencies
- `.gitignore` - Files excluded from version control

## Deployment

This app can be deployed on:
- Streamlit Community Cloud (recommended, free)
- Hugging Face Spaces
- Render / Railway
