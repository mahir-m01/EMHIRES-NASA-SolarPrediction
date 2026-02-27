# EMHIRES-NASA Solar Energy Prediction

A machine learning pipeline that predicts solar energy generation (capacity factor) across 29 European countries using historical solar output data from the EMHIRES dataset and hourly meteorological data from the NASA POWER API. Two models are trained and compared: Linear Regression and Random Forest Regressor.

---

## **Getting Started**

The best place to start is the **Walkthrough** folder. It contains a detailed Jupyter notebook that walks through the entire pipeline step by step, from loading the raw data to training, evaluating, and exporting both models. Every major decision and operation is documented with explanations.

- [Walkthrough/walkthrough.ipynb](Walkthrough/walkthrough.ipynb) - Full step-by-step pipeline walkthrough with notes and analysis.
- The EMHIRES reference PDF is also included in the same folder for further reading and understanding of the source dataset and its methodology.

---

## **Repository Structure**

### **Walkthrough/**
Contains the complete documented walkthrough notebook. Start here to understand what the project does, how the data flows, and what each step accomplishes. The notebook is self-contained and covers data loading, cleaning, merging, encoding, model training, evaluation, visualisation, and custom predictions.

### **Pipeline_Modules/**
A modular breakdown of the pipeline into individual scripts. Each subfolder contains a standalone Python file for one stage of the pipeline:
- Dataset Visualisation
- Cleaning and Transformation
- Merging
- Encoding
- Training and Evaluation (Linear Regression)
- Training and Evaluation (Random Forest Regressor)
- Analysis Visualisation (Linear Regression)
- Analysis Visualisation (Random Forest Regressor)

Use this folder if you want to understand or modify a specific stage in isolation.

### **Final_Pipeline/**
Contains `Final_Pipeline.py`, a single consolidated script that runs the entire pipeline end-to-end in one execution. This is the production-ready version that performs all steps from data loading through to model export.

### **NASA_Data_Fetch/**
Contains the `fetcher.py` script used to collect hourly weather data (Irradiance, Temperature, Wind Speed) from the NASA POWER API for all 29 EMHIRES countries from 2001 to 2015. This script takes approximately 20+ minutes to run due to the volume of API requests. The resulting CSV is already provided in the Datasets folder, so you do not need to run this unless you want to re-fetch the data.

### **Datasets/**
Contains the raw and processed datasets in compressed format:
- `solar_data.zip` - The EMHIRES solar capacity factor CSV and the NASA weather master CSV.
- `merged_encoded.zip` - The fully merged, cleaned, and one-hot-encoded dataset ready for model training.

### **Models/**
Contains `solar_models.zip` with the two trained model files:
- `solar_model_lr.pkl` - Trained Linear Regression model.
- `solar_model_rfr.pkl` - Trained Random Forest Regressor model.

These can be loaded directly with `joblib` for predictions without retraining.

### **Demo_and_Hosting/**
Contains the interactive Streamlit-based GUI for visualising solar energy forecasting results. The app provides real-time predictions, 24-hour generation profiles, feature importance analysis, seasonal patterns, and model performance metrics. See the folder's README for setup and deployment instructions.