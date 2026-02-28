# SolarIntel - Solar Energy Generation Prediction

An end-to-end machine learning system that forecasts hourly solar energy output across 29 European countries. The pipeline fuses 15 years of solar capacity factor data from the European Commission's EMHIRES dataset with hourly meteorological observations from NASA's POWER API, trains two regression models, and serves predictions through an interactive web application.

> **Live Demo:** [huggingface.co/spaces/priyanshutomar2024/SolarIntel](https://huggingface.co/spaces/priyanshutomar2024/SolarIntel)

---

## Highlights

| | |
|---|---|
| **Coverage** | 29 European countries, 2001 - 2015, hourly resolution |
| **Dataset** | ~3.8 million rows, 34 features |
| **Models** | Linear Regression (R2 = 0.788) and Random Forest Regressor (R2 = 0.911) |
| **Stack** | Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit |
| **Deployment** | Hugging Face Spaces (Streamlit) |

---

## Getting Started

If you are new to this project, start with the **Walkthrough** folder. It contains everything you need to understand the full pipeline from first principles:

| Resource | Description |
|----------|-------------|
| [`Walkthrough/walkthrough.ipynb`](Walkthrough/walkthrough.ipynb) | Interactive Jupyter notebook - full pipeline with code, explanations, and visualisations |
| [`Walkthrough/walkthrough.pdf`](Walkthrough/walkthrough.pdf) | Standalone PDF document covering every step in detail |
| [`EMHIRES PV Reference`](Walkthrough/emhirespv_gonzalezaparicioetal2017_newtemplate_corrected_last.pdf) | Original EMHIRES methodology paper by Gonzalez Aparicio et al. (2017) |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn (Linear Regression, Random Forest Regressor) |
| Visualisation | matplotlib, seaborn |
| Model Persistence | joblib |
| Web Application | Streamlit |
| Data Source | NASA POWER API, EMHIRES PV (JRC) |
| Hosting | Hugging Face Spaces |

---

## Repository Structure

```
EMHIRES-NASA-SolarPrediction/
├── Walkthrough/              # Start here - full documented walkthrough
├── Pipeline_Modules/         # Modular per-stage scripts
├── Final_Pipeline/           # Single-script end-to-end pipeline
├── NASA_Data_Fetch/          # NASA POWER API data collection
├── Datasets/                 # Raw and processed data (zipped)
├── Models/                   # Trained model files (zipped)
├── Demo_and_Hosting/         # Streamlit app for deployment
└── Report/                   # Project report (LaTeX + PDF)
```

### Walkthrough/
Complete documented walkthrough of the project. Contains the Jupyter notebook (`walkthrough.ipynb`), a formatted PDF version (`walkthrough.pdf`), and the original EMHIRES methodology paper. This is the recommended starting point for anyone looking to understand the pipeline, the data science decisions behind it, and the results.

### Pipeline_Modules/
A modular breakdown of the pipeline into individual scripts. Each subfolder contains a standalone Python file for one stage:

| Module | Stage |
|--------|-------|
| Dataset Visualisation | Raw data exploration and plotting |
| Cleaning and Transformation | Data integrity checks, reshaping, feature extraction |
| Merging | Fusing EMHIRES generation data with NASA weather data |
| Encoding | One-Hot Encoding of country labels |
| Training and Evaluation (LR) | Linear Regression model training and metrics |
| Training and Evaluation (RFR) | Random Forest Regressor training and metrics |
| Analysis Visualisation (LR) | Diagnostic plots for Linear Regression |
| Analysis Visualisation (RFR) | Diagnostic plots for Random Forest |

### Final_Pipeline/
Contains `Final_Pipeline.py` - a single consolidated script that runs the entire pipeline end-to-end in one execution, from data loading through to model export.

### NASA_Data_Fetch/
Contains `fetcher.py`, the script used to collect hourly weather data (Irradiance, Temperature, Wind Speed) from the NASA POWER API for all 29 countries from 2001 to 2015. The fetch takes approximately 20+ minutes due to API volume. The resulting CSV is already provided in the Datasets folder.

### Datasets/
Compressed raw and processed datasets:
- `solar_data.zip` - EMHIRES solar capacity factor CSV and NASA weather master CSV
- `merged_encoded.zip` - Fully merged, cleaned, and encoded dataset ready for training

### Models/
Contains `solar_models.zip` with the two trained model files (`solar_model_lr.pkl` and `solar_model_rfr.pkl`). Load directly with `joblib` for inference without retraining.

### Demo_and_Hosting/
The Streamlit web application deployed on Hugging Face Spaces. Provides real-time predictions, 24-hour generation profiles, feature importance analysis, seasonal patterns, and model performance metrics.

### Report/
Contains the formal project report in both LaTeX source (`Report.tex`) and compiled PDF (`Report.pdf`). Covers data sources, pipeline architecture, feature engineering, model training, evaluation results, deployment, and future work.

---

## Model Performance

| Metric | Linear Regression | Random Forest Regressor |
|--------|-------------------|-------------------------|
| MAE | 0.0534 | 0.0280 |
| RMSE | 0.0845 | 0.0547 |
| R2 | 0.7880 | 0.9112 |

> Evaluated on a held-out 20% test set (762,538 samples). `random_state=67` across all splits for reproducibility.

---

## Roadmap

- [x] Data pipeline (EMHIRES + NASA POWER fusion)
- [x] Model training and evaluation (LR + RFR)
- [x] Interactive Streamlit dashboard
- [x] Deployment on Hugging Face Spaces
- [ ] AI agent-based expansion for intelligent decision support

---

## License

This project uses publicly available data from the [European Commission Joint Research Centre (EMHIRES)](https://setis.ec.europa.eu/european-commission-services/emhires) and the [NASA POWER API](https://power.larc.nasa.gov/).