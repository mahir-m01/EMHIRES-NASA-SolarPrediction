# SolarIntel - Solar Energy Generation Prediction

An end-to-end machine learning and agentic AI system that forecasts hourly solar energy output across 29 European countries. The pipeline fuses 15 years of solar capacity factor data from the European Commission's EMHIRES dataset with hourly meteorological observations from NASA's POWER API, trains two regression models, and serves predictions through an interactive web application with an AI-powered grid advisory layer.

> **Live Demo:** [huggingface.co/spaces/priyanshutomar2024/SolarIntel](https://huggingface.co/spaces/priyanshutomar2024/SolarIntel)

---

## Highlights

| | |
|---|---|
| **Coverage** | 29 European countries, 2001 - 2015, hourly resolution |
| **Dataset** | ~3.8 million rows, 34 features |
| **Models** | Linear Regression (R2 = 0.788) and Random Forest Regressor (R2 = 0.911) |
| **Agent** | 4-node LangGraph pipeline with RAG, risk analysis, and LLM recommendations |
| **Stack** | Python, pandas, scikit-learn, LangGraph, LangChain, Chroma, Streamlit |
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
| Agentic AI | LangGraph, LangChain, LangChain-OpenAI |
| LLM | OpenRouter (openrouter/free — automatic free model routing) |
| Vector Store | Chroma + sentence-transformers/all-MiniLM-L6-v2 (local embeddings) |
| Weather Forecast | Open-Meteo API (free, no key required, up to 15 days ahead) |
| Visualisation | matplotlib, seaborn, Plotly |
| Model Persistence | joblib |
| Web Application | Streamlit |
| Data Source | NASA POWER API, EMHIRES PV (JRC) |
| Hosting | Hugging Face Spaces |

---

## Repository Structure

```
EMHIRES-NASA-SolarPrediction/
├── Agent_Pipeline/           # Agentic AI layer (Milestone 2)
├── Walkthrough/              # Start here - full documented walkthrough
├── Pipeline_Modules/         # Modular per-stage scripts
├── Final_Pipeline/           # Single-script end-to-end pipeline
├── NASA_Data_Fetch/          # NASA POWER API data collection
├── Datasets/                 # Raw and processed data (zipped)
├── Models/                   # Trained model files (zipped)
├── Demo_and_Hosting/         # Streamlit app for deployment
└── Report/                   # Project report (LaTeX + PDF)
```

### Agent_Pipeline/

A 4-node LangGraph pipeline that sits on top of the ML models and provides AI-driven grid management recommendations.

```
Agent_Pipeline/
├── state.py                  # GridAdvisorState TypedDict
├── graph.py                  # LangGraph StateGraph wiring
├── model_loader.py           # RFR model loader and feature builder
├── run_agent.py              # CLI entry point
├── 1_forecast/               # Node 1: fetches Open-Meteo weather, builds 24h CF profile
├── 2_risk/                   # Node 2: variability scoring and risk flag generation
├── 3_rag/                    # Node 3: Chroma vector store + retrieval
├── 4_recommendations/        # Node 4: OpenRouter LLM structured JSON output
├── knowledge_base/           # 4 plain-text grid management reference documents
├── models/                   # solar_model_rfr.pkl
└── weather/                  # Open-Meteo fetcher with 29-country coordinates
```

**Node flow:** Forecast -> Risk Analysis -> RAG Retrieval -> LLM Recommendations

**Output schema:**
```json
{
  "forecast_summary": "...",
  "risk_periods": [{"period": "...", "risk": "...", "severity": "high|medium|low"}],
  "strategies": [{"title": "...", "description": "...", "source": "retrieved|general"}],
  "responsible_ai_note": "..."
}
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

## Environment Setup

The ML pipeline and Streamlit app require no API keys. The Agent_Pipeline requires one key for the LLM node.

Create a `.env` file in `Agent_Pipeline/` (never commit this):

```
OPENROUTER_API_KEY=your_key_here
```

Get a free key at [openrouter.ai](https://openrouter.ai). The pipeline uses `openrouter/free` which automatically routes to whichever free model is available — no specific model needs to be selected or managed.

Install dependencies:

```bash
cd Agent_Pipeline
pip install -r requirements.txt
python run_agent.py
```

---

## Roadmap

- [x] Data pipeline (EMHIRES + NASA POWER fusion)
- [x] Model training and evaluation (LR + RFR)
- [x] Interactive Streamlit dashboard
- [x] Deployment on Hugging Face Spaces
- [x] Agentic AI layer — LangGraph pipeline with RAG and LLM grid recommendations

---

## Project Demo Video

Watch the full project demonstration here (phase-1):

[Video](https://drive.google.com/file/d/1PCLF0SfjMGW_zTBYoztBqh-fYx7ANlcm/view?usp=drive_link)


Watch the full project demonstration here (phase-2):

[Video](https://drive.google.com/file/d/1U1suAvY5Wzm5zf07u5_wTENIJ3HNzJk9/view?usp=sharing)

---

## License

This project uses publicly available data from the [European Commission Joint Research Centre (EMHIRES)](https://setis.ec.europa.eu/european-commission-services/emhires) and the [NASA POWER API](https://power.larc.nasa.gov/).
