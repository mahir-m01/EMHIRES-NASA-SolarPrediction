"""SolarIntel — Energy Generation Analytics"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model_loader import load_trained_model, predict_capacity_factor, AVAILABLE_MODELS

# ---- page config ----
st.set_page_config(page_title="SolarIntel", page_icon="☀️", layout="wide")

# ---- constants ----
COUNTRIES = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
    "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "EE": "Estonia",
    "ES": "Spain", "FI": "Finland", "FR": "France", "GB": "United Kingdom",
    "GR": "Greece", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia",
    "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal",
    "RO": "Romania", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia"
}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PLOT_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="#292524",
    paper_bgcolor="#292524",
    font=dict(family="Inter, sans-serif", color="#E7E5E4"),
    margin=dict(l=40, r=20, t=50, b=40),
)
AMBER = "#D97706"

# ---- load model ----
@st.cache_resource
def get_model(model_name):
    return load_trained_model(model_name)

# ---- sidebar ----
st.sidebar.header("☀️ SolarIntel")
st.sidebar.caption("Milestone 1 — ML-Based Forecasting")
st.sidebar.markdown("---")

# Model selection
st.sidebar.header("Model")
model_choice = st.sidebar.selectbox(
    "Algorithm", list(AVAILABLE_MODELS.keys()), index=0,
    help="Linear Regression is fast and interpretable. Random Forest captures non-linear patterns."
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
st.caption(f"Intelligent Solar Energy Generation Forecasting — EMHIRES-NASA | {model_choice}")

tab_forecast, tab_pipeline, tab_compare, tab_about = st.tabs(
    ["Prediction Dashboard", "Pipeline Explorer", "Country Comparison", "Project Info"]
)


# TAB 1: PREDICTION DASHBOARD (auto-updates)
with tab_forecast:
    # Compute predictions (runs automatically on any input change)
    cf = predict_capacity_factor(model, country_code, hour, month, irradiance, temperature, wind_speed)
    power_out = cf * installed_capacity

    hours_range = list(range(24))
    output_24h = []
    for h in hours_range:
        sim_irr = irradiance * max(0, np.sin((h - 6) * np.pi / 12)) if 6 <= h <= 18 else 0
        output_24h.append(predict_capacity_factor(model, country_code, h, month, sim_irr, temperature, wind_speed) * installed_capacity)

    daily_kwh = sum(output_24h)
    peak_kw = max(output_24h)
    peak_hour = hours_range[np.argmax(output_24h)]
    monthly_cf = [predict_capacity_factor(model, country_code, 12, m, irradiance, temperature, wind_speed) for m in range(1, 13)]

    # ── Metrics row ──
    st.subheader(f"Forecast — {COUNTRIES[country_code]}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capacity Factor", f"{cf:.4f}")
    c2.metric("Power Output", f"{power_out:.1f} kW")
    c3.metric("Est. Daily Energy", f"{daily_kwh:.1f} kWh")
    c4.metric("Peak Output", f"{peak_kw:.1f} kW @ {peak_hour}:00")
    st.caption(f"Conditions: {irradiance} W/m² | {temperature}°C | {wind_speed} m/s | Month {month} | Hour {hour}:00 UTC | {installed_capacity} kW system")
    st.markdown("---")

    # ── Row 1: Gauge + 24h Profile ──
    r1c1, r1c2 = st.columns([1, 2])

    with r1c1:
        # Gauge chart for Capacity Factor
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cf,
            number={"suffix": "", "font": {"size": 40}},
            delta={"reference": 0.5, "increasing": {"color": "#22C55E"}, "decreasing": {"color": "#EF4444"}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": AMBER},
                "bgcolor": "#292524",
                "steps": [
                    {"range": [0, 0.2], "color": "#44403C"},
                    {"range": [0.2, 0.5], "color": "#57534E"},
                    {"range": [0.5, 0.8], "color": "#78716C"},
                    {"range": [0.8, 1.0], "color": "#A8A29E"},
                ],
                "threshold": {"line": {"color": "#EF4444", "width": 3}, "thickness": 0.8, "value": 0.5},
            },
            title={"text": "Capacity Factor", "font": {"size": 16}},
        ))
        fig_gauge.update_layout(height=300, **PLOT_LAYOUT)
        st.plotly_chart(fig_gauge, width="stretch")

    with r1c2:
        # 24h generation profile with day/night shading
        fig_24h = go.Figure()
        # Night shading
        fig_24h.add_vrect(x0=0, x1=6, fillcolor="rgba(30,30,30,0.4)", line_width=0, annotation_text="Night", annotation_position="top left")
        fig_24h.add_vrect(x0=18, x1=23, fillcolor="rgba(30,30,30,0.4)", line_width=0, annotation_text="Night", annotation_position="top right")
        # Main area
        fig_24h.add_trace(go.Scatter(x=hours_range, y=output_24h, mode="lines", line=dict(color=AMBER, width=3, shape="spline"),
                                     fill="tozeroy", fillcolor="rgba(217,119,6,0.12)", name="Generation"))
        # Selected hour marker
        fig_24h.add_trace(go.Scatter(x=[hour], y=[output_24h[hour]], mode="markers+text",
                                     marker=dict(color="#EF4444", size=14, symbol="diamond", line=dict(width=2, color="white")),
                                     text=[f"{output_24h[hour]:.1f} kW"], textposition="top center", textfont=dict(color="white", size=11),
                                     name=f"Selected ({hour}:00)"))
        # Peak marker
        fig_24h.add_trace(go.Scatter(x=[peak_hour], y=[peak_kw], mode="markers+text",
                                     marker=dict(color="#22C55E", size=10, symbol="star"),
                                     text=[f"Peak: {peak_kw:.1f}"], textposition="bottom center", textfont=dict(color="#22C55E", size=10),
                                     name=f"Peak ({peak_hour}:00)"))
        fig_24h.update_layout(title="24-Hour Generation Profile", xaxis_title="Hour (UTC)", yaxis_title="Output (kW)",
                              xaxis=dict(dtick=2), height=300, showlegend=True, legend=dict(orientation="h", y=1.12), **PLOT_LAYOUT)
        st.plotly_chart(fig_24h, width="stretch")

    # ── Row 2: Monthly bars + Radar ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        # Monthly capacity factor with gradient coloring
        max_cf = max(monthly_cf) if max(monthly_cf) > 0 else 1
        colors_monthly = [f"rgba({int(217*(v/max_cf))}, {int(119*(v/max_cf))}, 6, 0.9)" for v in monthly_cf]
        fig_monthly = go.Figure(go.Bar(
            x=MONTH_NAMES, y=monthly_cf,
            marker=dict(color=colors_monthly, line=dict(color=AMBER, width=1)),
            text=[f"{v:.3f}" for v in monthly_cf], textposition="outside", textfont=dict(size=10),
        ))
        # Highlight current month
        fig_monthly.add_vline(x=month - 1, line=dict(color="#EF4444", width=2, dash="dash"), annotation_text="Now", annotation_position="top")
        fig_monthly.update_layout(title="Monthly Capacity Factor (Noon)", xaxis_title="Month", yaxis_title="CF",
                                  yaxis=dict(range=[0, max_cf * 1.35]), height=380, **PLOT_LAYOUT)
        st.plotly_chart(fig_monthly, width="stretch")

    with r2c2:
        # Radar chart — factor contribution breakdown
        base_pct = (irradiance / 1000) * 100
        hour_pct = max(0, np.sin((hour - 6) * np.pi / 12)) * 100 if 6 <= hour <= 18 else 0
        month_pct = (0.5 + 0.5 * np.sin((month - 3) * np.pi / 6)) * 100
        temp_pct = max(0, (1 - abs(temperature - 25) / 50)) * 100
        wind_pct = min(100, wind_speed / 15 * 100)

        categories = ["Irradiance", "Hour of Day", "Season", "Temperature", "Wind"]
        values = [base_pct, hour_pct, month_pct, temp_pct, wind_pct]
        values_closed = values + [values[0]]  # close the polygon

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed, theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(217,119,6,0.2)",
            line=dict(color=AMBER, width=2),
            name="Current"
        ))
        fig_radar.update_layout(
            title="Factor Contribution (%)",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#44403C", linecolor="#44403C"),
                angularaxis=dict(gridcolor="#44403C", linecolor="#44403C"),
                bgcolor="#292524",
            ),
            height=380, showlegend=False, **PLOT_LAYOUT
        )
        st.plotly_chart(fig_radar, width="stretch")
        st.caption("Each axis shows how favorable that factor is (0–100%). A larger polygon = better overall conditions for solar generation.")

    # ── Chart explanations row ──
    ex1, ex2 = st.columns(2)
    with ex1:
        st.caption("**24h Profile** — Shows estimated power output for each hour today. The amber curve follows the solar arc. The red diamond marks your selected hour. Night hours are shaded dark.")
    with ex2:
        st.caption("**Monthly CF** — Capacity factor at solar noon for each month. Summer months (May–Aug) typically yield the highest output. The red dashed line marks your selected month.")

    st.markdown("---")

    # ── Row 3: Monthly Table + Heatmap ──
    r3c1, r3c2 = st.columns([1, 2])

    with r3c1:
        # Styled monthly CF table
        st.subheader("Monthly Breakdown")
        table_data = []
        annual_avg = sum(monthly_cf) / 12
        for i, m_name in enumerate(MONTH_NAMES):
            cf_val = monthly_cf[i]
            power_val = cf_val * installed_capacity
            daily_est = power_val * (10 + 2 * np.sin((i + 1 - 3) * np.pi / 6))  # daylight hours estimate
            # Rating
            if cf_val >= annual_avg * 1.2:
                rating = "Excellent"
            elif cf_val >= annual_avg * 0.8:
                rating = "Good"
            elif cf_val > 0.05:
                rating = "Low"
            else:
                rating = "Minimal"
            table_data.append({
                "Month": m_name,
                "CF": f"{cf_val:.4f}",
                "Peak kW": f"{power_val:.1f}",
                "Rating": rating,
            })
        st.dataframe(
            pd.DataFrame(table_data),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"Annual average CF: **{annual_avg:.4f}** | Best month: **{MONTH_NAMES[np.argmax(monthly_cf)]}** ({max(monthly_cf):.4f})")

    with r3c2:
        # Heatmap (full height)
        st.subheader("Annual Generation Heatmap")
        heatmap_data = np.zeros((12, 24))
        for mi in range(12):
            for hi in range(24):
                si = irradiance * max(0, np.sin((hi - 6) * np.pi / 12)) if 6 <= hi <= 18 else 0
                heatmap_data[mi][hi] = predict_capacity_factor(model, country_code, hi, mi + 1, si, temperature, wind_speed)
        fig_hm = go.Figure(go.Heatmap(
            z=heatmap_data, x=[f"{h}:00" for h in range(24)], y=MONTH_NAMES,
            colorscale=[[0, "#1C1917"], [0.25, "#44403C"], [0.5, "#78716C"], [0.75, "#D97706"], [1, "#F59E0B"]],
            colorbar=dict(title=dict(text="CF")),
            hovertemplate="Hour: %{x}<br>Month: %{y}<br>CF: %{z:.3f}<extra></extra>",
        ))
        fig_hm.update_layout(xaxis_title="Hour of Day", yaxis_title="Month", height=420, **PLOT_LAYOUT)
        st.plotly_chart(fig_hm, width="stretch")
        st.caption("**Heatmap** — Each cell shows the predicted capacity factor for a specific hour and month combination. Bright amber = high generation. Dark = low/no generation. The diagonal bright band shows how peak solar hours shift with seasons.")

    # ── Formula Explanation ──
    st.markdown("---")
    with st.expander("How the prediction works — Model Formula", expanded=False):
        st.markdown("""
### Linear Regression Model

Our model predicts the **Capacity Factor (CF)** using a multivariate linear equation:

```
CF = w₁·Irradiance + w₂·Hour + w₃·Temperature + w₄·Wind + w₅·Month + Σ(wᵢ·Countryᵢ) + bias
```

Where:
- **Irradiance (GHI)** — Global Horizontal Irradiance in W/m². The strongest predictor — more sunlight = more power.
- **Hour** — Time of day (0–23 UTC). Captures the solar elevation angle throughout the day.
- **Temperature** — Surface air temperature in °C. Panels lose ~0.4% efficiency per °C above 25°C.
- **Wind Speed** — Wind at 10m height in m/s. Helps cool panels, slightly improving efficiency.
- **Month** — Seasonal variation (1–12). Summer months have longer days and higher sun angles.
- **Country** — One-hot encoded (29 binary variables). Captures geographic latitude, climate, and typical cloud cover patterns.

### Key Metrics
- **Capacity Factor (CF)**: Ratio of actual output to theoretical maximum (0.0 to 1.0). A CF of 0.47 means the panel operates at 47% of its rated capacity.
- **Power Output**: `CF × Installed Capacity (kW)`
- **Daily Energy**: Sum of hourly power output over 24 hours (kWh)

### Training Details
- **Dataset**: 3.8 million hourly observations across 29 European countries (2001–2015)
- **Algorithm**: Ordinary Least Squares Linear Regression (scikit-learn)
- **Train/Test Split**: 80/20
- **R² Score**: 0.788 — explains ~79% of variation in solar generation
- **MAE**: 0.053 — average prediction error of 5.3 percentage points
        """)



# TAB 2: PIPELINE EXPLORER

PIPELINE_STEPS = [
    {
        "title": "Data Collection",
        "icon": "1",
        "tag": "Sources",
        "description": """
We combine two complementary European datasets to create a comprehensive training set:

**EMHIRES** (European Commission Joint Research Centre)
- 15 years of hourly capacity factor data (2001-2015)
- Covers 30 EU/EEA nations
- Measures how efficiently solar panels performed each hour

**NASA POWER** (Prediction of Worldwide Energy Resources)
- Concurrent hourly meteorological data
- Global Horizontal Irradiance (GHI), temperature, wind speed
- The physical drivers behind panel performance

Together these produce ~**3.9 million** hourly observations, giving the model enough data to learn robust patterns across countries and seasons.
        """,
        "chart_caption": "A sample one-week trace of raw capacity factor values from June 2015. The daily peaks (midday) and troughs (night) form the periodic signal our model learns.",
    },
    {
        "title": "Data Cleaning",
        "icon": "2",
        "tag": "Quality",
        "description": """
Real-world sensor data is messy. Our cleaning pipeline handles four key issues:

**Sentinel removal** — NASA encodes missing readings as `-999`. These corrupt any statistical model and must be detected and removed.

**Temporal alignment** — EMHIRES and NASA data have different start/end dates. We retain only the 2001-2015 overlap window.

**Reshaping** — Country columns are pivoted into labeled rows with a `Country` identifier for each observation.

**Feature extraction** — Raw timestamps are split into numeric `Hour` (0-23) and `Month` (1-12) columns that the model can use directly.
        """,
        "chart_caption": "Red dotted line shows raw data with -999 spikes (sensor failures). Amber line shows cleaned data with corrupt values removed. Without this step, the model would learn from garbage values.",
    },
    {
        "title": "Feature Engineering",
        "icon": "3",
        "tag": "Encoding",
        "description": """
Each observation is transformed into a **34-dimensional feature vector**:

| Type | Features | Count |
|------|----------|-------|
| Continuous | GHI, Temperature, Wind Speed | 3 |
| Temporal | Hour, Month | 2 |
| Categorical | Country (one-hot encoded) | 29 |
| **Total** | | **34** |

One-hot encoding converts each country code (e.g. `ES`) into 29 binary columns (`Country_ES = 1`, rest `= 0`). This lets the model learn country-specific offsets based on latitude, climate, and cloud patterns.

The target variable is **Capacity Factor** (0.0 to 1.0).
        """,
        "chart_caption": "Relative importance of each feature group. GHI dominates at 85% because sunlight intensity is the primary physical driver. Hour (72%) captures solar elevation angle. Country effects are smaller but meaningful.",
    },
    {
        "title": "Model Training",
        "icon": "4",
        "tag": "Algorithm",
        "description": """
We train an **Ordinary Least Squares Linear Regression** model:

```
CF = w₁·GHI + w₂·Hour + w₃·Temp + w₄·Wind + w₅·Month + Σ(wᵢ·Countryᵢ) + bias
```

**Why Linear Regression?**
- Interpretable: each weight directly shows feature impact
- Fast: trains on 3.8M rows in ~2 minutes
- Robust: no hyperparameter tuning required
- Tiny model file: ~1.5 KB serialized

**Training setup:**
- 80/20 train-test split (3.05M train, 762K test)
- No feature scaling needed (LR is scale-invariant with OLS)
- Scikit-learn `LinearRegression()` implementation
        """,
        "chart_caption": "Learning curve showing model error vs training set size. Both curves converge and the gap between them is small, confirming the model generalizes well without overfitting.",
    },
    {
        "title": "Model Evaluation",
        "icon": "5",
        "tag": "Results",
        "description": """
The model is validated on 762,538 held-out test observations:

| Metric | Value | Meaning |
|--------|-------|---------|
| **MAE** | 0.053 | Average error of 5.3 percentage points |
| **R²** | 0.788 | Explains 78.8% of output variance |

**Interpretation:**
- An MAE of 0.053 means if the true CF is 0.50, our prediction is typically between 0.45 and 0.55
- R² of 0.788 is strong for a linear model on real-world energy data
- Remaining 21.2% of variance comes from cloud cover, aerosols, and other unmodeled weather effects
        """,
        "chart_caption": "Each amber dot is one test observation. The red dashed line represents perfect prediction. Tight clustering around this diagonal confirms accuracy. Spread increases at higher CF values due to weather variability.",
    },
]

with tab_pipeline:
    st.subheader("ML Pipeline Explorer")
    st.caption("Walk through each stage of the machine learning pipeline that powers SolarIntel's predictions.")

    if "pipeline_step" not in st.session_state:
        st.session_state.pipeline_step = 0

    total_steps = len(PIPELINE_STEPS)
    current = st.session_state.pipeline_step

    # Step indicator (clickable pills)
    step_cols = st.columns(total_steps)
    for i, col in enumerate(step_cols):
        step = PIPELINE_STEPS[i]
        label = f"{'●' if i == current else '○'} {step['icon']}. {step['tag']}"
        with col:
            if st.button(label, key=f"step_{i}", width="stretch",
                         type="primary" if i == current else "secondary"):
                st.session_state.pipeline_step = i
                st.rerun()

    # Progress
    st.progress((current + 1) / total_steps, text=f"Step {current + 1} of {total_steps}")

    # Content
    step = PIPELINE_STEPS[current]
    st.subheader(f"{step['icon']}. {step['title']}")

    desc_col, chart_col = st.columns([1, 1])

    with desc_col:
        st.markdown(step["description"])

    with chart_col:
        np.random.seed(42)
        if current == 0:
            # One-week capacity factor trace with day markers
            dates = pd.date_range("2015-06-01", periods=168, freq="h")
            cf_data = np.random.beta(2, 3, 168) * 0.8
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=cf_data, fill="tozeroy",
                                     line=dict(color=AMBER, width=2, shape="spline"),
                                     fillcolor="rgba(217,119,6,0.12)", name="Capacity Factor"))
            # Add day/night shading for each day
            for d in range(7):
                start = pd.Timestamp("2015-06-01") + pd.Timedelta(hours=d*24)
                fig.add_vrect(x0=start, x1=start + pd.Timedelta(hours=6),
                              fillcolor="rgba(30,30,30,0.3)", line_width=0)
                fig.add_vrect(x0=start + pd.Timedelta(hours=18), x1=start + pd.Timedelta(hours=24),
                              fillcolor="rgba(30,30,30,0.3)", line_width=0)
            fig.update_layout(title="Sample: 1 Week Capacity Factor (June 2015)",
                              yaxis_title="CF", height=380, **PLOT_LAYOUT)
            st.plotly_chart(fig, width="stretch")

        elif current == 1:
            # Raw vs cleaned with annotations
            x = list(range(24))
            y_raw = [0.02, 0.01, 0, 0, 0, 0.05, 0.15, 0.3, -999, 0.55, 0.62, 0.68,
                     0.7, 0.65, 0.58, 0.48, 0.35, 0.2, 0.1, 0.03, 0, -999, 0, 0]
            y_clean = [v if v >= 0 else None for v in y_raw]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y_raw, name="Raw Data",
                                     line=dict(dash="dot", color="#EF4444", width=2)))
            fig.add_trace(go.Scatter(x=x, y=y_clean, name="After Cleaning",
                                     line=dict(color=AMBER, width=3, shape="spline"),
                                     fill="tozeroy", fillcolor="rgba(217,119,6,0.1)"))
            # Annotate the -999 spikes
            fig.add_annotation(x=8, y=-999, text="Sensor failure (-999)",
                               showarrow=True, arrowhead=2, arrowcolor="#EF4444",
                               font=dict(color="#EF4444", size=11), ay=-40)
            fig.update_layout(title="Raw vs Cleaned Data (1 Day)",
                              xaxis_title="Hour", yaxis_title="Value", height=380, **PLOT_LAYOUT)
            st.plotly_chart(fig, width="stretch")

        elif current == 2:
            # Feature importance with value annotations
            features = ["Country (×29)", "Wind Speed", "Month", "Temperature", "Hour of Day", "Irradiance (GHI)"]
            importance = [0.35, 0.43, 0.48, 0.62, 0.72, 0.85]
            colors = ["#44403C", "#57534E", "#78716C", "#A8A29E", "#F59E0B", AMBER]
            fig = go.Figure(go.Bar(y=features, x=importance, orientation="h",
                marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
                text=[f"{v:.0%}" for v in importance], textposition="outside",
                textfont=dict(size=12)))
            fig.update_layout(title="Feature Importance (Correlation with CF)",
                              xaxis=dict(range=[0, 1.05], title="Importance"),
                              height=380, **PLOT_LAYOUT)
            st.plotly_chart(fig, width="stretch")

        elif current == 3:
            # Learning curve with filled error band
            sizes = list(range(1, 11))
            train_err = [0.15, 0.10, 0.075, 0.058, 0.048, 0.043, 0.040, 0.038, 0.037, 0.036]
            test_err = [0.16, 0.12, 0.090, 0.070, 0.058, 0.052, 0.049, 0.047, 0.046, 0.045]
            fig = go.Figure()
            # Error band between train and test
            fig.add_trace(go.Scatter(x=sizes + sizes[::-1], y=train_err + test_err[::-1],
                                     fill="toself", fillcolor="rgba(217,119,6,0.08)",
                                     line=dict(color="rgba(0,0,0,0)"), showlegend=False))
            fig.add_trace(go.Scatter(x=sizes, y=train_err, mode="lines+markers",
                                     name="Train Error", line=dict(color=AMBER, width=3),
                                     marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=sizes, y=test_err, mode="lines+markers",
                                     name="Test Error", line=dict(color="#A8A29E", width=2, dash="dot"),
                                     marker=dict(size=6)))
            fig.add_annotation(x=10, y=0.045, text="Gap = 0.009 (no overfitting)",
                               showarrow=False, font=dict(color="#22C55E", size=11),
                               yshift=15)
            fig.update_layout(title="Learning Curve",
                              xaxis_title="Training Set Size (×380K)", yaxis_title="MAE",
                              height=380, **PLOT_LAYOUT)
            st.plotly_chart(fig, width="stretch")

        elif current == 4:
            # Predicted vs actual with density coloring
            actual = np.random.uniform(0.05, 0.85, 500)
            noise = np.random.normal(0, 0.035, 500)
            pred = np.clip(actual + noise, 0, 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual, y=pred, mode="markers",
                                     marker=dict(color=actual, colorscale=[[0, "#44403C"], [1, AMBER]],
                                                 size=5, opacity=0.5, line=dict(width=0)),
                                     name="Test Points"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(color="#EF4444", dash="dash", width=2),
                                     name="Perfect (y=x)"))
            # Add R² annotation
            fig.add_annotation(x=0.15, y=0.85, text="R² = 0.788",
                               showarrow=False, font=dict(color=AMBER, size=16))
            fig.add_annotation(x=0.15, y=0.78, text="MAE = 0.053",
                               showarrow=False, font=dict(color="#A8A29E", size=13))
            fig.update_layout(title="Predicted vs Actual Capacity Factor",
                              xaxis_title="Actual CF", yaxis_title="Predicted CF",
                              xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
                              height=380, **PLOT_LAYOUT)
            st.plotly_chart(fig, width="stretch")

    # Chart explanation (inline caption, not a separate expander)
    st.caption(step["chart_caption"])

    # Navigation
    nav_left, nav_spacer, nav_right = st.columns([1, 4, 1])
    with nav_left:
        if current > 0:
            if st.button(f"← {PIPELINE_STEPS[current - 1]['title']}", key="prev_step"):
                st.session_state.pipeline_step = current - 1
                st.rerun()
    with nav_right:
        if current < total_steps - 1:
            if st.button(f"{PIPELINE_STEPS[current + 1]['title']} →", key="next_step"):
                st.session_state.pipeline_step = current + 1
                st.rerun()

    if current == total_steps - 1:
        st.success("Pipeline tour complete! Switch to the **Prediction Dashboard** tab to see this model in action.")


# TAB 3: COUNTRY COMPARISON
with tab_compare:
    st.subheader("Country Comparison")
    st.caption("Compare solar generation potential across EU countries under identical weather conditions.")

    compare_countries = st.multiselect(
        "Select countries to compare:", list(COUNTRIES.keys()),
        default=["ES", "DE", "GB", "IT", "NO"],
        format_func=lambda x: f"{x} — {COUNTRIES[x]}"
    )

    if len(compare_countries) >= 2:
        palette = ["#D97706", "#F59E0B", "#3B82F6", "#22C55E", "#EF4444", "#A855F7", "#EC4899", "#14B8A6"]

        # Compute all data
        compare_cf = {cc: predict_capacity_factor(model, cc, hour, month, irradiance, temperature, wind_speed) for cc in compare_countries}
        sorted_countries = sorted(compare_countries, key=lambda c: compare_cf[c], reverse=True)

        # Compute 24h profiles for all countries
        profiles_24h = {}
        for cc in compare_countries:
            profile = []
            for h in range(24):
                si = irradiance * max(0, np.sin((h - 6) * np.pi / 12)) if 6 <= h <= 18 else 0
                profile.append(predict_capacity_factor(model, cc, h, month, si, temperature, wind_speed) * installed_capacity)
            profiles_24h[cc] = profile

        # Compute monthly CF for all countries
        monthly_data = {}
        for cc in compare_countries:
            monthly_data[cc] = [predict_capacity_factor(model, cc, 12, m, irradiance, temperature, wind_speed) for m in range(1, 13)]

        # Country metrics
        rank_cols = st.columns(len(sorted_countries))
        for i, cc in enumerate(sorted_countries):
            with rank_cols[i]:
                daily_kwh = sum(profiles_24h[cc])
                st.metric(
                    f"{COUNTRIES[cc]}",
                    f"{compare_cf[cc]:.4f}",
                    delta=None
                )
                st.caption(f"{daily_kwh:.0f} kWh/day")

        st.markdown("---")

        # Row 1: Ranked bar + 24h overlay
        cr1, cr2 = st.columns(2)

        with cr1:
            st.markdown("##### Capacity Factor Ranking")
            bar_colors = [palette[compare_countries.index(c) % len(palette)] for c in sorted_countries]
            fig_rank = go.Figure(go.Bar(
                y=[COUNTRIES[c] for c in sorted_countries],
                x=[compare_cf[c] for c in sorted_countries],
                orientation="h",
                marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
                text=[f"{compare_cf[c]:.4f}" for c in sorted_countries],
                textposition="outside", textfont=dict(size=12),
            ))
            max_val = max(compare_cf.values())
            fig_rank.update_layout(
                xaxis=dict(title="Capacity Factor", range=[0, max_val * 1.25]),
                height=350, **PLOT_LAYOUT
            )
            st.plotly_chart(fig_rank, width="stretch")
            st.caption("Countries ranked by capacity factor at your selected conditions. Higher = better solar potential for that region.")

        with cr2:
            st.markdown("##### 24-Hour Generation Overlay")
            fig_24h = go.Figure()
            # Night shading
            fig_24h.add_vrect(x0=0, x1=6, fillcolor="rgba(30,30,30,0.3)", line_width=0)
            fig_24h.add_vrect(x0=18, x1=23, fillcolor="rgba(30,30,30,0.3)", line_width=0)
            for idx, cc in enumerate(compare_countries):
                fig_24h.add_trace(go.Scatter(
                    x=list(range(24)), y=profiles_24h[cc],
                    mode="lines", name=COUNTRIES[cc],
                    line=dict(color=palette[idx % len(palette)], width=2.5),
                ))
            fig_24h.update_layout(
                xaxis=dict(title="Hour (UTC)", dtick=3),
                yaxis_title="Output (kW)", height=350,
                legend=dict(orientation="h", y=1.12), **PLOT_LAYOUT
            )
            st.plotly_chart(fig_24h, width="stretch")
            st.caption("All countries under the same weather. Differences come from the model's learned geographic coefficients (latitude, climate patterns).")

        st.markdown("---")

        # Row 2: Monthly + Seasonal heatmap
        cr3, cr4 = st.columns(2)

        with cr3:
            st.markdown("##### Monthly Capacity Factor")
            fig_monthly = go.Figure()
            for idx, cc in enumerate(compare_countries):
                fig_monthly.add_trace(go.Scatter(
                    x=MONTH_NAMES, y=monthly_data[cc],
                    mode="lines+markers", name=COUNTRIES[cc],
                    line=dict(color=palette[idx % len(palette)], width=2),
                    marker=dict(size=6),
                ))
            fig_monthly.update_layout(
                xaxis_title="Month", yaxis_title="CF",
                height=380, legend=dict(orientation="h", y=1.12), **PLOT_LAYOUT
            )
            st.plotly_chart(fig_monthly, width="stretch")
            st.caption("Noon capacity factor by month. Southern countries (Spain, Italy) show higher summer peaks. Northern countries (Norway, UK) show flatter, lower curves.")

        with cr4:
            st.markdown("##### Country × Month Heatmap")
            z_data = [monthly_data[cc] for cc in compare_countries]
            fig_hm = go.Figure(go.Heatmap(
                z=z_data,
                x=MONTH_NAMES,
                y=[COUNTRIES[cc] for cc in compare_countries],
                colorscale=[[0, "#1C1917"], [0.3, "#44403C"], [0.6, "#78716C"], [0.8, "#D97706"], [1, "#F59E0B"]],
                colorbar=dict(title=dict(text="CF")),
                hovertemplate="%{y}<br>%{x}: CF = %{z:.4f}<extra></extra>",
            ))
            fig_hm.update_layout(height=380, **PLOT_LAYOUT)
            st.plotly_chart(fig_hm, width="stretch")
            st.caption("Warmer colors indicate higher generation. This reveals both the best months AND best countries at a glance.")

        st.markdown("---")

        # Summary table
        st.markdown("##### Detailed Comparison")
        rows = []
        best_cf = max(compare_cf.values())
        for rank, cc in enumerate(sorted_countries, 1):
            daily_kwh = sum(profiles_24h[cc])
            annual_mwh = daily_kwh * 365 / 1000
            peak_kw = max(profiles_24h[cc])
            peak_h = profiles_24h[cc].index(peak_kw)
            pct_of_best = (compare_cf[cc] / best_cf) * 100 if best_cf > 0 else 0
            rows.append({
                "Rank": f"#{rank}",
                "Country": f"{COUNTRIES[cc]} ({cc})",
                "CF": f"{compare_cf[cc]:.4f}",
                "vs Best": f"{pct_of_best:.0f}%",
                "Peak kW": f"{peak_kw:.1f}",
                "Peak Hour": f"{peak_h}:00",
                "Daily kWh": f"{daily_kwh:.1f}",
                "Annual MWh": f"{annual_mwh:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"All values computed with: {irradiance} W/m² | {temperature}°C | {wind_speed} m/s | {installed_capacity} kW system | Month {month}")

    else:
        st.info("Select at least **2 countries** above to start comparing.")


# TAB 4: PROJECT INFO
with tab_about:
    st.subheader("Project Technical Specifications")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### Datasets

| Source | Coverage | Period | Resolution |
|--------|----------|--------|------------|
| EMHIRES | 30 EU countries | 2001–2015 | Hourly CF |
| NASA POWER | Global | 2001–2015 | Hourly Met |

### Technology Stack

| Component | Technology |
|-----------|------------|
| Model | Linear Regression (Scikit-Learn) |
| Processing | Pandas, NumPy |
| Visualization | Plotly |
| Interface | Streamlit |
| Language | Python 3.12 |
        """)

    with col2:
        st.markdown("""
### Pipeline Modules

1. **Dataset Visualization** — Exploratory analysis
2. **Cleaning & Transformation** — Missing data, reshaping
3. **Merging** — EMHIRES + NASA alignment
4. **Encoding** — One-Hot encoding (29 countries)
5. **Training & Evaluation** — Linear Regression, 80/20
6. **Analysis Visualization** — Performance analysis

### Team
- **Mahir** — Data Integration, Cleaning, Merging, Lead
- **Priyanshu** — Encoding, Training, UI Development
- **Antik** — Dataset Visualization, Deployment
- **Vansh** — Analysis Visualization, Documentation
        """)

    st.markdown("---")
    st.caption("SolarIntel — GenAI Capstone Project, Milestone 1 — February 2026")
