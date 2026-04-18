"""SolarIntel — Energy Generation Analytics"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
load_dotenv()
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
st.sidebar.caption("Milestone 1 & 2 — ML Forecasting + Agentic AI")
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

tab_forecast, tab_compare, tab_advisor, tab_about = st.tabs(
    ["Prediction Dashboard", "Country Comparison", "Grid Advisor", "Project Info"]
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


# TAB 2: COUNTRY COMPARISON
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


# TAB 3: GRID ADVISOR
with tab_advisor:
    import sys as _sys
    import importlib.util as _ilu
    import datetime as _dt

    # Agent_Pipeline/ is at repo root, one level up from Demo_and_Hosting/
    _REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    _AP = os.path.join(_REPO_ROOT, "Agent_Pipeline")
    if _REPO_ROOT not in _sys.path:
        _sys.path.insert(0, _REPO_ROOT)
    if _AP not in _sys.path:
        _sys.path.insert(0, _AP)

    st.subheader("Grid Advisor — Agentic AI Optimization")
    st.caption(
        "Fetches real weather forecast data from Open-Meteo (up to 15 days ahead), runs a 4-node LangGraph pipeline, "
        "and returns structured grid management recommendations grounded in European best practices."
    )

    @st.cache_resource(show_spinner="Building knowledge base + compiling agent graph…")
    def get_agent_graph():
        # Pre-register store module as singleton before building graph
        _store_path = os.path.join(_AP, "3_rag", "store.py")
        _store_spec = _ilu.spec_from_file_location("store", _store_path)
        _store_mod = _ilu.module_from_spec(_store_spec)
        _sys.modules["store"] = _store_mod
        _store_spec.loader.exec_module(_store_mod)
        _store_mod.build_vector_store(os.path.join(_AP, "knowledge_base"))
        from graph import build_graph
        return build_graph()

    graph = get_agent_graph()

    st.markdown("#### Configuration")
    adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
    with adv_col1:
        adv_country = st.selectbox(
            "Country", list(COUNTRIES.keys()),
            index=list(COUNTRIES.keys()).index("ES"),
            format_func=lambda x: f"{x} — {COUNTRIES[x]}",
            key="adv_country",
        )
    with adv_col2:
        adv_capacity = st.number_input(
            "Installed Capacity (kW)", value=100.0, min_value=1.0, step=10.0, key="adv_capacity"
        )
    with adv_col3:
        adv_model = st.selectbox(
            "Model", ["Random Forest"], index=0, key="adv_model"
        )
    with adv_col4:
        _today = _dt.date.today()
        adv_date = st.date_input(
            "Forecast Date",
            value=_today + _dt.timedelta(days=1),
            min_value=_today + _dt.timedelta(days=1),
            max_value=_today + _dt.timedelta(days=15),
            key="adv_date",
        )

    run_btn = st.button("Run Grid Advisor", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running pipeline (forecast → risk → RAG → LLM)…"):
            try:
                agent_input = {
                    "country": adv_country,
                    "capacity_kw": adv_capacity,
                    "model_name": adv_model,
                    "forecast_date": str(adv_date),
                    "weather_forecast": [],
                    "cf_value": 0.0,
                    "hourly_profile": [],
                    "monthly_profile": [],
                    "risk_summary": {},
                    "risk_flags": [],
                    "retrieved_chunks": [],
                    "final_recommendations": {},
                    "error": None,
                }
                result = graph.invoke(agent_input)
                st.session_state["advisor_result"] = result
                st.session_state["advisor_country"] = adv_country
            except Exception as exc:
                st.error(f"Agent pipeline failed: {exc}")

    if "advisor_result" in st.session_state:
        result = st.session_state["advisor_result"]
        country_label = COUNTRIES.get(st.session_state.get("advisor_country", ""), "")
        rec = result.get("final_recommendations", {})
        risk = result.get("risk_summary", {})

        st.markdown("---")

        if rec.get("forecast_summary"):
            st.info(f"**Forecast Summary:** {rec['forecast_summary']}")
            st.markdown("<br>", unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Peak Capacity Factor", f"{result.get('cf_value', 0):.4f}")
        mc2.metric("Variability Score", f"{risk.get('variability_score', 0):.4f}")
        mc3.metric("Peak Generation Hours", str(len(risk.get('peak_hours', []))))
        mc4.metric("Low Generation Hours", str(len(risk.get('low_hours', []))))

        st.markdown("---")

        hourly = result.get("hourly_profile", [])
        if hourly:
            st.markdown("#### Hourly Generation Profile")
            fig_adv = go.Figure()
            fig_adv.add_vrect(x0=0, x1=6, fillcolor="rgba(30,30,30,0.4)", line_width=0,
                              annotation_text="Night", annotation_position="top left")
            fig_adv.add_vrect(x0=18, x1=23, fillcolor="rgba(30,30,30,0.4)", line_width=0,
                              annotation_text="Night", annotation_position="top right")
            fig_adv.add_trace(go.Scatter(
                x=list(range(24)), y=[cf * adv_capacity for cf in hourly],
                mode="lines", line=dict(color=AMBER, width=3, shape="spline"),
                fill="tozeroy", fillcolor="rgba(217,119,6,0.12)", name="Output (kW)",
            ))
            ramp_events = risk.get("ramp_events", [])
            if ramp_events:
                fig_adv.add_trace(go.Scatter(
                    x=ramp_events, y=[hourly[h] * adv_capacity for h in ramp_events],
                    mode="markers", marker=dict(color="#EF4444", size=10, symbol="x"),
                    name="Ramp Events",
                ))
            fig_adv.update_layout(
                xaxis=dict(title="Hour (UTC)", dtick=2),
                yaxis_title="Output (kW)", height=300,
                legend=dict(orientation="h", y=1.12), **PLOT_LAYOUT,
            )
            st.plotly_chart(fig_adv, width="stretch")
            st.caption(f"Real weather forecast from Open-Meteo for {country_label} | Date: {result.get('forecast_date', 'N/A')}")

        st.markdown("---")

        risk_periods = rec.get("risk_periods", [])
        if risk_periods:
            st.markdown("#### Risk Periods")
            sev_colors = {"high": "#EF4444", "medium": "#F59E0B", "low": "#22C55E"}
            
            for rp in risk_periods:
                sev = rp.get("severity", "low").lower()
                color = sev_colors.get(sev, "#78716C")
                st.markdown(
                    f"<div style='border-left:4px solid {color}; padding:8px 14px; margin-bottom:8px; "
                    f"background:#1C1917; border-radius:4px'>"
                    f"<strong style='color:{color}'>[{sev.upper()}]</strong> "
                    f"<strong>{rp.get('period','')}</strong> : {rp.get('risk','')}</div>",
                    unsafe_allow_html=True,
                )

        strategies = rec.get("strategies", [])
        if strategies:
            st.markdown("#### Recommended Strategies")
            for s in strategies:
                src = s.get("source", "general")
                badge_color = "#3B82F6" if src == "retrieved" else "#6B7280"
                st.markdown(
                    f"<div style='border:1px solid #44403C; padding:10px 14px; margin-bottom:8px; "
                    f"background:#1C1917; border-radius:6px'>"
                    f"<strong>{s.get('title','')}</strong> "
                    f"<span style='font-size:11px; background:{badge_color}; color:white; "
                    f"padding:1px 6px; border-radius:3px; margin-left:6px'>{src}</span><br>"
                    f"<span style='color:#A8A29E; font-size:13px'>{s.get('description','')}</span></div>",
                    unsafe_allow_html=True,
                )

        with st.expander("Raw Risk Flags", expanded=False):
            for flag in result.get("risk_flags", []):
                st.markdown(f"- {flag}")

        if rec.get("responsible_ai_note"):
            st.caption(f"**Responsible AI:** {rec['responsible_ai_note']}")

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
    st.caption("SolarIntel — GenAI Capstone Project, Milestone 1 & 2 — 2026")
