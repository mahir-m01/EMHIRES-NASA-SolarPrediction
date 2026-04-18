import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import COUNTRIES, MONTH_NAMES, PLOT_LAYOUT, AMBER
from model_loader import predict_capacity_factor

def render(model, country_code, hour, month, irradiance, temperature, wind_speed, installed_capacity):
    # Compute predictions
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
    with c1:
        st.metric("Capacity Factor", f"{cf:.4f}")
        st.caption("How much of the system's full potential is being used (0 = none, 1 = max)")
    with c2:
        st.metric("Power Output", f"{power_out:.1f} kW")
        st.caption("Estimated electricity being generated right now")
    with c3:
        st.metric("Est. Daily Energy", f"{daily_kwh:.1f} kWh")
        st.caption("Total energy expected over the full day")
    with c4:
        st.metric("Peak Output", f"{peak_kw:.1f} kW @ {peak_hour}:00")
        st.caption("Highest output and the hour it occurs")

    st.markdown("---")

    # ── Row 1: Gauge + 24h Profile ──
    r1c1, r1c2 = st.columns([1, 2])

    with r1c1:
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
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption("The needle shows how efficiently the solar system is performing. Above 0.5 (red line) is considered good.")

    with r1c2:
        fig_24h = go.Figure()
        fig_24h.add_vrect(x0=0, x1=6, fillcolor="rgba(30,30,30,0.4)", line_width=0, annotation_text="Night", annotation_position="top left")
        fig_24h.add_vrect(x0=18, x1=23, fillcolor="rgba(30,30,30,0.4)", line_width=0, annotation_text="Night", annotation_position="top right")
        fig_24h.add_trace(go.Scatter(x=hours_range, y=output_24h, mode="lines", line=dict(color=AMBER, width=3, shape="spline"),
                                     fill="tozeroy", fillcolor="rgba(217,119,6,0.12)", name="Generation"))
        fig_24h.add_trace(go.Scatter(x=[hour], y=[output_24h[hour]], mode="markers+text",
                                     marker=dict(color="#EF4444", size=14, symbol="diamond", line=dict(width=2, color="white")),
                                     text=[f"{output_24h[hour]:.1f} kW"], textposition="top center", textfont=dict(color="white", size=11),
                                     name=f"Selected ({hour}:00)"))
        fig_24h.add_trace(go.Scatter(x=[peak_hour], y=[peak_kw], mode="markers+text",
                                     marker=dict(color="#22C55E", size=10, symbol="star"),
                                     text=[f"Peak: {peak_kw:.1f}"], textposition="bottom center", textfont=dict(color="#22C55E", size=10),
                                     name=f"Peak ({peak_hour}:00)"))
        fig_24h.update_layout(title="24-Hour Generation Profile", xaxis_title="Hour (UTC)", yaxis_title="Output (kW)",
                              xaxis=dict(dtick=2), height=300, showlegend=True, legend=dict(orientation="h", y=1.12), **PLOT_LAYOUT)
        st.plotly_chart(fig_24h, use_container_width=True)
        st.caption("How much power your solar system produces throughout the day. The curve follows the sun — rising in the morning and dropping at sunset.")

    # ── Row 2: Monthly bars + Radar ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        max_cf = max(monthly_cf) if max(monthly_cf) > 0 else 1
        colors_monthly = [f"rgba({int(217*(v/max_cf))}, {int(119*(v/max_cf))}, 6, 0.9)" for v in monthly_cf]
        fig_monthly = go.Figure(go.Bar(
            x=MONTH_NAMES, y=monthly_cf,
            marker=dict(color=colors_monthly, line=dict(color=AMBER, width=1)),
            text=[f"{v:.3f}" for v in monthly_cf], textposition="outside", textfont=dict(size=10),
        ))
        fig_monthly.add_vline(x=month - 1, line=dict(color="#EF4444", width=2, dash="dash"), annotation_text="Now", annotation_position="top")
        fig_monthly.update_layout(title="Monthly Capacity Factor (Noon)", xaxis_title="Month", yaxis_title="CF",
                                  yaxis=dict(range=[0, max_cf * 1.35]), height=380, **PLOT_LAYOUT)
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.caption("Which months produce the most solar energy. Taller bars = better months for generation. The red dashed line marks your selected month.")

    with r2c2:
        base_pct = (irradiance / 1000) * 100
        hour_pct = max(0, np.sin((hour - 6) * np.pi / 12)) * 100 if 6 <= hour <= 18 else 0
        month_pct = (0.5 + 0.5 * np.sin((month - 3) * np.pi / 6)) * 100
        temp_pct = max(0, (1 - abs(temperature - 25) / 50)) * 100
        wind_pct = min(100, wind_speed / 15 * 100)

        categories = ["Irradiance", "Hour of Day", "Season", "Temperature", "Wind"]
        values = [base_pct, hour_pct, month_pct, temp_pct, wind_pct]
        values_closed = values + [values[0]]

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
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("What's helping or hurting solar generation right now. A bigger shape means better overall conditions.")
