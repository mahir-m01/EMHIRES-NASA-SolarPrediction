import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import COUNTRIES, MONTH_NAMES, PLOT_LAYOUT, AMBER
from model_loader import predict_capacity_factor

def render(model, country_code, hour, month, irradiance, temperature, wind_speed, installed_capacity):
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"All values computed with: {irradiance} W/m² | {temperature}°C | {wind_speed} m/s | {installed_capacity} kW system | Month {month}")

    else:
        st.info("Select at least **2 countries** above to start comparing.")
