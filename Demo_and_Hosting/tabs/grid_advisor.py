import streamlit as st
import os
import sys as _sys
import importlib.util as _ilu
import datetime as _dt
import plotly.graph_objects as go
from utils import COUNTRIES, PLOT_LAYOUT, AMBER

def render(installed_capacity):
    # Agent_Pipeline/ is at repo root, one level up from Demo_and_Hosting/
    _REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    _AP = os.path.join(_REPO_ROOT, "Agent_Pipeline")
    if _REPO_ROOT not in _sys.path:
        _sys.path.insert(0, _REPO_ROOT)
    if _AP not in _sys.path:
        _sys.path.insert(0, _AP)

    st.subheader("Grid Advisor : Agentic AI Optimization")

    @st.cache_resource(show_spinner="Building knowledge base + compiling agent graph…")
    def get_agent_graph():
        _store_path = os.path.join(_AP, "3_rag", "store.py")
        _store_spec = _ilu.spec_from_file_location("store", _store_path)
        _store_mod = _ilu.module_from_spec(_store_spec)
        _sys.modules["store"] = _store_mod
        _store_spec.loader.exec_module(_store_mod)
        _store_mod.build_vector_store(os.path.join(_AP, "knowledge_base"))
        from graph import build_graph
        return build_graph()

    graph = get_agent_graph()

    st.markdown("#### Settings")
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
            "System Size (kW)", value=installed_capacity, min_value=1.0, step=10.0, key="adv_capacity"
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
        with st.spinner("Analyzing solar forecast and generating recommendations…"):
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
                st.error(f"Something went wrong: {exc}")

    if "advisor_result" in st.session_state:
        result = st.session_state["advisor_result"]
        country_label = COUNTRIES.get(st.session_state.get("advisor_country", ""), "")
        rec = result.get("final_recommendations", {})
        risk = result.get("risk_summary", {})

        st.markdown("---")

        # Forecast Summary
        if rec.get("forecast_summary"):
            st.info(f"**Forecast Summary:** {rec['forecast_summary']}")
            st.markdown("")

        # KPIs with explanations
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Peak Capacity Factor", f"{result.get('cf_value', 0):.4f}")
            st.caption("Best efficiency the system can reach that day (closer to 1 = better)")
        with mc2:
            st.metric("Variability Score", f"{risk.get('variability_score', 0):.4f}")
            st.caption("How much the output fluctuates (lower = more stable)")
        with mc3:
            st.metric("Peak Generation Hours", str(len(risk.get('peak_hours', []))))
            st.caption("Hours when the system runs at full power")
        with mc4:
            st.metric("Low Generation Hours", str(len(risk.get('low_hours', []))))
            st.caption("Hours with very little or no solar output")

        st.markdown("---")

        # Hourly chart
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
                    name="Sudden Changes",
                ))
            fig_adv.update_layout(
                xaxis=dict(title="Hour (UTC)", dtick=2),
                yaxis_title="Output (kW)", height=300,
                legend=dict(orientation="h", y=1.12), **PLOT_LAYOUT,
            )
            st.plotly_chart(fig_adv, use_container_width=True)
            st.caption(f"Expected power output hour-by-hour for {country_label} on {result.get('forecast_date', 'N/A')}. Based on real weather forecast data.")

        st.markdown("---")

        # Risk Periods
        risk_periods = rec.get("risk_periods", [])
        if risk_periods:
            st.markdown("#### Risk Periods")
            st.caption("Times during the day that need extra attention for grid management.")
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

        # Strategies
        strategies = rec.get("strategies", [])
        if strategies:
            st.markdown("#### What You Can Do")
            for s in strategies:
                src = s.get("source", "general")
                rag_tag = " <span style='font-size:11px; background:#3B82F6; color:white; padding:1px 6px; border-radius:3px; margin-left:6px'>(RAG)</span>" if src == "retrieved" else ""
                st.markdown(
                    f"<div style='border:1px solid #44403C; padding:10px 14px; margin-bottom:8px; "
                    f"background:#1C1917; border-radius:6px'>"
                    f"<strong>{s.get('title','')}</strong>{rag_tag}<br>"
                    f"<span style='color:#A8A29E; font-size:13px'>{s.get('description','')}</span></div>",
                    unsafe_allow_html=True,
                )

        if rec.get("responsible_ai_note"):
            st.caption(f"⚠️ {rec['responsible_ai_note']}")
