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
            "Installed Capacity (kW)", value=installed_capacity, min_value=1.0, step=10.0, key="adv_capacity"
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
