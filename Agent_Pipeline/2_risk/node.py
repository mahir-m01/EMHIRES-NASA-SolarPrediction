import sys
import os
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from state import GridAdvisorState


def risk_analysis_node(state: GridAdvisorState) -> dict:
    profile = state["hourly_profile"]

    variability_score = round(statistics.stdev(profile), 4) if len(profile) > 1 else 0.0
    peak_hours = [h for h, cf in enumerate(profile) if cf > 0.6]
    low_hours = [h for h, cf in enumerate(profile) if cf < 0.05]
    ramp_events = [h for h in range(1, 24) if abs(profile[h] - profile[h - 1]) > 0.12]

    risk_flags = []
    if ramp_events:
        risk_flags.append(f"High variability window: hours {ramp_events[0]}-{ramp_events[-1]}")
    if peak_hours:
        risk_flags.append(f"Peak generation: hours {peak_hours[0]}-{peak_hours[-1]}")
    if low_hours:
        risk_flags.append(f"Minimal generation risk: {len(low_hours)} hours below 5% capacity")
    if variability_score < 0.05:
        risk_flags.append("Low variability day - stable but low output expected")

    return {
        "risk_summary": {
            "variability_score": variability_score,
            "peak_hours": peak_hours,
            "low_hours": low_hours,
            "ramp_events": ramp_events,
        },
        "risk_flags": risk_flags,
    }
