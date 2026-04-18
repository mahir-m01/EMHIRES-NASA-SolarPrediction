import os
import importlib.util

from langgraph.graph import StateGraph, END
from state import GridAdvisorState


def _load_node_func(folder: str, func_name: str):
    path = os.path.join(os.path.dirname(__file__), folder, "node.py")
    spec = importlib.util.spec_from_file_location(f"{folder}.node", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)


forecast_node = _load_node_func("1_forecast", "forecast_node")
risk_analysis_node = _load_node_func("2_risk", "risk_analysis_node")
rag_retrieval_node = _load_node_func("3_rag", "rag_retrieval_node")
recommendation_node = _load_node_func("4_recommendations", "recommendation_node")


def build_graph():
    g = StateGraph(GridAdvisorState)
    g.add_node("forecast", forecast_node)
    g.add_node("risk_analysis", risk_analysis_node)
    g.add_node("rag_retrieval", rag_retrieval_node)
    g.add_node("recommendation", recommendation_node)
    g.set_entry_point("forecast")
    g.add_edge("forecast", "risk_analysis")
    g.add_edge("risk_analysis", "rag_retrieval")
    g.add_edge("rag_retrieval", "recommendation")
    g.add_edge("recommendation", END)
    return g.compile()
