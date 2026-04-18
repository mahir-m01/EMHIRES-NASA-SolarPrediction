import os
import sys
import json
import importlib.util
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

# load store.py and register it in sys.modules BEFORE building the graph
# so that when 3_rag/node.py does "from store import retrieve_chunks"
# it gets the same instance where build_vector_store() already ran
_store_path = os.path.join(os.path.dirname(__file__), "3_rag", "store.py")
_store_spec = importlib.util.spec_from_file_location("store", _store_path)
_store_mod = importlib.util.module_from_spec(_store_spec)
sys.modules["store"] = _store_mod
_store_spec.loader.exec_module(_store_mod)

kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
_store_mod.build_vector_store(kb_path)

from graph import build_graph

graph = build_graph()

result = graph.invoke({
    "country": "ES",
    "capacity_kw": 100.0,
    "model_name": "Random Forest",
    "forecast_date": "",
    "weather_forecast": [],
    "cf_value": 0.0,
    "hourly_profile": [],
    "monthly_profile": [],
    "risk_summary": {},
    "risk_flags": [],
    "retrieved_chunks": [],
    "final_recommendations": {},
    "error": None,
})

print("\n--- Risk Analysis ---")
print("Variability score:", result["risk_summary"].get("variability_score"))
print("Risk flags:", result["risk_flags"])

print("\n--- Retrieved Chunks (first 100 chars each) ---")
for i, chunk in enumerate(result["retrieved_chunks"]):
    print(f"[{i+1}]", chunk[:100].replace("\n", " "))

print("\n--- Final Recommendations ---")
print(json.dumps(result["final_recommendations"], indent=2))
