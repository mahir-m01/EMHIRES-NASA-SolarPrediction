import sys
import os
import datetime

# add this folder so 'store' can be imported directly
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from store import retrieve_chunks
from state import GridAdvisorState

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def rag_retrieval_node(state: GridAdvisorState) -> dict:
    month_name = _MONTH_NAMES[datetime.date.today().month - 1]
    flags_snippet = " ".join(state["risk_flags"][:2])
    query = f"{state['country']} {month_name} solar grid {flags_snippet}"
    chunks = retrieve_chunks(query, k=4)
    return {"retrieved_chunks": chunks}
