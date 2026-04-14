"""
Agent 2 — Knowledge Graph Querier

Uses static KG + GraphRAG to retrieve:
  - Component-level failure modes
  - Direct sensor failure indicators
  - GraphRAG-retrieved subgraph context
"""

from src.kg.builder import (
    get_static_kg_context,
)
from src.graphrag.retriever import GraphRAGRetriever

_retriever: GraphRAGRetriever | None = None


def _get_retriever() -> GraphRAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GraphRAGRetriever()
    return _retriever


def build_anomaly_query(anomalies: list[dict]) -> str:
    """Build a natural language query for GraphRAG from anomaly list."""
    if not anomalies:
        return "normal engine operation"

    sensors = [a["sensor"] for a in anomalies[:5]]
    directions = [f"{a['sensor']} {a['direction']}" for a in anomalies[:5]]
    return (
        f"Anomalous sensors: {', '.join(sensors)}. "
        f"Readings: {', '.join(directions)}. "
        f"Engine degradation failure mode component diagnosis."
    )


def query_knowledge_graph(anomalies: list[dict]) -> str:
    """
    Main KG query function. Uses static KG + GraphRAG.

    Returns formatted context string.
    """
    if not anomalies:
        return "No anomalies to query. Engine appears healthy."

    sensor_ids = [a["sensor"] for a in anomalies]
    retriever  = _get_retriever()

    # ── GraphRAG retrieval ────────────────────────────────────────────────────
    graphrag_query   = build_anomaly_query(anomalies)
    graphrag_context = retriever.retrieve_formatted(
        query=graphrag_query,
        top_k=5,
        sensor_filter=sensor_ids,
    )

    # ── Static KG ─────────────────────────────────────────────────────────────
    neo4j_context = get_static_kg_context(sensor_ids)

    return f"Source: Static KG + GraphRAG\n\n{neo4j_context}\n\n{graphrag_context}"
