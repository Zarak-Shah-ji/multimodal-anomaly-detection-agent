"""
Agent 2 — Knowledge Graph Querier

Given anomalous sensors, queries Neo4j + GraphRAG to retrieve:
  - Component-level failure modes
  - Direct sensor failure indicators
  - GraphRAG-retrieved subgraph context

Falls back to static KG data if Neo4j is unavailable.
"""

import os
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.kg.builder import (
    query_kg_for_sensors,
    query_direct_failure_indicators,
    format_kg_context,
    get_static_kg_context,
)
from src.graphrag.retriever import GraphRAGRetriever

_retriever: GraphRAGRetriever | None = None
_driver = None


def _get_retriever() -> GraphRAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GraphRAGRetriever()
    return _retriever


def _get_neo4j_driver():
    global _driver
    if _driver is None:
        try:
            uri      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            username = os.environ.get("NEO4J_USERNAME", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "")
            _driver = GraphDatabase.driver(uri, auth=(username, password))
            # Verify connectivity
            _driver.verify_connectivity()
        except Exception:
            _driver = None
    return _driver


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
    Main KG query function. Tries Neo4j first, falls back to static.

    Returns formatted context string combining Neo4j + GraphRAG results.
    """
    if not anomalies:
        return "No anomalies to query. Engine appears healthy."

    sensor_ids = [a["sensor"] for a in anomalies]
    retriever  = _get_retriever()

    # ── GraphRAG retrieval (always available) ─────────────────────────────────
    graphrag_query   = build_anomaly_query(anomalies)
    graphrag_context = retriever.retrieve_formatted(
        query=graphrag_query,
        top_k=5,
        sensor_filter=sensor_ids,
    )

    # ── Neo4j query (optional) ────────────────────────────────────────────────
    driver = _get_neo4j_driver()

    if driver:
        try:
            comp_results   = query_kg_for_sensors(driver, sensor_ids)
            direct_results = query_direct_failure_indicators(driver, sensor_ids)
            neo4j_context  = format_kg_context(comp_results, direct_results)
            neo4j_available = True
        except (ServiceUnavailable, AuthError, Exception):
            neo4j_available = False
            neo4j_context   = get_static_kg_context(sensor_ids)
    else:
        neo4j_available = False
        neo4j_context   = get_static_kg_context(sensor_ids)

    source_note = (
        "Source: Live Neo4j graph + GraphRAG"
        if neo4j_available
        else "Source: Static KG + GraphRAG (Neo4j unavailable)"
    )

    return f"{source_note}\n\n{neo4j_context}\n\n{graphrag_context}"
