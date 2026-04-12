"""
LangGraph pipeline orchestrating the 3 agents.

State flow:
  sensor_data + logs
      │
      ▼
  [anomaly_detection_node]  → anomalies, anomaly_analysis
      │
      ▼
  [kg_query_node]           → kg_context
      │
      ▼
  [report_generation_node]  → report
"""

from __future__ import annotations
from typing import TypedDict, Optional
import pandas as pd
from langgraph.graph import StateGraph, END

from src.agents.anomaly_agent import detect_statistical_anomalies, analyze_anomalies_with_claude
from src.agents.kg_agent import query_knowledge_graph
from src.agents.report_agent import generate_diagnostic_report


# ── State ─────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    unit:             int
    sensor_df:        pd.DataFrame
    logs_df:          pd.DataFrame
    anomalies:        list[dict]
    anomaly_analysis: str
    kg_context:       str
    report:           str
    error:            Optional[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def anomaly_detection_node(state: PipelineState) -> PipelineState:
    """Detect anomalies statistically, then interpret with Claude."""
    try:
        anomalies = detect_statistical_anomalies(state["sensor_df"])
        analysis  = analyze_anomalies_with_claude(
            state["sensor_df"],
            state["logs_df"],
            anomalies,
            state["unit"],
        )
        return {**state, "anomalies": anomalies, "anomaly_analysis": analysis}
    except Exception as e:
        return {**state, "anomalies": [], "anomaly_analysis": "", "error": str(e)}


def kg_query_node(state: PipelineState) -> PipelineState:
    """Query KG + GraphRAG for relevant failure context."""
    try:
        context = query_knowledge_graph(state["anomalies"])
        return {**state, "kg_context": context}
    except Exception as e:
        return {**state, "kg_context": f"KG query failed: {e}", "error": str(e)}


def report_generation_node(state: PipelineState) -> PipelineState:
    """Generate the final diagnostic report with Claude."""
    try:
        cycle      = int(state["sensor_df"]["cycle"].max())
        recent_logs = state["logs_df"]["log_text"].tail(5).tolist() if not state["logs_df"].empty else []
        report = generate_diagnostic_report(
            unit=state["unit"],
            cycle=cycle,
            anomalies=state["anomalies"],
            anomaly_analysis=state["anomaly_analysis"],
            kg_context=state["kg_context"],
            recent_logs=recent_logs,
        )
        return {**state, "report": report}
    except Exception as e:
        return {**state, "report": f"Report generation failed: {e}", "error": str(e)}


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_pipeline():
    """Compile and return the LangGraph pipeline."""
    builder = StateGraph(PipelineState)

    builder.add_node("anomaly_detection", anomaly_detection_node)
    builder.add_node("kg_query",          kg_query_node)
    builder.add_node("report_generation", report_generation_node)

    builder.set_entry_point("anomaly_detection")
    builder.add_edge("anomaly_detection", "kg_query")
    builder.add_edge("kg_query",          "report_generation")
    builder.add_edge("report_generation", END)

    return builder.compile()


# ── Convenience runner ────────────────────────────────────────────────────────

def run_pipeline(
    unit: int,
    sensor_df: pd.DataFrame,
    logs_df: pd.DataFrame,
) -> PipelineState:
    """
    Run the full pipeline and return the final state.

    Args:
        unit:      Engine unit number
        sensor_df: Sensor time-series for this engine
        logs_df:   Maintenance logs for this engine

    Returns:
        Final PipelineState with anomalies, kg_context, and report populated.
    """
    pipeline = build_pipeline()
    initial_state: PipelineState = {
        "unit":             unit,
        "sensor_df":        sensor_df,
        "logs_df":          logs_df,
        "anomalies":        [],
        "anomaly_analysis": "",
        "kg_context":       "",
        "report":           "",
        "error":            None,
    }
    return pipeline.invoke(initial_state)
