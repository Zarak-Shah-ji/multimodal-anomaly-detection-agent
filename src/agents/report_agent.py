"""
Agent 3 — Diagnostic Report Generator

Takes anomaly analysis + KG context, calls Gemini (streaming) to
produce a structured diagnostic report in Markdown.
"""

import os
import google.generativeai as genai

_model = None


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _model = genai.GenerativeModel("gemini-2.0-flash")
    return _model


def _build_report_prompt(
    unit: int,
    cycle: int,
    anomalies: list[dict],
    anomaly_analysis: str,
    kg_context: str,
    recent_logs: list[str],
) -> str:
    anomaly_table = "\n".join(
        f"| {a['sensor']} | {a['latest_value']} | {a['z_score']:+.2f} | {a['direction']} | {a['severity']} |"
        for a in anomalies
    ) if anomalies else "| — | — | — | — | — |"

    log_section = "\n".join(f"- {l}" for l in recent_logs[-5:]) if recent_logs else "No recent maintenance logs."

    return f"""You are a senior aircraft engine health monitoring AI system.
Generate a professional diagnostic report in Markdown for the following engine data.

## Input Data

**Engine Unit:** {unit}
**Cycle:** {cycle}

### Detected Anomalies
| Sensor | Value | Z-Score | Direction | Severity |
|--------|-------|---------|-----------|----------|
{anomaly_table}

### AI Pattern Analysis
{anomaly_analysis}

### Knowledge Graph Context
{kg_context}

### Recent Maintenance Logs
{log_section}

## Report Requirements

Write a structured Markdown diagnostic report with these sections:
1. **Executive Summary** (2-3 sentences, plain language)
2. **Anomaly Analysis** (table + interpretation per sensor)
3. **Likely Failure Modes** (from KG context, ranked by severity)
4. **Affected Components** (which subsystems, why)
5. **Risk Assessment** (Low / Medium / High / Critical with justification)
6. **Recommended Actions** (numbered, specific, actionable)
7. **Monitoring Priorities** (top 3 sensors to watch next cycle)

Be precise and technical. Use the KG context to ground failure mode names.
Format cleanly in Markdown. Do not invent data not present in the input."""


def generate_diagnostic_report(
    unit: int,
    cycle: int,
    anomalies: list[dict],
    anomaly_analysis: str,
    kg_context: str,
    recent_logs: list[str],
) -> str:
    """
    Generate a full diagnostic report using Gemini with streaming.
    Returns the complete report as a Markdown string.
    """
    prompt = _build_report_prompt(unit, cycle, anomalies, anomaly_analysis, kg_context, recent_logs)
    model  = _get_model()

    response = model.generate_content(prompt, stream=True)
    return "".join(chunk.text for chunk in response)


def generate_report_streaming(
    unit: int,
    cycle: int,
    anomalies: list[dict],
    anomaly_analysis: str,
    kg_context: str,
    recent_logs: list[str],
):
    """
    Generator version for Streamlit — yields text chunks as they arrive.
    """
    prompt = _build_report_prompt(unit, cycle, anomalies, anomaly_analysis, kg_context, recent_logs)
    model  = _get_model()

    for chunk in model.generate_content(prompt, stream=True):
        if chunk.text:
            yield chunk.text
