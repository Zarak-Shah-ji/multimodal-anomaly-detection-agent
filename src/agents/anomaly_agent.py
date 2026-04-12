"""
Agent 1 — Anomaly Detector

Statistical detection (z-score) + Gemini reasoning over sensor time-series.
Returns a list of anomaly dicts with sensor, severity, description.
"""

import os
import google.generativeai as genai
import pandas as pd
from src.data.loader import SENSOR_NAMES, compute_rolling_stats

_model = None


def _get_model() -> genai.GenerativeModel:
    global _model
    if _model is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _model = genai.GenerativeModel("gemini-2.0-flash")
    return _model


def detect_statistical_anomalies(
    sensor_df: pd.DataFrame,
    zscore_threshold: float = 2.5,
    window: int = 10,
) -> list[dict]:
    """
    Z-score based anomaly detection over rolling window.

    Returns list of:
        {sensor, latest_value, z_score, mean, std, direction, severity}
    """
    df = compute_rolling_stats(sensor_df, window=window)
    latest = df.iloc[-1]
    anomalies = []

    for s in SENSOR_NAMES:
        if s not in df.columns:
            continue
        z_col = f"{s}_zscore"
        if z_col not in df.columns:
            continue

        z = latest.get(z_col, 0.0)
        if pd.isna(z):
            z = 0.0

        if abs(z) >= zscore_threshold:
            direction = "HIGH" if z > 0 else "LOW"
            severity  = "CRITICAL" if abs(z) >= 4.0 else "WARNING"
            anomalies.append({
                "sensor": s,
                "latest_value": round(float(latest[s]), 4),
                "z_score":      round(float(z), 3),
                "mean":         round(float(latest.get(f"{s}_mean", 0)), 4),
                "std":          round(float(latest.get(f"{s}_std", 0)), 6),
                "direction":    direction,
                "severity":     severity,
            })

    return sorted(anomalies, key=lambda x: abs(x["z_score"]), reverse=True)


def _build_anomaly_prompt(
    sensor_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    anomalies: list[dict],
    unit: int,
) -> str:
    """Build a compact prompt for Gemini to reason over detected anomalies."""
    n_cycles = len(sensor_df)
    latest_cycle = int(sensor_df["cycle"].max())

    anomaly_lines = "\n".join(
        f"  - {a['sensor']}: value={a['latest_value']}, z-score={a['z_score']} ({a['direction']}, {a['severity']})"
        for a in anomalies
    ) or "  None detected."

    recent_logs = logs_df.tail(3)["log_text"].tolist() if not logs_df.empty else []
    log_lines = "\n".join(f"  - {l}" for l in recent_logs) or "  No recent logs."

    return f"""You are an engine health monitoring AI.

Engine Unit: {unit}
Total Cycles Observed: {n_cycles}
Latest Cycle: {latest_cycle}

Statistically Anomalous Sensors (z-score > 2.5):
{anomaly_lines}

Recent Maintenance Logs:
{log_lines}

Task: In 3-4 sentences, explain what these sensor anomalies suggest about engine health.
Focus on: which subsystem is likely affected, trend direction, and urgency level.
Be specific and technical. Do not repeat raw numbers — interpret them."""


def analyze_anomalies_with_claude(
    sensor_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    anomalies: list[dict],
    unit: int,
) -> str:
    """
    Call Gemini to interpret the detected anomalies.
    Returns a short natural language analysis.
    """
    if not anomalies:
        return "No statistical anomalies detected. Engine appears to be operating within normal parameters."

    prompt = _build_anomaly_prompt(sensor_df, logs_df, anomalies, unit)
    model  = _get_model()

    response = model.generate_content(prompt)
    return response.text.strip()
