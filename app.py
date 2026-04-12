"""
Streamlit UI — Multimodal Anomaly Detection Agent

Layout:
  Sidebar  → configuration (engine, cycle range, thresholds)
  Main     → sensor plots + anomaly highlights + streaming report
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv()

from src.data.loader import generate_synthetic_cmapss, load_engine_data, SENSOR_NAMES
from src.agents.anomaly_agent import detect_statistical_anomalies
from src.agents.kg_agent import query_knowledge_graph
from src.agents.report_agent import generate_report_streaming

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal Anomaly Detection Agent",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multimodal Anomaly Detection Agent")
st.caption("LangGraph + GraphRAG + Neo4j + Claude — HP Advanced AI Scientist Demo")

# ── Session state: generate dataset once ──────────────────────────────────────

@st.cache_data(show_spinner="Generating synthetic CMAPSS dataset...")
def load_dataset():
    return generate_synthetic_cmapss(n_engines=50, seed=42)


sensor_df_all, logs_df_all = load_dataset()
available_units = sorted(sensor_df_all["unit"].unique().tolist())

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    unit = st.selectbox("Engine Unit", available_units, index=4)

    unit_sensor = sensor_df_all[sensor_df_all["unit"] == unit]
    max_cycle   = int(unit_sensor["cycle"].max())
    min_cycle   = int(unit_sensor["cycle"].min())

    cycle_range = st.slider(
        "Cycle Range",
        min_value=min_cycle,
        max_value=max_cycle,
        value=(min_cycle, max_cycle),
    )

    zscore_threshold = st.slider("Anomaly Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
    window = st.slider("Rolling Window (cycles)", 5, 30, 10)

    sensors_to_plot = st.multiselect(
        "Sensors to Plot",
        SENSOR_NAMES,
        default=["T24", "T30", "T50", "P30", "Ps30", "W31"],
    )

    st.divider()
    st.caption("**Stack:** Claude Opus 4.6 · LangGraph · GraphRAG · Neo4j (optional)")

# ── Load engine-specific data ─────────────────────────────────────────────────

sensor_df, logs_df = load_engine_data(sensor_df_all, logs_df_all, unit, cycle_range)

col1, col2, col3 = st.columns(3)
col1.metric("Engine Unit", unit)
col2.metric("Cycles Observed", f"{cycle_range[0]} – {cycle_range[1]}")
col3.metric("Maintenance Logs", len(logs_df))

# ── Anomaly detection (fast, no LLM needed) ───────────────────────────────────

anomalies = detect_statistical_anomalies(sensor_df, zscore_threshold=zscore_threshold, window=window)
anomalous_sensors = {a["sensor"] for a in anomalies}

# ── Sensor Plots ──────────────────────────────────────────────────────────────

st.subheader("📈 Sensor Time-Series")

if sensors_to_plot:
    n_cols = 2
    n_rows = -(-len(sensors_to_plot) // n_cols)  # ceiling div
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=sensors_to_plot,
        vertical_spacing=0.08,
    )

    for i, sensor in enumerate(sensors_to_plot):
        row = i // n_cols + 1
        col = i % n_cols + 1
        is_anomalous = sensor in anomalous_sensors
        color = "#ef4444" if is_anomalous else "#3b82f6"

        fig.add_trace(
            go.Scatter(
                x=sensor_df["cycle"],
                y=sensor_df[sensor],
                mode="lines",
                name=sensor,
                line=dict(color=color, width=1.5),
                showlegend=False,
            ),
            row=row, col=col,
        )

        # Mark anomaly region (last 15% of cycles)
        if is_anomalous:
            thresh = int(max_cycle * 0.85)
            fig.add_vrect(
                x0=max(thresh, cycle_range[0]),
                x1=cycle_range[1],
                fillcolor="rgba(239,68,68,0.1)",
                line_width=0,
                row=row, col=col,
            )

    fig.update_layout(height=200 * n_rows, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ── Anomaly Table ──────────────────────────────────────────────────────────────

st.subheader("⚠️ Detected Anomalies")

if anomalies:
    anomaly_display = pd.DataFrame([{
        "Sensor":     a["sensor"],
        "Value":      a["latest_value"],
        "Z-Score":    f"{a['z_score']:+.2f}",
        "Direction":  a["direction"],
        "Severity":   a["severity"],
    } for a in anomalies])

    def highlight_severity(row):
        color = "#fee2e2" if row["Severity"] == "CRITICAL" else "#fef3c7"
        return [f"background-color: {color}"] * len(row)

    st.dataframe(
        anomaly_display.style.apply(highlight_severity, axis=1),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("✅ No anomalies detected in selected range. Engine operating normally.")

# ── Maintenance Logs ───────────────────────────────────────────────────────────

with st.expander("📋 Maintenance Logs (multimodal text input)"):
    if logs_df.empty:
        st.info("No logs in selected cycle range.")
    else:
        for _, row in logs_df.tail(10).iterrows():
            st.text(f"[Cycle {int(row['cycle'])}] {row['log_text']}")

# ── KG Context Preview ─────────────────────────────────────────────────────────

with st.expander("🔗 Knowledge Graph Context (GraphRAG)"):
    if anomalies:
        with st.spinner("Querying KG + GraphRAG..."):
            kg_context = query_knowledge_graph(anomalies)
        st.code(kg_context, language="text")
    else:
        st.info("No anomalies — KG query skipped.")

# ── Report Generation ──────────────────────────────────────────────────────────

st.subheader("📄 Diagnostic Report")

if not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY not set. Add it to your .env file.")
elif not anomalies:
    st.info("No anomalies detected. Run on an engine near end-of-life (high cycle count) to trigger anomalies.")
else:
    if st.button("🚀 Generate Diagnostic Report", type="primary"):
        with st.spinner("Running Agent 1 (anomaly analysis)..."):
            from src.agents.anomaly_agent import analyze_anomalies_with_claude
            anomaly_analysis = analyze_anomalies_with_claude(sensor_df, logs_df, anomalies, unit)

        with st.spinner("Running Agent 2 (KG query)..."):
            kg_context = query_knowledge_graph(anomalies)

        st.info("Running Agent 3 (report generation — streaming)...")
        cycle = int(sensor_df["cycle"].max())
        recent_logs = logs_df["log_text"].tail(5).tolist() if not logs_df.empty else []

        report_placeholder = st.empty()
        report_text = ""

        for chunk in generate_report_streaming(
            unit=unit,
            cycle=cycle,
            anomalies=anomalies,
            anomaly_analysis=anomaly_analysis,
            kg_context=kg_context,
            recent_logs=recent_logs,
        ):
            report_text += chunk
            report_placeholder.markdown(report_text)

        st.success("Report complete.")
        st.download_button(
            "⬇️ Download Report",
            data=report_text,
            file_name=f"diagnostic_report_unit{unit}_cycle{cycle}.md",
            mime="text/markdown",
        )
