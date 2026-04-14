# Project Overview: Multimodal Anomaly Detection Agent

## What It Does
A demo system for detecting anomalies in turbofan engine sensor data (NASA C-MAPSS dataset) using a 3-agent LangGraph pipeline. Sensor time-series and maintenance text logs flow through sequential agents — statistical detection → knowledge graph enrichment → AI-generated diagnostic report — all surfaced through a Streamlit UI.

## Tech Stack
- **Language:** Python 3
- **UI:** Streamlit + Plotly
- **Agent orchestration:** LangGraph (`StateGraph`)
- **LLM:** Google Gemini (`google-generativeai`) for anomaly analysis & report generation
- **Graph DB:** Neo4j (`neo4j>=5.0`) for knowledge graph
- **RAG:** TF-IDF-based GraphRAG over KG subgraph descriptions (`scikit-learn`)
- **Data:** Synthetic C-MAPSS sensor data via `pandas`, `numpy`, `scipy`
- **Config:** `python-dotenv` — requires `.env` with `GEMINI_API_KEY` + `NEO4J_*`

## Architecture

**Agent pipeline pattern** — LangGraph `StateGraph` with typed `PipelineState` flowing through 3 sequential nodes:

```
Sensor Data + Logs
       │
[anomaly_detection_node]  → Z-score stats + Gemini interpretation
       │
[kg_query_node]           → Neo4j Cypher + GraphRAG TF-IDF retrieval
       │
[report_generation_node]  → Streaming diagnostic report (Gemini)
```

The Streamlit app (`app.py`) runs the pipeline interactively, letting users pick an engine unit and trigger report generation.

## Key Files & Entry Points

| File | Role |
|------|------|
| `app.py` | Streamlit UI — entry point (`streamlit run app.py`) |
| `src/pipeline.py` | LangGraph `StateGraph` wiring all 3 agent nodes |
| `src/agents/anomaly_agent.py` | Z-score detection + Gemini anomaly analysis |
| `src/agents/kg_agent.py` | Neo4j Cypher queries + GraphRAG retrieval |
| `src/agents/report_agent.py` | Streaming report generation via Gemini |
| `src/data/loader.py` | Synthetic C-MAPSS data generator + loader |
| `src/kg/builder.py` | Builds the Neo4j knowledge graph |
| `src/graphrag/retriever.py` | TF-IDF index over KG subgraph descriptions |
| `.env.example` | Template for required secrets |

## How to Run

```bash
# Install
pip install -r requirements.txt

# Configure secrets
cp .env.example .env   # fill in GEMINI_API_KEY + NEO4J_*

# Run UI
streamlit run app.py   # → http://localhost:8501
```

## What to Know Before Changing Code
- **No Neo4j = degraded mode**: the KG agent gracefully skips if Neo4j is unreachable; set `NEO4J_URI=bolt://localhost:7687` and run Neo4j locally or via Docker
- **`=0.8.0`** in the project root is a stray pip install log — ignore it (gitignored)
- The pipeline in `src/pipeline.py` is used by tests/scripts; `app.py` calls agents directly for streaming support (bypasses LangGraph for the report step)
