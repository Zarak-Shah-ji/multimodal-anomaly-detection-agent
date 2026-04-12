# Multimodal Anomaly Detection Agent

**HP Advanced AI Scientist Interview Demo**

A production-quality demo combining time-series sensor data, text maintenance logs, a Neo4j knowledge graph, GraphRAG retrieval, and a 3-agent LangGraph pipeline — all surfaced through a Streamlit UI.

```
Architecture:
  Sensor Data (time-series)  ─┐
  Maintenance Logs (text)    ─┤─► LangGraph Pipeline ─► Diagnostic Report
  Knowledge Graph (Neo4j)   ─┘
       │
       ▼
  [Agent 1] Anomaly Detector    (Z-score + Claude reasoning)
       │
       ▼
  [Agent 2] KG Querier          (Neo4j Cypher + GraphRAG TF-IDF retrieval)
       │
       ▼
  [Agent 3] Report Generator    (Claude Opus 4.6 streaming)
```

## Gaps Covered

| Gap | How |
|-----|-----|
| Multimodal AI (telemetry + text) | Sensor time-series + maintenance logs ingested together |
| Knowledge Graphs | Neo4j KG: Sensor → Component → FailureMode |
| GraphRAG | TF-IDF index over KG subgraph descriptions, retrieved by anomaly query |
| LangGraph | 3-node agent pipeline with typed state |
| LLM fine-tuning concepts | Adaptive thinking on Claude Opus 4.6 |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add:
#   ANTHROPIC_API_KEY=your_key
#   NEO4J_URI=bolt://localhost:7687
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=your_password
```

### 3. Start Neo4j (optional but recommended)
```bash
docker run -d \
  --name neo4j-demo \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

### 4. Build the Knowledge Graph
```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from src.kg.builder import KGBuilder
kg = KGBuilder(os.environ['NEO4J_URI'], os.environ['NEO4J_USERNAME'], os.environ['NEO4J_PASSWORD'])
kg.build()
kg.close()
"
```
> **Note:** If Neo4j is unavailable, the app falls back to a static KG — GraphRAG still works.

### 5. Run the app
```bash
streamlit run app.py
```

## Project Structure

```
multimodal_anomaly_detection_agent/
├── app.py                    # Streamlit UI
├── requirements.txt
├── .env.example
├── data/                     # Generated datasets (gitignored)
└── src/
    ├── data/
    │   └── loader.py         # Synthetic CMAPSS dataset generator
    ├── kg/
    │   └── builder.py        # Neo4j KG schema + queries
    ├── graphrag/
    │   └── retriever.py      # TF-IDF GraphRAG retriever
    ├── agents/
    │   ├── anomaly_agent.py  # Agent 1: statistical detection + Claude
    │   ├── kg_agent.py       # Agent 2: Neo4j + GraphRAG
    │   └── report_agent.py   # Agent 3: Claude streaming report
    └── pipeline.py           # LangGraph orchestration
```

## Dataset

Synthetic NASA C-MAPSS-style turbofan engine data:
- **21 sensors**: temperature, pressure, speed, flow metrics
- **3 operational settings** (clustered)
- **50 engines**, 100-250 cycles each
- **Text maintenance logs** generated at key intervals (multimodal)
- **Anomalies injected** in final 15% of engine lifetime

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  [Engine Select] [Cycle Range] [Threshold] [Plot]       │
└────────────────────────┬────────────────────────────────┘
                         │ user triggers pipeline
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Pipeline (src/pipeline.py)       │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Node 1: Anomaly Detection                        │  │
│  │  • Rolling z-score over 21 sensors               │  │
│  │  • Claude Opus 4.6 (adaptive thinking)           │  │
│  │    interprets pattern                            │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │ anomalies[]                       │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │ Node 2: KG Query                                 │  │
│  │  • Neo4j Cypher: sensor→component→failure_mode   │  │
│  │  • GraphRAG TF-IDF retrieval from KG subgraphs   │  │
│  │  • Fallback to static KG if Neo4j unavailable    │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │ kg_context (text)                 │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │ Node 3: Report Generation                        │  │
│  │  • Claude Opus 4.6 streaming                     │  │
│  │  • Structured Markdown report                    │  │
│  │  • Risk level + recommended actions              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Diagnostic Report  │
              │  (Markdown, stream) │
              └─────────────────────┘
```

## Key Talking Points for HP Interview

1. **Multimodal**: Combines continuous sensor readings (time-series) with discrete text logs — two modalities fused at the agent level
2. **Knowledge Graphs**: Domain knowledge encoded as a property graph (Neo4j) rather than embedded in prompts — more maintainable and queryable
3. **GraphRAG**: Retrieves relevant KG subgraphs using semantic similarity rather than exact-match Cypher — handles natural language anomaly descriptions
4. **LangGraph**: Typed state machine with explicit data flow — production pattern for multi-agent systems
5. **LLM usage**: Claude Opus 4.6 with adaptive thinking for both interpretation (Agent 1) and report generation (Agent 3 streaming)
