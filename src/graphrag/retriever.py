"""
GraphRAG retriever: builds a TF-IDF index over KG subgraph descriptions,
retrieves relevant context based on anomaly descriptions.

This is the 'Graph' in GraphRAG — we represent subgraphs as text,
embed them, and retrieve the most relevant ones given an anomaly query.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.kg.builder import (
    SENSORS, COMPONENTS, FAILURE_MODES,
    SENSOR_COMPONENT, COMPONENT_FAILURE, SENSOR_FAILURE,
)


def _build_subgraph_documents() -> list[dict]:
    """
    Convert KG relationships into natural-language text documents.
    Each document describes a sensor → component → failure mode path.
    """
    docs = []
    sensor_map   = {s["id"]: s for s in SENSORS}
    comp_map     = {c["id"]: c for c in COMPONENTS}
    fm_map       = {f["id"]: f for f in FAILURE_MODES}

    # Path: sensor → component → failure_mode
    for sensor_id, comp_ids in SENSOR_COMPONENT.items():
        s = sensor_map[sensor_id]
        for comp_id in comp_ids:
            c = comp_map[comp_id]
            for fm_id in COMPONENT_FAILURE.get(comp_id, []):
                fm = fm_map[fm_id]
                text = (
                    f"Sensor {s['id']} ({s['name']}, type: {s['type']}) "
                    f"measures the {c['name']} component in the {c['subsystem']} subsystem. "
                    f"The {c['name']} can experience {fm['name']} "
                    f"(severity {fm['severity']}/5): {fm['description']}"
                )
                docs.append({
                    "text": text,
                    "sensor_id": sensor_id,
                    "component_id": comp_id,
                    "failure_mode_id": fm_id,
                    "sensor_name": s["name"],
                    "component_name": c["name"],
                    "failure_mode_name": fm["name"],
                    "severity": fm["severity"],
                    "fm_description": fm["description"],
                })

    # Direct sensor → failure_mode paths
    for sensor_id, fm_ids in SENSOR_FAILURE.items():
        s = sensor_map[sensor_id]
        for fm_id in fm_ids:
            fm = fm_map[fm_id]
            text = (
                f"Sensor {s['id']} ({s['name']}) directly indicates {fm['name']} "
                f"(severity {fm['severity']}/5): {fm['description']}"
            )
            docs.append({
                "text": text,
                "sensor_id": sensor_id,
                "component_id": None,
                "failure_mode_id": fm_id,
                "sensor_name": s["name"],
                "component_name": None,
                "failure_mode_name": fm["name"],
                "severity": fm["severity"],
                "fm_description": fm["description"],
            })

    return docs


class GraphRAGRetriever:
    """
    TF-IDF based retriever over KG subgraph descriptions.

    Usage:
        retriever = GraphRAGRetriever()
        context = retriever.retrieve(
            query="T50 temperature anomaly, LPT outlet overheating",
            top_k=5
        )
    """

    def __init__(self):
        self.docs = _build_subgraph_documents()
        self.corpus = [d["text"] for d in self.docs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, query: str, top_k: int = 6, sensor_filter: list[str] | None = None) -> list[dict]:
        """
        Retrieve top-k most relevant subgraph descriptions for the query.

        Args:
            query:         Natural language description of anomaly.
            top_k:         Number of results to return.
            sensor_filter: If set, only return results for these sensor IDs.

        Returns:
            List of relevant subgraph dicts with text and metadata.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        ranked_indices = np.argsort(scores)[::-1]
        results = []
        seen_paths = set()

        for idx in ranked_indices:
            doc = self.docs[idx]
            if sensor_filter and doc["sensor_id"] not in sensor_filter:
                continue
            path_key = (doc["sensor_id"], doc["failure_mode_id"])
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            results.append({**doc, "score": float(scores[idx])})
            if len(results) >= top_k:
                break

        return results

    def format_context(self, results: list[dict]) -> str:
        """Format retrieval results as a readable context block."""
        if not results:
            return "No relevant KG context found."

        lines = ["=== GraphRAG Retrieved Context ===\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r['text']}\n"
                f"     Relevance score: {r['score']:.3f}"
            )
        return "\n".join(lines)

    def retrieve_formatted(
        self,
        query: str,
        top_k: int = 6,
        sensor_filter: list[str] | None = None,
    ) -> str:
        results = self.retrieve(query, top_k=top_k, sensor_filter=sensor_filter)
        return self.format_context(results)
