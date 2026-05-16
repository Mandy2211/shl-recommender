"""
retriever.py  (app/services/retriever.py)
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.catalog_loader import load_catalog

catalog = load_catalog()
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_index = faiss.read_index("app/data/faiss.index")

_MAX_DISTANCE = 1.5


def retrieve_assessments(query: str, top_k: int = 10) -> list[dict]:
    if not query.strip():
        return []

    top_k = min(top_k, len(catalog))

    query_embedding = np.array(_embed_model.encode([query])).astype("float32")
    distances, indices = _index.search(query_embedding, top_k)

    recommendations = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(catalog):
            continue
        if dist > _MAX_DISTANCE:
            continue

        item = catalog[idx]

        # ── Map actual catalog field names ───────────────────────────────────
        # link       → url
        # keys       → test_type  (list, e.g. ["Knowledge & Skills"])
        # job_levels → list of applicable levels
        keys = item.get("keys") or []
        test_type = ", ".join(keys) if keys else "Unknown"

        job_levels = item.get("job_levels") or []

        recommendations.append({
            "name":        item.get("name", ""),
            "url":         item.get("link", ""),
            "test_type":   test_type,
            "description": item.get("description", ""),
            "job_levels":  job_levels,
            "duration":    item.get("duration", ""),
            "remote":      item.get("remote", ""),
            "adaptive":    item.get("adaptive", ""),
        })

    return recommendations