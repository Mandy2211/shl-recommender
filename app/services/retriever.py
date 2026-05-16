import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.catalog_loader import load_catalog


catalog = load_catalog()

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("app/data/faiss.index")


def retrieve_assessments(query, top_k=5):

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    recommendations = []

    for idx in indices[0]:

        item = catalog[idx]

        recommendations.append({
            "name": item.get("name"),
            "url": item.get("url"),
            "test_type": item.get("test_type", "Unknown")
        })

    return recommendations