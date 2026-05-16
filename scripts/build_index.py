"""
build_index.py  (scripts/build_index.py)
Run once to build the FAISS index from your catalog.
  python -m scripts.build_index
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.catalog_loader import load_catalog

catalog = load_catalog()
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
for item in catalog:
    keys = item.get("keys") or []
    test_type = ", ".join(keys) if keys else ""

    job_levels = item.get("job_levels") or []
    levels_str = ", ".join(job_levels) if job_levels else ""

    text = f"""
    Name: {item.get('name', '')}
    Description: {item.get('description', '')}
    Test Type: {test_type}
    Job Levels: {levels_str}
    Duration: {item.get('duration', '')}
    Remote: {item.get('remote', '')}
    Adaptive: {item.get('adaptive', '')}
    """
    texts.append(text)

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)  # type: ignore

faiss.write_index(index, "app/data/faiss.index")
np.save("app/data/embeddings.npy", embeddings)

print(f"FAISS index built: {len(texts)} entries, dimension {dimension}")