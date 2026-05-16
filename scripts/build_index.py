import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.services.catalog_loader import load_catalog


catalog = load_catalog()

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []

for item in catalog:

    text = f"""
    Name: {item.get('name', '')}

    Description: {item.get('description', '')}

    Category: {item.get('category', '')}

    Test Type: {item.get('test_type', '')}
    """

    texts.append(text)


embeddings = model.encode(texts)

embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings) # type: ignore

faiss.write_index(index, "app/data/faiss.index")

np.save("app/data/embeddings.npy", embeddings)

print("FAISS index created successfully")