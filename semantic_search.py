import hnswlib
from sentence_transformers import SentenceTransformer
import pickle

from config_loader import load_config
config = load_config()

file_cfg = config["filestorage"]

model = SentenceTransformer(file_cfg["sentence_tranformer"])
print("Model loaded")


print("Loading model")
import numpy as np
with open(file_cfg["document_pickle"], "rb") as f:
    chunks = pickle.load(f)
    
embeddings = np.load(file_cfg["embedding_file"])

# Load hnswlib index
dim = embeddings.shape[1]
index = hnswlib.Index(space='cosine', dim=dim)
index.load_index(file_cfg["hnslib_index"])
index.set_ef(50)


def semantic_search(query, top_k=10):
    query_emb = model.encode([query])
    labels, distances = index.knn_query(query_emb, k=top_k)
    results = []
    for idx, dist in zip(labels[0], distances[0]):
        results.append((chunks[idx], dist))
    return results
