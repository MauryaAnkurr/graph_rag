import hnswlib
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pickle
import numpy as np
import requests

from config_loader import load_config
config = load_config()

file_cfg = config["filestorage"]


text = requests.get(file_cfg["file_url"]).text

# Save the text to a file
with open(file_cfg["local_path"], "w", encoding="utf-8") as file:
    file.write(text)
    
def split_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits the input text into smaller chunks for processing.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return text_splitter.split_text(text)

# Use your existing chunking function
chunks = split_text(text)  

print("Length of chunks create", len(chunks))

# Load embedding model
model = SentenceTransformer(file_cfg["sentence_tranformer"])

# Embed all chunks
embeddings = model.encode(chunks, show_progress_bar=True)

# Initialize hnswlib index
dim = embeddings.shape[1]
num_elements = len(embeddings)
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=num_elements, ef_construction=200, M=16)
index.add_items(embeddings, list(range(num_elements)))
index.set_ef(50) 


with open(file_cfg["document_pickle"], "wb") as f:
    pickle.dump(chunks, f)
print("Documents saved")

np.save(file_cfg["embedding_file"], embeddings)
print("Embeddings saved")

index.save_index(file_cfg["hnslib_index"])
print("HNSWlin index saved")

