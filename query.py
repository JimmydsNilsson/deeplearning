# ------------------------------------------------------------
# query.py – Hämta relevanta textbitar från FAISS
# ------------------------------------------------------------

import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Ladda embeddings-modellen
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ladda FAISS-index
index = faiss.read_index("models/faiss_index.bin")

# Ladda chunks (textbitar)
with open("models/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


def search(query, k=3):
    """
    Tar en fråga (query) och returnerar de k mest relevanta textbitarna.
    """

    # Skapa embedding för frågan
    query_embedding = model.encode([query])

    # Sök i FAISS
    distances, indices = index.search(query_embedding, k)

    # Hämta texten från de matchande chunksen
    results = [chunks[i].page_content for i in indices[0]]

    return results


# Testkörning
if __name__ == "__main__":
    svar = search("Vad handlar rapporten om?")
    for i, text in enumerate(svar):
        print(f"\n--- Resultat {i+1} ---\n{text}\n")