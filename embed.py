# ------------------------------------------------------------
# embed.py – Skapar embeddings och FAISS-index från IKEA:s hållbarhetsrapport
# ------------------------------------------------------------

# För att läsa PDF-dokument och omvandla dem till text
from langchain_community.document_loaders import PyPDFLoader

# För att dela upp text i mindre bitar (chunks) som LLM:er kan hantera
from langchain_text_splitters import RecursiveCharacterTextSplitter

# För att skapa embeddings (text -> numeriska vektorer)
from sentence_transformers import SentenceTransformer

# För att skapa och söka i en vektordatabas (FAISS)
import faiss

# För att spara och ladda metadata (t.ex. chunks och embeddings)
import pickle

# För att hantera API-nycklar via .env-fil
from dotenv import load_dotenv
import os

# För att anropa OpenAI:s modeller (GPT-4o-mini m.fl.)
from openai import OpenAI


# ------------------------------------------------------------
# Skapa embeddings + FAISS-index för IKEA:s hållbarhetsrapport
# ------------------------------------------------------------

def create_embeddings():

    # 1. Ladda PDF
    print("Läser PDF...")
    loader = PyPDFLoader("data/ikea_sustainability_report_fy_23.pdf")
    documents = loader.load()

    # 2. Dela upp text i chunks
    print("Chunkar text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 3. Skapa embeddings med SentenceTransformer
    print("Skapar embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts)

    # 4. Skapa FAISS-index
    print("Bygger FAISS-index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 5. Spara index + metadata
    print("Sparar index och metadata...")
    faiss.write_index(index, "models/faiss_index.bin")

    with open("models/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Klart! Embeddings + FAISS sparade.")


if __name__ == "__main__":
    create_embeddings()