RAG‑pipeline på IKEA:s hållbarhetsrapport

Detta projekt implementerar en enkel RAG‑modell (Retrieval‑Augmented Generation) som kan svara på frågor baserat på IKEA:s hållbarhetsrapport för FY23. Modellen använder embeddings, FAISS‑indexering och en språkmodell (GPT‑4o‑mini) för att generera faktabaserade svar.

Projektstruktur
kunskapskontroll_2/
│
├── data/
│   └── ikea_sustainability_report_fy_23.pdf
│
├── models/
│   ├── faiss_index.bin        ← Skapas av embed.py (läggs ej upp på GitHub)
│   └── chunks.pkl             ← Skapas av embed.py (läggs ej upp på GitHub)
│
├── src/
│   ├── embed.py               ← Skapar embeddings + FAISS-index
│   ├── query.py               ← Söker relevanta textbitar
│   └── rag_chatbot.py         ← Enkel RAG-chatbot
│
├── .env.example               ← Mall för API-nyckel
└── README.md


Hur man kör projektet
pip install -r requirements.txt

Skapa .env:
OPENAI_API_KEY=din_nyckel_här

Skapa embeddings + FAISS-index:
python src/embed.py

Testa sökfunktionen:
python src/query.py

Starta RAG‑chatboten:
python src/rag_chatbot.py

Modelutvärdering:
python src/evaluate.py

Tekniker som används
LangChain – PDF‑laddning och textsplittring

SentenceTransformers – Embeddings

FAISS – Vektordatabas

OpenAI GPT‑4o‑mini – Generering av svar

Python – All logik
