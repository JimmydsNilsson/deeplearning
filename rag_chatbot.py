# ------------------------------------------------------------
# rag_chatbot.py – Enkel RAG-chatbot
# ------------------------------------------------------------

from query import search
from openai import OpenAI
import os
from dotenv import load_dotenv

# Ladda API-nyckel
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rag_answer(question):
    """
    Tar en fråga, hämtar relevanta textbitar och skickar allt till GPT-4o-mini.
    """

    # 1. Hämta relevanta chunks
    relevant_chunks = search(question, k=3)

    # 2. Bygg prompten
    context_text = "\n\n".join(relevant_chunks)

    prompt = f"""
Du är en hjälpsam assistent. Använd endast informationen nedan för att svara.

--- Kontext från dokumentet ---
{context_text}
--------------------------------

Fråga: {question}

Svara tydligt och kortfattat.
"""

    # 3. Skicka till GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Du är en faktabaserad assistent."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message["content"]


# Testkörning
if __name__ == "__main__":
    fråga = input("Ställ en fråga om IKEA-rapporten: ")
    svar = rag_answer(fråga)
    print("\n--- Svar från RAG-chatboten ---\n")
    print(svar)

    # ------------------------------------------------------------
# REFLEKTION – Hur modellen kan användas i verkligheten
# ------------------------------------------------------------

# Denna RAG-modell skulle kunna användas i företag för att snabbt söka fram
# relevant information ur stora dokument, t.ex. hållbarhetsrapporter,
# policydokument eller manualer. Istället för att en anställd behöver läsa
# hundratals sidor kan modellen ge snabba, faktabaserade svar.

# Möjligheter:
# - Effektivare informationssökning
# - Minskad arbetsbelastning
# - Bättre beslutsunderlag
# - Kan integreras i interna system eller chattbotar

# Utmaningar:
# - Kräver att dokumenten är uppdaterade och korrekta
# - Risk för att modellen misstolkar om kontexten är dålig
# - Etiska frågor kring datalagring och integritet
# - Kräver teknisk kompetens för att underhålla

# Affärsmässigt kan detta ge stora tidsbesparingar och bättre kvalitet i
# informationshantering, men kräver ansvarstagande och tydliga riktlinjer.