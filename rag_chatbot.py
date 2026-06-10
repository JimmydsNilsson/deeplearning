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

# ------------------------------------------------------------
# FÖRDJUPAD – Kritisk analys av användning, möjligheter och utmaningar
# ------------------------------------------------------------

# Denna RAG-modell visar hur företag kan använda AI för att effektivisera
# informationssökning i stora dokument. I praktiken kan en sådan lösning
# spara mycket tid genom att snabbt ge faktabaserade svar utan att en
# användare behöver läsa hundratals sidor. Detta kan vara värdefullt för
# exempelvis beslutsfattare, kundsupport, HR, hållbarhetsarbete eller
# intern kommunikation.

# En tydlig möjlighet är att modellen kan integreras i interna system,
# intranät eller chattbotar och fungera som ett kunskapsnav. Det kan
# förbättra kvaliteten på beslutsunderlag, minska arbetsbelastning och
# skapa mer enhetliga svar inom organisationen. RAG-tekniken gör dessutom
# att modellen baserar sina svar på specifika dokument, vilket minskar
# risken för hallucinationer jämfört med en vanlig LLM.

# Samtidigt finns viktiga utmaningar. En central risk är att modellen bara
# är så bra som dokumenten den bygger på. Om informationen är gammal,
# felaktig eller ofullständig kan modellen ge missvisande svar. Det finns
# också etiska aspekter kring datalagring och integritet, särskilt om
# dokumenten innehåller känslig information. Organisationer måste därför
# ha tydliga riktlinjer för vilka dokument som får användas och hur de ska
# hanteras.

# En annan utmaning är att användare kan överskatta modellens förmåga och
# tro att den alltid har rätt. Det kräver utbildning och förståelse för
# att modellen inte ersätter mänsklig expertis, utan fungerar som ett
# stöd. Tekniskt sett kräver systemet också underhåll, uppdatering av
# dokument och övervakning av prestanda för att fortsätta vara relevant.

# Affärsmässigt kan RAG-lösningar ge stora vinster i effektivitet, men de
# kräver investeringar i både teknik och kompetens. Organisationer behöver
# också ta hänsyn till bias i dokumenten, eftersom modellen kan förstärka
# befintliga perspektiv eller brister i materialet.

# Sammanfattningsvis erbjuder RAG-modellen en kraftfull och praktiskt
# användbar lösning för informationssökning, men den måste implementeras
# med medvetenhet om risker, ansvar och etiska överväganden för att ge
# maximal nytta i verkliga verksamheter.
