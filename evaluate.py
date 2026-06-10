# ------------------------------------------------------------
# evaluate.py – Enkel utvärdering av RAG-chatboten
# ------------------------------------------------------------

from rag_chatbot import rag_answer

# Några testfrågor och förväntade nyckelord i svaren
tests = [
    {
        "question": "Vad handlar IKEA:s hållbarhetsrapport om?",
        "expected_keywords": ["hållbarhet", "sustainability", "miljö"]
    },
    {
        "question": "Vilket år gäller rapporten?",
        "expected_keywords": ["2023", "FY23"]
    },
    {
        "question": "Vilka fokusområden nämns i rapporten?",
        "expected_keywords": ["klimat", "resurser", "människor"]
    }
]

def evaluate():
    print("\n--- Utvärdering av RAG-modellen ---\n")

    for i, test in enumerate(tests):
        print(f"Test {i+1}: {test['question']}")
        answer = rag_answer(test["question"])
        print("Svar:", answer)

        # Kontrollera om något av nyckelorden finns i svaret
        passed = any(keyword.lower() in answer.lower() for keyword in test["expected_keywords"])

        if passed:
            print("Resultat: ✔ Godkänt – svaret innehåller relevanta nyckelord.\n")
        else:
            print("Resultat: ✘ Underkänt – svaret matchar inte förväntad kontext.\n")

if __name__ == "__main__":
    evaluate()
