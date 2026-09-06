from utils.ollama_client import ask_ollama


def answer_question(vector_store, question):

    try:
        documents = vector_store.similarity_search(
            question,
            k=5
        )

        if not documents:
            return "This information is not available in the uploaded document."

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are an AI College Assistant.

Answer the user's question using only the college notes provided below.

COLLEGE NOTES:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Give a simple and clear answer.
- Use only the information from the college notes.
- You may combine information from multiple sections.
- Do not invent information.
- If the answer genuinely cannot be found in the notes, say:
"This information is not available in the uploaded document."

ANSWER:
"""

        return ask_ollama(prompt)

    except Exception as e:
        return f"❌ RAG error: {str(e)}"