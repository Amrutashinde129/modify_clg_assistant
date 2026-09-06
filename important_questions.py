from utils.ollama_client import ask_ollama


def generate_important_questions(
    text,
    number_of_questions=10
):

    # Limit text to avoid sending a huge PDF
    text = text[:6000]

    prompt = f"""
You are an AI College Assistant helping students
prepare for examinations.

Study Material:
{text}

Generate {number_of_questions} important
examination questions.

Divide them into:

1. Short Answer Questions
2. Long Answer Questions

For every question provide:
- Question
- Topic
- Priority: High / Medium / Low

Rules:
- Use only the provided study material.
- Focus on important concepts and definitions.
- Avoid duplicate questions.
- Do not invent information.
- Keep questions suitable for college examinations.
"""

    return ask_ollama(prompt)