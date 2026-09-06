import time

try:
    from utils.ollama_client import ask_gemini
except ImportError:
    ask_gemini = None

from utils.ollama_client import ask_ollama


def generate_mcqs(text, number_of_questions=5):
    """Generate well-formatted MCQs from uploaded study material.

    The input is limited to keep Gemini requests smaller and more reliable.
    Temporary 503/UNAVAILABLE responses are retried automatically.
    """

    if not text or not str(text).strip():
        return "⚠️ No study material is available. Please upload a PDF first."

    # Keep the request reasonably small. This also matches the approach used
    # by the working summarizer and reduces temporary model overload failures.
    study_material = str(text)[:12000]

    prompt = f"""
You are an AI College Assistant.

Generate exactly {number_of_questions} multiple-choice questions from the
college study material below.

STUDY MATERIAL:
{study_material}

OUTPUT FORMAT — FOLLOW EXACTLY:

### Question 1

Question text here?

**A)** Option A

**B)** Option B

**C)** Option C

**D)** Option D

**Correct Answer:** B

**Explanation:** Short explanation.

### Question 2

Question text here?

**A)** Option A

**B)** Option B

**C)** Option C

**D)** Option D

**Correct Answer:** A

**Explanation:** Short explanation.

Continue this format until exactly {number_of_questions} questions are given.

IMPORTANT RULES:
- Use ONLY information present in the study material.
- Do not invent facts.
- Make questions suitable for college students.
- Avoid duplicate or nearly duplicate questions.
- Every question must have exactly four options: A, B, C and D.
- Put every question on its own section.
- Put every option on a separate line.
- Put Correct Answer on a separate line.
- Put Explanation on a separate line.
- Leave a blank line between each item.
- Use Markdown formatting as shown above.
- Do not put multiple questions, options, answers, or explanations on one line.
- Do not use code fences.
"""

    # Gemini can temporarily return 503 when a model is under high demand.
    # Retry with increasing delays before reporting the error to the user.
    last_response = None

    for attempt in range(3):
        try:
            if ask_gemini is not None:
                response = ask_gemini(prompt)
            else:
                response = ask_ollama(prompt)

            response_text = str(response).strip()
            last_response = response_text

            if "503" not in response_text and "UNAVAILABLE" not in response_text:
                return response_text

        except Exception as e:
            last_response = str(e)

        if attempt < 2:
            time.sleep(2 ** attempt * 2)

    return (
        "⚠️ Gemini is temporarily busy (503 UNAVAILABLE).\n\n"
        "The MCQ request was retried automatically, but the model is still "
        "under high demand. Please wait 10–30 seconds and click "
        "**Generate MCQs** again.\n\n"
        "Details: {last_response}"
    )
