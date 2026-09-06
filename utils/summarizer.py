from utils.ollama_client import ask_ollama
import re


def summarize_text(text):

    # Limit the amount of text sent to Ollama
    max_chars = 12000
    text = text[:max_chars]

    prompt = f"""
You are an AI College Assistant.

Create study notes from the following college material.

College Material:
{text}

IMPORTANT:
Divide the material into meaningful topics.

For EACH topic use exactly this format:

### TOPIC: Topic Name

## 📚 Topic Overview
- Short explanation

## 🔑 Important Concepts
- Point
- Point

## 📖 Key Definitions
- Definition

## ⭐ Important Points
- Point
- Point

## 💡 Examples
- Example

## 📝 Quick Revision
- Short revision points

### END TOPIC

Rules:
- Create multiple topics whenever the material contains different concepts.
- Use simple language.
- Use bullet points.
- Keep each topic concise.
- Use ONLY the provided material.
- Do not invent information.
- Do not combine unrelated concepts into one topic.
"""

    result = ask_ollama(prompt)

    return result