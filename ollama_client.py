import os
import streamlit as st
from google import genai


# =========================================================
# GET GEMINI API KEY
# =========================================================

def get_gemini_api_key():

    try:
        if "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]

            if key:
                return str(key).strip()

    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY")

    if key:
        return key.strip()

    return None


# =========================================================
# CREATE GEMINI CLIENT
# =========================================================

def get_gemini_client():

    api_key = get_gemini_api_key()

    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)

    except Exception:
        return None


# =========================================================
# FIND AVAILABLE GEMINI MODEL
# =========================================================

def get_available_model():

    client = get_gemini_client()

    if client is None:
        return None

    try:

        models = client.models.list()

        # Preferred models, in order
        preferred_models = [
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
        ]

        available = []

        for model in models:

            name = getattr(model, "name", "")

            if not name:
                continue

            # Remove "models/" prefix
            clean_name = name.replace("models/", "")

            # Check generateContent support
            supported_actions = getattr(
                model,
                "supported_actions",
                []
            )

            if (
                "generateContent" in supported_actions
                or not supported_actions
            ):
                available.append(clean_name)

        # First try preferred models
        for preferred in preferred_models:

            if preferred in available:
                return preferred

        # Otherwise use any Flash model
        for model in available:

            if "flash" in model.lower():
                return model

        # Last resort
        if available:
            return available[0]

        return None

    except Exception:
        return None


# =========================================================
# ASK GEMINI
# =========================================================

def ask_gemini(prompt):

    api_key = get_gemini_api_key()

    # -----------------------------------------------------
    # Check API key
    # -----------------------------------------------------

    if not api_key:

        return (
            "❌ Gemini API key is missing.\n\n"
            "Please add your Gemini API key to:\n\n"
            ".streamlit/secrets.toml"
        )

    # Detect placeholder
    if api_key in [
        "YOUR_KEY",
        "YOUR_API_KEY",
        "YOUR_GEMINI_API_KEY",
        "PASTE_YOUR_KEY_HERE"
    ]:

        return (
            "❌ Gemini API key is still a placeholder.\n\n"
            "Replace YOUR_KEY with your actual Gemini API key."
        )

    # -----------------------------------------------------
    # Check prompt
    # -----------------------------------------------------

    if prompt is None:
        return "⚠️ Empty prompt received."

    prompt = str(prompt).strip()

    if not prompt:
        return "⚠️ Empty prompt received."

    # -----------------------------------------------------
    # Create client
    # -----------------------------------------------------

    try:

        client = genai.Client(
            api_key=api_key
        )

    except Exception as e:

        return (
            "❌ Unable to initialize Gemini.\n\n"
            f"{str(e)}"
        )

    # -----------------------------------------------------
    # Find available model
    # -----------------------------------------------------

    model_name = get_available_model()

    if not model_name:

        return (
            "❌ No Gemini text-generation model is available "
            "for this API key.\n\n"
            "Please check your Google AI Studio project "
            "and API key permissions."
        )

    # -----------------------------------------------------
    # Generate response
    # -----------------------------------------------------

    try:

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

        return "⚠️ Gemini returned an empty response."

    except Exception as e:

        error_message = str(e)
        error_lower = error_message.lower()

        if (
            "401" in error_message
            or "api key" in error_lower
            or "api_key" in error_lower
            or "unauthenticated" in error_lower
            or "authentication" in error_lower
        ):

            return (
                "❌ Gemini Authentication Error.\n\n"
                "The API key was rejected by Google.\n\n"
                "Create a new API key in Google AI Studio "
                "and update .streamlit/secrets.toml."
            )

        if (
            "403" in error_message
            or "permission_denied" in error_lower
            or "permission denied" in error_lower
        ):

            return (
                "❌ Gemini Permission Error.\n\n"
                "Your API key/project does not have permission "
                "to use the selected Gemini model."
            )

        if (
            "404" in error_message
            or "not_found" in error_lower
            or "not found" in error_lower
        ):

            return (
                "❌ Gemini Model Error.\n\n"
                f"The automatically selected model "
                f"'{model_name}' is unavailable.\n\n"
                "Please check the models available to your API key."
            )

        if (
            "429" in error_message
            or "resource_exhausted" in error_lower
            or "quota" in error_lower
        ):

            return (
                "⚠️ Gemini quota/rate limit reached.\n\n"
                "Please wait and try again."
            )

        return (
            "❌ Gemini API Error.\n\n"
            f"{error_message}"
        )


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

def ask_ollama(prompt):

    return ask_gemini(prompt)


# =========================================================
# TEST CONNECTION
# =========================================================

def test_gemini_connection():

    api_key = get_gemini_api_key()

    if not api_key:
        return False, "GEMINI_API_KEY is missing."

    client = get_gemini_client()

    if client is None:
        return False, "Unable to initialize Gemini client."

    model_name = get_available_model()

    if not model_name:
        return False, "No usable Gemini model found."

    try:

        response = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: Gemini connection successful."
        )

        if response and response.text:

            return True, (
                f"Model: {model_name}\n"
                f"Response: {response.text.strip()}"
            )

        return False, "Gemini returned an empty response."

    except Exception as e:

        return False, str(e)