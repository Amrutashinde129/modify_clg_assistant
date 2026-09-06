import streamlit as st
import os
import re
from datetime import datetime, date, timedelta

# =========================================================
# IMPORTS
# =========================================================

from db_manager import (
    get_student_profile,
    get_chat_history,
    create_table,
    save_student_profile,
    save_chat,
)

from utils.pdf_processor import extract_text_from_pdf
from utils.ollama_client import ask_gemini
from utils.mcq_generator import generate_mcqs
from utils.important_questions import generate_important_questions
from utils.rag import answer_question
from utils.summarizer import summarize_text


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DATABASE
# =========================================================

create_table()

profile = get_student_profile() or {}
history = get_chat_history()[:5]

name = profile.get("name") or "Student"
branch = profile.get("branch") or "Computer Engineering"
semester = profile.get("semester") or "Semester Not Set"
college = profile.get("college") or "College Not Set"

raw_exam_date = profile.get("exam_date")
exam_date = None

if raw_exam_date:
    try:
        if isinstance(raw_exam_date, date):
            exam_date = raw_exam_date
        else:
            exam_date = datetime.strptime(
                str(raw_exam_date),
                "%Y-%m-%d"
            ).date()
    except (ValueError, TypeError):
        exam_date = None

try:
    study_hours = float(profile.get("study_hours") or 0)
except (ValueError, TypeError):
    study_hours = 0.0


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "page": "dashboard",
    "chat_messages": [],
    "rag_chat": [],
    "study_plan": None,
    "uploaded_files": [],
    "pdf_text": "",
    "vector_store": None,
    "completed_tasks": set(),
    "generated_tasks": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPERS
# =========================================================

def safe_html(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_html(content):
    """
    Render custom HTML using Streamlit HTML renderer.
    """
    st.html(content)


def go_to(page):
    st.session_state.page = page
    st.rerun()


def render_back_button():
    if st.button(
        "⬅️ Back to Dashboard",
        key=f"back_{st.session_state.page}",
    ):
        go_to("dashboard")


def create_default_tasks():
    return [
        {
            "id": "task_1",
            "title": "Review today's lecture notes",
            "subject": branch,
        },
        {
            "id": "task_2",
            "title": "Practice 10 important questions",
            "subject": "Exam Preparation",
        },
        {
            "id": "task_3",
            "title": "Complete one focused study session",
            "subject": "Study Goal",
        },
        {
            "id": "task_4",
            "title": "Ask AI Assistant one doubt",
            "subject": "Smart Learning",
        },
    ]


# =========================================================
# IMPORTANT QUESTIONS FORMATTER
# =========================================================

def render_important_questions(text):
    """
    Converts AI-generated important-question text into
    separate attractive question cards.
    """

    if not text:
        st.warning("No questions were generated.")
        return

    # Remove code fences
    text = re.sub(
        r"```(?:markdown|text)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "").strip()

    # -----------------------------------------------------
    # Extract sections
    # -----------------------------------------------------

    short_match = re.search(
        r"(?:###\s*)?(?:\*\*)?\s*1\.\s*Short Answer Questions.*?"
        r"(?=(?:###\s*)?(?:\*\*)?\s*2\.\s*Long Answer Questions|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    long_match = re.search(
        r"(?:###\s*)?(?:\*\*)?\s*2\.\s*Long Answer Questions.*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    short_text = short_match.group(0) if short_match else ""
    long_text = long_match.group(0) if long_match else ""

    # If AI did not create sections
    if not short_text and not long_text:
        short_text = text

    # -----------------------------------------------------
    # Extract individual questions
    # -----------------------------------------------------

    def extract_questions(section_text):

        questions = []

        pattern = re.compile(
            r"(?:####\s*)?\*?\*?Question\s*(\d+)\*?\*?"
            r"(.*?)(?=(?:####\s*)?\*?\*?Question\s*\d+|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )

        matches = pattern.findall(section_text)

        for number, content in matches:

            content = content.strip()

            # Question
            q_match = re.search(
                r"\*?\*?Question:\s*\*?\*?(.*?)(?="
                r"\*?\*?Topic:|\*?\*?Priority:|$)",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )

            # Topic
            topic_match = re.search(
                r"\*?\*?Topic:\s*\*?\*?(.*?)(?="
                r"\*?\*?Priority:|$)",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )

            # Priority
            priority_match = re.search(
                r"\*?\*?Priority:\s*\*?\*?(.*?)(?=\n|$)",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if q_match:
                question = q_match.group(1).strip()
            else:
                question = content

            if topic_match:
                topic = topic_match.group(1).strip()
            else:
                topic = "General"

            if priority_match:
                priority = priority_match.group(1).strip()
            else:
                priority = "Medium"

            # Remove Markdown
            question = re.sub(
                r"\*\*",
                "",
                question
            )

            topic = re.sub(
                r"\*\*",
                "",
                topic
            )

            priority = re.sub(
                r"\*\*",
                "",
                priority
            )

            # Remove unwanted bullets
            question = re.sub(
                r"^[\-\*\•\s]+",
                "",
                question
            )

            topic = re.sub(
                r"^[\-\*\•\s]+",
                "",
                topic
            )

            priority = re.sub(
                r"^[\-\*\•\s]+",
                "",
                priority
            )

            # Normalize spaces
            question = re.sub(
                r"\s+",
                " ",
                question
            ).strip()

            topic = re.sub(
                r"\s+",
                " ",
                topic
            ).strip()

            priority = re.sub(
                r"\s+",
                " ",
                priority
            ).strip()

            questions.append(
                {
                    "number": number,
                    "question": question,
                    "topic": topic,
                    "priority": priority,
                }
            )

        return questions

    short_questions = extract_questions(short_text)
    long_questions = extract_questions(long_text)

    # -----------------------------------------------------
    # Render section
    # -----------------------------------------------------

    def render_section(title, icon, questions):

        if not questions:
            return

        render_html(
            f"""
            <div class="question-section-title">

                <div>
                    {icon}
                    {title}
                </div>

                <span class="question-count">
                    {len(questions)} Questions
                </span>

            </div>
            """
        )

        for q in questions:

            priority = q["priority"].lower()

            if "high" in priority:

                priority_class = "priority-high"
                priority_icon = "🔴"

            elif "medium" in priority:

                priority_class = "priority-medium"
                priority_icon = "🟡"

            else:

                priority_class = "priority-low"
                priority_icon = "🟢"

            render_html(
                f"""
                <div class="important-question-card">

                    <div class="question-top">

                        <div class="question-number">
                            Q{safe_html(q["number"])}
                        </div>

                        <div class="question-priority {priority_class}">
                            {priority_icon}
                            {safe_html(q["priority"])}
                        </div>

                    </div>

                    <div class="question-text">
                        {safe_html(q["question"])}
                    </div>

                    <div class="question-topic">
                        📚 <b>Topic:</b>
                        {safe_html(q["topic"])}
                    </div>

                </div>
                """
            )

    # -----------------------------------------------------
    # Display sections
    # -----------------------------------------------------

    render_section(
        "Short Answer Questions",
        "📝",
        short_questions,
    )

    render_section(
        "Long Answer Questions",
        "📖",
        long_questions,
    )

    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    if not short_questions and not long_questions:

        render_html(
            f"""
            <div class="card">

                <div style="
                    font-weight:700;
                    color:#dc2626;
                    margin-bottom:10px;
                ">
                    ⚠️ Could not format the generated questions.
                </div>

                <div style="
                    color:#475569;
                    white-space:pre-wrap;
                ">
                    {safe_html(text)}
                </div>

            </div>
            """
        )


# =========================================================
# CUSTOM CSS
# =========================================================

CSS = """
<style>

/* =====================================================
   GLOBAL
===================================================== */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.stApp {
    background: #f8fafc;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}


/* =====================================================
   SIDEBAR
===================================================== */

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: #f8fafc !important;
    color: #475569 !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    text-align: left !important;
    font-weight: 600 !important;
    padding: 12px 14px !important;
    margin: 4px 0 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #eef2ff !important;
    color: #4f46e5 !important;
    border-color: #c7d2fe !important;
}

.sidebar-logo {
    font-size: 1.35rem;
    font-weight: 800;
    color: #4f46e5;
    padding: 10px 2px 2px;
}

.sidebar-subtitle {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-bottom: 18px;
}

.sidebar-mini {
    background: #f8fafc;
    color: #334155;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 15px;
    margin-top: 14px;
}


/* =====================================================
   TITLES
===================================================== */

.main-title {
    font-size: 2.25rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 5px;
}

.subtitle {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 20px;
}

.section-header {
    font-size: 1.25rem;
    font-weight: 750;
    color: #1e293b;
    margin: 28px 0 15px;
}


/* =====================================================
   DASHBOARD HERO
===================================================== */

.dash-header {
    background: linear-gradient(
        135deg,
        #4f46e5 0%,
        #7c3aed 100%
    );

    border-radius: 24px;
    padding: 32px;
    color: white;
    margin-bottom: 25px;

    box-shadow:
        0 18px 35px rgba(79,70,229,0.16);

    position: relative;
    overflow: hidden;
}

.dash-header::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -80px;
    top: -100px;
    border-radius: 50%;
    background: rgba(255,255,255,0.10);
}

.dash-welcome {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.dash-date {
    font-size: 1rem;
    opacity: 0.9;
}


/* =====================================================
   STAT CARDS
===================================================== */

.stat-card {
    background: #ffffff;
    border-radius: 17px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    min-height: 145px;

    box-shadow:
        0 5px 12px rgba(15,23,42,0.05);

    transition: 0.2s;
}

.stat-card:hover {
    transform: translateY(-3px);

    box-shadow:
        0 12px 25px rgba(15,23,42,0.08);
}

.stat-icon {
    font-size: 1.8rem;
    display: block;
}

.stat-label {
    color: #64748b;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-top: 8px;
}

.stat-value {
    color: #1e293b;
    font-size: 1.7rem;
    font-weight: 800;
    margin: 4px 0;
}

.stat-sub {
    color: #94a3b8;
    font-size: 0.8rem;
}


/* =====================================================
   PDF UPLOAD
===================================================== */

.pdf-upload-card {
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #f5f3ff 100%
    );

    border: 2px dashed #818cf8;
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 15px;

    box-shadow:
        0 8px 20px rgba(79,70,229,0.07);
}

.pdf-upload-icon {
    font-size: 2.8rem;
    margin-bottom: 7px;
}

.pdf-upload-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #312e81;
}

.pdf-upload-text {
    color: #6366f1;
    font-size: 0.92rem;
    line-height: 1.6;
    margin-top: 5px;
}

.pdf-feature {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #e0e7ff;
    height: 100%;
}

.pdf-feature-icon {
    font-size: 1.7rem;
}

.pdf-feature-title {
    font-weight: 750;
    color: #1e293b;
    margin-top: 7px;
}

.pdf-feature-text {
    color: #64748b;
    font-size: 0.85rem;
    line-height: 1.8;
}


/* =====================================================
   FOCUS CARD
===================================================== */

.focus-card {
    background: linear-gradient(
        135deg,
        #fff1f2 0%,
        #ffe4e6 100%
    );

    border: 1px solid #fecdd3;
    border-radius: 16px;
    padding: 24px;
    color: #be123c;
}

.focus-title {
    font-weight: 800;
    font-size: 1.2rem;
}

.focus-desc {
    opacity: 0.8;
    margin-top: 5px;
}


/* =====================================================
   QUICK ACTIONS
===================================================== */

.action-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 16px 10px;
    text-align: center;
    min-height: 90px;
}

.action-icon {
    font-size: 1.5rem;
    display: block;
}

.action-label {
    font-weight: 650;
    color: #334155;
    font-size: 0.82rem;
    margin-top: 6px;
}


/* =====================================================
   TASKS
===================================================== */

.task-item {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
}

.task-title {
    font-weight: 650;
    color: #1e293b;
}

.task-subject {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 3px;
}


/* =====================================================
   SUBJECT PROGRESS
===================================================== */

.subject-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;

    box-shadow:
        0 3px 8px rgba(15,23,42,0.03);
}

.subject-name {
    font-weight: 800;
    color: #1e293b;
    font-size: 1.1rem;
}

.subject-full {
    color: #64748b;
    font-size: 0.85rem;
    margin-bottom: 12px;
}

.progress-bar-bg {
    background: #f1f5f9;
    height: 8px;
    border-radius: 5px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;

    background: linear-gradient(
        90deg,
        #4f46e5,
        #818cf8
    );

    border-radius: 5px;
}

.progress-text {
    font-size: 0.8rem;
    color: #64748b;
    text-align: right;
    margin-top: 5px;
}


/* =====================================================
   GENERAL CARDS
===================================================== */

.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;

    box-shadow:
        0 2px 6px rgba(15,23,42,0.03);
}

.info-card {
    background: #eff6ff;
    border: 1px solid #dbeafe;
    border-radius: 12px;
    padding: 15px;
    color: #1e40af;
    margin-bottom: 20px;
}


/* =====================================================
   FILE CARD
===================================================== */

.file-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 10px;
}

.file-name {
    font-weight: 700;
    color: #1e293b;
}

.file-meta {
    color: #64748b;
    font-size: 0.8rem;
}


/* =====================================================
   IMPORTANT QUESTIONS
===================================================== */

.question-section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;

    font-size: 1.35rem;
    font-weight: 800;
    color: #1e293b;

    margin-top: 30px;
    margin-bottom: 15px;
}

.question-count {
    background: #eef2ff;
    color: #4f46e5;

    padding: 6px 12px;
    border-radius: 20px;

    font-size: 0.75rem;
    font-weight: 700;
}

.important-question-card {
    background: #ffffff;

    border: 1px solid #e2e8f0;
    border-radius: 16px;

    padding: 20px;
    margin-bottom: 14px;

    box-shadow:
        0 3px 10px rgba(15,23,42,0.04);

    transition: 0.2s;
}

.important-question-card:hover {
    transform: translateY(-2px);

    box-shadow:
        0 10px 25px rgba(15,23,42,0.08);

    border-color: #c7d2fe;
}

.question-top {
    display: flex;
    align-items: center;
    justify-content: space-between;

    margin-bottom: 12px;
}

.question-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;

    min-width: 45px;
    height: 34px;

    padding: 0 12px;

    background: #eef2ff;
    color: #4f46e5;

    border-radius: 9px;

    font-size: 0.85rem;
    font-weight: 800;
}

.question-priority {
    padding: 6px 11px;

    border-radius: 20px;

    font-size: 0.75rem;
    font-weight: 800;
}

.priority-high {
    background: #fee2e2;
    color: #dc2626;
}

.priority-medium {
    background: #fef3c7;
    color: #d97706;
}

.priority-low {
    background: #dcfce7;
    color: #16a34a;
}

.question-text {
    color: #1e293b;

    font-size: 1rem;
    font-weight: 700;

    line-height: 1.65;

    margin-bottom: 14px;
}

.question-topic {
    background: #f8fafc;

    border-radius: 9px;

    padding: 9px 12px;

    color: #64748b;

    font-size: 0.82rem;
}


/* =====================================================
   BUTTONS
===================================================== */

.stButton > button {
    border-radius: 10px !important;
    font-weight: 650 !important;
    transition: 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
}


/* =====================================================
   CHAT
===================================================== */

[data-testid="stChatMessage"] {
    border-radius: 14px;
}


/* =====================================================
   INPUTS
===================================================== */

.stTextInput input,
.stNumberInput input,
.stSelectbox,
.stMultiSelect,
.stDateInput input {
    border-radius: 10px !important;
}


/* =====================================================
   MOBILE
===================================================== */

@media (max-width: 768px) {

    .dash-welcome {
        font-size: 1.5rem;
    }

    .main-title {
        font-size: 1.7rem;
    }

    .dash-header {
        padding: 24px;
    }

    .question-section-title {
        font-size: 1.1rem;
    }

    .question-count {
        font-size: 0.68rem;
    }

}

</style>
"""


# =========================================================
# APPLY CSS
# =========================================================

st.html(CSS)


# =========================================================
# LANGCHAIN / RAG
# =========================================================

try:

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    LANGCHAIN_AVAILABLE = True

except ImportError:

    LANGCHAIN_AVAILABLE = False


# =========================================================
# UPLOAD DIRECTORY
# =========================================================

os.makedirs("uploads", exist_ok=True)


# =========================================================
# PDF PROCESSING
# =========================================================

def process_uploaded_pdf(uploaded_file):

    try:

        text = extract_text_from_pdf(uploaded_file)

        if text and text.strip():

            st.session_state.pdf_text += (
                f"\n\n--- {uploaded_file.name} ---\n{text}"
            )

        if LANGCHAIN_AVAILABLE and text and text.strip():

            try:

                api_key = st.secrets.get(
                    "GEMINI_API_KEY",
                    ""
                )

                if api_key:

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                    )

                    chunks = splitter.create_documents(
                        [text]
                    )

                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=api_key,
                    )

                    new_store = FAISS.from_documents(
                        chunks,
                        embeddings,
                    )

                    if st.session_state.vector_store:

                        try:

                            st.session_state.vector_store.merge_from(
                                new_store
                            )

                        except Exception:

                            st.session_state.vector_store = new_store

                    else:

                        st.session_state.vector_store = new_store

            except Exception as e:

                st.warning(
                    f"RAG setup skipped: {e}"
                )

        return text or ""

    except Exception as e:

        st.error(
            f"Error processing PDF: {e}"
        )

        return ""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-logo">
            🎓 AI College Assistant
        </div>

        <div class="sidebar-subtitle">
            Your intelligent study companion
        </div>

        <div style="
            font-size:.82rem;
            color:#64748b;
            margin-bottom:14px;
        ">
            Your Smart Academic Companion
        </div>
        """
    )

    pages = [
        ("dashboard", "🏠 Dashboard"),
        ("pdf_upload", "📄 Study Materials"),
        ("smart_chat", "💬 Smart Chat"),
        ("summarizer", "📝 Summarizer"),
        ("mcq_generator", "✅ MCQ Generator"),
        ("important_questions", "❓ Important Questions"),
        ("study_plan", "📅 Study Planner"),
        ("settings", "⚙️ Student Profile"),
    ]

    for key, label in pages:

        if st.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
        ):

            go_to(key)

    st.markdown("---")

    render_html(
        f"""
        <div class="sidebar-mini">

            <div style="
                font-size:.75rem;
                color:#64748b;
            ">
                SIGNED IN AS
            </div>

            <div style="
                font-size:1.05rem;
                font-weight:750;
                margin-top:4px;
            ">
                {safe_html(name)}
            </div>

            <div style="
                font-size:.8rem;
                color:#64748b;
                margin-top:3px;
            ">
                {safe_html(branch)}
            </div>

        </div>
        """
    )


    # ---------------------------------------------------------
    # LOGOUT / SIGN OUT
    # ---------------------------------------------------------
    st.markdown("---")

    if st.button(
        "🚪 Logout / Sign Out",
        key="logout_button",
        use_container_width=True,
    ):
        # Clear the current Streamlit session data.
        # The saved student profile and chat history in SQLite are kept.
        keys_to_clear = [
            "page",
            "chat_messages",
            "rag_chat",
            "study_plan",
            "uploaded_files",
            "pdf_text",
            "vector_store",
            "completed_tasks",
            "generated_tasks",
        ]

        for key in keys_to_clear:
            st.session_state.pop(key, None)

        # Mark the current browser session as signed out.
        st.session_state["logged_out"] = True
        st.rerun()


# =========================================================
# PAGE 1 — DASHBOARD
# =========================================================

if st.session_state.page == "dashboard":

    today = date.today()

    display_name = (
        name.strip()
        if name and name.strip()
        else "Student"
    )

    days_left = (
        max(
            (exam_date - today).days,
            0
        )
        if exam_date
        else 0
    )

    tasks = st.session_state.get(
        "generated_tasks",
        []
    )

    if not isinstance(tasks, list) or not tasks:

        tasks = create_default_tasks()

        st.session_state.generated_tasks = tasks

    total_tasks = len(tasks)

    completed_count = len(
        st.session_state.get(
            "completed_tasks",
            set()
        )
    )

    task_progress = (
        round(
            completed_count / total_tasks * 100
        )
        if total_tasks
        else 0
    )

    # =====================================================
    # HERO
    # =====================================================

    render_html(
        f"""
        <div class="dash-header">

            <div class="dash-welcome">
                Hello, {safe_html(display_name)}! 👋
            </div>

            <div class="dash-date">
                📅 {today.strftime('%A, %d %B %Y')}
                &nbsp; • &nbsp;
                🎯 Your personal academic command center
            </div>

        </div>
        """
    )

    # =====================================================
    # STATISTICS
    # =====================================================

    stats_cols = st.columns(4)

    stat_data = [
        (
            "🔥",
            "Study Streak",
            "7 days",
            "Keep momentum",
            "#f59e0b",
        ),
        (
            "⏳",
            "Exam Countdown",
            f"{days_left} days",
            "Stay consistent",
            "#3b82f6",
        ),
        (
            "📄",
            "Study Materials",
            str(len(st.session_state.uploaded_files)),
            "PDFs uploaded",
            "#6366f1",
        ),
        (
            "📈",
            "Overall Progress",
            f"{task_progress}%",
            "Learning journey",
            "#10b981",
        ),
    ]

    for col, data in zip(
        stats_cols,
        stat_data
    ):

        icon, label, value, sub, border_color = data

        with col:

            render_html(
                f"""
                <div class="stat-card"
                     style="border-top:4px solid {border_color};">

                    <span class="stat-icon">
                        {icon}
                    </span>

                    <div class="stat-label">
                        {label}
                    </div>

                    <div class="stat-value">
                        {value}
                    </div>

                    <div class="stat-sub">
                        {sub}
                    </div>

                </div>
                """
            )

    # =====================================================
    # STUDY MATERIALS
    # =====================================================

    render_html(
        '<div class="section-header">📄 Study Materials</div>'
    )

    pdf_left, pdf_right = st.columns([2.2, 1])

    with pdf_left:

        render_html(
            """
            <div class="pdf-upload-card">

                <div class="pdf-upload-icon">
                    📚
                </div>

                <div class="pdf-upload-title">
                    Upload Your Study Materials
                </div>

                <div class="pdf-upload-text">
                    Add lecture notes, textbooks,
                    question papers and other PDF
                    materials to make your AI
                    Assistant smarter.
                </div>

            </div>
            """
        )

        if st.button(
            "📄 Upload PDF / Study Material",
            key="dashboard_pdf_upload",
            use_container_width=True,
            type="primary",
        ):

            go_to("pdf_upload")

    with pdf_right:

        render_html(
            """
            <div class="pdf-feature">

                <div class="pdf-feature-icon">
                    🤖
                </div>

                <div class="pdf-feature-title">
                    What can AI do?
                </div>

                <div class="pdf-feature-text">
                    • Ask questions from PDFs<br>
                    • Generate MCQs<br>
                    • Create summaries<br>
                    • Generate important questions
                </div>

            </div>
            """
        )

    # =====================================================
    # MAIN CONTENT
    # =====================================================

    main_left, main_right = st.columns([2, 1])

    with main_left:

        render_html(
            '<div class="section-header">🎯 Today\'s Priority</div>'
        )

        render_html(
            """
            <div class="focus-card">

                <div class="focus-title">
                    📚 Deep Study Session
                </div>

                <div class="focus-desc">
                    Recommended: 45 minutes
                    • One topic
                    • Zero distractions
                </div>

            </div>
            """
        )

        # =================================================
        # QUICK ACTIONS
        # =================================================

        render_html(
            '<div class="section-header">⚡ Quick Actions</div>'
        )

        action_cols = st.columns(5)

        actions = [
            ("📄", "Study Materials", "pdf_upload"),
            ("🤖", "Smart Chat", "smart_chat"),
            ("📝", "Summarizer", "summarizer"),
            ("✅", "MCQ Gen", "mcq_generator"),
            ("❓", "Important Qs", "important_questions"),
        ]

        for col, action in zip(
            action_cols,
            actions
        ):

            icon, label, page_name = action

            with col:

                render_html(
                    f"""
                    <div class="action-card">

                        <span class="action-icon">
                            {icon}
                        </span>

                        <span class="action-label">
                            {label}
                        </span>

                    </div>
                    """
                )

                if st.button(
                    "Open",
                    key=f"dashboard_action_{page_name}",
                    use_container_width=True,
                ):

                    go_to(page_name)

        # =================================================
        # TASKS
        # =================================================

        render_html(
            '<div class="section-header">📋 Today\'s Tasks</div>'
        )

        for i, task in enumerate(tasks):

            task_id = task.get(
                "id",
                f"task_{i}"
            )

            is_completed = (
                task_id
                in st.session_state.completed_tasks
            )

            new_value = st.checkbox(
                task.get(
                    "title",
                    "Task"
                ),
                value=is_completed,
                key=f"check_{task_id}",
            )

            if new_value != is_completed:

                if new_value:

                    st.session_state.completed_tasks.add(
                        task_id
                    )

                else:

                    st.session_state.completed_tasks.discard(
                        task_id
                    )

                st.rerun()

            border = (
                "#10b981"
                if is_completed
                else "#cbd5e1"
            )

            text_style = (
                "text-decoration:line-through;color:#94a3b8;"
                if is_completed
                else ""
            )

            render_html(
                f"""
                <div class="task-item"
                     style="border-left:4px solid {border};">

                    <div class="task-title"
                         style="{text_style}">
                        {safe_html(task.get("title", "Task"))}
                    </div>

                    <div class="task-subject">
                        {safe_html(task.get("subject", ""))}
                    </div>

                </div>
                """
            )

    # =====================================================
    # RIGHT SIDE
    # =====================================================

    with main_right:

        render_html(
            '<div class="section-header">📚 Subject Progress</div>'
        )

        subjects = [
            ("DBMS", "Database Management", 78),
            ("OOP", "Object Oriented Programming", 68),
            ("DSU", "Data Structures", 61),
            ("DTE", "Digital Techniques", 52),
        ]

        for short, full, prog in subjects:

            render_html(
                f"""
                <div class="subject-card">

                    <div class="subject-name">
                        {short}
                    </div>

                    <div class="subject-full">
                        {full}
                    </div>

                    <div class="progress-bar-bg">

                        <div class="progress-bar-fill"
                             style="width:{prog}%;">
                        </div>

                    </div>

                    <div class="progress-text">
                        {prog}% Completed
                    </div>

                </div>
                """
            )

        render_html("<br>")

        render_html(
            """
            <div class="info-card">

                <b>🤖 AI Coach Tip</b>

                <br><br>

                Based on your recent activity,
                try focusing on
                <b>Data Structures</b> today.

                <br><br>

                Use the
                <b>MCQ Generator</b>
                after studying to test your knowledge.

            </div>
            """
        )


# =========================================================
# PAGE 2 — PDF UPLOAD
# =========================================================

elif st.session_state.page == "pdf_upload":

    render_html(
        '<h1 class="main-title">📄 Study Materials</h1>'
    )

    render_html(
        '<p class="subtitle">'
        'Upload lecture notes and PDFs to power your AI learning tools.'
        '</p>'
    )

    render_back_button()

    render_html(
        """
        <div class="pdf-upload-card">

            <div class="pdf-upload-icon">
                📚
            </div>

            <div class="pdf-upload-title">
                Upload Study Materials
            </div>

            <div class="pdf-upload-text">
                Upload one or multiple PDF files.
                Your documents can be used for
                Smart Chat, summaries, MCQs and
                important exam questions.
            </div>

        </div>
        """
    )

    uploaded = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded:

        for file in uploaded:

            existing_names = [
                f.name
                for f in st.session_state.uploaded_files
            ]

            if file.name not in existing_names:

                with st.spinner(
                    f"Processing {file.name}..."
                ):

                    st.session_state.uploaded_files.append(
                        file
                    )

                    text = process_uploaded_pdf(file)

                    save_path = os.path.join(
                        "uploads",
                        os.path.basename(file.name)
                    )

                    with open(
                        save_path,
                        "wb"
                    ) as f:

                        f.write(
                            file.getbuffer()
                        )

                st.success(
                    f"✅ Processed: {file.name} "
                    f"({len(text):,} characters)"
                )

    # =====================================================
    # FILE LIST
    # =====================================================

    if st.session_state.uploaded_files:

        st.markdown("### 📚 Your Uploaded Files")

        st.info(
            f"You currently have "
            f"**{len(st.session_state.uploaded_files)} PDF(s)** uploaded."
        )

        for index, file in enumerate(
            st.session_state.uploaded_files
        ):

            file_col, delete_col = st.columns([6, 1])

            with file_col:

                render_html(
                    f"""
                    <div class="file-card">

                        <div class="file-name">
                            📄 {safe_html(file.name)}
                        </div>

                        <div class="file-meta">
                            {file.size / 1024:.1f} KB
                            • PDF Study Material
                        </div>

                    </div>
                    """
                )

            with delete_col:

                if st.button(
                    "🗑️",
                    key=f"delete_pdf_{index}",
                ):

                    marker = (
                        f"\n\n--- {file.name} ---\n"
                    )

                    if marker in st.session_state.pdf_text:

                        before, after = (
                            st.session_state.pdf_text.split(
                                marker,
                                1
                            )
                        )

                        if "\n\n--- " in after:

                            after = (
                                "\n\n--- "
                                + after.split(
                                    "\n\n--- ",
                                    1
                                )[1]
                            )

                        else:

                            after = ""

                        st.session_state.pdf_text = (
                            before + after
                        )

                    st.session_state.uploaded_files.pop(
                        index
                    )

                    physical_path = os.path.join(
                        "uploads",
                        os.path.basename(file.name)
                    )

                    try:

                        if os.path.exists(
                            physical_path
                        ):

                            os.remove(
                                physical_path
                            )

                    except Exception:
                        pass

                    st.session_state.vector_store = None

                    st.rerun()

        render_html("<br>")

        if st.button(
            "🗑️ Clear All Files",
            use_container_width=True,
        ):

            for file in st.session_state.uploaded_files:

                physical_path = os.path.join(
                    "uploads",
                    os.path.basename(file.name)
                )

                try:

                    if os.path.exists(
                        physical_path
                    ):

                        os.remove(
                            physical_path
                        )

                except Exception:
                    pass

            st.session_state.uploaded_files = []
            st.session_state.pdf_text = ""
            st.session_state.vector_store = None
            st.session_state.rag_chat = []

            st.success(
                "All uploaded files cleared."
            )

            st.rerun()

        if (
            LANGCHAIN_AVAILABLE
            and st.session_state.vector_store
        ):

            st.success(
                "🤖 RAG vector store is ready for Smart Chat."
            )

        else:

            st.info(
                "ℹ️ PDF text extraction is ready. "
                "RAG will be used when the required "
                "embedding configuration is available."
            )

    else:

        render_html(
            """
            <div class="card"
                 style="text-align:center;padding:40px;">

                <div style="font-size:3rem;">
                    📂
                </div>

                <h3>
                    No study materials uploaded yet
                </h3>

                <p style="color:#64748b;">
                    Upload your first PDF above
                    to start learning with AI.
                </p>

            </div>
            """
        )


# =========================================================
# PAGE 3 — SMART CHAT
# =========================================================

elif st.session_state.page == "smart_chat":

    render_html(
        '<h1 class="main-title">💬 Smart Chat</h1>'
    )

    render_html(
        '<p class="subtitle">'
        'Ask questions about your uploaded study materials.'
        '</p>'
    )

    render_back_button()

    if not st.session_state.uploaded_files:

        st.warning(
            "📄 Please upload PDFs first from Study Materials."
        )

        if st.button(
            "📄 Go to Study Materials",
            type="primary",
        ):

            go_to("pdf_upload")

    else:

        st.success(
            f"📚 Using "
            f"{len(st.session_state.uploaded_files)} "
            f"uploaded PDF(s)."
        )

        for msg in st.session_state.rag_chat:

            with st.chat_message(
                msg["role"]
            ):

                st.markdown(
                    msg["content"]
                )

        prompt = st.chat_input(
            "Ask anything about your uploaded notes..."
        )

        if prompt:

            st.session_state.rag_chat.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            with st.chat_message("user"):

                st.markdown(prompt)

            with st.chat_message("assistant"):

                with st.spinner(
                    "🔍 Searching your documents..."
                ):

                    try:

                        if st.session_state.vector_store:

                            response = answer_question(
                                st.session_state.vector_store,
                                prompt,
                            )

                        else:

                            response = ask_gemini(
                                "Answer based on this study material:\n"
                                f"{st.session_state.pdf_text}\n\n"
                                f"Question: {prompt}"
                            )

                    except Exception as e:

                        response = (
                            f"AI response failed: {e}"
                        )

                    st.markdown(response)

                    save_chat(
                        prompt,
                        response
                    )

            st.session_state.rag_chat.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )


# =========================================================
# PAGE 4 — SUMMARIZER
# =========================================================

elif st.session_state.page == "summarizer":

    render_html(
        '<h1 class="main-title">📝 Study Notes Summarizer</h1>'
    )

    render_html(
        '<p class="subtitle">'
        'Turn long study materials into concise revision notes.'
        '</p>'
    )

    render_back_button()

    if not st.session_state.uploaded_files:

        st.warning(
            "📄 Please upload PDFs first."
        )

        if st.button(
            "📄 Upload Study Material",
            type="primary",
        ):

            go_to("pdf_upload")

    else:

        if st.button(
            "📝 Generate Study Notes",
            use_container_width=True,
            type="primary",
        ):

            with st.spinner(
                "Creating summary..."
            ):

                try:

                    summary = summarize_text(
                        st.session_state.pdf_text
                    )

                    with st.container():
                        st.markdown("### 📝 Generated Study Notes")
                        st.markdown(summary)

                    save_chat(
                        "Generate study notes",
                        summary,
                    )

                except Exception as e:

                    st.error(
                        f"Summary generation failed: {e}"
                    )


# =========================================================
# PAGE 5 — MCQ GENERATOR
# =========================================================

elif st.session_state.page == "mcq_generator":

    render_html(
        '<h1 class="main-title">✅ MCQ Generator</h1>'
    )

    render_html(
        '<p class="subtitle">'
        'Generate practice questions from your uploaded materials.'
        '</p>'
    )

    render_back_button()

    if not st.session_state.uploaded_files:

        st.warning(
            "📄 Please upload PDFs first."
        )

        if st.button(
            "📄 Upload Study Material",
            type="primary",
        ):

            go_to("pdf_upload")

    else:

        num_questions = st.slider(
            "Number of Questions",
            3,
            15,
            5,
        )

        if st.button(
            "🎯 Generate MCQs",
            use_container_width=True,
            type="primary",
        ):

            with st.spinner(
                "Creating MCQs..."
            ):

                try:

                    mcqs = generate_mcqs(
                        st.session_state.pdf_text,
                        num_questions,
                    )

                    with st.container():
                        st.markdown("### ✅ Generated MCQs")
                        st.markdown(mcqs)

                    save_chat(
                        f"Generate {num_questions} MCQs",
                        mcqs,
                    )

                except Exception as e:

                    st.error(
                        f"MCQ generation failed: {e}"
                    )


# =========================================================
# PAGE 6 — IMPORTANT QUESTIONS
# =========================================================

elif st.session_state.page == "important_questions":

    render_html(
        '<h1 class="main-title">❓ Important Exam Questions</h1>'
    )

    render_html(
        '<p class="subtitle">'
        'Generate high-priority questions for examination preparation.'
        '</p>'
    )

    render_back_button()

    if not st.session_state.uploaded_files:

        st.warning(
            "📄 Please upload PDFs first."
        )

        if st.button(
            "📄 Upload Study Material",
            type="primary",
        ):

            go_to("pdf_upload")

    else:

        num_questions = st.slider(
            "Number of Questions",
            5,
            20,
            10,
        )

        if st.button(
            "📋 Generate Important Questions",
            use_container_width=True,
            type="primary",
        ):

            with st.spinner(
                "Creating exam questions..."
            ):

                try:

                    questions = generate_important_questions(
                        st.session_state.pdf_text,
                        num_questions,
                    )

                    # =================================================
                    # FIXED DISPLAY
                    # =================================================

                    render_important_questions(
                        questions
                    )

                    # Save original AI response
                    save_chat(
                        f"Generate {num_questions} important questions",
                        questions,
                    )

                except Exception as e:

                    st.error(
                        "Question generation failed: "
                        f"{e}"
                    )


# =========================================================
# PAGE 7 — STUDY PLANNER
# =========================================================

elif st.session_state.page == "study_plan":

    render_html(
        '<h1 class="main-title">📅 Smart Study Planner</h1>'
    )

    render_html(
        '<p class="subtitle">'
        'Create a structured preparation plan and track your daily tasks.'
        '</p>'
    )

    render_back_button()

    default_exam = (
        exam_date
        or (
            date.today()
            + timedelta(days=12)
        )
    )

    if default_exam < date.today():

        default_exam = (
            date.today()
            + timedelta(days=12)
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        planner_exam_date = st.date_input(
            "Exam Date",
            value=default_exam,
            min_value=date.today(),
            max_value=(
                date.today()
                + timedelta(days=3650)
            ),
        )

    with c2:

        default_hours = max(
            1,
            min(
                8,
                int(study_hours)
                if study_hours
                else 3
            )
        )

        hours_per_day = st.slider(
            "Study Hours / Day",
            1,
            8,
            default_hours,
        )

    with c3:

        subjects = st.multiselect(
            "Subjects",
            [
                "Data Structures",
                "Operating Systems",
                "DBMS",
                "Mathematics",
                "Computer Networks",
                "AI / ML",
                "OOP Using C++",
                "Other",
            ],
            default=[
                "Data Structures",
                "DBMS",
            ],
        )

    if st.button(
        "🎯 Generate Smart Plan",
        use_container_width=True,
        type="primary",
    ):

        days = (
            planner_exam_date
            - date.today()
        ).days

        if days <= 0:

            st.error(
                "Exam date must be in the future."
            )

        elif not subjects:

            st.warning(
                "Select at least one subject."
            )

        else:

            plan = []

            plan_days = min(
                days,
                14
            )

            for day_number in range(
                1,
                plan_days + 1
            ):

                current_date = (
                    date.today()
                    + timedelta(
                        days=day_number
                    )
                )

                subject = subjects[
                    (day_number - 1)
                    % len(subjects)
                ]

                next_subject = (
                    subjects[
                        day_number
                        % len(subjects)
                    ]
                    if len(subjects) > 1
                    else subject
                )

                plan.append(
                    {
                        "day": day_number,
                        "date": current_date.strftime(
                            "%A, %d %b"
                        ),
                        "subjects": [
                            subject,
                            next_subject,
                        ],
                        "tasks": [
                            f"Study {subject} concepts",
                            f"Practice questions from {subject}",
                            "Take a short self-review",
                        ],
                    }
                )

            st.session_state.study_plan = plan

            generated = []

            for p in plan[:7]:

                for index, task in enumerate(
                    p["tasks"]
                ):

                    generated.append(
                        {
                            "id": f"plan_{p['day']}_{index}",
                            "title": task,
                            "subject": ", ".join(
                                p["subjects"]
                            ),
                        }
                    )

            st.session_state.generated_tasks = generated

            st.session_state.completed_tasks = set()

            st.success(
                "✅ Smart study plan generated successfully."
            )

    # =====================================================
    # DISPLAY PLAN
    # =====================================================

    if st.session_state.study_plan:

        st.markdown(
            "### 📚 Your Plan"
        )

        for p in st.session_state.study_plan:

            task_text = " • ".join(
                p["tasks"]
            )

            render_html(
                f"""
                <div class="card">

                    <div style="
                        font-size:1.1rem;
                        font-weight:800;
                        color:#6d28d9;
                    ">
                        📅 Day {p['day']}
                        — {p['date']}
                    </div>

                    <div style="margin-top:8px;">
                        <b>📚 Subjects:</b>
                        {safe_html(", ".join(p["subjects"]))}
                    </div>

                    <div style="margin-top:7px;">
                        <b>⏱️ Duration:</b>
                        {hours_per_day} hour(s)
                    </div>

                    <div style="margin-top:7px;">
                        <b>✅ Tasks:</b>
                        {safe_html(task_text)}
                    </div>

                </div>
                """
            )

        st.info(
            "Your first 7 days of tasks are also available "
            "on the Dashboard for progress tracking."
        )


# =========================================================
# PAGE 8 — STUDENT PROFILE
# =========================================================

elif st.session_state.page == "settings":

    render_html(
        '<h1 class="main-title">⚙️ Student Profile</h1>'
    )

    render_html(
        '<p class="subtitle">'
        'Manage your academic information and study preferences.'
        '</p>'
    )

    render_back_button()

    render_html(
        """
        <div class="info-card">

            <b>🎓 Why complete your profile?</b>

            <br><br>

            Your branch, semester, exam date and
            study goal help the dashboard and
            Study Planner provide more relevant
            information.

        </div>
        """
    )

    profile_exam_default = (
        exam_date
        or (
            date.today()
            + timedelta(days=12)
        )
    )

    if profile_exam_default < date.today():

        profile_exam_default = (
            date.today()
            + timedelta(days=12)
        )

    with st.form("profile_form"):

        p1, p2 = st.columns(2)

        with p1:

            new_name = st.text_input(
                "👤 Student Name",
                value=(
                    name
                    if name != "Student"
                    else ""
                ),
            )

            new_branch = st.text_input(
                "💻 Branch",
                value=branch,
            )

            new_semester = st.text_input(
                "📚 Semester",
                value=semester,
            )

        with p2:

            new_college = st.text_input(
                "🏫 College",
                value=(
                    college
                    if college != "College Not Set"
                    else ""
                ),
            )

            new_exam_date = st.date_input(
                "📅 Exam Date",
                value=profile_exam_default,
                min_value=date.today(),
                max_value=(
                    date.today()
                    + timedelta(days=3650)
                ),
            )

            new_hours = st.number_input(
                "⏱️ Daily Study Goal (Hours)",
                min_value=0.0,
                max_value=12.0,
                value=max(
                    0.0,
                    min(
                        study_hours,
                        12.0
                    )
                ),
                step=0.5,
            )

            current_time = profile.get(
                "preferred_study_time",
                "Morning"
            )

            time_options = [
                "Morning",
                "Afternoon",
                "Evening",
                "Night",
            ]

            preferred_study_time = st.selectbox(
                "🕒 Preferred Study Time",
                time_options,
                index=(
                    time_options.index(
                        current_time
                    )
                    if current_time in time_options
                    else 0
                ),
            )

        save_profile = st.form_submit_button(
            "💾 Save Student Profile",
            use_container_width=True,
        )

    if save_profile:

        if not new_name.strip():

            st.error(
                "⚠️ Please enter your name."
            )

        else:

            try:

                save_student_profile(
                    new_name.strip(),
                    new_branch.strip(),
                    new_semester.strip(),
                    new_college.strip(),
                    new_exam_date.strftime(
                        "%Y-%m-%d"
                    ),
                    float(new_hours),
                    preferred_study_time,
                )

                st.success(
                    "✅ Student profile saved successfully!"
                )

                st.session_state.page = "dashboard"

                st.rerun()

            except Exception as e:

                st.error(
                    "❌ Could not save the profile. "
                    f"Database error: {e}"
                )