import sqlite3
import os

DB_NAME = "study_assistant.db"


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create student profile table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            branch TEXT,
            semester TEXT,
            college TEXT,
            exam_date TEXT,
            study_hours REAL,
            preferred_study_time TEXT DEFAULT 'Morning'
        )
    """)

    # Create chat history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # -------------------------------------------------
    # DATABASE MIGRATION
    # Adds columns if an older database already exists
    # -------------------------------------------------
    cursor.execute("PRAGMA table_info(student_profile)")
    existing_columns = {
        row["name"] for row in cursor.fetchall()
    }

    required_columns = {
        "name": "TEXT",
        "branch": "TEXT",
        "semester": "TEXT",
        "college": "TEXT",
        "exam_date": "TEXT",
        "study_hours": "REAL",
        "preferred_study_time": "TEXT DEFAULT 'Morning'"
    }

    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            try:
                cursor.execute(
                    f"ALTER TABLE student_profile "
                    f"ADD COLUMN {column_name} {column_type}"
                )
            except sqlite3.OperationalError:
                pass

    conn.commit()
    conn.close()


def save_student_profile(
    name,
    branch,
    semester,
    college,
    exam_date,
    study_hours,
    preferred_study_time="Morning"
):
    """
    Save or update the student's profile.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # Make sure database tables/columns exist
    cursor.execute("PRAGMA table_info(student_profile)")
    existing_columns = {
        row["name"] for row in cursor.fetchall()
    }

    # Add missing preferred_study_time column if necessary
    if "preferred_study_time" not in existing_columns:
        cursor.execute("""
            ALTER TABLE student_profile
            ADD COLUMN preferred_study_time TEXT DEFAULT 'Morning'
        """)

    # Check whether a profile already exists
    cursor.execute(
        "SELECT id FROM student_profile LIMIT 1"
    )
    row = cursor.fetchone()

    if row:
        # -----------------------------
        # UPDATE EXISTING PROFILE
        # -----------------------------
        cursor.execute("""
            UPDATE student_profile
            SET
                name = ?,
                branch = ?,
                semester = ?,
                college = ?,
                exam_date = ?,
                study_hours = ?,
                preferred_study_time = ?
            WHERE id = ?
        """, (
            name,
            branch,
            semester,
            college,
            exam_date,
            study_hours,
            preferred_study_time,
            row["id"]
        ))

    else:
        # -----------------------------
        # INSERT NEW PROFILE
        # -----------------------------
        cursor.execute("""
            INSERT INTO student_profile (
                name,
                branch,
                semester,
                college,
                exam_date,
                study_hours,
                preferred_study_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            branch,
            semester,
            college,
            exam_date,
            study_hours,
            preferred_study_time
        ))

    conn.commit()
    conn.close()


def get_student_profile():
    """
    Get the saved student profile.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM student_profile
        LIMIT 1
    """)

    row = cursor.fetchone()

    conn.close()

    if row:
        return dict(row)

    return None


def save_chat(question, answer):
    """
    Save a chat question and answer.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chat_history (question, answer)
        VALUES (?, ?)
    """, (
        question,
        answer
    ))

    conn.commit()
    conn.close()


def get_chat_history(limit=5):
    """
    Get recent chat history.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            question,
            answer,
            timestamp
        FROM chat_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]


# -------------------------------------------------
# CREATE / UPDATE DATABASE WHEN FILE IS LOADED
# -------------------------------------------------

create_table()