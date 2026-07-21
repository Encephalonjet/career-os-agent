"""
database.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Handles local SQLite database operations for the Kanban Board.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import sqlite3
import json
import uuid
import os

DB_FILE = "jobs.db"

def init_db():
    """Initializes the database and creates the table if it doesn't exist."""
    # Added timeout=10 to safely handle parallel concurrent thread writes
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id TEXT PRIMARY KEY,
            company TEXT,
            title TEXT,
            status TEXT,
            score INTEGER,
            full_data TEXT
        )
    ''')
    
    # NEW: Create a permanent table to store the user's base CV
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_profile (
            id TEXT PRIMARY KEY,
            base_cv TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_job(company: str, title: str, status: str, score: int, full_data: dict):
    """Saves a newly evaluated job to the database."""
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    job_id = str(uuid.uuid4())
    c.execute(
        "INSERT INTO applications (id, company, title, status, score, full_data) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, company, title, status, score, json.dumps(full_data))
    )
    conn.commit()
    conn.close()
    return job_id

def get_all_jobs():
    """Fetches all jobs for the Kanban board."""
    if not os.path.exists(DB_FILE):
        init_db()
        
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM applications")
    rows = c.fetchall()
    conn.close()
    
    jobs = []
    for row in rows:
        jobs.append({
            "id": row["id"],
            "company": row["company"],
            "title": row["title"],
            "status": row["status"],
            "score": row["score"],
            "full_data": json.loads(row["full_data"])
        })
    return jobs

def update_job_status(job_id: str, new_status: str):
    """Moves a job between Kanban columns."""
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    c.execute("UPDATE applications SET status = ? WHERE id = ?", (new_status, job_id))
    conn.commit()
    conn.close()

def delete_job(job_id: str):
    """Permanently deletes a job from the database."""
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    c.execute("DELETE FROM applications WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# NEW: PERMANENT USER PROFILE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def save_user_cv(cv_text: str):
    """Permanently saves or updates the user's base CV."""
    conn = sqlite3.connect(DB_FILE, timeout=10)
    c = conn.cursor()
    # INSERT OR REPLACE handles both new users and updating existing users effortlessly
    c.execute("INSERT OR REPLACE INTO user_profile (id, base_cv) VALUES ('default_user', ?)", (cv_text,))
    conn.commit()
    conn.close()

def get_user_cv() -> str:
    """Retrieves the user's permanently saved CV."""
    if not os.path.exists(DB_FILE):
        init_db()
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT base_cv FROM user_profile WHERE id = 'default_user'")
    row = c.fetchone()
    conn.close()
    return row["base_cv"] if row else ""

init_db()