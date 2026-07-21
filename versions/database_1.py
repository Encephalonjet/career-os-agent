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
    conn = sqlite3.connect(DB_FILE)
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
    conn.commit()
    conn.close()

def add_job(company: str, title: str, status: str, score: int, full_data: dict):
    """Saves a newly evaluated job to the database."""
    conn = sqlite3.connect(DB_FILE)
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
    # Ensure DB exists before fetching
    if not os.path.exists(DB_FILE):
        init_db()
        
    conn = sqlite3.connect(DB_FILE)
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
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE applications SET status = ? WHERE id = ?", (new_status, job_id))
    conn.commit()
    conn.close()

def delete_job(job_id: str):
    """Permanently deletes a job from the database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM applications WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()

init_db()