import sqlite3
import json
import uuid
import os
import hashlib

DB_FILE = "jobs.db"

def init_db():
    """Initializes the database and creates tables for jobs and multi-tier users."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Phase 5: Users table for multi-tenant access
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT
        )
    ''')
    
    # Applications table (now including user_id)
    c.execute('''
        CREATE TABLE IF NOT EXISTS applications (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            company TEXT,
            title TEXT,
            status TEXT,
            score INTEGER,
            full_data TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Migration: Add user_id column if updating from Phase 1-3
    try:
        c.execute("ALTER TABLE applications ADD COLUMN user_id TEXT DEFAULT 'default_user'")
    except sqlite3.OperationalError:
        pass # Column already exists, safe to ignore
        
    # Seed a default admin user for backward compatibility
    c.execute("INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (?, ?, ?)", 
              ("default_user", "admin", hash_password("admin")))
              
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Simple SHA-256 hash for basic authentication."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str):
    """Creates a new user for multi-tier access."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    user_id = str(uuid.uuid4())
    try:
        c.execute("INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
                  (user_id, username, hash_password(password)))
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None # Username already exists
    finally:
        conn.close()

def authenticate_user(username: str, password: str):
    """Authenticates a user and returns their unique user_id."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ? AND password_hash = ?", 
              (username, hash_password(password)))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def add_job(company: str, title: str, status: str, score: int, full_data: dict, user_id: str = "default_user"):
    """Saves a newly evaluated job tied strictly to a specific user."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    job_id = str(uuid.uuid4())
    c.execute(
        "INSERT INTO applications (id, user_id, company, title, status, score, full_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (job_id, user_id, company, title, status, score, json.dumps(full_data))
    )
    conn.commit()
    conn.close()
    return job_id

def get_all_jobs(user_id: str = "default_user"):
    """Fetches all jobs specifically for the logged-in user's Kanban board."""
    if not os.path.exists(DB_FILE):
        init_db()
        
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM applications WHERE user_id = ?", (user_id,))
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

def update_job_status(job_id: str, new_status: str, user_id: str = "default_user"):
    """Moves a job between Kanban columns, ensuring it belongs to the active user."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE applications SET status = ? WHERE id = ? AND user_id = ?", (new_status, job_id, user_id))
    conn.commit()
    conn.close()

def delete_job(job_id: str, user_id: str = "default_user"):
    """Permanently deletes a job, ensuring the user owns it."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM applications WHERE id = ? AND user_id = ?", (job_id, user_id))
    conn.commit()
    conn.close()

init_db()