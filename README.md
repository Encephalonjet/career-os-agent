💼 Personal Career OS & Job AI Agent

A powerful, semi-automated AI agent designed to help you land your target roles. Built with a React Frontend, a FastAPI/SQLite Backend, and an intelligent LangGraph Agent, this tool evaluates your base CV against job descriptions, bypasses bot-blockers to read URLs, scores your fit across 12 critical dimensions, and automatically generates ATS-optimized documents.

✨ Features

Intelligent File & URL Parsing: Upload PDFs/Docx files, or paste direct URLs to job boards. The agent automatically bypasses bot protections (using Jina AI) to scrape job descriptions.

12-Dimension Scoring: Grades your CV against the JD for tech stack match, salary alignment, company stage, work-life balance, and more.

Automated Document Tailoring: Rewrites your CV and drafts a cover letter specifically tailored to the target job.

Skill Gap Analysis & Coach: Identifies missing skills, provides a curated learning roadmap, writes mock interview questions, and suggests salary negotiation strategies.

Kanban Board Tracking: Automatically saves all your generated evaluations to a local SQLite database for easy tracking and review.

🚀 Setup & Installation

1. Clone the repository:

git clone <your-repo-url>
cd personal-job-ai-agent


2. Install dependencies:

pip install -r requirements.txt


3. Configure Environment Variables:
Create a .env file in the root directory and add your API keys. Do not commit this file to version control.

# Closed-Source APIs
GOOGLE_API_KEY="AIza-your-gemini-key"
OPENAI_API_KEY="sk-your-openai-key"

# Local Model Configuration (Optional)
LOCAL_MODEL_KEY="sk-local-key"
LOCAL_MODEL_BASE_URL="[http://127.0.0.1](http://127.0.0.1):port/v1"


4. Run the Application Backend:
Start the FastAPI server (this handles the database and AI processing):

uvicorn api:app --reload --port 8000


5. Open the UI:
Double-click the job_agent_ui.html file to open it directly in your web browser. It will automatically connect to your running backend.

📁 Project Structure

api.py: FastAPI server handling file uploads, API endpoints, and database connection.

react_backend.py: The core LangGraph AI brain. Handles multi-agent evaluation, generation, and coaching.

database.py: SQLite database initialization and interaction functions.

job_agent_ui.html: Standalone React/Tailwind UI connected to the backend.

jobs.db: Your local database tracking all job applications (auto-generated).