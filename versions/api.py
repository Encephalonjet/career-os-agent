"""
api.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The FastAPI server. This connects your React UI to the LangGraph
backend and the SQLite database.

Run this using: uvicorn api:app --reload --port 8000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import database

# IMPORTANT: Ensure your react_backend.py is in the same folder
# and has the run_agent function we built previously.
try:
    from react_backend import run_agent
except ImportError:
    print("WARNING: react_backend.py not found. API will run, but agent evaluation will fail.")

app = FastAPI(title="Job AI Agent API")

# Allow the React HTML file to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluateRequest(BaseModel):
    base_cv: str
    jd_text: str
    company: str
    title: str
    provider: str

class StatusUpdateRequest(BaseModel):
    status: str

@app.get("/api/jobs")
def get_jobs():
    """Returns all saved jobs for the Kanban board."""
    return {"jobs": database.get_all_jobs()}

@app.put("/api/jobs/{job_id}/status")
def update_status(job_id: str, request: StatusUpdateRequest):
    """Updates the Kanban column status of a job."""
    database.update_job_status(job_id, request.status)
    return {"status": "success"}

@app.post("/api/evaluate")
def evaluate_job(req: EvaluateRequest):
    """Runs the LangGraph agent and saves the result to the DB."""
    try:
        # Run the agent (from react_backend.py)
        result_state = run_agent(
            cv_text=req.base_cv, 
            jd_text=req.jd_text, 
            company=req.company, 
            title=req.title, 
            provider=req.provider
        )
        
        # Extract data from the LangGraph state
        evals = result_state.get("evaluations", [{}])[0]
        score = getattr(evals, "overall_score", 0)
        
        # Save to database automatically under "To Apply" status
        job_id = database.add_job(
            company=req.company,
            title=req.title,
            status="To Apply",
            score=score,
            full_data=result_state # Save all generated CVs, Cover Letters, etc.
        )
        
        return {"status": "success", "job_id": job_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))