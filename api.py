import traceback
import io
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import database

# File Parsing
import pdfplumber
import docx

try:
    from react_backend import run_agent
except ImportError as e:
    print("\n" + "!"*60)
    print(f"CRITICAL ERROR LOADING react_backend.py: {e}")
    print("Run: pip install -r requirements.txt")
    print("!"*60 + "\n")
    def run_agent(*args, **kwargs):
        raise HTTPException(status_code=500, detail="react_backend.py failed to load. Check server console.")

app = FastAPI(title="Job AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StatusUpdateRequest(BaseModel):
    status: str

@app.get("/api/jobs")
def get_jobs():
    return {"jobs": database.get_all_jobs()}

@app.put("/api/jobs/{job_id}/status")
def update_status(job_id: str, request: StatusUpdateRequest):
    database.update_job_status(job_id, request.status)
    return {"status": "success"}

def parse_uploaded_file(file: UploadFile) -> str:
    """Helper function to extract text from PDF or DOCX files in memory."""
    if not file or not file.filename:
        return ""
        
    try:
        content = file.file.read()
        if file.filename.lower().endswith('.pdf'):
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif file.filename.lower().endswith(('.docx', '.doc')):
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file.filename}. Please upload PDF or DOCX.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file {file.filename}: {str(e)}")

@app.post("/api/evaluate")
async def evaluate_job(
    file: UploadFile = File(None),         
    jd_file: UploadFile = File(None),      
    base_cv_text: str = Form(""),
    jd_text: str = Form(""),
    company: str = Form(""),               # Now completely optional at the API level
    title: str = Form(""),                 # Now completely optional at the API level
    provider: str = Form("gemini")
):
    """
    Accepts physical files OR raw text for both the CV and the JD.
    Company and Title can be empty; the backend AI will auto-extract them.
    """
    try:
        # 1. Parse Base CV
        extracted_cv_text = parse_uploaded_file(file)
        if not extracted_cv_text:
            extracted_cv_text = base_cv_text

        if not extracted_cv_text.strip():
            raise HTTPException(status_code=400, detail="No CV text could be extracted or provided. Please provide a CV.")

        # 2. Parse JD File
        extracted_jd_text = parse_uploaded_file(jd_file)
        final_jd_text = f"{extracted_jd_text}\n\n{jd_text}".strip()

        if not final_jd_text:
            raise HTTPException(status_code=400, detail="No Job Description could be extracted or provided. Please provide a JD.")

        # 3. RUN THE LANGGRAPH AGENT
        result_state = run_agent(
            cv_text=extracted_cv_text, 
            jd_text=final_jd_text, 
            company=company, 
            title=title, 
            provider=provider
        )
        
        # 4. SAVE TO DATABASE
        evals = result_state.get("evaluations", [{}])[0]
        score = getattr(evals, "overall_score", 0)
        
        # If company/title were blank, grab the auto-extracted ones from the AI state
        processed_job = result_state.get("jobs_to_process", [{}])[0]
        final_company = processed_job.company_name if hasattr(processed_job, 'company_name') else company
        final_title = processed_job.job_title if hasattr(processed_job, 'job_title') else title
        
        job_id = database.add_job(
            company=final_company or "Unknown Company",
            title=final_title or "Unknown Title",
            status="To Apply",
            score=score,
            full_data=result_state 
        )
        return {"status": "success", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR IN EVALUATE:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
