import traceback
import io
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import database

# Phase 3 & 4 Imports
from auto_fetch import run_auto_fetch_pipeline
import notion_sync

# File Parsing
import pdfplumber
import docx

try:
    from react_backend import run_agent, chat_with_agent
except ImportError as e:
    print("\n" + "!"*60)
    print(f"CRITICAL ERROR LOADING react_backend.py: {e}")
    print("Run: pip install -r requirements.txt")
    print("!"*60 + "\n")
    def run_agent(*args, **kwargs):
        raise HTTPException(status_code=500, detail="react_backend.py failed to load. Check server console.")
    def chat_with_agent(*args, **kwargs):
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

# Schema for Chat processing
class ChatRequest(BaseModel):
    message: str
    history: list
    provider: str = "gemini"

@app.get("/api/jobs")
def get_jobs():
    return {"jobs": database.get_all_jobs()}

@app.put("/api/jobs/{job_id}/status")
def update_status(job_id: str, request: StatusUpdateRequest):
    database.update_job_status(job_id, request.status)
    return {"status": "success"}

@app.delete("/api/jobs/{job_id}")
def delete_job_endpoint(job_id: str):
    database.delete_job(job_id)
    return {"status": "success"}

# Chat Endpoint wired to the LLM
@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    try:
        reply = chat_with_agent(request.message, request.history, request.provider)
        return {"reply": reply}
    except Exception as e:
        print("ERROR IN CHAT:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/profile/cv")
def get_profile_cv():
    return {"cv_text": database.get_user_cv()}

# NOTION SYNC ENDPOINT (Phase 4)
@app.post("/api/notion/sync")
def sync_notion_endpoint():
    try:
        synced_amount = notion_sync.sync_to_notion()
        return {"status": "success", "synced_count": synced_amount}
    except Exception as e:
        print("ERROR IN NOTION SYNC:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# AUTO-FETCH ENDPOINT (Restored Phase 3 Logic)
@app.post("/api/auto-fetch")
def auto_fetch_endpoint(
    file: UploadFile = File(None),         
    base_cv_text: str = Form(""),
    target_title: str = Form(...),
    location: str = Form(""), 
    provider: str = Form("gemini"),
    max_jobs: int = Form(5),
    job_board: str = Form("All Supported Boards"),
    work_model: str = Form("Any"),
    job_type: str = Form("Any"),
    visa: str = Form("Any")
):
    try:
        extracted_cv_text = parse_uploaded_file(file)
        if not extracted_cv_text:
            extracted_cv_text = base_cv_text

        if not extracted_cv_text.strip():
            raise HTTPException(status_code=400, detail="A Base CV is required. Please add a CV to your workspace first.")
            
        # Permanent CV Storage
        database.save_user_cv(extracted_cv_text)
            
        added_count, new_job_ids = run_auto_fetch_pipeline(
            base_cv_text=extracted_cv_text,
            target_title=target_title,
            location=location,
            provider=provider,
            max_jobs=max_jobs,
            job_board=job_board,
            work_model=work_model,
            job_type=job_type,
            visa=visa
        )
        return {"status": "success", "added_jobs": added_count, "new_job_ids": new_job_ids}
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR IN AUTO-FETCH:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def parse_uploaded_file(file: UploadFile) -> str:
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
    company: str = Form(""),               
    title: str = Form(""),                 
    provider: str = Form("gemini")
):
    try:
        extracted_cv_text = parse_uploaded_file(file)
        if not extracted_cv_text:
            extracted_cv_text = base_cv_text

        if not extracted_cv_text.strip():
            raise HTTPException(status_code=400, detail="No CV text could be extracted or provided. Please provide a CV.")

        # Permanent CV Storage
        database.save_user_cv(extracted_cv_text)

        extracted_jd_text = parse_uploaded_file(jd_file)
        final_jd_text = f"{extracted_jd_text}\n\n{jd_text}".strip()

        if not final_jd_text:
            raise HTTPException(status_code=400, detail="No Job Description could be extracted or provided. Please provide a JD.")

        result_state = run_agent(
            cv_text=extracted_cv_text, 
            jd_text=final_jd_text, 
            company=company, 
            title=title, 
            provider=provider
        )
        
        evals = result_state.get("evaluations", [{}])[0]
        score = getattr(evals, "overall_score", 0)
        
        processed_job = result_state.get("jobs_to_process", [{}])[0]
        final_company = getattr(processed_job, 'company_name', company)
        final_title = getattr(processed_job, 'job_title', title)
        
        job_id = database.add_job(
            company=final_company or "Unknown Company",
            title=final_title or "Unknown Title",
            status="To Apply",
            score=score,
            full_data=jsonable_encoder(result_state),
            source="Manual Upload" # Tag manual uploads
        )
        return {"status": "success", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR IN EVALUATE:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))