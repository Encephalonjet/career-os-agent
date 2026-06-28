import os
import operator
import requests
import re
import urllib.parse as urlparse
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Literal, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA MODELS & SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class JobInput(BaseModel):
    id: str
    raw_text: str
    company_name: str
    job_title: str

class JobMetadata(BaseModel):
    """Used for auto-extracting missing company and title"""
    company_name: str = Field(description="The name of the hiring company")
    job_title: str = Field(description="The exact title of the role")

class JobEvaluation(BaseModel):
    tech_stack_match: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    role_seniority: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    salary_alignment: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    remote_policy: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    company_stage: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    domain_fit: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    growth_opportunities: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    work_life_balance: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    commute_timezone: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    ats_keyword_density: Literal['A', 'B', 'C', 'D', 'F'] = 'C'
    overall_score: int = Field(default=50)
    hire_verdict: str = Field(default="PENDING")
    gap_analysis: List[str] = Field(default=[])
    strengths: List[str] = Field(default=[])

class GeneratedDocs(BaseModel):
    tailored_cv_markdown: str = Field(description="The full ATS optimized CV")
    cover_letter_markdown: str = Field(description="The full cover letter")
    learning_roadmap: List[Dict[str, str]] = Field(default=[])

class CoachOutput(BaseModel):
    salary_strategy: str = Field(default="")
    interview_questions: str = Field(default="")
    prep_plan: str = Field(default="")

class AgentState(TypedDict):
    base_cv: str
    model_provider: str 
    jobs_to_process: List[JobInput] 
    evaluations: Annotated[List[JobEvaluation], operator.add]
    generated_docs: Annotated[List[GeneratedDocs], operator.add]
    coach_outputs: Annotated[List[CoachOutput], operator.add]

# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(provider: str):
    provider = provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Fixed model version to 3.0
        return ChatGoogleGenerativeAI(model="gemini-3.0-flash", temperature=0.1)
    elif provider == "local (llama.cpp)":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=os.getenv("LOCAL_MODEL_KEY", "sk-local-123"),
            base_url=os.getenv("LOCAL_MODEL_BASE_URL", "http://127.0.0.1:58125/v1"),
            model="local-model",
            temperature=0.1
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_urls_and_scrape(raw_input: str) -> str:
    """
    Finds URLs in the text, cleans LinkedIn search URLs, and scrapes them using Jina Reader API.
    """
    urls = re.findall(r'(https?://[^\s]+)', raw_input)
    
    if not urls:
        return raw_input
        
    extracted_texts = []
    
    for url in urls:
        # CLEAN LINKEDIN SEARCH URLS
        if "linkedin.com/jobs/search" in url and "currentJobId=" in url:
            parsed = urlparse.urlparse(url)
            qs = urlparse.parse_qs(parsed.query)
            job_id = qs.get("currentJobId", [None])[0]
            if job_id:
                url = f"https://www.linkedin.com/jobs/view/{job_id}"

        try:
            # Bypass blockers with Jina AI Reader
            jina_url = f"https://r.jina.ai/{url}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(jina_url, headers=headers, timeout=15)
            response.raise_for_status()
            extracted_texts.append(f"\n--- Scraped Job Data from {url} ---\n{response.text}\n")
        except Exception as e:
            # We raise this so the UI catches it and displays it as a Toast error
            raise ValueError(f"Failed to scrape URL ({url}). Error: {str(e)}")
            
    clean_original_text = re.sub(r'(https?://[^\s]+)', '', raw_input)
    return clean_original_text + "\n" + "".join(extracted_texts)

# ─────────────────────────────────────────────────────────────────────────────
# 4. LANGGRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata_node(state: AgentState):
    """
    Scrapes the JD text. If company or title are missing, it uses a lightweight LLM call to extract them.
    """
    job = state["jobs_to_process"][0]
    
    # 1. Scrape URLs immediately
    actual_jd = extract_urls_and_scrape(job.raw_text)
    job.raw_text = actual_jd
    
    # 2. Auto-detect missing title/company
    if not job.company_name or not job.job_title:
        try:
            llm = get_llm(state["model_provider"])
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract the hiring company name and the exact job title from the provided job description text. If you absolutely cannot find them, return 'Unknown'."),
                ("human", "{jd}")
            ])
            chain = prompt | llm.with_structured_output(JobMetadata)
            # Send only the first 3000 chars to save tokens on this quick check
            res = chain.invoke({"jd": actual_jd[:3000]})
            
            if not job.company_name:
                job.company_name = res.company_name
            if not job.job_title:
                job.job_title = res.job_title
        except Exception as e:
            print(f"Auto-extraction failed: {e}")
            if not job.company_name: job.company_name = "Unknown Company"
            if not job.job_title: job.job_title = "Unknown Title"

    return {"jobs_to_process": [job]}

def evaluate_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter. Grade the CV against the JD on the 12 specified dimensions (A-F). Provide an overall_score (0-100), hire_verdict, strengths, and gap_analysis."),
        ("human", "CV:\n{cv}\n\nJD:\n{jd}")
    ])
    
    # No more silent fallbacks. Will fail loudly and inform UI if schema is broken
    structured_llm = llm.with_structured_output(JobEvaluation)
    chain = prompt | structured_llm
    result = chain.invoke({"cv": state["base_cv"], "jd": job.raw_text})
    
    return {"evaluations": [result]}

def generate_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    evaluation = state["evaluations"][-1]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write an ATS-optimized tailored CV, a Cover Letter, and a learning_roadmap (list of dicts with 'type', 'topic', 'url') based on gaps. The URL can be '#' if unknown."),
        ("human", "Job: {job}\nBase CV: {cv}\nGaps: {gaps}")
    ])
    
    structured_llm = llm.with_structured_output(GeneratedDocs)
    chain = prompt | structured_llm
    result = chain.invoke({"cv": state["base_cv"], "job": job.raw_text, "gaps": "\n".join(evaluation.gap_analysis)})
    return {"generated_docs": [result]}

def coach_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    evaluation = state["evaluations"][-1]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate interview_questions, salary_strategy, and prep_plan based on the candidate's gaps."),
        ("human", "JD:\n{job}\nGaps: {gaps}")
    ])
    
    structured_llm = llm.with_structured_output(CoachOutput)
    chain = prompt | structured_llm
    result = chain.invoke({"job": job.raw_text, "gaps": "\n".join(evaluation.gap_analysis)})
    return {"coach_outputs": [result]}

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD & RUN AGENT
# ─────────────────────────────────────────────────────────────────────────────

def build_job_agent():
    builder = StateGraph(AgentState)
    builder.add_node("extract_metadata", extract_metadata_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("generate", generate_node)
    builder.add_node("coach", coach_node)
    
    builder.add_edge(START, "extract_metadata")
    builder.add_edge("extract_metadata", "evaluate")
    builder.add_edge("evaluate", "generate")
    builder.add_edge("generate", "coach")
    builder.add_edge("coach", END)
    
    return builder.compile(checkpointer=MemorySaver())

def run_agent(cv_text: str, jd_text: str, company: str, title: str, provider: str = "gemini") -> dict:
    agent = build_job_agent()
    initial_state = {
        "base_cv": cv_text,
        "model_provider": provider,
        "jobs_to_process": [JobInput(id="1", company_name=company, job_title=title, raw_text=jd_text)],
        "evaluations": [], 
        "generated_docs": [],
        "coach_outputs": []
    }
    config = {"configurable": {"thread_id": "api_session"}}
    for _ in agent.stream(initial_state, config=config):
        pass
    return agent.get_state(config).values