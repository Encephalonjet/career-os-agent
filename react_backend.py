import os
import operator
import requests
import re
from bs4 import BeautifulSoup
from typing import TypedDict, List, Dict, Literal, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class JobInput(BaseModel):
    id: str
    raw_text: str
    company_name: str
    job_title: str

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

def get_llm(provider: str):
    provider = provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
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

def extract_urls_and_scrape(raw_input: str) -> str:
    """
    Finds URLs in the text and scrapes them using Jina Reader API to bypass bot blockers.
    This works on Greenhouse, Workday, Lever, and even handles many LinkedIn limits.
    """
    urls = re.findall(r'(https?://[^\s]+)', raw_input)
    
    if not urls:
        return raw_input
        
    extracted_texts = []
    
    for url in urls:
        try:
            # Jina Reader API automatically converts websites to clean Markdown and bypasses generic bot protections
            jina_url = f"https://r.jina.ai/{url}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(jina_url, headers=headers, timeout=15)
            response.raise_for_status()
            extracted_texts.append(f"\n--- Scraped Job Data from {url} ---\n{response.text}\n")
        except Exception as e:
            extracted_texts.append(f"\n[SYSTEM WARNING: Failed to scrape {url}: {str(e)}]\n")
            
    # Combine original text (with URLs removed) with the newly scraped content
    clean_original_text = re.sub(r'(https?://[^\s]+)', '', raw_input)
    return clean_original_text + "\n" + "".join(extracted_texts)

def evaluate_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    
    # Process multi-URL scraping logic intelligently
    actual_jd = extract_urls_and_scrape(job.raw_text)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter. Grade the CV against the JD on the 12 specified dimensions (A-F). Provide an overall_score (0-100), hire_verdict, strengths, and gap_analysis."),
        ("human", "CV:\n{cv}\n\nJD:\n{jd}")
    ])
    
    # Removed mock fallbacks. Will fail loudly if schema is wrong.
    structured_llm = llm.with_structured_output(JobEvaluation)
    chain = prompt | structured_llm
    result = chain.invoke({"cv": state["base_cv"], "jd": actual_jd})
    
    # Save the scraped text back to state so downstream nodes use the real text
    state["jobs_to_process"][0].raw_text = actual_jd
    return {"evaluations": [result]}

def generate_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    evaluation = state["evaluations"][-1]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write an ATS-optimized tailored CV, a Cover Letter, and a learning_roadmap (list of dicts with 'type', 'topic', 'url') based on gaps."),
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

def build_job_agent():
    builder = StateGraph(AgentState)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("generate", generate_node)
    builder.add_node("coach", coach_node)
    builder.add_edge(START, "evaluate")
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