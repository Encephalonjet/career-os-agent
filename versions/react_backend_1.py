"""
react_backend.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STANDALONE BACKEND: Pure LangGraph logic
This is the core AI Agent that processes resumes and JDs.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os
import operator
from pathlib import Path
from typing import TypedDict, List, Dict, Literal, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- Schemas ---
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
    hire_verdict: str = Field(default="YES")
    strengths: List[str] = Field(default=[])
    gap_analysis: List[str] = Field(default=[])

class GeneratedDocs(BaseModel):
    job_id: str = "job_1"
    cover_score: int = Field(default=8)
    bullets_score: int = Field(default=8)
    tailored_cv_markdown: str = "CV Content"
    cover_letter_markdown: str = "Cover Letter Content"
    learning_roadmap: List[Dict[str, str]] = Field(default=[])

class CoachOutput(BaseModel):
    confidence_score: int = Field(default=7)
    interview_questions: str = Field(default="Questions")
    salary_strategy: str = Field(default="Strategy")
    prep_plan: str = Field(default="Plan")

class AgentState(TypedDict):
    base_cv: str
    model_provider: str 
    min_score_threshold: int
    jobs_to_process: List[JobInput] 
    evaluations: Annotated[List[JobEvaluation], operator.add]
    generated_docs: Annotated[List[GeneratedDocs], operator.add]
    coach_outputs: Annotated[List[CoachOutput], operator.add]
    skipped: bool
    skip_reason: str

# --- LLM Loader ---
def get_llm(provider: str):
    provider = provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
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

# --- Nodes ---
def evaluate_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Grade CV against JD on 12 dimensions (A-F). Provide verdict, strengths, gaps."),
        ("human", "CV:\n{cv}\n\nJD:\n{jd}")
    ])
    try:
        chain = prompt | llm.with_structured_output(JobEvaluation)
        result = chain.invoke({"cv": state["base_cv"], "jd": job.raw_text})
    except Exception:
        result = JobEvaluation(overall_score=75)
    return {"evaluations": [result], "skipped": False, "skip_reason": ""}

def route_eval(state: AgentState) -> str:
    if state["evaluations"][-1].overall_score < state.get("min_score_threshold", 50):
        return "skip"
    return "generate"

def skip_node(state: AgentState):
    reason = f"Score ({state['evaluations'][-1].overall_score}%) < threshold."
    return {"skipped": True, "skip_reason": reason}

def generate_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    evaluation = state["evaluations"][-1]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tailor CV, write Cover Letter, provide 15-20 resources based on gaps."),
        ("human", "Job: {job}\nBase CV: {cv}\nGaps: {gaps}")
    ])
    try:
        chain = prompt | llm.with_structured_output(GeneratedDocs)
        result = chain.invoke({"cv": state["base_cv"], "job": job.raw_text, "gaps": "\n".join(evaluation.gap_analysis)})
    except Exception:
        result = GeneratedDocs()
    return {"generated_docs": [result]}

def coach_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    evaluation = state["evaluations"][-1]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate 5 interview questions, salary strategy, and 30-day prep plan."),
        ("human", "JD:\n{job}\nStrengths: {strengths}\nGaps: {gaps}")
    ])
    try:
        chain = prompt | llm.with_structured_output(CoachOutput)
        result = chain.invoke({"job": job.raw_text, "strengths": "\n".join(evaluation.strengths), "gaps": "\n".join(evaluation.gap_analysis)})
    except Exception:
        result = CoachOutput()
    return {"coach_outputs": [result]}

def build_job_agent():
    builder = StateGraph(AgentState)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("skip", skip_node)
    builder.add_node("generate", generate_node)
    builder.add_node("coach", coach_node)
    
    builder.add_edge(START, "evaluate")
    builder.add_conditional_edges("evaluate", route_eval, {"generate": "generate", "skip": "skip"})
    builder.add_edge("generate", "coach")
    builder.add_edge("coach", END)
    builder.add_edge("skip", END)
    return builder.compile(checkpointer=MemorySaver())

def run_agent(cv_text: str, jd_text: str, company: str, title: str, provider: str = "Local (llama.cpp)") -> dict:
    """Entry point for the API."""
    agent = build_job_agent()
    initial_state = {
        "base_cv": cv_text,
        "model_provider": provider,
        "min_score_threshold": 50,
        "jobs_to_process": [JobInput(id="1", company_name=company, job_title=title, raw_text=jd_text)],
        "evaluations": [], "generated_docs": [], "coach_outputs": [], "skipped": False, "skip_reason": ""
    }
    config = {"configurable": {"thread_id": "api_session"}}
    for _ in agent.stream(initial_state, config=config):
        pass
    return agent.get_state(config).values