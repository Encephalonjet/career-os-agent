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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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
    hire_verdict: str = Field(default="Moderate Fit")
    gap_analysis: List[str] = Field(default=[])
    strengths: List[str] = Field(default=[])

class LearningResource(BaseModel):
    type: str = Field(description="Type/Cost of resource (e.g., Free Course, Paid Video, Free Article)")
    topic: str = Field(description="The specific skill or topic covered")
    url: str = Field(description="The exact URL to the resource. Must be a valid link.")

class GeneratedDocs(BaseModel):
    tailored_cv_markdown: str = Field(description="The full ATS optimized CV")
    cover_letter_markdown: str = Field(description="The full cover letter")
    changes_made: List[str] = Field(description="List of specific changes made to optimize the CV", default=[])
    learning_roadmap: List[LearningResource] = Field(default=[])

class CoachOutput(BaseModel):
    salary_strategy: str = Field(description="A concise 1-2 paragraph salary negotiation strategy.")
    interview_questions: str = Field(description="Exactly 5 specific interview questions with brief sample answers, using simple markdown bullet points.")
    prep_plan: str = Field(description="A concise 30-day prep plan using simple markdown bullet points.")

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
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
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
    urls = re.findall(r'(https?://[^\s]+)', raw_input)
    if not urls:
        return raw_input
        
    extracted_texts = []
    for url in urls:
        if "linkedin.com/jobs/search" in url and "currentJobId=" in url:
            parsed = urlparse.urlparse(url)
            qs = urlparse.parse_qs(parsed.query)
            job_id = qs.get("currentJobId", [None])[0]
            if job_id:
                url = f"https://www.linkedin.com/jobs/view/{job_id}"

        try:
            jina_url = f"https://r.jina.ai/{url}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(jina_url, headers=headers, timeout=15)
            response.raise_for_status()
            extracted_texts.append(f"\n--- Scraped Job Data from {url} ---\n{response.text}\n")
        except Exception as e:
            raise ValueError(f"Failed to scrape URL ({url}). Error: {str(e)}")
            
    clean_original_text = re.sub(r'(https?://[^\s]+)', '', raw_input)
    return clean_original_text + "\n" + "".join(extracted_texts)

def chat_with_agent(message: str, history: list, provider: str = "gemini"):
    """Standalone function to power the UI Chat Agent."""
    llm = get_llm(provider)
    
    # UNRESTRICTED FIELD-AGNOSTIC SYSTEM PROMPT
    system_prompt = (
        "You are Career OS, a highly intelligent, expert AI Career Coach and Technical Assistant. "
        "You are adaptable to ANY industry (Product Management, HR, Finance, Engineering, Sales, Marketing, Design, etc.). "
        "You must provide expert-level guidance, answer technical/domain-specific questions, suggest learning resources, "
        "and help the user upskill in their specific field. Use markdown formatting."
    )
    
    messages = [SystemMessage(content=system_prompt)]
    
    # Load previous conversation
    for msg in history:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "agent":
            messages.append(AIMessage(content=msg.get("content", "")))
            
    messages.append(HumanMessage(content=message))
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"I encountered an error connecting to the LLM: {str(e)}"

# ─────────────────────────────────────────────────────────────────────────────
# 4. LANGGRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata_node(state: AgentState):
    job = state["jobs_to_process"][0]
    actual_jd = extract_urls_and_scrape(job.raw_text)
    job.raw_text = actual_jd
    
    if not job.company_name or not job.job_title:
        try:
            llm = get_llm(state["model_provider"])
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract the hiring company name and the exact job title from the provided job description text. If you absolutely cannot find them, return 'Unknown'."),
                ("human", "{jd}")
            ])
            chain = prompt | llm.with_structured_output(JobMetadata)
            res = chain.invoke({"jd": actual_jd[:3000]})
            
            if not job.company_name: job.company_name = res.company_name
            if not job.job_title: job.job_title = res.job_title
        except Exception as e:
            if not job.company_name: job.company_name = "Unknown Company"
            if not job.job_title: job.job_title = "Unknown Title"

    return {"jobs_to_process": [job]}

def evaluate_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical recruiter. Grade the CV against the JD on the 12 specified dimensions (A-F). CRITICAL: If information (like salary or visa) is missing from the JD, default to 'C' (Unknown). Never fail a dimension unless explicitly contradicted. Keep the hire_verdict constructive and encouraging (e.g., 'Strong Fit', 'Moderate Fit', 'Developing Match'). Provide an overall_score (0-100), hire_verdict, strengths, and gap_analysis."),
        ("human", "CV:\n{cv}\n\nJD:\n{jd}")
    ])
    
    structured_llm = llm.with_structured_output(JobEvaluation)
    chain = prompt | structured_llm
    result = chain.invoke({"cv": state["base_cv"], "jd": job.raw_text})
    return {"evaluations": [result]}

def generate_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    evaluation = state["evaluations"][-1]
    
    # STRICT EDITOR PROMPT FOR ATS CV GENERATION
    system_instructions = (
        "You are an Expert ATS Optimizer. DO NOT rewrite the entire CV. Act as an editor. Maintain your voice, and humanize any added content. "
        "CRITICAL FORMATTING: You MUST place the candidate's Name, Job Title, Phone Number, and Email on strictly separate lines using double newlines (\\n\\n) at the very top of the document. Do not put them on a single line. "
        "Reformat the extracted text into a beautiful, professional Markdown CV. You MUST use newlines. Place the Name, Title, and Contact Info on separate lines. "
        "You MUST use Markdown bullet points (-) for all Experience and Skills. NEVER output a wall of text.\n"
        "ONLY append or slightly modify existing bullet points to inject missing ATS keywords and quantifiable metrics "
        "(percentages, dollar amounts, time saved). Do not over-quantify; make sure quantities are necessary and measurable. "
        "Where only optimized words are needed, just add the words without over-exaggerating. If exact numbers aren't provided in the base CV, use reasonable contextual scale.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. For EACH identified gap, you MUST provide exactly 5 free resources and 2 paid resources in the learning roadmap. "
        "The topics MUST strictly relate to the specific skill gap required for THIS EXACT JOB ROLE. NEVER hallucinate unrelated industries. "
        "If a specific URL isn't known, provide a highly relevant Coursera or edX search link.\n"
        "2. You MUST populate 'changes_made' with a list of specific optimizations. Tell the user EXACTLY which job role and bullet point you modified "
        "(e.g., 'Added asynchronous programming to your 2023 Jireh Computers role')."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
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
    
    system_prompt = (
        "You are an expert career coach. Generate a concise salary strategy, exactly 5 interview questions, and a brief 30-day prep plan based on the candidate's gaps. "
        "You MUST generate all responses exclusively in English"
        "CRITICAL AVOID TIMEOUTS: Keep your text concise and directly to the point. "
        "Use simple markdown bullet points (-). Do not use complex nested formatting, tables, or excessive line breaks that break JSON parsing."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
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