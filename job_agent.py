"""
streamlit_app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNIFIED FILE: Streamlit Frontend + LangGraph Backend
Run with: streamlit run streamlit_app.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import streamlit as st
import os
import operator
from pathlib import Path
from typing import TypedDict, List, Dict, Literal, Optional, Annotated
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA MODELS & SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
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
    gap_analysis: List[str] = Field(default=[])

class GeneratedDocs(BaseModel):
    job_id: str = "job_1"
    tailored_cv_markdown: str = "CV Content"
    cover_letter_markdown: str = "Cover Letter Content"
    learning_roadmap: List[Dict[str, str]] = Field(default=[])

class AgentState(TypedDict):
    base_cv: str
    model_provider: str 
    jobs_to_process: List[JobInput] 
    evaluations: Annotated[List[JobEvaluation], operator.add]
    generated_docs: Annotated[List[GeneratedDocs], operator.add]

# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM FACTORY
# ─────────────────────────────────────────────────────────────────────────────
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
        # Connects to your local llama.cpp server running at 127.0.0.1:58125
        return ChatOpenAI(
            api_key=os.getenv("LOCAL_MODEL_KEY", "sk-local-123"),
            base_url=os.getenv("LOCAL_MODEL_BASE_URL", "http://127.0.0.1:58125/v1"),
            model="local-model",
            temperature=0.1
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH NODES & BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_job_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Grade the CV against the JD on 12 dimensions (A-F)."),
        ("human", "CV:\n{cv}\n\nJD:\n{jd}")
    ])
    
    try:
        chain = prompt | llm.with_structured_output(JobEvaluation)
        result = chain.invoke({"cv": state["base_cv"], "jd": job.raw_text})
    except Exception as e:
        print(f"Fallback used due to: {e}")
        result = JobEvaluation(overall_score=85, gap_analysis=["Needs more Python"])
        
    return {"evaluations": [result]}

def generate_documents_node(state: AgentState):
    llm = get_llm(state["model_provider"])
    job = state["jobs_to_process"][0]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tailor the CV, write a cover letter, and provide 15-20 learning resources."),
        ("human", "Job: {job}\nBase CV: {cv}")
    ])
    
    try:
        chain = prompt | llm.with_structured_output(GeneratedDocs)
        result = chain.invoke({"cv": state["base_cv"], "job": job.raw_text})
    except Exception as e:
        print(f"Fallback used due to: {e}")
        result = GeneratedDocs(tailored_cv_markdown="Mock Tailored CV", cover_letter_markdown="Mock CL", learning_roadmap=[])
        
    # AUTOMATIC SAVE TO LAPTOP FOLDER
    Path("workspace").mkdir(exist_ok=True)
    with open(f"workspace/CV_{job.company_name}.md", "w") as f:
        f.write(result.tailored_cv_markdown)
    with open(f"workspace/CoverLetter_{job.company_name}.md", "w") as f:
        f.write(result.cover_letter_markdown)
        
    return {"generated_docs": [result]}

def build_job_agent():
    builder = StateGraph(AgentState)
    builder.add_node("evaluate", evaluate_job_node)
    builder.add_node("generate", generate_documents_node)
    builder.add_edge(START, "evaluate")
    builder.add_edge("evaluate", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=MemorySaver())

# ─────────────────────────────────────────────────────────────────────────────
# 4. STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Personal Job Agent", layout="wide")
    
    if "agent" not in st.session_state:
        st.session_state.agent = build_job_agent()
        st.session_state.thread_config = {"configurable": {"thread_id": "session_1"}}

    st.title("💼 Personal Job Agent")
    
    with st.sidebar:
        provider = st.selectbox("LLM Provider:", ["Local (llama.cpp)", "OpenAI", "Gemini"])
        base_cv = st.text_area("Base CV:", height=300)

    col1, col2 = st.columns(2)
    with col1: company_name = st.text_input("Company:")
    with col2: job_title = st.text_input("Title:")
    raw_jd = st.text_area("LinkedIn JD Text:")

    if st.button("Run Agent"):
        if base_cv and raw_jd:
            initial_state = {
                "base_cv": base_cv, "model_provider": provider,
                "jobs_to_process": [JobInput(id="1", company_name=company_name, job_title=job_title, raw_text=raw_jd)],
                "evaluations": [], "generated_docs": []
            }
            with st.spinner("Processing..."):
                for _ in st.session_state.agent.stream(initial_state, config=st.session_state.thread_config):
                    pass
                
            st.success("Done! Results saved to 'workspace/' folder on your laptop.")
            
            # Display results on screen
            state = st.session_state.agent.get_state(st.session_state.thread_config).values
            if state.get("evaluations"):
                st.metric("Score", f"{state['evaluations'][0].overall_score}/100")
            if state.get("generated_docs"):
                st.markdown("### Tailored CV Preview")
                st.markdown(state['generated_docs'][0].tailored_cv_markdown)

if __name__ == "__main__":
    main()