"""
auto_fetch.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3: Auto-Fetch Job Pipeline (MCP/Search Integration)
Features:
- Programmatically searches for job postings matching a title/location.
- Dynamic query routing for specific ATS Job Boards to prevent Jina API failures.
- Prompt-injected Filters: Passes your exact Visa/Model constraints to LangGraph.
- DuckDuckGo Fallback: Automatically catches API blocks and fails over.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import requests
import time
from typing import List
from database import add_job, get_all_jobs
from react_backend import run_agent
from duckduckgo_search import DDGS

def search_job_urls(job_title: str, location: str, job_board: str, max_results: int = 5) -> List[str]:
    """
    Uses Jina AI to bypass bot-blockers, with a DuckDuckGo failover.
    Dynamically adjusts the search string based on your target Job Board.
    """
    print(f"🔍 Searching for '{job_title}' roles in '{location}'...", flush=True)
    
    board_mapping = {
        "LinkedIn": "site:linkedin.com",
        "Greenhouse": "site:greenhouse.io",
        "Lever": "site:lever.co",
        "Workday": "site:myworkdayjobs.com",
        "Ashby": "site:ashbyhq.com",
        "Jobright": "site:jobright.ai"
    }
    
    if job_board in board_mapping:
        boards = board_mapping[job_board]
        print(f"🎯 Streamlining search specifically to: {job_board}", flush=True)
    else:
        boards = "site:linkedin.com OR site:greenhouse.io OR site:lever.co OR site:myworkdayjobs.com OR site:ashbyhq.com OR site:jobright.ai"
        
    query = f"{job_title} jobs in {location} {boards}"
    search_url = f"https://s.jina.ai/{query.replace(' ', '%20')}"
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'CareerOS-AutoFetcher/1.0'
    }
    
    valid_domains = ["linkedin.com/jobs", "greenhouse.io", "lever.co", "myworkdayjobs.com", "ashbyhq.com", "jobright.ai"]
    urls = []
    
    try:
        response = requests.get(search_url, headers=headers, timeout=20)
        
        print(f"📡 Raw Jina AI Status Code: {response.status_code}", flush=True)
        print(f"📡 Raw Jina AI Response (First 300 chars): {response.text[:300]}", flush=True)
        
        response.raise_for_status()
        data = response.json()
        
        for item in data.get('data', []):
            url = item.get('url', '')
            if any(domain in url for domain in valid_domains):
                if url not in urls:
                    urls.append(url)
            if len(urls) >= max_results:
                break
                
        if urls:
            print(f"✅ Found {len(urls)} job URLs via Jina AI.", flush=True)
            return urls
        else:
            print("⚠️ Jina AI returned 0 valid results. Initiating Fallback...", flush=True)
            
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code if http_err.response is not None else "Unknown"
        print(f"⚠️ Jina AI Blocked Request (Status: {status_code}). Initiating Fallback...", flush=True)
    except Exception as e:
        print(f"⚠️ Jina AI Request Failed: {e}. Initiating Fallback...", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # DUCKDUCKGO FALLBACK LOGIC
    # ─────────────────────────────────────────────────────────────────────────
    print("🔄 Switching to DuckDuckGo Search Fallback...", flush=True)
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=15)
            urls = []
            
            if results:
                for r in results:
                    url = r.get('href', '')
                    if any(domain in url for domain in valid_domains):
                        if url not in urls:
                            urls.append(url)
                    if len(urls) >= max_results:
                        break
                        
            print(f"✅ Found {len(urls)} job URLs via DuckDuckGo.", flush=True)
            return urls
    except Exception as fallback_e:
        print(f"❌ Fallback also failed: {fallback_e}", flush=True)
        return []

def run_auto_fetch_pipeline(base_cv_text: str, target_title: str, location: str, provider: str = "gemini", max_jobs: int = 5, job_board: str = "All Supported Boards", work_model: str = "Any", job_type: str = "Any", visa: str = "Any"):
    """
    Hunts for jobs, injects your filters into the LangGraph prompt, and saves them.
    """
    print("🚀 Initiating Auto-Fetch Pipeline...", flush=True)
    
    found_urls = search_job_urls(target_title, location, job_board, max_results=max_jobs)
    
    if not found_urls:
        print("⚠️ No new jobs found. Try broadening the search terms or picking 'All Supported Boards'.", flush=True)
        return 0

    existing_jobs = get_all_jobs()
    existing_urls = [job.get('full_data', {}).get('jobs_to_process', [{}])[0].get('raw_text', '') for job in existing_jobs]
    
    processed_count = 0
    
    for url in found_urls:
        if any(url in existing_text for existing_text in existing_urls):
            print(f"⏭️ Skipping {url} (Already in database)", flush=True)
            continue
            
        print(f"\n🧠 Evaluating new job: {url}", flush=True)
        
        try:
            injected_jd_payload = (
                f"USER PREFERENCES & CONSTRAINTS:\n"
                f"- Work Model Required: {work_model}\n"
                f"- Job Type Required: {job_type}\n"
                f"- Visa/Sponsorship Rules: {visa}\n"
                f"CRITICAL AI INSTRUCTION: If this job explicitly contradicts these user preferences, "
                f"you MUST severely penalize the 'overall_score' and fail the 'domain_fit' dimension.\n"
                f"{'-'*50}\n\n"
                f"{url}"
            )

            result_state = run_agent(
                cv_text=base_cv_text,
                jd_text=injected_jd_payload, 
                company="Auto-Fetched Company", 
                title=target_title,
                provider=provider
            )
            
            evals = result_state.get("evaluations", [{}])[0]
            score = getattr(evals, "overall_score", 0)
            
            processed_job = result_state.get("jobs_to_process", [{}])[0]
            final_company = getattr(processed_job, 'company_name', "Unknown Company")
            final_title = getattr(processed_job, 'job_title', "Unknown Title")
            
            add_job(
                company=final_company,
                title=final_title,
                status="To Apply",
                score=score,
                full_data=result_state
            )
            
            print(f"💾 Saved to Kanban Board: {final_company} - {final_title} (Score: {score}/100)", flush=True)
            processed_count += 1
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ Failed to process {url}: {e}", flush=True)

    print(f"\n🎉 Auto-Fetch Complete! Added {processed_count} new optimized jobs to your Kanban board.", flush=True)
    return processed_count

if __name__ == "__main__":
    pass