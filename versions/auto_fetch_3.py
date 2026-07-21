"""
auto_fetch.py
*************************************************************************************
Phase 3: Auto-Fetch Job Pipeline (MCP/Search Integration)
Features:
- Programmatically searches for job postings matching a title/location.
- Dynamic query routing for specific ATS Job Boards to prevent Jina API failures.
- Prompt-injected Filters: Passes your exact Visa/Model constraints to LangGraph.
- DuckDuckGo Fallback: Automatically catches API blocks and fails over.
- Parallel Processing: Slashes processing time using ThreadPoolExecutor.
*************************************************************************************
"""
import requests
import time
import concurrent.futures
from typing import List, Tuple
from urllib.parse import urlparse
from database import add_job, get_all_jobs
from react_backend import run_agent
from duckduckgo_search import DDGS

def get_source_from_url(url: str) -> str:
    """Extracts a clean brand name from the raw job URL to display in the UI."""
    url_lower = url.lower()
    if "linkedin.com" in url_lower: return "LinkedIn"
    if "greenhouse.io" in url_lower: return "Greenhouse"
    if "lever.co" in url_lower: return "Lever"
    if "myworkdayjobs.com" in url_lower: return "Workday"
    if "ashbyhq.com" in url_lower: return "Ashby"
    if "jobright.ai" in url_lower: return "Jobright"
    return "Web Scrape"

def search_job_urls(job_title: str, location: str, job_board: str, max_results: int = 5) -> List[str]:
    """
    Uses Jina AI to bypass bot-blockers, with a DuckDuckGo failover.
    Dynamically adjusts the search string based on your target Job Board.
    """
    loc_display = location if location and location.strip() else "Anywhere"
    print(f"🔍 Searching for '{job_title}' roles in '{loc_display}'...", flush=True)
    
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
        
    # Strictly quote the location to prevent DuckDuckGo/Jina from ignoring it in favor of the title
    loc_query = f'"{location}"' if location and location.strip() else ""
    query = f"{job_title} jobs {loc_query} {boards}".strip()
    
    search_url = f"https://s.jina.ai/{query.replace(' ', '%20')}"
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'CareerOS-AutoFetcher/1.0'
    }
    
    valid_domains = ["linkedin.com/jobs", "greenhouse.io", "lever.co", "myworkdayjobs.com", "ashbyhq.com", "jobright.ai"]
    urls = []
    
    try:
        response = requests.get(search_url, headers=headers, timeout=20)
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

# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL PROCESSING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def process_single_job(url: str, base_cv_text: str, target_title: str, location: str, provider: str, work_model: str, job_type: str, visa: str) -> str:
    """
    Helper function to run a single job evaluation in an isolated thread.
    Returns the new job_id if successfully processed, or None if failed.
    """
    print(f"\n🧠 Evaluating new job: {url}", flush=True)
    try:
        # Conditionally inject location requirement only if the user specified one
        location_constraint = f"- Location Required: {location}\n" if location and location.strip() else ""
        
        injected_jd_payload = (
            f"USER PREFERENCES & CONSTRAINTS:\n"
            f"{location_constraint}"
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
        source_tag = get_source_from_url(url)
        
        job_id = add_job(
            company=final_company,
            title=final_title,
            status="To Apply",
            score=score,
            full_data=result_state,
            source=source_tag
        )
        
        print(f"💾 Saved to Kanban Board: {final_company} - {final_title} (Score: {score}/100)", flush=True)
        return job_id
    except Exception as e:
        print(f"❌ Failed to process {url}: {e}", flush=True)
        return None


def run_auto_fetch_pipeline(base_cv_text: str, target_title: str, location: str, provider: str = "gemini", max_jobs: int = 5, job_board: str = "All Supported Boards", work_model: str = "Any", job_type: str = "Any", visa: str = "Any") -> Tuple[int, List[str]]:
    """
    Hunts for jobs, injects your filters into the LangGraph prompt, and saves them.
    Returns a tuple: (number_of_jobs_added, list_of_new_job_ids)
    """
    print("🚀 Initiating Auto-Fetch Pipeline...", flush=True)
    
    found_urls = search_job_urls(target_title, location, job_board, max_results=max_jobs)
    
    if not found_urls:
        print("⚠️ No new jobs found. Try broadening the search terms or picking 'All Supported Boards'.", flush=True)
        return 0, []

    existing_jobs = get_all_jobs()
    existing_urls = [job.get('full_data', {}).get('jobs_to_process', [{}])[0].get('raw_text', '') for job in existing_jobs]
    
    new_urls = []
    for url in found_urls:
        if any(url in existing_text for existing_text in existing_urls):
            print(f"⏭️ Skipping {url} (Already in database)", flush=True)
        else:
            new_urls.append(url)
    
    processed_count = 0
    new_job_ids = []
    
    if new_urls:
        print(f"⚡ Launching parallel processing for {len(new_urls)} jobs...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    process_single_job, url, base_cv_text, target_title, location, provider, work_model, job_type, visa
                ) for url in new_urls
            ]
            
            for future in concurrent.futures.as_completed(futures):
                job_id = future.result()
                if job_id:
                    processed_count += 1
                    new_job_ids.append(job_id)

    print(f"\n🎉 Auto-Fetch Complete! Added {processed_count} new optimized jobs to your Kanban board.", flush=True)
    return processed_count, new_job_ids

if __name__ == "__main__":
    pass