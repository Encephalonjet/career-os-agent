"""
auto_fetch.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3: Auto-Fetch Job Pipeline (MCP/Search Integration)
Features:
- Programmatically searches for job postings matching a title/location.
- Bypasses blockers using semantic search (Jina AI Search).
- Automatically routes discovered jobs through the LangGraph evaluator.
- Deposits evaluated jobs directly into the local SQLite Kanban DB.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import re
import requests
import time
from typing import List

# Import our existing Phase 1 & 2 logic
from database import add_job, get_all_jobs
from react_backend import run_agent

def search_job_urls(job_title: str, location: str, max_results: int = 3) -> List[str]:
    """
    Acts as an MCP search tool. Uses Jina AI's search endpoint to bypass 
    traditional scraping blocks and find highly relevant job posting URLs.
    """
    print(f"🔍 Searching for '{job_title}' roles in '{location}'...")
    
    # We restrict the search to known job boards to get high-quality URLs
    query = f"{job_title} jobs in {location} site:linkedin.com/jobs/view/ OR site:boards.greenhouse.io"
    search_url = f"https://s.jina.ai/{query}"
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'CareerOS-AutoFetcher/1.0'
    }
    
    try:
        response = requests.get(search_url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        
        urls = []
        # Jina Search returns a 'data' array with search results
        for item in data.get('data', []):
            url = item.get('url', '')
            # Filter to ensure we only grab actual job listing pages
            if "linkedin.com/jobs/view" in url or "boards.greenhouse.io" in url:
                if url not in urls:
                    urls.append(url)
            if len(urls) >= max_results:
                break
                
        print(f"✅ Found {len(urls)} job URLs.")
        return urls
        
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []

def run_auto_fetch_pipeline(base_cv_text: str, target_title: str, location: str, provider: str = "gemini", max_jobs: int = 3):
    """
    The main automation pipeline.
    1. Hunts for jobs.
    2. Checks if we already applied/evaluated them.
    3. Runs the LangGraph AI agent.
    4. Saves to the Kanban database.
    """
    print("🚀 Initiating Auto-Fetch Pipeline...")
    
    # 1. Fetch URLs
    found_urls = search_job_urls(target_title, location, max_results=max_jobs)
    
    if not found_urls:
        print("⚠️ No new jobs found. Try broadening the search terms.")
        return

    # Load existing jobs to prevent duplicate evaluations
    existing_jobs = get_all_jobs()
    existing_urls = [job.get('full_data', {}).get('jobs_to_process', [{}])[0].get('raw_text', '') for job in existing_jobs]
    
    processed_count = 0
    
    for url in found_urls:
        # Simple duplicate check (if the URL is already in our DB, skip it)
        if any(url in existing_text for existing_text in existing_urls):
            print(f"⏭️ Skipping {url} (Already in database)")
            continue
            
        print(f"\n🧠 Evaluating new job: {url}")
        
        try:
            # 2. Run the heavy LangGraph Evaluation & Document Generation
            # We pass the URL as the 'jd_text'. The agent's extract_metadata_node will auto-scrape it!
            result_state = run_agent(
                cv_text=base_cv_text,
                jd_text=url, 
                company="Auto-Fetched Company", 
                title=target_title,
                provider=provider
            )
            
            # 3. Extract evaluation score
            evals = result_state.get("evaluations", [{}])[0]
            score = getattr(evals, "overall_score", 0)
            
            # Extract actual company/title found by the AI metadata node
            processed_job = result_state.get("jobs_to_process", [{}])[0]
            final_company = getattr(processed_job, 'company_name', "Unknown Company")
            final_title = getattr(processed_job, 'job_title', "Unknown Title")
            
            # 4. Save directly to the local SQLite Kanban board
            job_id = add_job(
                company=final_company,
                title=final_title,
                status="To Apply",
                score=score,
                full_data=result_state
            )
            
            print(f"💾 Saved to Kanban Board: {final_company} - {final_title} (Score: {score}/100)")
            processed_count += 1
            
            # Sleep briefly to avoid rate-limiting our LLM API or scraping endpoints
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ Failed to process {url}: {e}")

    print(f"\n🎉 Auto-Fetch Complete! Added {processed_count} new optimized jobs to your Kanban board.")

# Example Usage:
if __name__ == "__main__":
    # To test this locally, you would read your CV from a file and run the pipeline:
    # 
    # with open("my_base_cv.txt", "r") as f:
    #     my_cv = f.read()
    # 
    # run_auto_fetch_pipeline(
    #     base_cv_text=my_cv,
    #     target_title="AI Engineer",
    #     location="London, UK",
    #     provider="gemini",
    #     max_jobs=2
    # )
    pass