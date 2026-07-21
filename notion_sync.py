import os
import database
from notion_client import Client

def sync_to_notion():
    """
    Reads all jobs from the local SQLite database and upserts them into the Notion Database.
    Prevents duplicates by checking the unique 'Job ID' field.
    """
    notion_key = os.getenv("NOTION_API_KEY")
    database_id = os.getenv("NOTION_DATABASE_ID")

    if not notion_key or not database_id:
        raise Exception("Missing Notion configuration. Please add NOTION_API_KEY and NOTION_DATABASE_ID to your .env file.")

    # Initialize the Notion Client
    notion = Client(auth=notion_key)
    
    # NEW NOTION SDK COMPATIBILITY: Retrieve the database to get its data_source_id
    try:
        db_info = notion.databases.retrieve(database_id=database_id)
        if "data_sources" in db_info and len(db_info["data_sources"]) > 0:
            data_source_id = db_info["data_sources"][0]["id"]
        else:
            # Fallback if the API returns the old format or defaults
            data_source_id = database_id
    except Exception as e:
        raise Exception(f"Failed to retrieve Notion Database info. Ensure the integration is connected. Error: {e}")
    
    # Fetch all local jobs
    local_jobs = database.get_all_jobs()
    synced_count = 0

    for job in local_jobs:
        job_id = job["id"]
        
        # 1. Check if this specific job already exists in Notion using the new Data Sources API
        response = notion.data_sources.query(
            data_source_id=data_source_id,
            filter={
                "property": "Job ID",
                "rich_text": {
                    "equals": job_id
                }
            }
        )
        
        # 2. Map local data to Notion Properties exactly as named in the Setup Guide
        properties = {
            "Company": {
                "title": [{"text": {"content": job.get("company", "Unknown")}}]
            },
            "Job Title": {
                "rich_text": [{"text": {"content": job.get("title", "Unknown")}}]
            },
            "Status": {
                "select": {"name": job.get("status", "To Apply")}
            },
            "Score": {
                "number": job.get("score", 0)
            },
            "Job ID": {
                "rich_text": [{"text": {"content": job_id}}]
            }
        }
        
        # 3. Upsert (Update if exists, Create if new)
        if response["results"]:
            # Job exists, update the status and score
            page_id = response["results"][0]["id"]
            notion.pages.update(page_id=page_id, properties=properties)
        else:
            # New job, create a new row using the original database_id as parent
            notion.pages.create(parent={"database_id": database_id}, properties=properties)
            
        synced_count += 1
        
    return synced_count