from playwright.sync_api import sync_playwright
import time
import os

def autofill_job_application(
    job_url: str, 
    user_name: str, 
    user_email: str, 
    user_phone: str, 
    user_linkedin: str, 
    resume_path: str = None
):
    """
    Launches a visible Chromium browser, navigates to the job URL, and attempts 
    to map user profile data to common ATS form fields. Pauses for human review.
    """
    print(f"🚀 Launching Playwright to navigate to: {job_url}")
    
    with sync_playwright() as p:
        # Headless MUST be False so you can see the browser and manually submit
        browser = p.chromium.launch(headless=False, args=['--start-maximized'])
        context = browser.new_context(no_viewport=True)
        page = context.new_page()
        
        try:
            page.goto(job_url, wait_until="networkidle", timeout=30000)
            print("✅ Successfully loaded page. Scanning for form fields...")
            
            # Wait a brief moment for dynamic ATS scripts (like Greenhouse/Lever) to load forms
            page.wait_for_timeout(3000)
            
            # 1. Fill Name
            try:
                # Looks for standard name fields in Greenhouse, Lever, Workday, etc.
                name_field = page.locator('input[name*="name" i], input[id*="name" i], input[autocomplete*="name" i]').first
                if name_field.is_visible(timeout=1000):
                    name_field.fill(user_name)
                    print(f"✏️ Filled Name: {user_name}")
            except Exception:
                pass

            # 2. Fill Email
            try:
                email_field = page.locator('input[type="email" i], input[name*="email" i], input[id*="email" i]').first
                if email_field.is_visible(timeout=1000):
                    email_field.fill(user_email)
                    print(f"✏️ Filled Email: {user_email}")
            except Exception:
                pass

            # 3. Fill Phone
            try:
                phone_field = page.locator('input[type="tel" i], input[name*="phone" i], input[id*="phone" i]').first
                if phone_field.is_visible(timeout=1000):
                    phone_field.fill(user_phone)
                    print(f"✏️ Filled Phone: {user_phone}")
            except Exception:
                pass

            # 4. Fill LinkedIn
            try:
                linkedin_field = page.locator('input[name*="linkedin" i], input[id*="linkedin" i], input[name*="urls[LinkedIn]" i]').first
                if linkedin_field.is_visible(timeout=1000):
                    linkedin_field.fill(user_linkedin)
                    print(f"✏️ Filled LinkedIn: {user_linkedin}")
            except Exception:
                pass

            # 5. Attach Resume PDF
            if resume_path and os.path.exists(resume_path):
                try:
                    # Target file upload inputs
                    file_input = page.locator('input[type="file" i]').first
                    if file_input.is_attached(timeout=1000):
                        file_input.set_input_files(resume_path)
                        print(f"📎 Attached Resume: {resume_path}")
                except Exception:
                    print("⚠️ Could not auto-attach resume. Please upload manually.")

            print("\n🛑 PAUSING FOR MANUAL REVIEW 🛑")
            print("1. Review the filled fields.")
            print("2. Complete any CAPTCHAs.")
            print("3. Click the final 'Submit' button.")
            print("4. Close the browser window when finished to end the script.")
            
            # This halts the script indefinitely and opens the Playwright inspector
            # You must close the browser manually to move on.
            page.pause()
            
        except Exception as e:
            print(f"❌ Automation encountered an error: {e}")
            page.pause()
            
        finally:
            browser.close()

if __name__ == "__main__":
    # Test script block. You can run this directly via `python autofill.py`
    test_url = "https://www.linkedin.com/jobs"
    autofill_job_application(
        job_url=test_url,
        user_name="Jane Doe",
        user_email="jane.doe@example.com",
        user_phone="555-0199",
        user_linkedin="https://linkedin.com/in/janedoe"
    )