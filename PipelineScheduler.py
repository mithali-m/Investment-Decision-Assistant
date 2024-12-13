from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import FetchData
import os
import sys
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

python_executable = sys.executable

def run_incremental_load():
    """Function to run IncrementalLoad.py"""
    try:
        print("Running Incremental Load...")
        subprocess.run(["python", "IncrementalLoad.py"], check=True)
        print("Incremental Load completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during Incremental Load: {e}")

# Create the scheduler
scheduler = BlockingScheduler()
scheduler.add_job(run_incremental_load, 'interval', seconds=1)

print("Scheduler started. Incremental Load will run every 5 minutes.")
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    print("Scheduler stopped.")