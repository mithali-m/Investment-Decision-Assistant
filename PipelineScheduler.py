import schedule
import time
import IncrementalLoad

def job():
    IncrementalLoad.main()
    print("Incremental load ran successfully")

#schedule.every().day.at("06:00").do(job)  # Schedule to run daily at 6:00 AM
schedule.every(5).minutes.do(job) # Schedule to run every 15 seconds

while True:
    schedule.run_pending()
    time.sleep(100)
