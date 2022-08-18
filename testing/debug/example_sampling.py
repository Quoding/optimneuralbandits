import sqlite3 as sql
import threading
import time

con = sql.connect("chinook.db", check_same_thread=False)

statement = "SELECT * FROM EMPLOYEES;"

current_data = []
next_data = []
time_to_sample = True


def fetch_data(con):
    # Fetch the data from the database
    global time_to_sample
    global next_data
    cur = con.cursor()
    next_data = cur.execute(statement).fetchall()
    time.sleep(10)
    time_to_sample = True
    print("Fetched data")


fetch_data(con)
current_data = next_data

for i in range(100):

    # Sample if a thread isn't already sampling the database
    if time_to_sample:
        print("Fetching data")
        db_thread = threading.Thread(target=fetch_data, args=[con])
        time_to_sample = False
        db_thread.start()

    # Update current_data to the data we just fetched with the thread
    current_data = next_data
    time.sleep(1)
    print(f"Executed {i}th loop")
