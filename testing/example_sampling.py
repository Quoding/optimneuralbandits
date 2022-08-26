import threading
import time
from queue import Queue
import sqlite3 as sql
import pandas as pd

con = sql.connect("chinook.db", check_same_thread=False)
statement = "SELECT * FROM EMPLOYEES;"

algo_over = False


def fetch_data(con: sql.Connection, output: Queue):
    global algo_over
    cur = con.cursor()
    cnt = 0
    while not algo_over:
        cnt += 1
        df = pd.read_sql(statement, con=con)[:cnt]
        time.sleep(10)
        output.put(df)
        print("Fetched data")


def main():
    global algo_over

    q = Queue()
    db_thread = threading.Thread(target=fetch_data, args=[con, q])
    db_thread.start()
    current_data = None

    # Make sure we have a first sample before starting
    # while current_data is None:
    #     if q:
    #         current_data = q.get()

    for i in range(100):
        time.sleep(1)
        if not q.empty():
            current_data = q.get()
        print(f"Executed {i}")

    algo_over = True


try:
    main()
except:
    # If anything happens and main crashes, make sure the thread exits
    algo_over = True
