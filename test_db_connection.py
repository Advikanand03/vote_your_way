import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

try:
    with psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.getenv("PGDATABASE", "VoteYourWay"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD"),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_user, current_database();")
            print("Connection successful!")
            print(cur.fetchone())

except Exception as e:
    print("Connection failed:", e)