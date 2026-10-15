import json
import os
import unicodedata
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "data" / "processed" / "aap_2020_atomic_promises_clean.json"

load_dotenv(ROOT / ".env")


def normalize(text):
    """Normalize Unicode and whitespace for diagnostic comparison."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


with open(JSON_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

with psycopg.connect(
    host=os.getenv("PGHOST", "localhost"),
    port=os.getenv("PGPORT", "5432"),
    dbname=os.getenv("PGDATABASE", "VoteYourWay"),
    user=os.getenv("PGUSER", "postgres"),
    password=os.getenv("PGPASSWORD"),
) as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                s.title,
                p.original_text,
                p.atomic_promise
            FROM promises p
            JOIN sections s ON s.section_id = p.section_id
            JOIN promise_sources ps ON ps.promise_id = p.promise_id
            WHERE ps.manifesto_id = %s
        """, ("30398bb3-18f7-4332-adfd-fbadbdcbfb12",))

        existing = cur.fetchall()

existing_by_section = {}

for section, original, atomic in existing:
    existing_by_section.setdefault(section, []).append(
        (original, atomic)
    )

exact_matches = 0
normalized_matches = 0
unmatched = []

for record in records:
    section = record["source"]["section_name"]

    for promise in record["atomic_promises"]:
        original = promise["original_text"]
        atomic = promise["atomic_promise"]

        candidates = existing_by_section.get(section, [])

        if (original, atomic) in candidates:
            exact_matches += 1
        elif any(
            normalize(original) == normalize(db_original)
            and normalize(atomic) == normalize(db_atomic)
            for db_original, db_atomic in candidates
        ):
            normalized_matches += 1
        else:
            unmatched.append({
                "section": section,
                "original_text": original,
                "atomic_promise": atomic,
            })

print("Validated JSON promises:", sum(
    len(r["atomic_promises"]) for r in records
))
print("Existing database records linked to manifesto:", len(existing))
print("Exact matches:", exact_matches)
print("Matches after Unicode/whitespace normalization:", normalized_matches)
print("Unmatched JSON promises:", len(unmatched))

for item in unmatched[:10]:
    print("\n--- UNMATCHED ---")
    print("Section:", item["section"])
    print("Original:", item["original_text"])
    print("Atomic:", item["atomic_promise"])