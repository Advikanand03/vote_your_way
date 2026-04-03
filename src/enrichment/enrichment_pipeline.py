import pandas as pd
import json
import time
import re
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# CONFIG
# -------------------------
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
MODEL = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

# -------------------------
# LOAD YOUR DATASET
# -------------------------
df = pd.read_csv("data/processed/karnataka_inc_promises_cleaned.csv")

# -------------------------
# SAFE JSON PARSER (CRITICAL FIX)
# -------------------------
def safe_parse_json(text):
    try:
        # remove markdown
        text = text.replace("```json", "").replace("```", "").strip()

        # extract JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())

        return {}

    except Exception as e:
        print("JSON Parse Error:", e)
        return {}

# -------------------------
# LLM FUNCTION
# -------------------------
def enrich_promise(promise_text, category):

    prompt = f"""
    Analyze the following government promise.

    Promise:
    {promise_text}

    Category:
    {category}

    Extract the following fields:

    - sector (choose from: Economy, Agriculture, Health, Education, Infrastructure, Welfare, Governance, Energy, Transport, Other)
    - sub_sector
    - quantifiable (Yes/No)
    - target_value
    - timeline_mentioned (Yes/No)
    - target_year
    - commitment_type

    Return ONLY a valid JSON object.
    Do NOT include explanation or extra text.

    {{
        "sector":"",
        "sub_sector":"",
        "quantifiable":"",
        "target_value":"",
        "timeline_mentioned":"",
        "target_year":"",
        "commitment_type":""
    }}
    """

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        result = completion.choices[0].message.content

        # DEBUG (optional)
        # print("RAW OUTPUT:", result)

        return safe_parse_json(result)

    except Exception as e:
        print("API Error:", e)
        return {}

# -------------------------
# APPLY ENRICHMENT
# -------------------------
enriched_rows = []

for i, row in df.iterrows():

    print(f"Processing {row['promise_id']}")

    extra = enrich_promise(row["promise_text"], row["category"])

    enriched_rows.append({
        "promise_id": row["promise_id"],
        "category": row["category"],
        "promise_text": row["promise_text"],
        "sector": extra.get("sector", ""),
        "sub_sector": extra.get("sub_sector", ""),
        "quantifiable": extra.get("quantifiable", "No"),
        "target_value": extra.get("target_value", ""),
        "timeline_mentioned": extra.get("timeline_mentioned", "No"),
        "target_year": extra.get("target_year", ""),
        "commitment_type": extra.get("commitment_type", "")
    })

    time.sleep(1)  # avoid rate limits

# -------------------------
# SAVE FINAL DATASET
# -------------------------
final_df = pd.DataFrame(enriched_rows)

final_df.to_csv("data/processed/final_enriched_dataset.csv", index=False)

print("Final dataset created successfully!")