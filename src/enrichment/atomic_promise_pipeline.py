import json
import os
from dotenv import load_dotenv

load_dotenv()
import re
import time
import pandas as pd
from groq import Groq


# -------------------------
# CONFIG
# -------------------------
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
MODEL = "llama-3.1-8b-instant"
SLEEP_SECONDS = 1

INPUT_PATH = "data/processed/karnataka_inc_promises_cleaned.csv"
OUTPUT_PATH = "data/processed/karnataka_inc_promises_atomic.csv"
PARTIAL_OUTPUT_PATH = "data/processed/karnataka_inc_promises_atomic_partial.csv"

client = Groq(api_key=GROQ_API_KEY)


def build_prompt(category, promise_text):
    return f"""
You are processing rows from a political manifesto dataset.
Each row contains:
- category (heading)
- promise_text (raw extracted text, may contain context, criticism, and actual commitments mixed together)
Your task is to transform each row into one or more clean, atomic, actionable promises.

### Instructions:

1. Identify actionable commitments
   - Extract ONLY the parts of the text that describe a concrete action the government/political party intends to take.
   - Ignore background context, opinions, criticism, ideology, or justification.

2. Split multiple promises
   - If a single input contains multiple commitments, split them into separate promises.
   - Each output row must contain ONLY ONE promise.

3. Rewrite into clean format
   - Convert each extracted promise into a short, clear, action-oriented sentence.
   - Start with a strong action verb (e.g., Provide, Increase, Establish, Launch, Ensure, Implement).
   - Remove filler phrases like:
     - "The party is committed to..."
     - "We believe..."
     - "Accordingly..."
   - Keep meaning intact, do NOT hallucinate or add new information.

4. Preserve key details
   - Keep important numerical values (money, %, counts)
   - Keep target groups (e.g., women, minorities)
   - Keep timelines if mentioned

5. Output format
Return a JSON array where each item has:
{{
  "category": "<same as input>",
  "clean_promise": "<one atomic actionable promise>"
}}

### Input Row:
category: {category}
promise_text: {promise_text}

### Important Rules:
- DO NOT return explanations
- DO NOT include non-actionable text
- DO NOT merge multiple promises
- DO NOT drop important quantitative details
"""


def parse_json_array(text):
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "")).strip()
            clean_promise = str(item.get("clean_promise", "")).strip()
            if clean_promise:
                out.append({"category": category, "clean_promise": clean_promise})
        return out
    except Exception as e:
        print("JSON parse error:", e)
        return []


def to_atomic_promises(category, promise_text, retries=3):
    prompt = build_prompt(category, promise_text)
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            result = completion.choices[0].message.content
            parsed = parse_json_array(result)
            if parsed:
                return parsed
        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            time.sleep(2)

    return []


def load_partial_state():
    if not os.path.exists(PARTIAL_OUTPUT_PATH):
        return [], set()
    try:
        df_partial = pd.read_csv(PARTIAL_OUTPUT_PATH)
        records = df_partial.to_dict("records")
        done_ids = set(df_partial["source_promise_id"].astype(str).tolist())
        print(f"Loaded partial state with {len(records)} rows.")
        return records, done_ids
    except Exception as e:
        print("Failed to load partial file:", e)
        return [], set()


def save_partial(rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(PARTIAL_OUTPUT_PATH, index=False)
    print(f"Saved partial -> {PARTIAL_OUTPUT_PATH} ({len(rows)} rows)")


def is_valid_promise(text):
    if len(text) < 10:
        return False
    if text.lower().startswith(("we believe", "it is important", "there is a need")):
        return False
    return True

def main():
    df = pd.read_csv(INPUT_PATH)
    rows, done_ids = load_partial_state()

    for _, row in df.iterrows():
        source_id = str(row["promise_id"])
        if source_id in done_ids:
            continue

        category = str(row.get("category", "")).strip()
        promise_text = str(row.get("promise_text", "")).strip()
        print(f"Processing {source_id}")

        atomic_items = to_atomic_promises(category, promise_text)
        added_any = False
        if atomic_items:
            for item in atomic_items:
                clean = item.get("clean_promise", "").strip()
                if not is_valid_promise(clean):
                    continue

                rows.append(
                    {
                        "source_promise_id": source_id,
                        "category": item.get("category", category) or category,
                        "clean_promise": clean,
                        "is_atomic": True
                    }
                )
                added_any = True

        if not added_any:
            print(f"⚠️ Fallback used for {source_id}")
            rows.append(
                {
                    "source_promise_id": source_id,
                    "category": category,
                    "clean_promise": promise_text,
                    "is_atomic": False
                }
            )

        done_ids.add(source_id)
        save_partial(rows)
        time.sleep(SLEEP_SECONDS)

    final_df = pd.DataFrame(rows)
    final_df = final_df[final_df["clean_promise"].str.len() > 0]
    final_df = final_df.drop_duplicates(subset=["source_promise_id", "clean_promise"])
    final_df = final_df.reset_index(drop=True)
    final_df.insert(0, "atomic_promise_id", [f"A{i}" for i in range(1, len(final_df) + 1)])
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Atomic promise dataset created -> {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
