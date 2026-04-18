import argparse
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

load_dotenv()

GROQ_API_KEY_DIFF = os.environ.get("GROQ_API_KEY_DIFF")
MODEL = "llama-3.1-8b-instant"
SLEEP_SECONDS = 0

client = Groq(api_key=GROQ_API_KEY_DIFF)


def clean_text(text):
    """Clean raw act text by removing OCR noise and normalizing whitespace."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, max_words=800):
    """Split text into chunks of up to max_words words."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)

    return chunks


def build_prompt(chunk):
    """Build the LLM prompt for extracting high-level policy actions."""
    return f"""
You are analyzing a government act.

Your task is to extract a SMALL set of HIGH-LEVEL policy actions.

### VERY IMPORTANT RULES:

1. DO NOT extract repetitive or similar actions

2. GROUP similar changes into ONE single action

3. DO NOT list variations (like different years, percentages, slabs separately)

4. Focus on BIG policy changes, not detailed values

5. Keep total outputs between 5 to 10 actions MAX

6. Every action MUST be directly traceable to the text
7. Do NOT reinterpret numerical tables as \"increase\" or \"decrease\"
8. If the text defines rates or percentages, describe it as:
  \"Define tax rates...\" or \"Specify tax slabs...\"
9. NEVER use words like:
  - increase
  - decrease
  - reduce
UNLESS those exact words appear in the text

10. DO NOT infer or assume anything not explicitly written
11. Every action must be directly supported by the text
12. Avoid vague outputs like "define tax rates" — be specific to context

### Think like this:

Instead of:

- \"Increase tax for 2 years\"

- \"Increase tax for 3 years\"

→ Write:

- \"Revise tax structure based on vehicle age\"

### Output must:

- Be concise

- Be policy-level

- Be non-redundant

### Output format:

Return ONLY a JSON array of strings.

### Text:

{chunk}
"""


def parse_json_array(text):
    """Parse a JSON array from the LLM output."""
    try:
        cleaned = (
            text.replace("```json", "")
            .replace("```python", "")
            .replace("```", "")
            .strip()
        )
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []

        data = json.loads(match.group())
        if isinstance(data, list):
            return [str(x).strip() for x in data if isinstance(x, str) and str(x).strip()]
        return []
    except Exception as e:
        print("Parse error:", e)
        return []


def consolidate_actions(actions):
    """Consolidate extracted actions into a small set of unique high-level actions."""
    if not actions:
        return []

    prompt = f"""
You are given a list of policy actions.

Your job:
- Remove duplicates
- Merge similar actions
- Keep only 5–10 high-level actions
- Remove vague or generic actions (e.g., "define tax rates")
- Ensure each action is specific and grounded in the given list
- Remove any action that uses inferred words like:
  \"increase\", \"reduce\", \"waive\", unless explicitly present
- Prefer neutral wording like:
  \"define\", \"specify\", \"introduce\", \"amend\"
### STRICT RULES:
- Return ONLY a JSON array
- DO NOT write code
- DO NOT include ``` or python
- DO NOT include explanations
- Output must start with [ and end with ]
- Do not introduce new actions not present in input list

Actions:
{actions[:20]}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.choices[0].message.content
        print("Raw consolidation output:", raw)

        parsed = parse_json_array(raw)
        if not parsed:
            return list(dict.fromkeys(actions))
        return parsed
    except Exception as e:
        print("Consolidation error:", e)
        return list(dict.fromkeys(actions))


def extract_actions_from_text(text, timeout_seconds=60):
    """Extract high-level policy actions from act text (single-call version)."""

    if not text or len(text.strip()) == 0:
        return []

    try:
        # 🔥 SINGLE CALL (no chunking)
        prompt = build_prompt(text)

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout_seconds,
        )

        output = response.choices[0].message.content
        print(f"\n[DEBUG] Raw LLM output:\n{output}\n")
        
        actions = parse_json_array(output)
        print(f"[DEBUG] Parsed actions: {actions}")

        # Simple grounding filter
        filtered_actions = []
        for act in actions:
            if not isinstance(act, str):
                continue
            if len(act) < 15:
                continue
            if any(word in act.lower() for word in ["some", "various", "certain activities"]):
                continue
            filtered_actions.append(act)

        # Remove duplicates
        filtered_actions = list(dict.fromkeys(filtered_actions))

        # Consolidate only if too many
        if len(filtered_actions) > 10:
            return consolidate_actions(filtered_actions)

        return filtered_actions

    except Exception as e:
        print(f"[ERROR] Exception in extract_actions_from_text: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_file(input_path, output_path, text_column="act_text", min_text_length=100, skip_filled=True, limit=None):
    """
    Process act text to extract actions.
    
    Args:
        input_path: Path to input CSV with act text
        output_path: Path to output CSV with actions
        text_column: Name of the column containing text to process
        min_text_length: Minimum text length to process
        skip_filled: (Deprecated) This parameter is ignored; all rows are processed and overwritten.
        limit: Max number of rows to process (for testing)
    """
    print(f"\n[STARTUP] process_file called")
    print(f"[STARTUP] Input: {input_path}")
    print(f"[STARTUP] Output: {output_path}")
    print(f"[STARTUP] Text column: {text_column}")
    print(f"[STARTUP] API Key set: {bool(GROQ_API_KEY_DIFF)}")
    if limit:
        print(f"[STARTUP] LIMIT: Processing only first {limit} rows (TEST MODE)\n")
    else:
        print()
    
    df_input = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df_input)} rows from input")
    
    # Validate that the text column exists
    if text_column not in df_input.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {list(df_input.columns)}")
    
    # Always use the input file as base (it has act_text)
    # The output file is only for reference if we want to resume
    df_output = df_input.copy()
    if "actions" not in df_output.columns:
        df_output["actions"] = None
    
    print(f"[INFO] Starting fresh from input file (which has act_text column)")
    
    # Identify rows to process (all rows; overwrite existing actions)
    rows_to_process = df_output.index.tolist()
    if limit:
        rows_to_process = rows_to_process[:limit]
    
    print(f"[INFO] Processing {len(rows_to_process)} rows (overwriting existing actions)...\n")
    
    # Process only the rows that need it
    for idx in tqdm(rows_to_process, desc="Processing acts"):
        row = df_output.iloc[idx]
        text = clean_text(row.get(text_column, ""))
        
        print(f"\n[ROW {idx}] Title: {row.get('title')[:50]}... | Text length: {len(text)}")
        
        if len(text) < min_text_length:
            print(f"[ROW {idx}] SKIPPED: Text too short ({len(text)} < {min_text_length})")
            df_output.at[idx, "actions"] = "[]"
            continue
        
        print(f"[ROW {idx}] Calling LLM...")
        actions = extract_actions_from_text(text)
        df_output.at[idx, "actions"] = json.dumps(actions if isinstance(actions, list) else [])
        print(f"[ROW {idx}] Actions stored: {actions}")
    
    df_output.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved extracted actions to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract high-level policy actions from PRS act text using an LLM."
    )
    parser.add_argument(
        "--input",
        default="data/prs_datasets/prs_karnataka_acts_processed.csv",
        help="Input CSV path containing act text (output from prs_pdf_processor_acts.py).",
    )
    parser.add_argument(
        "--output",
        default="data/prs_datasets/prs_karnataka_acts_with_actions.csv",
        help="Output CSV path for extracted actions.",
    )
    parser.add_argument(
        "--text-column",
        default="act_text",
        help="Name of the column containing text to process (default: act_text).",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=100,
        help="Minimum act text length to process.",
    )
    parser.add_argument(
        "--fill-all",
        action="store_true",
        help="Process all rows (default: skip rows with non-empty actions).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N rows (for testing)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("[STARTUP] Script started")
    args = parse_args()
    print(f"[STARTUP] Arguments parsed: input={args.input}, output={args.output}")
    process_file(
        args.input,
        args.output,
        text_column=args.text_column,
        min_text_length=args.min_text_length,
        skip_filled=not args.fill_all,
        limit=args.limit
    )
