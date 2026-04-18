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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"
SLEEP_SECONDS = 1

client = Groq(api_key=GROQ_API_KEY)


def clean_text(text):
    """Clean raw bill text by removing OCR noise and normalizing whitespace."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, max_words=400):
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
You are analyzing a government bill.

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
        cleaned = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return []

        data = json.loads(match.group())
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return []
    except Exception:
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


def extract_actions_from_text(text):
    """Extract high-level policy actions from bill text."""
    print("Function started")
    chunks = chunk_text(text)
    all_actions = []

    for chunk in chunks:
        try:
            prompt = build_prompt(chunk)
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            output = response.choices[0].message.content
            print("Raw model output:", output)
            actions = parse_json_array(output)
            print("Parsed actions:", actions)
            all_actions.extend(actions)
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print("Error:", e)

    final_actions = consolidate_actions(all_actions)
    print("Final actions:", final_actions)
    return final_actions


def process_file(input_path, output_path, text_column="bill_text", min_text_length=100, skip_filled=True):
    """
    Process bill text to extract actions.
    
    Args:
        input_path: Path to input CSV with bill text
        output_path: Path to output CSV with actions
        text_column: Name of the column containing text to process
        min_text_length: Minimum text length to process
        skip_filled: If True, skip rows with non-empty actions (default: True)
    """
    df_input = pd.read_csv(input_path)
    
    # Validate that the text column exists
    if text_column not in df_input.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {list(df_input.columns)}")
    
    # Load existing output if file exists
    if os.path.exists(output_path):
        df_output = pd.read_csv(output_path)
        print(f"Loading existing output: {output_path}")
    else:
        df_output = df_input.copy()
        if "actions" not in df_output.columns:
            df_output["actions"] = None
        print(f"Creating new output file")
    
    # Identify rows to process
    if skip_filled:
        # Filter for rows with empty/None/NaN actions
        rows_to_process = df_output[
            (df_output["actions"].isna()) | 
            (df_output["actions"] == "") | 
            (df_output["actions"] == "[]")
        ].index.tolist()
        print(f"Skipping filled rows. Found {len(rows_to_process)} rows with empty actions. Processing only these...")
    else:
        rows_to_process = df_output.index.tolist()
        print(f"Processing all {len(rows_to_process)} rows")
    
    # Process only the rows that need it
    for idx in tqdm(rows_to_process, desc="Processing bills"):
        row = df_output.iloc[idx]
        text = clean_text(row.get(text_column, ""))
        
        if len(text) < min_text_length:
            df_output.at[idx, "actions"] = "[]"
            continue
        
        actions = extract_actions_from_text(text)
        df_output.at[idx, "actions"] = json.dumps(actions)
    
    df_output.to_csv(output_path, index=False)
    print(f"Saved extracted actions to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract high-level policy actions from PRS bill text using an LLM."
    )
    parser.add_argument(
        "--input",
        default="data/prs_datasets/prs_karnataka_bills_processed_partial.csv",
        help="Input CSV path containing bill text.",
    )
    parser.add_argument(
        "--output",
        default="data/prs_datasets/prs_karnataka_bills_with_actions.csv",
        help="Output CSV path for extracted actions.",
    )
    parser.add_argument(
        "--text-column",
        default="bill_text",
        help="Name of the column containing text to process (default: bill_text).",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=100,
        help="Minimum bill text length to process.",
    )
    parser.add_argument(
        "--fill-all",
        action="store_true",
        help="Process all rows (default: skip rows with non-empty actions).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_file(
        args.input,
        args.output,
        text_column=args.text_column,
        min_text_length=args.min_text_length,
        skip_filled=not args.fill_all
    )
