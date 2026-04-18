import os
import io
import time
import requests
import pandas as pd
import pdfplumber

# -------------------------
# CONFIG
# -------------------------
INPUT_CSV = "data/prs_datasets/prs_karnataka_acts.csv"
OUTPUT_CSV = "data/prs_datasets/prs_karnataka_acts_processed.csv"
PARTIAL_CSV = "data/prs_datasets/prs_karnataka_acts_processed_partial.csv"

SLEEP_SECONDS = 0.5
TEXT_LIMIT = 5000

# -------------------------
# HELPER: detect Kannada
# -------------------------
def is_kannada(text):
    # Kannada unicode range
    for ch in text:
        if '\u0C80' <= ch <= '\u0CFF':
            return True
    return False

# -------------------------
# PDF TEXT EXTRACTION
# -------------------------
def extract_pdf_text(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            full_text = ""
            english_started = False

            for page in pdf.pages:
                text = page.extract_text() or ""

                # Skip Kannada pages until English starts
                if not english_started:
                    if is_kannada(text):
                        continue
                    else:
                        english_started = True

                full_text += text + "\n"

                if len(full_text) > TEXT_LIMIT:
                    break

        return full_text.strip()

    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return ""

# -------------------------
# LOAD PARTIAL STATE
# -------------------------
def load_partial():
    if not os.path.exists(PARTIAL_CSV):
        return [], set()

    df = pd.read_csv(PARTIAL_CSV)
    records = df.to_dict("records")
    done = set(df["pdf_url"].tolist())

    print(f"Loaded partial: {len(records)} rows")
    return records, done

# -------------------------
# SAVE PARTIAL
# -------------------------
def save_partial(rows):
    if not rows:
        return

    pd.DataFrame(rows).to_csv(PARTIAL_CSV, index=False)
    print(f"Saved partial: {len(rows)} rows")

# -------------------------
# MAIN PROCESS
# -------------------------
def main():
    df = pd.read_csv(INPUT_CSV)

    rows, done_urls = load_partial()

    for _, row in df.iterrows():
        pdf_url = row.get("pdf_url")

        if pdf_url in done_urls:
            continue

        print(f"Processing: {row.get('title')}")

        text = extract_pdf_text(pdf_url)

        rows.append({
            "title": row.get("title"),
            "year": row.get("year"),
            "state": row.get("state"),
            "pdf_url": pdf_url,
            "act_text": text
        })

        done_urls.add(pdf_url)

        # Save every 20 rows
        if len(rows) % 20 == 0:
            save_partial(rows)

        time.sleep(SLEEP_SECONDS)

    final_df = pd.DataFrame(rows)
    final_df = final_df[final_df["act_text"].str.len() > 0]
    final_df = final_df.drop_duplicates(subset=["pdf_url"]).reset_index(drop=True)

    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nFinal dataset saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
