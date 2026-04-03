import os
import re
import pandas as pd
from pdf2image import convert_from_path
from pytesseract import pytesseract


# -------------------------------
# 1. Find PDF
# -------------------------------
def find_pdf_file(folder_path):
    for file in os.listdir(folder_path):
        if file.lower().endswith('.pdf'):
            return os.path.join(folder_path, file)
    return None


# -------------------------------
# 2. OCR Extraction
# -------------------------------
def extract_text_from_pdf(pdf_path):
    print(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path)

    extracted_pages = []
    for idx, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        extracted_pages.append(text)
        print(f"Page {idx + 1}: Extracted {len(text)} characters")

    return extracted_pages


# -------------------------------
# 3. Heading Detection
# -------------------------------
def is_heading(line):
    if not line:
        return False

    line = line.strip()

    if len(line) < 3 or len(line) > 80:
        return False

    if line[0].islower():
        return False

    words = line.split()

    # ALL CAPS
    if line.isupper() and len(words) <= 8:
        return True

    # Title Case
    if all(w.istitle() for w in words if w.isalpha()) and len(words) <= 8:
        return True

    # Ends with :
    if line.endswith(':') and len(words) <= 8:
        return True

    return False


# -------------------------------
# 4. Bullet Detection (STRICT)
# -------------------------------
def is_bullet_line(line):
    if not line:
        return False

    s = line.strip()

    bullets = ['•', '-', '*', '◦', '▪', '→', '✓', '✗', '¢', '·']

    if any(s.startswith(b) for b in bullets):
        return True

    if re.match(r'^(\d+|[a-zA-Z])[\.|\)]\s+', s):
        return True

    return False


# -------------------------------
# 5. Clean Text
# -------------------------------
def clean_text(text):
    if text is None:
        return ""

    text = re.sub(r'\s+', ' ', text).strip()

    fixes = {
        'infrastractural': 'infrastructural',
        'Rajastan': 'Rajasthan',
        'Himac': 'Himachal',
    }

    for wrong, right in fixes.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)

    return text


# -------------------------------
# 6. Parse Lines (CORE LOGIC)
# -------------------------------
def parse_lines(text_pages):
    parsed = []

    for page_text in text_pages:
        lines = page_text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Detect heading
            if is_heading(line):
                current_category = clean_text(line)
                i += 1

                current_promise = None

                while i < len(lines):
                    next_line = lines[i].strip()

                    if not next_line:
                        i += 1
                        continue

                    if is_heading(next_line):
                        break

                    # New bullet → start new promise
                    if is_bullet_line(next_line):
                        if current_promise:
                            parsed.append({
                                "category": current_category,
                                "text": current_promise
                            })

                        txt = re.sub(r'^[•\-*\u25E6\u25AA\u2192\u2713\u2717\u00A2\u00B7]+\s*', '', next_line)
                        txt = re.sub(r'^(\d+|[a-zA-Z])[\.|\)]\s*', '', txt)

                        current_promise = txt.strip()

                    else:
                        # CONTINUATION logic (FIXED)
                        if current_promise:
                            if next_line[0].islower() or len(next_line.split()) < 6:
                                current_promise += ' ' + next_line
                            else:
                                # treat as new promise (fallback)
                                parsed.append({
                                    "category": current_category,
                                    "text": current_promise
                                })
                                current_promise = next_line
                        else:
                            current_promise = next_line

                    i += 1

                if current_promise:
                    parsed.append({
                        "category": current_category,
                        "text": current_promise
                    })

            else:
                i += 1

    return parsed


# -------------------------------
# 7. Build Dataset
# -------------------------------
def build_dataset(parsed):
    cleaned = []

    for item in parsed:
        text = clean_text(item["text"])

        if len(text) >= 40:  # remove weak fragments
            cleaned.append({
                "category": item["category"],
                "promise_text": text
            })

    # Remove duplicates
    seen = set()
    final = []

    for row in cleaned:
        key = (row["category"].lower(), row["promise_text"].lower())
        if key not in seen:
            seen.add(key)
            final.append(row)

    # Add IDs
    for i, row in enumerate(final, 1):
        row["promise_id"] = f"P{i}"

    df = pd.DataFrame(final)
    df = df[["promise_id", "category", "promise_text"]]

    return df


# -------------------------------
# 8. Final Cleaning (CRITICAL)
# -------------------------------
def post_clean(df):
    df = df[df["promise_text"].str.len() > 40]
    df = df[~df["promise_text"].str.match(r"^[a-z]\s")]

    df["promise_text"] = df["promise_text"].str.lstrip(":;,- ")
    df["promise_text"] = df["promise_text"].str.replace(r"\s+", " ", regex=True)
    df["promise_text"] = df["promise_text"].str.strip()

    return df


# -------------------------------
# 9. Save
# -------------------------------
def save_dataset(df):
    output = "data/processed/karnataka_inc_promises_cleaned.csv"
    df.to_csv(output, index=False)
    print(f"Saved dataset → {output}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    pdf_file = find_pdf_file(folder)

    if not pdf_file:
        print("No PDF found")
        return

    print(f"Processing: {pdf_file}")

    pages = extract_text_from_pdf(pdf_file)

    parsed = parse_lines(pages)
    print(f"Parsed entries: {len(parsed)}")

    df = build_dataset(parsed)
    df = post_clean(df)

    print(f"Final dataset size: {len(df)}")
    print(df.head())

    save_dataset(df)


if __name__ == "__main__":
    main()