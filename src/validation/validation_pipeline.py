import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time
from bs4 import BeautifulSoup
import pdfplumber
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import tempfile
import os
import math
import re
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# CONFIG
# -------------------------
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
MODEL = "llama-3.1-8b-instant"
MAX_PRS_PDF_READS_PER_PROMISE = 3
TOP_K_EVIDENCE = 6
UNCERTAIN_THRESHOLD = 0.55
PARTIAL_SAVE_INTERVAL = 10
PARTIAL_OUTPUT_PATH = "outputs/final_results_partial.csv"
FINAL_OUTPUT_PATH = "outputs/final_results.csv"

# Additional config
USE_CROSS_ENCODER = False
USE_LLM_FILTER = True

try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDINGS_AVAILABLE = True
except Exception:
    EMBEDDINGS_AVAILABLE = False

client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2") if EMBEDDINGS_AVAILABLE else None

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/processed/final_enriched_dataset.csv")

# -------------------------
# QUERY
# -------------------------
def generate_queries(promise):
    prompt = f"""
    Generate 4 different search queries for this promise covering:
    1. launch
    2. funding
    3. implementation
    4. current status

    Promise: {promise}

    Return as a list.
    """
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        text = res.choices[0].message.content
        return [q.strip("- ").strip() for q in text.split("\n") if q.strip()]
    except:
        return [promise + " Karnataka scheme"]
# -------------------------
# LLM FILTER
# -------------------------
def filter_evidence(promise, evidence_items):
    filtered = []
    for item in evidence_items:
        prompt = f"""
        Promise: {promise}
        Evidence: {item["text"]}

        Does this help determine progress of the promise? Yes or No.
        """
        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            if "Yes" in res.choices[0].message.content:
                filtered.append(item)
        except:
            filtered.append(item)
    return filtered

# -------------------------
# GOOGLE NEWS
# -------------------------
def get_news(query):
    url = f"https://news.google.com/rss/search?q={query}"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)

        return [item.find("title").text for item in root.findall(".//item")[:5]]
    except:
        return []

# -------------------------
# GOVT RESULTS
# -------------------------
def get_govt(query):
    url = f"https://www.google.com/search?q={query} Karnataka government"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        return [g.find("h3").text for g in soup.select("div.g")[:5] if g.find("h3")]
    except:
        return []

# -------------------------
# PRS LINKS
# -------------------------
def get_prs_links():
    urls = [
        "https://prsindia.org/bills/states?state=Karnataka&year=2026",
        "https://prsindia.org/bills/states?state=Karnataka&year=2025",
        "https://prsindia.org/bills/states?state=Karnataka&year=2024",
        "https://prsindia.org/bills/states?state=Karnataka&year=2023",
    ]

    links = []
    seen = set()

    for url in urls:
        try:
            soup = BeautifulSoup(requests.get(url, timeout=5).text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if ".pdf" in href.lower():
                    full = href if href.startswith("http") else "https://prsindia.org" + href
                    if full not in seen:
                        seen.add(full)
                        title = a.get_text(" ", strip=True) or href
                        links.append({"title": title, "url": full})
        except:
            pass

    return links

# -------------------------
# PDF TEXT
# -------------------------
def extract_pdf_text(url):
    temp_path = None
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(r.content)
            temp_path = f.name

        text = ""

        try:
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    try:
                        text += page.extract_text() or ""
                    except Exception:
                        continue   # skip bad pages

        except Exception as e:
            # Some PRS PDFs can fail parsing (for example FontBBox issues).
            print("PDF parse error:", e)
            return ""

        return text[:1500]

    except Exception as e:
        print("PDF fetch error:", e)
        return ""
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

# -------------------------
# LLM STANCE
# -------------------------
def get_stance(promise, text):

    prompt = f"""
    Promise: {promise}

    Evidence: {text}

    Classify as:
    Completed / In Progress / Not Done

    Return only one.
    """

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except:
        return "In Progress"


def _tokenize(text):
    return set(re.findall(r"[a-zA-Z]{3,}", str(text).lower()))


def retrieve_relevant_evidence(promise, evidence_items, top_k=TOP_K_EVIDENCE):
    """
    Minimal semantic retrieval (RAG-style):
    - Prefer embeddings cosine similarity when available
    - Fallback to lexical overlap score when embeddings are unavailable
    """
    if not evidence_items:
        return []

    if EMBEDDINGS_AVAILABLE and embedding_model is not None:
        try:
            query_emb = embedding_model.encode(promise, convert_to_tensor=True)
            text_embs = embedding_model.encode(
                [item["text"] for item in evidence_items],
                convert_to_tensor=True
            )
            sims = util.cos_sim(query_emb, text_embs)[0]
            ranked = sorted(
                zip(evidence_items, sims),
                key=lambda x: float(x[1]),
                reverse=True
            )
            return [item for item, _ in ranked[:top_k]]
        except Exception as e:
            print("Embedding retrieval fallback:", e)

    # Fallback retrieval: token overlap
    promise_tokens = _tokenize(promise)
    scored = []
    for item in evidence_items:
        item_tokens = _tokenize(item["text"])
        overlap = len(promise_tokens.intersection(item_tokens))
        scored.append((item, overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored[:top_k]]

# -------------------------
# PRS CHECK
# -------------------------
def prs_check(promise, prs_links):
    tokens = [w for w in promise.lower().split() if len(w) > 3]
    pdf_reads = 0

    for bill in prs_links:
        link = bill["url"]
        title = bill["title"].lower()

        # Keep title usage as first-level filter; fallback to URL match.
        if not any(word in title or word in link.lower() for word in tokens):
            continue

        if pdf_reads >= MAX_PRS_PDF_READS_PER_PROMISE:
            break

        text = extract_pdf_text(link)
        pdf_reads += 1
        if not text:
            continue

        prompt = f"""
        Promise: {promise}
        Bill content: {text}

        Does this bill implement the promise? Yes or No.
        """

        try:
            res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )

            if "Yes" in res.choices[0].message.content:
                return True
        except:
            pass

    return False

# -------------------------
# SCORING
# -------------------------
def compute_probabilistic_verdict(stances, prs_match):
    """
    Minimal probabilistic verdict model:
    - Build class logits from source-weighted stance evidence
    - Convert logits to probabilities via softmax
    - Keep PRS as strong positive evidence (not a hard override)
    """
    if not stances and not prs_match:
        probs = {"Completed": 0.0, "In Progress": 1.0, "Not Done": 0.0}
        return "In Progress", 0.0, probs

    source_weights = {"news": 1.0, "government": 1.5, "prs": 2.0}
    logits = {"Completed": 0.0, "In Progress": 0.0, "Not Done": 0.0}

    for stance, source in stances:
        w = source_weights.get(source, 1.0)
        if stance == "Completed":
            logits["Completed"] += 1.0 * w
            logits["In Progress"] += 0.2 * w
        elif stance == "Not Done":
            logits["Not Done"] += 1.0 * w
            logits["In Progress"] += 0.2 * w
        else:
            logits["In Progress"] += 0.8 * w
            logits["Completed"] += 0.1 * w
            logits["Not Done"] += 0.1 * w

    if prs_match:
        logits["Completed"] += 2.5

    max_logit = max(logits.values())
    exp_vals = {k: math.exp(v - max_logit) for k, v in logits.items()}
    denom = sum(exp_vals.values()) or 1.0
    probs = {k: exp_vals[k] / denom for k in logits}

    verdict = max(probs, key=probs.get)
    confidence = round(float(probs[verdict]), 2)

    # Uncertainty bucket for low-confidence outcomes.
    if confidence < UNCERTAIN_THRESHOLD:
        return "Uncertain", confidence, probs

    return verdict, confidence, probs

# -------------------------
# PROCESS FUNCTION (PARALLEL)
# -------------------------
def process_row(row):

    promise = row["promise_text"]
    print(f"Processing {row['promise_id']}")

    queries = generate_queries(promise)

    news, govt = [], []
    for q in queries:
        news.extend(get_news(q))
        govt.extend(get_govt(q))

    evidence_pool = [{"text": n, "source": "news"} for n in news] + [
        {"text": g, "source": "government"} for g in govt
    ]
    selected_evidence = retrieve_relevant_evidence(promise, evidence_pool, top_k=TOP_K_EVIDENCE)
    if USE_LLM_FILTER:
        selected_evidence = filter_evidence(promise, selected_evidence)

    stances = []
    for item in selected_evidence:
        stances.append((get_stance(promise, item["text"]), item["source"]))

    prs_match = prs_check(promise, prs_links)

    verdict, confidence, probs = compute_probabilistic_verdict(stances, prs_match)

    return {
        "promise_id": row["promise_id"],
        "promise_text": promise,
        "verdict": verdict,
        "confidence": confidence,
        "p_completed": round(probs["Completed"], 3),
        "p_in_progress": round(probs["In Progress"], 3),
        "p_not_done": round(probs["Not Done"], 3),
    }


def save_partial_results(results):
    if not results:
        return
    pd.DataFrame(results).to_csv(PARTIAL_OUTPUT_PATH, index=False)
    print(f"Checkpoint saved: {len(results)} rows -> {PARTIAL_OUTPUT_PATH}")


def load_existing_partial():
    if not os.path.exists(PARTIAL_OUTPUT_PATH):
        return []
    try:
        existing = pd.read_csv(PARTIAL_OUTPUT_PATH)
        records = existing.to_dict("records")
        print(f"Loaded existing partial: {len(records)} rows from {PARTIAL_OUTPUT_PATH}")
        return records
    except Exception as e:
        print("Could not load existing partial file:", e)
        return []

# -------------------------
# MAIN (PARALLEL EXECUTION)
# -------------------------
prs_links = get_prs_links()

results = load_existing_partial()
processed_ids = {row["promise_id"] for row in results if "promise_id" in row}
pending_rows = [row for _, row in df.iterrows() if row["promise_id"] not in processed_ids]

print(f"Already processed: {len(processed_ids)} | Pending: {len(pending_rows)}")

completed_since_checkpoint = 0

with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_row = {executor.submit(process_row, row): row for row in pending_rows}

    for future in as_completed(future_to_row):
        row = future_to_row[future]
        promise_id = row["promise_id"]
        try:
            result = future.result()
            results.append(result)
            processed_ids.add(promise_id)
            completed_since_checkpoint += 1
        except Exception as e:
            # Continue processing other rows while preserving current progress.
            print(f"Error processing {promise_id}: {e}")
            continue

        if completed_since_checkpoint >= PARTIAL_SAVE_INTERVAL:
            save_partial_results(results)
            completed_since_checkpoint = 0

# Final checkpoint before writing final file
save_partial_results(results)

# -------------------------
# SAVE
# -------------------------
pd.DataFrame(results).to_csv(FINAL_OUTPUT_PATH, index=False)
print(f"Final results saved: {FINAL_OUTPUT_PATH}")

print("Final validation complete!")