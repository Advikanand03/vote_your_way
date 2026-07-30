import pandas as pd
import requests
import xml.etree.ElementTree as ET
import time
from bs4 import BeautifulSoup
from groq import Groq
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import re
import os
import json
from dotenv import load_dotenv
import ast
from urllib.parse import quote_plus

load_dotenv()

# -------------------------
# CONFIG
# -------------------------
MODEL = "llama-3.1-8b-instant"
TOP_K_EVIDENCE = 10
PARTIAL_SAVE_INTERVAL = 10
PARTIAL_OUTPUT_PATH = "outputs/news_evidence_partial.csv"
FINAL_OUTPUT_PATH = "outputs/news_evidence_dataset.csv"

# Additional config
USE_LLM_FILTER = False

QUERY_CACHE = {}
RELEVANCE_CACHE = {}

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    EMBEDDINGS_AVAILABLE = True
except Exception:
    EMBEDDINGS_AVAILABLE = False

GROQ_API_KEY_THIRD = os.environ.get("GROQ_API_KEY_THIRD")
client = Groq(api_key=GROQ_API_KEY_THIRD)

def _gen_content(prompt, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(2)
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response
        except Exception as e:
            err = str(e)
            if "429" in err:
                wait_time = 15 + attempt * 15
                print(f"Groq rate limit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                raise
    raise Exception("Max retries exceeded for Groq call")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2") if EMBEDDINGS_AVAILABLE else None
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") if EMBEDDINGS_AVAILABLE else None

def _get(url, headers=None, timeout=8, retries=2):
    for _ in range(retries + 1):
        try:
            return requests.get(url, headers=headers, timeout=timeout)
        except Exception:
            time.sleep(1)
    raise

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/processed/final_enriched_dataset.csv")

# Remove old partial file if exists
if os.path.exists(PARTIAL_OUTPUT_PATH):
    os.remove(PARTIAL_OUTPUT_PATH)

# -------------------------
# QUERY
# -------------------------
def generate_queries(promise):
    if promise in QUERY_CACHE:
        return QUERY_CACHE[promise]

    base = promise.lower()
    queries = [
        f"{base} Karnataka implementation",
        f"{base} Karnataka scheme launched",
        f"{base} Karnataka budget allocation",
        f"{base} Karnataka latest news"
    ]

    QUERY_CACHE[promise] = queries
    return queries
# -------------------------
# LLM FILTER
# -------------------------
def filter_evidence(promise, evidence_items):
    if not evidence_items:
        return []

    cache_key = (promise, tuple(item["text"] for item in evidence_items))
    if cache_key in RELEVANCE_CACHE:
        return RELEVANCE_CACHE[cache_key]

    # batch in groups of 5
    batches = [evidence_items[i:i+5] for i in range(0, len(evidence_items), 5)]
    scored = []

    for batch in batches:
        texts = "\n".join([f"{i+1}. {item.get('text','')}" for i, item in enumerate(batch)])

        prompt = f"""
Rate relevance between the promise and each evidence.

Promise: {promise}

Evidences:
{texts}

Return ONLY a Python list of floats between 0 and 1 in the same order.
"""

        try:
            time.sleep(0.2)
            res = _gen_content(prompt)
            arr = []
            txt = res.choices[0].message.content
            txt = (txt or "").strip()
            try:
                match = re.search(r'\[.*\]', txt, re.DOTALL)
                if match:
                    arr = ast.literal_eval(match.group())
                else:
                    arr = []
            except Exception:
                arr = []

            for item, score in zip(batch, arr):
                try:
                    s = float(score)
                except Exception:
                    s = 0.0
                if s >= 0.3:
                    item["score"] = s
                    scored.append(item)
        except Exception:
            continue

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    final = scored[:TOP_K_EVIDENCE]
    RELEVANCE_CACHE[cache_key] = final
    return final

# -------------------------
# GOOGLE NEWS
# -------------------------
PREFERRED_SOURCES = ["The Hindu", "The Indian Express", "The Times of India", "Hindustan Times"]

def get_news(query):
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = _get(url, timeout=8)
        root = ET.fromstring(response.content)

        results = []

        for item in root.findall(".//item"):
            title = item.find("title").text if item.find("title") is not None else ""
            desc = item.find("description").text if item.find("description") is not None else ""
            desc = re.sub(r"<.*?>", "", desc)

            text = f"{title}. {desc}"

            source = ""
            if " - " in title:
                source = title.split(" - ")[-1]

            results.append({
                "text": text,
                "source": source
            })

        # prioritize preferred sources
        preferred = [r for r in results if r["source"] in PREFERRED_SOURCES]
        others = [r for r in results if r["source"] not in PREFERRED_SOURCES]

        return (preferred + others)[:10]

    except Exception as e:
        print("News fetch error:", e)
        return []




def _tokenize(text):
    return set(re.findall(r"[a-zA-Z]{3,}", str(text).lower()))


def retrieve_relevant_evidence(promise, evidence_items, top_k=10):
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

def rerank_evidence(promise, evidence_items, top_k=10):
    """
    Reranks a list of evidence items using a cross-encoder model.
    """
    if not evidence_items:
        return []
        
    if EMBEDDINGS_AVAILABLE and cross_encoder is not None:
        try:
            pairs = [[promise, item["text"]] for item in evidence_items]
            scores = cross_encoder.predict(pairs)
            
            for i, score in enumerate(scores):
                evidence_items[i]["rerank_score"] = float(score)
                
            evidence_items.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return evidence_items[:top_k]
        except Exception as e:
            print("Cross-encoder rerank fallback:", e)
            
    # Fallback to returning original top K
    return evidence_items[:top_k]

# -------------------------
# PROCESS FUNCTION (PARALLEL)
# -------------------------
def process_row(row):

    promise = row["promise_text"]
    print(f"Processing {row['promise_id']}")

    queries = generate_queries(promise)

    news = []
    for q in queries:
        news.extend(get_news(q))

    news = [
        item for item in news
        if any(k in item["text"].lower() for k in ["karnataka", "bengaluru"])
    ]

    # deduplicate
    news = list({item["text"]: item for item in news}.values())

    # 1. full_tuple
    full_tuple = news

    # 2. around 20 similar articles using miniLM
    retrieved_20 = retrieve_relevant_evidence(promise, full_tuple, top_k=20)

    # 3. Reranking of the articles using cross-encoder (BERT)
    reranked = rerank_evidence(promise, retrieved_20, top_k=10)

    # 4. Perform a relevancy check mechanism by assigning scores
    final_relevant = filter_evidence(promise, reranked) if USE_LLM_FILTER else reranked

    # Split back into respective columns
    news_texts = [item["text"] for item in final_relevant]
    news_sources = [item.get("source", "") for item in final_relevant]

    return {
        "promise_id": row["promise_id"],
        "promise_text": promise,
        "category": row.get("category"),
        "evidence_googlenews": news_texts,
        "news_sources": news_sources,
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
# Always start fresh
results = []
processed_ids = set()
pending_rows = list(df.iterrows())

print(f"Starting fresh run. Total rows: {len(pending_rows)}")

completed_since_checkpoint = 0

with ThreadPoolExecutor(max_workers=1) as executor:
    future_to_row = {executor.submit(process_row, row): row for _, row in pending_rows}

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
# SAVE JSON
# -------------------------
json_output_path = "outputs/news_evidence_dataset.json"

final_results = [
    r for r in results if r.get("evidence_googlenews")
]

with open(json_output_path, "w") as f:
    json.dump(final_results, f, indent=2)

print(f"Final JSON saved: {json_output_path}")
print("News evidence collection complete!")