import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder


# =========================
# LOAD DATA
# =========================

# Promises
promises_df = pd.read_csv("data/processed/final_enriched_dataset.csv")

# PRS Bills actions
bills_df = pd.read_csv("data/prs_datasets/prs_karnataka_bills_with_actions.csv")
bills_df["actions"] = bills_df["actions"].apply(ast.literal_eval)

# (Optional) PRS Acts actions
try:
    acts_df = pd.read_csv("data/prs_datasets/prs_karnataka_acts_with_actions.csv")
    acts_df["actions"] = acts_df["actions"].apply(ast.literal_eval)
except:
    acts_df = pd.DataFrame()

# =========================
# BUILD EVIDENCE POOLS
# =========================

acts_evidence = []
bills_evidence = []

# Acts
if not acts_df.empty:
    for actions in acts_df["actions"]:
        acts_evidence.extend(actions)

# Bills
for actions in bills_df["actions"]:
    bills_evidence.extend(actions)

# Remove duplicates while preserving order
acts_evidence = list(dict.fromkeys(acts_evidence))
bills_evidence = list(dict.fromkeys(bills_evidence))

print(f"Total acts evidence items: {len(acts_evidence)}")
print(f"Total bills evidence items: {len(bills_evidence)}")

# =========================
# LOAD MODEL
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

# BERT Cross-Encoder for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Encode evidence pools separately
acts_embeddings = model.encode(acts_evidence, convert_to_tensor=True) if acts_evidence else None
bills_embeddings = model.encode(bills_evidence, convert_to_tensor=True) if bills_evidence else None


# =========================
# RETRIEVAL FUNCTIONS
# =========================

def clean_evidence(evidence_list):
    cleaned = []
    for e in evidence_list:
        if not isinstance(e, str):
            continue
        e = e.strip()

        if len(e) < 20:
            continue

        if any(word in e.lower() for word in ["define", "provision", "act shall", "section"]):
            continue

        cleaned.append(e)

    return list(dict.fromkeys(cleaned))


def retrieve_evidence(query, evidence_list, embeddings, top_k=20):
    if not evidence_list or embeddings is None:
        return []
        
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]

    top_results = scores.topk(k=top_k)
    results = [evidence_list[idx] for idx in top_results.indices]

    # Clean results
    results = clean_evidence(results)

    return results


def rerank(query, candidates, top_k=5):
    if not candidates:
        return []

    pairs = [(query, c) for c in candidates]

    scores = reranker.predict(pairs, batch_size=16)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [r[0] for r in ranked[:top_k]]

# =========================
# BUILD DATASET
# =========================

output_rows = []

for i, row in promises_df.iterrows():
    promise = row["promise_text"]

    try:
        # Retrieve from acts
        acts_candidates = retrieve_evidence(promise, acts_evidence, acts_embeddings, top_k=20)[:15]
        if not acts_candidates:
            top_acts_evidence = []
        else:
            top_acts_evidence = rerank(promise, acts_candidates, top_k=5)

        # Retrieve from bills
        bills_candidates = retrieve_evidence(promise, bills_evidence, bills_embeddings, top_k=20)[:15]
        if not bills_candidates:
            top_bills_evidence = []
        else:
            top_bills_evidence = rerank(promise, bills_candidates, top_k=5)

        output_rows.append({
            "promise_id": row.get("promise_id"),
            "promise_text": promise,
            "category": row.get("category"),
            # "sector": row.get("sector"),
            # "sub_sector": row.get("sub_sector"),
            # "quantifiable": row.get("quantifiable"),
            # "timeline_mentioned": row.get("timeline_mentioned"),
            # "target_year": row.get("target_year"),
            # "commitment_type": row.get("commitment_type"),
            "prs_evidences_acts": top_acts_evidence,
            "prs_evidences_bills": top_bills_evidence
        })

        if i % 10 == 0:
            print(f"Processed {i} promises")

    except Exception as e:
        print(f"Error at row {i}: {e}")


# =========================
# SAVE OUTPUT
# =========================

output_df = pd.DataFrame(output_rows)

output_df.to_csv("outputs/promise_evidence_dataset.csv", index=False)

print("Saved → outputs/promise_evidence_dataset.csv")
print(f"Columns: {list(output_df.columns)}")
print(f"Total rows: {len(output_df)}")