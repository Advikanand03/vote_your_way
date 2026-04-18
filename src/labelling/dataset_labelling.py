# ============================================================
# DATASET GENERATION PIPELINE (FOR ML TRAINING)
# ============================================================

# GOAL:
# Generate labeled dataset for training models for:
# 1. Evidence Filtering → Useful / Not Useful
# 2. Evidence Reranking → Relevance Score

# INPUT:
# CSV: data/processed/final_enriched_dataset.csv
# Columns include: promise_id, promise_text

# ------------------------------------------------------------
# STEP 1: QUERY GENERATION
# ------------------------------------------------------------
# For each promise:
# - Use LLM (Groq) to generate 4 queries:
#   - launch
#   - funding
#   - implementation
#   - current status

# ------------------------------------------------------------
# STEP 2: DATA COLLECTION
# ------------------------------------------------------------
# For each query:
# - Get evidence from:
#   - Google News RSS
#   - Google search (government results)
# - Collect ~10–20 evidence texts per promise
# - Store: evidence_text + source (news/government)

# ------------------------------------------------------------
# STEP 3: DATA STRUCTURE
# ------------------------------------------------------------
# For each evidence:
# {
#   "promise_id": ...,
#   "promise_text": ...,
#   "evidence_text": ...,
#   "source": ...
# }

# ------------------------------------------------------------
# STEP 4: LLM LABELING
# ------------------------------------------------------------
# For each (promise, evidence):

# A. Filtering Label:
# Prompt:
# "Does this evidence indicate implementation, progress, or completion
#  of the promise? Ignore opinions, criticism, or general discussion.
#  Answer Yes or No."
# Output:
# useful_label = 1 (Yes) or 0 (No)

# B. Relevance Label:
# Prompt:
# "How relevant is this evidence to the promise? Give a score between 0 and 1."
# Output:
# relevance_score (float)

# ------------------------------------------------------------
# STEP 5: SAVE DATASET
# ------------------------------------------------------------
# Save final dataset to:
# data/training/evidence_dataset.csv

# Columns:
# - promise_id
# - promise_text
# - evidence_text
# - source
# - useful_label
# - relevance_score

# ------------------------------------------------------------
# STEP 6: CLEANING
# ------------------------------------------------------------
# - Remove duplicate evidence_text
# - Remove empty or very short entries

# ------------------------------------------------------------
# STEP 7: PARTIAL SAVING
# ------------------------------------------------------------
# - Save every 50 promises
# - File: data/training/evidence_dataset_partial.csv

# ------------------------------------------------------------
# STEP 8: PARALLEL PROCESSING
# ------------------------------------------------------------
# - Use ThreadPoolExecutor
# - max_workers = 5

# ------------------------------------------------------------
# STEP 9: ERROR HANDLING
# ------------------------------------------------------------
# - Skip failed API calls
# - Continue execution without stopping pipeline

# ------------------------------------------------------------
# FUNCTIONS TO IMPLEMENT
# ------------------------------------------------------------
# - generate_queries()
# - get_news()
# - get_govt()
# - label_with_llm()
# - process_row()

# ------------------------------------------------------------
# NOTE:
# This pipeline ONLY generates dataset.
# Training of ML models will be done separately.
# ============================================================