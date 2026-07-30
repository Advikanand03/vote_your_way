# Vote Your Way

Vote Your Way is a political promise tracking project built around one central idea:

take a party manifesto, extract each promise, structure it into a machine-readable dataset, collect supporting evidence from legislative and public sources, and estimate whether each promise is completed, in progress, not done, or uncertain.

This repository is focused on the Indian National Congress (INC) Karnataka Assembly Election 2023 manifesto and builds a full research pipeline around that document.

## What The Project Is Trying To Do

Political manifestos are usually long PDFs written for humans, not for analysis. This project turns them into a structured accountability workflow:

1. Read the manifesto PDF with OCR.
2. Detect categories and bullet-point promises.
3. Clean and optionally atomize promises into one-action-per-row statements.
4. Enrich each promise with metadata such as sector, quantifiability, target year, and commitment type.
5. Gather evidence from PRS legislative records, Google News RSS, and government-search style results.
6. Compare evidence to promises with embeddings, reranking, and LLM judgments.
7. Produce a final verdict dataset.
8. Show the result in a simple Streamlit tracker.

## What Has Already Been Done In This Repo

This repository already contains a substantial end-to-end prototype and several generated datasets.

- The Karnataka manifesto PDF is already present in `data/raw/`.
- A cleaned manifesto promise dataset has already been created with 268 rows.
- An atomic promise dataset has already been created with 268 rows.
- An enriched promise dataset has already been created with 268 rows.
- PRS Karnataka bills have been scraped into a 192-row dataset.
- PRS Karnataka acts have been scraped into a 44-row dataset.
- Bill PDF text has been processed into a 177-row text dataset.
- Act PDF text has been processed into a 44-row text dataset.
- PRS action summaries have already been extracted for bills and acts.
- A news evidence dataset and PRS evidence dataset already exist in `outputs/`.
- A final promise verdict file already exists in `outputs/final_results.csv`.
- A Streamlit app exists for browsing the final results.

In short: the project is no longer just an idea. The repo contains the first working version of the full research pipeline plus intermediate artifacts.

## End-To-End Pipeline

### Stage 1: Manifesto ingestion and OCR

Input:

- `data/raw/Assembly Election Karnataka Manifesto - 2023-2.pdf`

Main script:

- `src/extraction/manifesto_pipeline.py`

What happens:

- The script finds a PDF.
- It converts each page into images with `pdf2image`.
- It runs OCR with `pytesseract`.
- It detects section headings.
- It detects bullet lines and continuation lines.
- It converts raw text into category + promise pairs.
- It removes weak fragments and duplicates.
- It assigns promise IDs like `P1`, `P2`, and so on.

Output:

- `data/processed/karnataka_inc_promises_cleaned.csv`

Typical schema:

- `promise_id`
- `category`
- `promise_text`

### Stage 2: Atomic promise generation

Input:

- `data/processed/karnataka_inc_promises_cleaned.csv`

Main script:

- `src/enrichment/atomic_promise_pipeline.py`

What happens:

- Each extracted promise is sent to Groq.
- The model splits multi-part promises into cleaner, action-oriented units.
- The script expects JSON output and validates it.
- If the model fails, it falls back to the original promise text.
- Partial progress is checkpointed.
- Final rows get IDs like `A1`, `A2`, and so on.

Outputs:

- `data/processed/karnataka_inc_promises_atomic.csv`
- `data/processed/karnataka_inc_promises_atomic_partial.csv`

Typical schema:

- `atomic_promise_id`
- `source_promise_id`
- `category`
- `clean_promise`

### Stage 3: Promise enrichment

Input:

- `data/processed/karnataka_inc_promises_atomic.csv`

Main script:

- `src/enrichment/enrichment_pipeline.py`

What happens:

- Each atomic promise is classified with an LLM.
- The script fills metadata fields such as sector, sub-sector, target year, and commitment type.
- The enriched dataset becomes the main structured promise table used by later stages.

Output:

- `data/processed/final_enriched_dataset.csv`

Typical schema:

- `promise_id`
- `source_promise_id`
- `category`
- `promise_text`
- `sector`
- `sub_sector`
- `quantifiable`
- `target_value`
- `timeline_mentioned`
- `target_year`
- `commitment_type`

### Stage 4: PRS legislative scraping

Inputs:

- PRS India bill and act listing pages for Karnataka, 2023 to 2026

Main script:

- `src/extraction/prs_data_scraper.py`

What happens:

- The scraper requests PRS state bills and acts pages.
- It extracts titles and PDF links.
- It deduplicates entries across paginated pages.
- It saves separate bill and act datasets.

Outputs:

- `data/prs_datasets/prs_karnataka_bills.csv`
- `data/prs_datasets/prs_karnataka_acts.csv`

### Stage 5: PRS PDF text extraction

Inputs:

- `data/prs_datasets/prs_karnataka_bills.csv`
- `data/prs_datasets/prs_karnataka_acts.csv`

Main scripts:

- `src/enrichment/prs_pdf_processor.py`
- `src/enrichment/prs_pdf_processor_acts.py`

What happens:

- Each PRS PDF is downloaded.
- Text is extracted with `pdfplumber`.
- Kannada-heavy front matter is skipped until English text starts.
- Text is truncated to a manageable size.
- Partial progress is written so long runs can be resumed.

Outputs:

- `data/prs_datasets/prs_karnataka_bills_processed.csv`
- `data/prs_datasets/prs_karnataka_bills_processed_partial.csv`
- `data/prs_datasets/prs_karnataka_acts_processed.csv`
- `data/prs_datasets/prs_karnataka_acts_processed_partial.csv`

### Stage 6: PRS action extraction

Inputs:

- `data/prs_datasets/prs_karnataka_bills_processed.csv`
- `data/prs_datasets/prs_karnataka_acts_processed.csv`

Main scripts:

- `src/enrichment/prs_action_extraction.py`
- `src/enrichment/prs_acts_action_extraction.py`

What happens:

- The full bill or act text is cleaned.
- The text is sent to Groq with a prompt asking for a small set of high-level policy actions.
- The script parses a JSON list of actions.
- Duplicates and weak actions are filtered.
- When too many actions are produced, similar actions are consolidated.

Outputs:

- `data/prs_datasets/prs_karnataka_bills_with_actions.csv`
- `data/prs_datasets/prs_karnataka_acts_with_actions.csv`

Why this stage matters:

- Raw legislative PDFs are too long and noisy for direct promise matching.
- These action lists become a compact evidence layer for semantic retrieval.

### Stage 7: PRS evidence retrieval for promises

Inputs:

- `data/processed/final_enriched_dataset.csv`
- `data/prs_datasets/prs_karnataka_bills_with_actions.csv`
- `data/prs_datasets/prs_karnataka_acts_with_actions.csv`

Main script:

- `src/validation/evidence_builder.py`

What happens:

- The script flattens all extracted PRS actions into evidence pools.
- It embeds evidence with `sentence-transformers`.
- It retrieves top PRS evidence candidates for each promise.
- It reranks those candidates with a cross-encoder.

Output:

- `outputs/promise_evidence_dataset.csv`

Typical schema:

- `promise_id`
- `promise_text`
- `category`
- `prs_evidences_acts`
- `prs_evidences_bills`

### Stage 8: News evidence retrieval

Input:

- `data/processed/final_enriched_dataset.csv`

Main script:

- `src/validation/newsevidence.py`

What happens:

- For each promise, the script generates or derives search-style queries.
- It pulls article titles and snippets from Google News RSS.
- It favors a few preferred Indian news sources.
- It optionally reranks evidence using embeddings and a cross-encoder.
- It writes checkpoint files during long runs.

Current artifact in the repo:

- `outputs/news_evidence_dataset.csv`
- `outputs/news_evidence_partial.csv`

Important note:

- The current version of `src/validation/newsevidence.py` ends by writing `outputs/news_evidence_dataset.json`, while the repository currently contains a CSV artifact. That means the shipped output was generated by an earlier or modified run.

### Stage 9: Combined validation and verdict generation

Input:

- `data/processed/final_enriched_dataset.csv`

Main script:

- `src/validation/validation_pipeline.py`

What happens:

- The script generates promise-specific evidence queries.
- It collects public evidence from news and government-search style results.
- It retrieves the most relevant evidence snippets.
- It asks an LLM to assign evidence-level stance labels such as `Completed`, `In Progress`, or `Not Done`.
- It separately checks PRS PDFs for likely legislative support.
- It combines all stance signals into a probabilistic verdict.
- It writes periodic checkpoints.

Outputs:

- `outputs/final_results_partial.csv`
- `outputs/final_results.csv`

Typical schema:

- `promise_id`
- `promise_text`
- `verdict`
- `confidence`
- `p_completed`
- `p_in_progress`
- `p_not_done`

### Stage 10: Tracker UI

Inputs:

- `outputs/final_results.csv`
- `data/processed/final_enriched_dataset.csv`

Main script:

- `src/tracker_app.py`

What happens:

- Streamlit loads final verdicts.
- It merges verdicts with enriched metadata.
- The user can filter by verdict, sector, category, search term, and confidence.
- The dashboard shows simple metrics and a table of promises.

Goal:

- Give citizens and researchers a lightweight interface over the pipeline outputs.

## Data Lineage In One View

```text
Manifesto PDF
  -> src/extraction/manifesto_pipeline.py
  -> data/processed/karnataka_inc_promises_cleaned.csv
  -> src/enrichment/atomic_promise_pipeline.py
  -> data/processed/karnataka_inc_promises_atomic.csv
  -> src/enrichment/enrichment_pipeline.py
  -> data/processed/final_enriched_dataset.csv

PRS state listings
  -> src/extraction/prs_data_scraper.py
  -> data/prs_datasets/prs_karnataka_bills.csv
  -> data/prs_datasets/prs_karnataka_acts.csv
  -> src/enrichment/prs_pdf_processor.py
  -> src/enrichment/prs_pdf_processor_acts.py
  -> *_processed.csv
  -> src/enrichment/prs_action_extraction.py
  -> src/enrichment/prs_acts_action_extraction.py
  -> *_with_actions.csv
  -> src/validation/evidence_builder.py
  -> outputs/promise_evidence_dataset.csv

Enriched promises
  -> src/validation/newsevidence.py
  -> news evidence artifacts
  -> src/validation/validation_pipeline.py
  -> outputs/final_results.csv
  -> src/tracker_app.py
  -> Streamlit dashboard
```

## How To Run The Pipeline

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add environment variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_key
GROQ_API_KEY_DIFF=your_key
GROQ_API_KEY_THIRD=your_key
```

The code currently uses three variable names for Groq access because different scripts were developed separately.

### 3. Run the pipeline stages

```bash
python3 src/extraction/manifesto_pipeline.py
python3 src/enrichment/atomic_promise_pipeline.py
python3 src/enrichment/enrichment_pipeline.py
python3 src/extraction/prs_data_scraper.py
python3 src/enrichment/prs_pdf_processor.py
python3 src/enrichment/prs_pdf_processor_acts.py
python3 src/enrichment/prs_action_extraction.py --input data/prs_datasets/prs_karnataka_bills_processed.csv --output data/prs_datasets/prs_karnataka_bills_with_actions.csv
python3 src/enrichment/prs_acts_action_extraction.py --input data/prs_datasets/prs_karnataka_acts_processed.csv --output data/prs_datasets/prs_karnataka_acts_with_actions.csv
python3 src/validation/evidence_builder.py
python3 src/validation/newsevidence.py
python3 src/validation/validation_pipeline.py
streamlit run src/tracker_app.py
```

## Repository File Guide

This section explains every important file currently present in the repository snapshot.

### Root files

| File | Purpose |
| --- | --- |
| `README.md` | Main project documentation, pipeline explanation, and repo guide. |
| `requirements.txt` | Minimal dependency list used by the scripts in this repository. |

### Source code

| File | Purpose |
| --- | --- |
| `src/tracker_app.py` | Streamlit dashboard that loads final verdicts and enriched metadata, then exposes filters and summary charts. |
| `src/extraction/manifesto_pipeline.py` | OCR-based manifesto extraction pipeline that builds the first clean promise dataset. |
| `src/extraction/prs_data_scraper.py` | Scrapes Karnataka PRS bill and act listing pages and stores titles plus PDF URLs. |
| `src/enrichment/atomic_promise_pipeline.py` | Converts extracted manifesto text into cleaner one-action-per-row promises using Groq. |
| `src/enrichment/enrichment_pipeline.py` | Adds structured metadata fields such as sector, timeline, quantifiability, and commitment type. |
| `src/enrichment/prs_pdf_processor.py` | Downloads PRS bill PDFs and extracts English bill text into a CSV. |
| `src/enrichment/prs_pdf_processor_acts.py` | Downloads PRS act PDFs and extracts English act text into a CSV. |
| `src/enrichment/prs_action_extraction.py` | Converts processed bill text into a short list of policy actions using an LLM. |
| `src/enrichment/prs_acts_action_extraction.py` | Converts processed act text into a short list of policy actions using an LLM. |
| `src/validation/evidence_builder.py` | Uses sentence embeddings and a cross-encoder to match promises with PRS actions. |
| `src/validation/newsevidence.py` | Collects and reranks Google News RSS evidence for each promise. |
| `src/validation/validation_pipeline.py` | Performs the final verdict computation by combining retrieved evidence, PRS matching, and LLM stance classification. |
| `src/labelling/dataset_labelling.py` | Design-note style script describing how a future labeled dataset for evidence filtering and reranking could be generated. It is more of a pipeline sketch than a fully implemented runner. |

### Notebook

| File | Purpose |
| --- | --- |
| `notebooks/prs_action_extraction.ipynb` | Exploratory notebook version of the PRS bill action-extraction workflow. Useful for prompt iteration and debugging before script hardening. |

### Data artifacts in `data/raw/`

| File | Purpose |
| --- | --- |
| `data/raw/Assembly Election Karnataka Manifesto - 2023-2.pdf` | Source manifesto PDF used as the starting point for promise extraction. |

### Data artifacts in `data/processed/`

| File | Purpose |
| --- | --- |
| `data/processed/karnataka_inc_promises_cleaned.csv` | First structured promise dataset created from OCR and rule-based parsing. |
| `data/processed/karnataka_inc_promises_atomic.csv` | Atomic version of the promise dataset with one cleaned promise per row. |
| `data/processed/karnataka_inc_promises_atomic_partial.csv` | Checkpoint file written while atomic promise generation is in progress. |
| `data/processed/final_enriched_dataset.csv` | Main enriched promise table used by evidence and validation stages. |

### Data artifacts in `data/evidence/`

| File | Purpose |
| --- | --- |
| `data/evidence/evidence_dataset.csv` | Stored evidence rows for promises, likely used as a training or analysis artifact. |
| `data/evidence/evidence_dataset_partial.csv` | Partial checkpoint version of the evidence dataset. |

### Data artifacts in `data/prs_datasets/`

| File | Purpose |
| --- | --- |
| `data/prs_datasets/prs_karnataka_bills.csv` | Scraped PRS Karnataka bills index with titles and PDF URLs. |
| `data/prs_datasets/prs_karnataka_acts.csv` | Scraped PRS Karnataka acts index with titles and PDF URLs. |
| `data/prs_datasets/prs_karnataka_bills_processed.csv` | Extracted bill text for bills whose PDFs were successfully parsed. |
| `data/prs_datasets/prs_karnataka_bills_processed_partial.csv` | Checkpoint file for bill text extraction runs. |
| `data/prs_datasets/prs_karnataka_bills_with_actions.csv` | Bill text plus extracted policy-action summaries for retrieval. |
| `data/prs_datasets/prs_karnataka_acts_processed.csv` | Extracted act text for acts whose PDFs were successfully parsed. |
| `data/prs_datasets/prs_karnataka_acts_processed_partial.csv` | Checkpoint file for act text extraction runs. |
| `data/prs_datasets/prs_karnataka_acts_with_actions.csv` | Act text plus extracted policy-action summaries for retrieval. |

### Output artifacts in `outputs/`

| File | Purpose |
| --- | --- |
| `outputs/promise_evidence_dataset.csv` | Top PRS evidence candidates linked to each promise. |
| `outputs/news_evidence_dataset.csv` | Stored news evidence artifact for promises from an earlier or alternate run. |
| `outputs/news_evidence_partial.csv` | Checkpoint file for news evidence collection. |
| `outputs/news_gov_evidence_dataset.csv` | Historical combined news and government-site evidence artifact. No generating script for this exact file is present in the current repository snapshot. |
| `outputs/news_gov_evidence_partial.csv` | Partial checkpoint for the historical combined news and government evidence flow. |
| `outputs/final_results.csv` | Final promise verdicts and confidence values. |
| `outputs/final_results_partial.csv` | Checkpoint file for the verdict-generation stage. |

## Current State And Known Gaps

This repo is a strong working prototype, but the current snapshot is not perfectly standardized yet. These points are worth knowing before you extend it.

### 1. The tracker currently cannot fully join the shipped result files

- `src/tracker_app.py` merges `outputs/final_results.csv` with `data/processed/final_enriched_dataset.csv` on `promise_id`.
- In the current repository snapshot, `final_results.csv` uses IDs like `P1`, while `final_enriched_dataset.csv` uses IDs like `A1`.
- That means the shipped files do not currently produce metadata matches in the tracker without aligning IDs first.

### 2. `manifesto_pipeline.py` and the raw PDF location are not fully aligned

- The script currently looks for a PDF inside its own script directory.
- The actual manifesto PDF in this repo is stored under `data/raw/`.
- You may need to move the PDF, duplicate it, or update the script path before re-running extraction.

### 3. `newsevidence.py` and the stored news artifact are from different pipeline moments

- The script currently writes a JSON file at the end.
- The repository currently contains `outputs/news_evidence_dataset.csv`.
- That suggests the committed output was produced by an older version or a local adaptation of the script.

### 4. `outputs/news_gov_evidence_dataset.csv` appears to be a legacy artifact

- The file exists and contains historical results.
- No source script in this snapshot generates that exact output end-to-end.
- Some rows also contain noisy irrelevant evidence, so it should be treated as an experimental artifact rather than a clean production dataset.

### 5. `requirements.txt` is minimal

- The scripts use extra packages such as `tqdm` in PRS action extraction.
- If a stage fails after setup, compare imports against `requirements.txt` and install anything missing.

### 6. Multiple API key names are used

- `GROQ_API_KEY`
- `GROQ_API_KEY_DIFF`
- `GROQ_API_KEY_THIRD`

This is a sign that the repo evolved stage by stage rather than through a single unified config layer.

## Why This Project Matters

The core value of Vote Your Way is not only extraction or scraping. Its real value is building a repeatable accountability pipeline:

- manifesto text becomes structured data
- structured data becomes searchable promises
- promises are matched against legislative and public evidence
- evidence is turned into interpretable verdicts
- verdicts are exposed through a simple citizen-facing interface

That makes this repository a useful base for political accountability research, civic-tech dashboards, manifesto comparison systems, and future promise-tracking models.
