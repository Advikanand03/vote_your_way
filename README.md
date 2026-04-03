# Vote Your Way

Manifesto extraction and one-by-one promise validation pipeline for Karnataka Congress 2023, with a simple tracker web UI.

## Pipeline Outputs

- `data/processed/karnataka_inc_promises_cleaned.csv` - extracted promises
- `data/processed/final_enriched_dataset.csv` - enriched promise metadata
- `outputs/final_results.csv` - validation verdicts and confidence

## Run the Pipeline

1. Extraction:
   - `python src/extraction/manifesto_pipeline.py`
2. Atomic promise cleanup (optional but recommended):
   - `python src/enrichment/atomic_promise_pipeline.py`
   - Output: `data/processed/karnataka_inc_promises_atomic.csv`
3. Enrichment:
   - `python src/enrichment/enrichment_pipeline.py`
4. Validation:
   - `python src/validation/validation_pipeline.py`

## Run the Tracker Website

Install dependencies:

- `pip install -r requirements.txt`

Start the app:

- `streamlit run src/tracker_app.py`

The tracker provides:

- Overview metrics (Total, Completed, In Progress, Not Done)
- Filters by verdict, sector, and category
- Confidence threshold slider
- Promise text search
- Full promise table with enrichment fields
