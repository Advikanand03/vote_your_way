# Vote Your Way: Political Promise Tracker & Validator

A comprehensive end-to-end pipeline for extracting, enriching, validating, and tracking political promises made in election manifestos.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Project Overview](#project-overview)
3. [Key Features](#key-features)
4. [Architecture & Data Flow](#architecture--data-flow)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [Pipeline Stages](#pipeline-stages)
8. [Running the Project](#running-the-project)
9. [Output & Tracking Dashboard](#output--tracking-dashboard)
10. [Data Dictionary](#data-dictionary)

---

## Problem Statement

**Challenge**: Political parties make numerous promises in election manifestos, but there is no systematic way to:
- Extract and parse these promises from PDF manifestos
- Categorize and structure them consistently
- Track their implementation status over time
- Validate claims with actual government actions and legislative records

**Goal**: Build an automated system to extract political promises, enrich them with metadata, validate them against government records and evidence, and provide an intuitive dashboard for citizens to track promise fulfillment.

**Context**: This project focuses on the **Indian National Congress (INC) manifestos from the 2023 Karnataka elections**, analyzing promises across various sectors and tracking their status through parliamentary records and public evidence.

---

## Project Overview

Vote Your Way is a **promise tracking and validation platform** that:

1. **Extracts** political promises from election manifesto PDFs using OCR and intelligent text parsing
2. **Enriches** extracted promises with metadata (sector, category, quantifiability, timeline)
3. **Validates** promises against real-world government actions using:
   - Parliamentary Records System (PRS) data (bills, acts)
   - Evidence datasets (news, reports, official announcements)
   - AI-powered semantic matching and verdict determination
4. **Tracks** promise status with confidence scores and visual analytics

### Use Cases

- **Citizens**: Understand which political promises are being fulfilled
- **Researchers**: Analyze party commitment patterns across sectors
- **Policymakers**: Identify promise-reality gaps and accountability mechanisms
- **Media**: Fact-check political claims with data-driven evidence

---

## Key Features

### 1. **Automated Promise Extraction**
- OCR-based text extraction from PDF manifestos
- Intelligent heading detection to identify promise sections
- Automatic cleaning and normalization

### 2. **Promise Enrichment**
- Sector and category classification (Healthcare, Education, Infrastructure, etc.)
- Quantifiability assessment (measurable vs. qualitative promises)
- Timeline extraction (promise target years)
- Commitment type identification

### 3. **Evidence-Based Validation**
- Multi-source evidence gathering (PRS bills/acts, news, reports)
- Semantic similarity matching between promises and evidence
- AI-powered verdict generation with confidence scoring
- Support for multiple validation states: Fulfilled, In Progress, Not Done, Uncertain

### 4. **Interactive Dashboard**
- Real-time filtering and search capabilities
- Sector and category-based breakdowns
- Confidence threshold controls
- Detailed promise-level intelligence with evidence links

---

## Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│         MANIFESTO EXTRACTION STAGE                          │
│  PDF → OCR → Text Parsing → Heading Detection → Promises   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
         karnataka_inc_promises_cleaned.csv
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   [ATOMIC CLEANUP]        [ENRICHMENT STAGE]
   (Optional)              Structure promises with:
        │                  • Sector/Category
        │                  • Quantifiability
        │                  • Timeline
        │                  • Commitment Type
        │                     │
        └─────────┬───────────┘
                  ▼
       final_enriched_dataset.csv
                  │
        ┌─────────┴──────────────┐
        ▼                        ▼
   [EVIDENCE GATHERING]    [VALIDATION STAGE]
   • PRS Bills/Acts        • Semantic Matching
   • News Archives         • LLM-based Verdict
   • Public Records        • Confidence Scoring
        │                        │
        └─────────┬──────────────┘
                  ▼
         final_results.csv
              (VERDICTS)
                  │
                  ▼
        ┌─────────────────────┐
        │  TRACKER DASHBOARD  │
        │  (Streamlit Web UI) │
        └─────────────────────┘
```

---

## Project Structure

```
vote_your_way/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
│
├── src/                               # Source code
│   ├── tracker_app.py                # Streamlit dashboard application
│   │
│   ├── extraction/                   # Promise extraction stage
│   │   ├── manifesto_pipeline.py     # PDF extraction & parsing
│   │   └── prs_data_scraper.py       # Parliamentary records scraper
│   │
│   ├── enrichment/                   # Promise enrichment stage
│   │   ├── enrichment_pipeline.py    # Metadata enrichment (LLM-based)
│   │   └── atomic_promise_pipeline.py# Promise atomization & cleanup
│   │
│   ├── labelling/                    # Labelling utilities
│   │   └── dataset_labelling.py      # Manual/semi-automatic labelling
│   │
│   └── validation/                   # Promise validation stage
│       └── validation_pipeline.py    # Verdict generation & evidence matching
│
├── data/                              # Data directory
│   ├── raw/                          # Raw input files
│   │   └── [manifesto PDFs]
│   │
│   ├── evidence/                     # Evidence datasets
│   │   ├── evidence_dataset.csv      # Full evidence collection
│   │   └── evidence_dataset_partial.csv # Sample evidence
│   │
│   ├── prs_datasets/                 # Parliamentary records
│   │   ├── prs_karnataka_bills.csv   # Bills from PRS
│   │   └── prs_karnataka_acts.csv    # Acts from PRS
│   │
│   └── processed/                    # Processed datasets
│       ├── karnataka_inc_promises_cleaned.csv
│       ├── karnataka_inc_promises_atomic.csv (optional)
│       └── final_enriched_dataset.csv
│
├── outputs/                           # Final output files
│   ├── final_results.csv             # Validation verdicts
│   └── final_results_partial.csv     # Partial checkpoints
│
└── notebooks/                         # Jupyter notebooks (exploratory)
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR (for PDF text extraction)
- GROQ API key (for LLM-powered enrichment & validation)

### Step 1: Clone and Navigate
```bash
cd /path/to/vote_your_way
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `pandas` - Data manipulation
- `streamlit` - Web dashboard
- `pdfplumber` / `pdf2image` - PDF processing
- `pytesseract` - OCR text extraction
- `sentence-transformers` - Semantic embeddings
- `groq` - LLM API for enrichment & validation
- `beautifulsoup4` - Web scraping
- `requests` - HTTP requests
- `python-dotenv` - Environment configuration

### Step 4: Configure Environment
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_api_key_here
```

### Step 5: Install Tesseract (System-level)
**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download installer from https://github.com/UB-Mannheim/tesseract/wiki

---

## Pipeline Stages

### Stage 1: Extraction (`extraction/manifesto_pipeline.py`)

**Input:** Manifesto PDF files

**Process:**
1. **PDF to Images**: Convert PDF pages to high-resolution images
2. **OCR**: Use Tesseract to extract text from images
3. **Heading Detection**: Identify promise section headers using heuristics:
   - Line length constraints (3-80 characters)
   - Capital letter requirement
   - Common promise keywords detection
4. **Text Normalization**: Clean extracted text, handle encoding issues
5. **Promise Segmentation**: Split text into individual promise units

**Output:** `data/processed/karnataka_inc_promises_cleaned.csv`
- Columns: `promise_id`, `promise_text`, `page_number`, `source`

**Challenges Addressed:**
- OCR accuracy issues (typical 85-95% accuracy)
- Inconsistent formatting across different manifesto versions
- Overlapping text from document layouts

---

### Stage 2: Enrichment (`enrichment/enrichment_pipeline.py`)

**Input:** `karnataka_inc_promises_cleaned.csv`

**Process:**
1. **LLM Analysis**: Use Groq's Llama 3.1 model to analyze each promise and extract:
   - **Sector**: Healthcare, Education, Infrastructure, Agriculture, etc.
   - **Category**: Type of commitment (policy, investment, institution-building, etc.)
   - **Quantifiability**: Whether the promise includes measurable targets
   - **Timeline**: Mentioned implementation period/target year
   - **Commitment Type**: Direct action, legislative, infrastructural, etc.

2. **JSON Parsing**: Safely extract structured metadata from LLM responses
3. **Validation**: Ensure all extracted fields are properly populated

**Output:** `data/processed/final_enriched_dataset.csv`
- All columns from cleaned dataset +
- `sector`, `category`, `sub_sector`, `quantifiable`, `timeline_mentioned`, `target_year`, `commitment_type`

**LLM Prompt Focus:**
- Systematic categorization using predefined taxonomies
- High consistency through structured output templates
- Fast processing using optimized model (Llama 3.1 8B)

---

### Stage 2b: Atomic Promise Cleanup (Optional) (`enrichment/atomic_promise_pipeline.py`)

**Input:** `karnataka_inc_promises_cleaned.csv`

**Process:**
- Break down compound promises into atomic units
- Each atomic promise addresses a single, specific commitment
- Remove duplicates and near-duplicates
- Standardize promise phrasing

**Output:** `data/processed/karnataka_inc_promises_atomic.csv`

**Note:** This stage is optional but recommended for improved validation accuracy.

---

### Stage 3: Validation (`validation/validation_pipeline.py`)

**Input:** 
- `data/processed/final_enriched_dataset.csv` (promises)
- `data/evidence/` (evidence datasets)
- `data/prs_datasets/` (parliamentary records)

**Process:**

1. **Evidence Gathering**:
   - Query PRS database for relevant bills and acts
   - Search evidence dataset for mentions of promise topics
   - Retrieve news articles and official announcements

2. **Semantic Matching**:
   - Generate embeddings for promises using sentence-transformers
   - Generate embeddings for evidence pieces
   - Calculate similarity scores to find relevant evidence
   - Filter to top-K most relevant pieces (default: 6)

3. **LLM-Based Verdict Generation**:
   - Present promise + top evidence to Llama 3.1 model
   - Generate structured verdict:
     - **Status**: Fulfilled / In Progress / Not Done / Uncertain
     - **Confidence**: 0.0-1.0 confidence score
     - **Reasoning**: Explanation for the verdict
     - **Evidence Summary**: How evidence supports the verdict

4. **Confidence Thresholding**:
   - Verdicts with confidence < 0.55 marked as "Uncertain"
   - Allows users to filter by confidence in dashboard

5. **Checkpointing**:
   - Saves partial results every 10 promises
   - Enables resumption of interrupted runs
   - Prevents data loss on API failures

**Output:** `outputs/final_results.csv`
- Columns: `promise_id`, `promise_text`, `verdict`, `confidence`, `reasoning`, `evidence_summary`

**Validation States:**
- ✅ **Fulfilled**: Strong evidence of promise implementation
- 🔄 **In Progress**: Partial implementation or active work
- ❌ **Not Done**: No evidence of implementation
- ❓ **Uncertain**: Insufficient evidence for clear verdict

---

## Running the Project

### Full Pipeline Execution

```bash
# Activate virtual environment
source .venv/bin/activate

# 1. Extract promises from manifesto PDFs
python src/extraction/manifesto_pipeline.py

# 2. (Optional) Atomize and clean promises
python src/enrichment/atomic_promise_pipeline.py

# 3. Enrich promises with metadata
python src/enrichment/enrichment_pipeline.py

# 4. Validate promises against evidence
python src/validation/validation_pipeline.py
```

### Running Individual Stages

If you already have intermediate files, you can run specific stages:

```bash
# Only run validation on existing enriched dataset
python src/validation/validation_pipeline.py

# Only run enrichment on existing cleaned promises
python src/enrichment/enrichment_pipeline.py
```

### Launching the Dashboard

```bash
# Start the Streamlit web application
streamlit run src/tracker_app.py

# Application will open at: http://localhost:8501
```

---

## Output & Tracking Dashboard

The **Vote Your Way Dashboard** (`tracker_app.py`) provides:

### Dashboard Sections

1. **Overview Metrics** (Top cards)
   - Total Promises: Count of all tracked promises
   - Fulfilled: Count with "Fulfilled" verdict
   - In Progress: Count with "In Progress" verdict
   - Not Done: Count with "Not Done" verdict

2. **Filters & Controls** (Sidebar)
   - **Verdict Filter**: Select which verdicts to display
   - **Sector Filter**: Filter by sector (Healthcare, Education, etc.)
   - **Category Filter**: Filter by promise category
   - **Confidence Slider**: Adjust minimum confidence threshold (0.0-1.0)
   - **Text Search**: Search promise text with regex/keywords

3. **Main Table**
   - Sortable columns with full promise details
   - Enrichment metadata (sector, category, timeline, quantifiability)
   - Validation verdict and confidence score
   - Evidence summary
   - Reasoning for the verdict

### How to Use the Dashboard

1. **View Overview**: See high-level fulfillment statistics
2. **Filter by Sector**: Explore promise fulfillment by policy area
3. **Adjust Confidence**: Show/hide uncertain verdicts as needed
4. **Search Promises**: Find specific promises by text
5. **Drill Down**: Click rows to see full details with evidence

---

## Data Dictionary

### Core Promise Fields

| Field | Type | Description |
|-------|------|-------------|
| `promise_id` | String | Unique identifier for each promise |
| `promise_text` | String | Full text of the promise |
| `page_number` | Integer | Page number in original manifesto |
| `source` | String | Source document name |

### Enrichment Fields

| Field | Type | Description |
|-------|------|-------------|
| `sector` | String | Policy sector (Healthcare, Education, Agriculture, etc.) |
| `category` | String | Promise category (Policy, Investment, Infrastructure, etc.) |
| `sub_sector` | String | Specific sub-category within sector |
| `quantifiable` | Boolean | Whether promise has measurable targets |
| `timeline_mentioned` | Boolean | Whether promise mentions implementation timeline |
| `target_year` | Integer | Year by which promise should be fulfilled |
| `commitment_type` | String | Type of commitment (Direct, Legislative, Infrastructure) |

### Validation Fields

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | String | Status: Fulfilled / In Progress / Not Done / Uncertain |
| `confidence` | Float | Confidence score (0.0-1.0) in the verdict |
| `reasoning` | String | Explanation for the verdict |
| `evidence_summary` | String | Summary of evidence used for verdict |
| `evidence_sources` | List | Links to evidence documents/records |

---

## Extending the Project

### Adding New Data Sources
1. Add evidence datasets to `data/evidence/`
2. Update `validation_pipeline.py` to include new sources
3. Ensure consistent schema: `[promise_related_text, source, date]`

### Customizing Categories
1. Edit sector/category lists in `enrichment_pipeline.py`
2. Update LLM prompt to reflect new taxonomy
3. Re-run enrichment pipeline

### Improving Validation Accuracy
1. Add more evidence sources
2. Fine-tune LLM prompts in `validation_pipeline.py`
3. Adjust confidence threshold based on validation results
4. Implement cross-validation with manual labels

---

## Performance & Limitations

### Current Performance
- **Extraction**: ~2-5 seconds per manifesto page
- **Enrichment**: ~2-3 seconds per promise (API-limited)
- **Validation**: ~5-10 seconds per promise (including evidence search)

### Known Limitations
- OCR accuracy depends on manifesto PDF quality
- Validation relies on availability and quality of evidence data
- LLM verdicts are probabilistic (not legal determinations)
- Promise interpretation can be subjective

### Recommendations
- Use confidence scores to filter uncertain verdicts
- Cross-validate with manual review for critical promises
- Regularly update evidence datasets for accurate tracking
- Consider sector-specific validation heuristics

---

## Future Enhancements

- [ ] Real-time monitoring of promise implementation
- [ ] Automated evidence gathering from news APIs
- [ ] Comparative analysis across multiple elections/manifestos
- [ ] Performance scoring by political party and sector
- [ ] Citizen feedback integration for verdict validation
- [ ] Mobile app for promise tracking
- [ ] Integration with government open data portals

---

## Contributing

To contribute to this project:
1. Create a new branch for your feature
2. Follow the existing code structure
3. Test your changes with sample data
4. Submit a pull request with documentation

---

## License

[Add appropriate license]

---

## Contact & Support

For questions or issues, please open an issue in the repository or contact the project maintainers.

**Last Updated**: April 18, 2026
