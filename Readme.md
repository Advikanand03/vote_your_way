Vote Your Way

Project Overview

Vote Your Way is a continuously updating AI-powered political accountability platform.

It converts manifesto promises into structured entities, gathers real-world evidence about them, evaluates that evidence using hybrid retrieval and agentic RAG, and presents transparent, evidence-backed promise statuses through a public website.

The project is being rebuilt from scratch. The goal is not to reproduce the old batch-processing implementation, but to build a persistent knowledge system that can continuously ingest new evidence and update the assessment of political promises over time.

⸻

1. Core Idea

Old Approach

Manifesto
    ↓
Scripts
    ↓
CSV Files
    ↓
LLM
    ↓
Final Results

The previous implementation relied heavily on scripts and CSV files, batch processing, unstable identifiers, narrow evidence sources, simplistic news retrieval, and excessive responsibility placed on the LLM.

This made the system difficult to scale, update, audit, and maintain.

New Approach

Manifestos
    ↓
Persistent Promise Database
    +
Evidence Knowledge Base
    ↓
Hybrid Retrieval
    ↓
Agentic RAG
    ↓
Evidence-Based Verdict
    ↓
Website

The database becomes the source of truth.

AI components operate on structured data and retrieved evidence rather than treating the LLM as the primary source of state or truth.

The fundamental philosophy is:

Evidence first. Reasoning second. Verdict last.

⸻

2. System Goals

The new system should:

* Track manifesto promises using permanent stable IDs.
* Store elections, parties, manifestos, promises, documents, evidence, and verdicts in a persistent database.
* Continuously ingest new evidence from controlled sources.
* Retrieve relevant evidence using semantic search, keyword search, and metadata filtering.
* Use agentic RAG for iterative evidence discovery and reasoning.
* Explicitly represent supporting and contradictory evidence.
* Preserve a timeline of how a promise’s status changes.
* Make every verdict traceable to its underlying evidence.
* Provide a public website for browsing and comparing political accountability data.
* Keep AI components modular so that models/providers can be changed without redesigning the system.
* Make results understandable and verifiable by humans.

⸻

3. High-Level Architecture

                         ┌─────────────────────┐
                         │      Manifestos      │
                         │       (PDFs)        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Text / OCR        │
                         │    Extraction       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Promise Extraction  │
                         │ Atomic Splitting    │
                         │ Metadata Enrichment │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Promise Database    │
                         │    PostgreSQL       │
                         └──────────┬──────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
      ┌───────────────────┐                  ┌───────────────────┐
      │ Evidence Sources  │                  │ Documents / Chunks│
      │                   │                  │                   │
      │ Government        │                  │ Metadata          │
      │ Legislative       │                  │ Embeddings        │
      │ Financial         │                  │ Text              │
      │ News              │                  │                   │
      │ Audit / Research  │                  │                   │
      └─────────┬─────────┘                  └─────────┬─────────┘
                │                                      │
                └──────────────────┬───────────────────┘
                                   ▼
                       ┌─────────────────────────┐
                       │ Evidence Knowledge Base │
                       │   PostgreSQL + pgvector │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │    Hybrid Retrieval     │
                       │                         │
                       │ Vector + Keyword + Meta │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │       Agentic RAG        │
                       │ Search / Assess / Refine │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │      Verdict Engine     │
                       │                         │
                       │ Status                   │
                       │ Confidence               │
                       │ Evidence                 │
                       │ Reasoning                │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │        Website          │
                       │                         │
                       │ Dashboard                │
                       │ Search                   │
                       │ Promises                 │
                       │ Evidence                 │
                       │ Comparisons              │
                       └─────────────────────────┘

⸻

4. Core Entities

The central entity in the entire system is the Promise.

Every promise receives a permanent stable unique ID and remains linked to its origin, evidence, verdicts, and historical timeline.

Main Entities

Election

Represents an election cycle.

Possible information:

* Election ID
* Election name
* Election type
* Country
* State/region
* Election date
* Term period
* Other election metadata

⸻

Party

Represents a political party.

Possible information:

* Party ID
* Party name
* Abbreviation
* Country
* State/region
* Party metadata

⸻

Manifesto

Represents a manifesto published by a party for an election.

Possible information:

* Manifesto ID
* Party
* Election
* Publication date
* Document location
* Original file
* Extracted text
* Processing status

⸻

Promise

Represents an atomic political commitment extracted from a manifesto.

A promise is the most important entity in the system.

Possible information:

* Promise ID
* Manifesto ID
* Party ID
* Election ID
* Promise text
* Sector
* Sub-sector
* Commitment type
* Target value
* Target unit
* Target year
* Geography
* Responsible department
* Source page/location
* Creation timestamp

⸻

Document

Represents an external source document containing potentially relevant evidence.

Examples:

* Government report
* Budget document
* Parliamentary report
* News article
* Audit report
* Government order

⸻

DocumentChunk

A searchable section of a document.

Documents are split into smaller chunks so that relevant passages can be retrieved without sending the entire document to an LLM.

Possible metadata:

* Chunk ID
* Document ID
* Text
* Page
* Section
* Publication date
* Source
* Geography
* Department
* Embedding

⸻

Source

Represents the publisher or provider of a document.

Examples:

* Government ministry
* Parliament
* PRS
* Newspaper
* CAG
* Research institution

Source metadata can later be used for source-quality scoring.

⸻

Evidence

Represents a specific factual claim or passage relevant to one or more promises.

Evidence should retain its connection to the original document and source.

⸻

Verdict

Represents an assessment of a promise based on available evidence.

A verdict should never exist independently of its supporting evidence.

⸻

Timeline Event

Represents the state of a promise at a particular point in time.

This allows the system to track how a promise changes over time rather than storing only one final status.

⸻

5. Promise Model

Promises should be atomic wherever possible.

For example, a manifesto may say:

“We will construct 500 new schools and recruit 10,000 teachers.”

This should not necessarily be treated as one promise.

Instead:

P-001 → Construct 500 new schools.
P-002 → Recruit 10,000 teachers.

This allows each commitment to be independently evaluated.

Possible Promise Metadata

A promise may contain:

Promise ID
Manifesto ID
Party
Election
Promise Text
Sector
Sub-sector
Commitment Type
Target Value
Target Unit
Target Year
Geography
Responsible Department
Source Page
Source Location

Promise IDs must remain stable even when new evidence or new verdicts are added.

⸻

6. Evidence Knowledge Base

The evidence layer is intended to grow continuously.

Potential evidence categories include:

Government Evidence

* Government orders
* Ministry documents
* Department documents
* Press releases
* Implementation reports
* Official dashboards
* Government schemes
* Government statistics
* Project completion reports

Legislative Evidence

* Bills
* Acts
* Parliamentary documents
* Legislative documents
* Committee reports
* Parliamentary questions
* PRS material

Financial Evidence

* Union budgets
* State budgets
* Expenditure reports
* Funding documents
* Financial statements
* Department allocations
* Spending reports

News Evidence

* National newspapers
* Regional newspapers
* News agencies
* Local publications

Audit / Research Evidence

* CAG reports
* Audit reports
* Official statistics
* Institutional reports
* Research publications
* Independent evaluations

The first implementation should use a limited and controlled source set.

Source coverage should expand only after ingestion and retrieval quality have been validated.

⸻

7. Document Processing Pipeline

The intended document pipeline is:

Document
    ↓
Extraction / OCR
    ↓
Cleaning
    ↓
Chunking
    ↓
Metadata Enrichment
    ↓
Embedding
    ↓
Knowledge Base

Each document chunk should retain enough metadata to make the evidence auditable.

Possible metadata:

Document ID
Source ID
Publication Date
Document Type
Page
Section
Geography
Department
Organization
Election Relevance
Text
Embedding

⸻

8. Embeddings

Embeddings are used for retrieval, not for deciding whether a promise was fulfilled.

An embedding converts a piece of text into a numerical vector representing semantic characteristics of the text.

This allows the system to identify passages that are conceptually related even when they use different words.

For example:

Promise

“Construct 500 new schools.”

Evidence

“327 educational institutions were completed.”

A traditional keyword search might not consider these highly related because the wording is different.

An embedding-based search can recognize that both statements concern the construction/completion of educational institutions.

Therefore:

Text
 ↓
Embedding Model
 ↓
Numerical Vector
 ↓
Vector Database

The vector is primarily used to find relevant information.

It does not determine whether the promise was actually fulfilled.

⸻

9. Hybrid Retrieval

A reliable retrieval system should not depend solely on embeddings.

The system should combine multiple retrieval strategies.

9.1 Semantic / Vector Search

Finds conceptually similar passages.

Useful when:

* wording differs
* synonyms are used
* the evidence uses indirect language
* exact keywords are absent

⸻

9.2 Keyword / Full-Text Search

Useful for:

* exact names
* numbers
* project names
* department names
* specific schemes
* exact phrases

⸻

9.3 Metadata Filtering

Results can be restricted using metadata such as:

* Date
* Geography
* State
* Sector
* Department
* Election
* Party
* Source type
* Document type

⸻

9.4 Fusion and Reranking

Results from different retrieval methods can be combined.

Conceptually:

Vector Search
      +
Keyword Search
      +
Metadata Filters
      ↓
Candidate Evidence
      ↓
Result Fusion
      ↓
Reranking
      ↓
Best Evidence

The final ranked evidence is then passed to the reasoning layer.

⸻

10. PostgreSQL + pgvector

The initial architecture should use:

PostgreSQL
    +
pgvector

rather than immediately introducing a separate vector database.

This allows structured information and embeddings to exist in the same system.

For example:

PostgreSQL
Election
Party
Manifesto
Promise
Document
Evidence
Verdict
Timeline
        +
pgvector
Document Embeddings
Evidence Embeddings

A separate vector database can be considered later if scale or architecture requires it.

⸻

11. Agentic RAG

Agentic RAG is an iterative evidence retrieval and reasoning process.

It is not simply a chatbot.

The system should be able to recognize when the first retrieval is insufficient and perform additional searches.

Conceptually:

Promise
   ↓
Initial Search
   ↓
Retrieve Evidence
   ↓
Assess Evidence Sufficiency
   ↓
Is Evidence Sufficient?
   │
   ├── YES ──────────────┐
   │                     │
   └── NO                │
        ↓                │
Generate / Refine        │
Additional Queries       │
        ↓                │
Retrieve More Evidence   │
        │                │
        └────────────────┘
                 ↓
       Supporting Evidence
                 +
       Contradictory Evidence
                 ↓
          Evidence Synthesis
                 ↓
          Structured Verdict

The system should not simply search once and immediately ask an LLM to make a judgment.

Instead:

1. Understand the promise.
2. Generate initial search queries.
3. Retrieve evidence.
4. Evaluate whether the evidence is sufficient.
5. Generate additional searches if necessary.
6. Look for supporting evidence.
7. Look for contradictory evidence.
8. Evaluate source quality.
9. Synthesize the evidence.
10. Produce a structured verdict.

⸻

12. Contradictory Evidence

Contradictory evidence is a first-class concept.

The system must not only search for information supporting an initial conclusion.

For example:

Promise
   ↓
Evidence A → Supports implementation
Evidence B → Supports implementation
Evidence C → Shows delay
Evidence D → Shows incomplete implementation

The reasoning system should consider all of these.

The final verdict should explain the conflict instead of hiding it.

This is critical for political accountability because government claims, independent reports, audits, and news reports can sometimes disagree.

⸻

13. Verdict Model

The initial set of possible statuses is:

NOT_STARTED
IN_PROGRESS
PARTIALLY_FULFILLED
FULFILLED
NOT_FULFILLED
UNCERTAIN

The final schema can evolve as the project becomes more mature.

Example Verdict

{
  "verdict": "PARTIALLY_FULFILLED",
  "confidence": 0.82,
  "progress": {
    "completed": 312,
    "target": 500
  },
  "supporting_evidence": [
    {
      "evidence_id": "EV-001",
      "document_id": "DOC-042"
    }
  ],
  "contradictory_evidence": [
    {
      "evidence_id": "EV-007",
      "document_id": "DOC-091"
    }
  ],
  "reasoning": "The available evidence indicates substantial implementation, but the stated target has not yet been fully achieved."
}

Every verdict must be traceable to the evidence used to reach it.

⸻

14. Confidence

Confidence should not simply mean:

“The LLM feels 82% confident.”

It should eventually incorporate measurable factors such as:

* Evidence quality
* Source reliability
* Number of independent sources
* Evidence consistency
* Evidence recency
* Retrieval quality
* Whether contradictory evidence exists
* Whether the promise has a measurable target
* Whether the available evidence directly addresses the promise

The exact confidence model can be designed later.

⸻

15. Timeline

Promise status is temporal.

A promise should not simply have one permanent status.

For example:

2024 → NOT_STARTED
2025 → IN_PROGRESS
2026 → PARTIALLY_FULFILLED
2027 → FULFILLED

The system should preserve these historical states.

This allows users to understand not just where the promise ended up, but how implementation progressed.

⸻

16. Continuous Updates

The long-term system should continuously ingest new evidence.

Conceptually:

New Evidence
      ↓
Ingestion
      ↓
Extraction
      ↓
Chunking
      ↓
Embedding
      ↓
Knowledge Base
      ↓
Identify Affected Promises
      ↓
Re-evaluate
      ↓
Updated Verdict
      ↓
Timeline Update
      ↓
Website

For example:

A new government report is published in 2027.

The system identifies that it relates to P-001.

It retrieves the existing evidence.

It evaluates the new information.

If the new evidence changes the assessment:

Old Status: IN_PROGRESS
        ↓
New Evidence
        ↓
New Status: PARTIALLY_FULFILLED

The old assessment remains part of the historical timeline.

⸻

17. Website

The website is the public-facing layer of the system.

Home Dashboard

The homepage can contain:

* Elections
* Parties
* Accountability statistics
* Recent updates
* Recently evaluated promises
* Search
* Featured comparisons

⸻

18. Election Explorer

Basic navigation:

Election
   ↓
Parties
   ↓
Manifestos
   ↓
Promises

Users should be able to select an election and explore the commitments made by each party.

⸻

19. Manifesto Explorer

Users should be able to browse promises by:

* Sector
* Sub-sector
* Commitment type
* Status
* Target year
* Geography
* Party

Example:

Education
├── Schools
├── Universities
├── Teachers
├── Scholarships
└── Infrastructure

⸻

20. Promise Detail Page

The Promise Detail Page is the most important page on the website.

It should contain:

Promise
Party
Election
        ↓
Current Status
        ↓
Confidence
        ↓
Progress
        ↓
Supporting Evidence
        ↓
Contradictory Evidence
        ↓
Timeline
        ↓
AI Assessment
        ↓
Underlying Documents
        ↓
Original Sources

The user should be able to inspect exactly why a promise received a particular status.

⸻

21. Party Comparison

Users should eventually be able to compare parties.

For example:

Party A
Education      65% fulfilled
Healthcare     48% fulfilled
Infrastructure 72% fulfilled
Party B
Education      54% fulfilled
Healthcare     61% fulfilled
Infrastructure 58% fulfilled

However, aggregated metrics must always be drillable.

A user should be able to click:

65% fulfilled
      ↓
List of promises
      ↓
Individual promise
      ↓
Evidence
      ↓
Original document

This prevents summary statistics from becoming black boxes.

⸻

22. Search

The eventual search system should cover:

* Promises
* Parties
* Elections
* Topics
* Documents
* Evidence

Example queries:

"school construction"
"teacher recruitment"
"healthcare spending"
"metro projects"
"employment promises"

Search can eventually use the same hybrid retrieval infrastructure used by the accountability engine.

⸻

23. Technology Stack

Frontend

Next.js
TypeScript
Tailwind CSS
shadcn/ui

Backend

FastAPI
Python

Database

PostgreSQL
pgvector

AI

The AI layer should use a provider abstraction.

The core system should not be hard-coded around a single LLM or embedding provider.

Conceptually:

Application
     ↓
AI Provider Interface
     ↓
┌───────────────┬───────────────┐
│ LLM Provider  │ Embedding     │
│               │ Provider      │
└───────────────┴───────────────┘

This allows models to be changed later without rewriting the application.

Background Processing

Potential stack:

Redis
Celery

These can be introduced when asynchronous processing becomes necessary.

Document Storage

Potential options:

S3-compatible storage
MinIO
AWS S3

MinIO can be useful for local development while S3 can be used for production.

⸻

24. Development Roadmap

Phase 0 — Product and System Design

Before writing significant code:

* Define requirements.
* Finalize architecture.
* Define entities.
* Define relationships.
* Design ER diagram.
* Define promise model.
* Define evidence model.
* Define verdict schema.
* Plan APIs.
* Define evaluation strategy.

⸻

Phase 1 — Backend Foundation

Build the core backend.

Tasks:

* Create FastAPI application.
* Set up PostgreSQL.
* Enable pgvector.
* Implement database models.
* Create migrations.
* Establish configuration.
* Build foundational APIs.
* Add validation.
* Add basic error handling.

Expected result:

FastAPI
    ↓
PostgreSQL
    ↓
Election / Party / Manifesto / Promise

⸻

Phase 2 — Manifesto Pipeline

Build the first intelligence pipeline.

Tasks:

* PDF ingestion.
* Text extraction.
* OCR where necessary.
* Text cleaning.
* Promise detection.
* Atomic promise splitting.
* Promise metadata extraction.
* Persist promises in PostgreSQL.

Pipeline:

Manifesto PDF
     ↓
Text / OCR
     ↓
Clean Text
     ↓
Promise Detection
     ↓
Atomic Splitting
     ↓
Metadata
     ↓
PostgreSQL

⸻

Phase 3 — Evidence Knowledge Base

Build the evidence infrastructure.

Tasks:

* Source ingestion.
* Document ingestion.
* Text extraction.
* Cleaning.
* Chunking.
* Metadata storage.
* Embedding generation.
* pgvector storage.

Pipeline:

Evidence Source
      ↓
Document
      ↓
Extraction
      ↓
Chunks
      ↓
Embeddings
      ↓
PostgreSQL + pgvector

⸻

Phase 4 — Retrieval

Build retrieval before building sophisticated reasoning.

Tasks:

* Semantic search.
* Keyword/full-text search.
* Metadata filters.
* Candidate generation.
* Result fusion.
* Reranking.
* Retrieval API.

Expected flow:

Promise
   ↓
Query Generation
   ↓
Vector Search
+
Keyword Search
+
Metadata Filtering
   ↓
Candidate Evidence
   ↓
Fusion
   ↓
Reranking
   ↓
Relevant Evidence

⸻

Phase 5 — Retrieval Evaluation

This phase is extremely important.

Before relying on an AI system to produce political accountability verdicts, we need to know whether it can actually retrieve the right evidence.

Create a labelled evaluation dataset.

For example:

Promise P-001
Relevant:
DOC-001
DOC-017
DOC-034
Not Relevant:
DOC-005
DOC-022
DOC-051

Then measure retrieval quality.

Possible metrics:

* Recall@5
* Recall@10
* Precision@k
* MRR
* NDCG where useful

Perform error analysis.

Questions to investigate:

* Which promises retrieve poorly?
* Which source types are missed?
* Are numbers retrieved correctly?
* Are contradictory documents found?
* Does metadata filtering help?
* Does reranking improve results?

Only after retrieval quality is reasonably strong should the system move to sophisticated verdict generation.

⸻

Phase 6 — Agentic RAG

Build the iterative reasoning layer.

Tasks:

* Query formulation.
* Initial retrieval.
* Evidence sufficiency assessment.
* Additional query generation.
* Iterative retrieval.
* Supporting evidence identification.
* Contradiction detection.
* Evidence synthesis.

Expected flow:

Promise
   ↓
Search
   ↓
Retrieve
   ↓
Evaluate
   ↓
Enough evidence?
   ├── Yes → Synthesize
   │
   └── No
        ↓
   New queries
        ↓
   Retrieve again
        ↓
   Evaluate again
        ↓
   Synthesize

⸻

Phase 7 — Verdict Engine

Tasks:

* Implement structured verdict schema.
* Generate evidence-backed statuses.
* Generate confidence estimates.
* Link verdicts to evidence.
* Store verdict history.
* Store reasoning.
* Add consistency checks.

The verdict engine should never produce a conclusion without recording the evidence used to support it.

⸻

Phase 8 — Website

Build the public interface.

Tasks:

* Dashboard.
* Election explorer.
* Party pages.
* Manifesto explorer.
* Promise detail pages.
* Evidence viewer.
* Search.
* Party comparison.
* Timeline visualization.

⸻

Phase 9 — Continuous Updates

Automate the system.

Tasks:

* Scheduled ingestion.
* Background jobs.
* New-document detection.
* Document deduplication.
* Affected-promise detection.
* Automatic re-evaluation.
* Timeline updates.
* Website updates.

⸻

Phase 10 — Analytics and Expansion

Potential future work:

* Richer party comparisons.
* Sector analysis.
* Geographic analysis.
* Historical elections.
* Cross-election comparisons.
* Researcher APIs.
* Public datasets.
* More evidence sources.
* Better source-quality modelling.
* Advanced contradiction detection.

⸻

25. Initial MVP

The first MVP should deliberately be much smaller than the final vision.

The first goal is:

Manifesto
    ↓
Promise Extraction
    ↓
Persistent Promise Database
    ↓
Evidence Knowledge Base
    ↓
Embeddings
    ↓
Hybrid Retrieval
    ↓
Relevant Evidence

The MVP should answer:

Given a manifesto promise, can our system reliably find the relevant real-world evidence?

The MVP should not immediately attempt to solve every problem.

In particular, automated verdict generation should come only after retrieval quality has been measured.

⸻

26. What We Should NOT Do Initially

Avoid prematurely building:

* A huge autonomous agent.
* Multiple vector databases.
* Dozens of evidence sources.
* Complex multi-agent architectures.
* Fully automated political verdicts.
* Large-scale web crawling.
* Complicated microservices.
* Excessive frontend features.
* Provider-specific AI logic throughout the codebase.

First build a strong foundation.

The priority is:

Good Data
   ↓
Good Retrieval
   ↓
Good Evidence
   ↓
Good Reasoning
   ↓
Good Verdicts
   ↓
Good User Experience

⸻

27. Design Principles

1. Evidence Before Verdict

Never start with the desired conclusion and search for supporting information.

⸻

2. Retrieval Before Sophistication

A sophisticated agent cannot compensate for poor retrieval.

⸻

3. Database as Source of Truth

The database should contain the canonical state of the system.

CSV files may be generated as exports or analysis artifacts, but they should not be the primary source of state.

⸻

4. Stable IDs

Promises must have permanent IDs.

Avoid unstable identifiers such as:

P1
P2
A1
A2

being regenerated differently across processing runs.

⸻

5. Explainability

Every important output should be explainable.

A user should be able to move from:

Verdict
   ↓
Evidence
   ↓
Document
   ↓
Original Source

⸻

6. Contradictions Are Explicit

Supporting and contradictory evidence should both be represented.

⸻

7. Source Quality Matters

A government implementation report, an independent audit, and a random article should not necessarily have identical evidentiary weight.

⸻

8. Temporal Awareness

Promises evolve over time.

The system should preserve historical assessments instead of overwriting them.

⸻

9. Modular AI

LLMs and embedding models should be replaceable.

⸻

10. Human-Verifiable Output

The final product should help people inspect the evidence themselves rather than simply trusting an AI-generated conclusion.

⸻

28. Long-Term Vision

Vote Your Way should evolve from a static manifesto-analysis project into a persistent political accountability knowledge system.

Ultimately, it should be able to answer questions such as:

* What did a party promise?
* What exactly did the promise mean?
* What evidence exists about implementation?
* What has actually been completed?
* What remains incomplete?
* What evidence contradicts the apparent progress?
* How has the status changed over time?
* How does one party compare with another?
* How reliable is the available evidence?
* How confident should we be in the assessment?

The goal is not to make an LLM simply declare whether a politician kept a promise.

The goal is to build a system where evidence is:

Collected
   ↓
Organized
   ↓
Stored
   ↓
Retrieved
   ↓
Evaluated
   ↓
Cited
   ↓
Explained

so that people can inspect the basis for the conclusion themselves.

⸻

29. Final Product Philosophy

Vote Your Way should ultimately function less like a chatbot and more like a living political accountability database.

The system should continuously connect:

Political Promise
      ↓
Real-World Evidence
      ↓
Historical Context
      ↓
Evidence Evaluation
      ↓
Accountability Status

The website is only the visible layer.

The real product is the underlying political accountability knowledge base.

⸻

30. One-Sentence Definition

Vote Your Way is a continuously updating AI-powered political accountability platform that converts manifesto promises into structured entities, gathers real-world evidence about them, uses hybrid retrieval and agentic RAG to evaluate that evidence, and presents transparent, evidence-backed promise statuses through a public website.