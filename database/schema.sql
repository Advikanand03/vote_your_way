-- VoteYourWay Knowledge Base
-- PostgreSQL V1 Schema
-- Run this script while connected to the VoteYourWay database.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- 1. PARTIES
-- ============================================================

CREATE TABLE parties (
    party_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    abbreviation VARCHAR(50),
    country VARCHAR(100),
    region VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_parties_name_country UNIQUE (name, country)
);

-- ============================================================
-- 2. ELECTIONS
-- ============================================================

CREATE TABLE elections (
    election_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    election_type VARCHAR(100),
    country VARCHAR(100),
    region VARCHAR(255),
    election_date DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_elections_country_region
    ON elections (country, region);

-- ============================================================
-- 3. MANIFESTOS
-- A manifesto connects one party to one election.
-- ============================================================

CREATE TABLE manifestos (
    manifesto_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    party_id UUID NOT NULL,
    election_id UUID NOT NULL,
    title VARCHAR(500),
    publication_date DATE,
    source_url TEXT,
    document_path TEXT,
    document_hash VARCHAR(128),
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_manifestos_party
        FOREIGN KEY (party_id)
        REFERENCES parties (party_id)
        ON DELETE RESTRICT,

    CONSTRAINT fk_manifestos_election
        FOREIGN KEY (election_id)
        REFERENCES elections (election_id)
        ON DELETE RESTRICT,

    CONSTRAINT chk_manifestos_version
        CHECK (version >= 1),

    CONSTRAINT uq_manifestos_party_election_version
        UNIQUE (party_id, election_id, version)
);

CREATE INDEX idx_manifestos_party
    ON manifestos (party_id);

CREATE INDEX idx_manifestos_election
    ON manifestos (election_id);

-- ============================================================
-- 4. SECTIONS
-- Reconstructed logical structure of a manifesto.
-- ============================================================

CREATE TABLE sections (
    section_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    manifesto_id UUID NOT NULL,
    parent_section_id UUID,
    title VARCHAR(500),
    page_start INTEGER,
    page_end INTEGER,
    section_order INTEGER NOT NULL,
    text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_sections_manifesto
        FOREIGN KEY (manifesto_id)
        REFERENCES manifestos (manifesto_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_sections_parent
        FOREIGN KEY (parent_section_id)
        REFERENCES sections (section_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_sections_pages
        CHECK (
            page_start IS NULL
            OR page_start >= 1
        ),

    CONSTRAINT chk_sections_page_range
        CHECK (
            page_start IS NULL
            OR page_end IS NULL
            OR page_end >= page_start
        ),

    CONSTRAINT chk_sections_order
        CHECK (section_order >= 1)
);

CREATE INDEX idx_sections_manifesto
    ON sections (manifesto_id);

CREATE INDEX idx_sections_parent
    ON sections (parent_section_id);

-- ============================================================
-- 5. LOCATIONS
-- Hierarchical geography.
-- ============================================================

CREATE TABLE locations (
    location_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    location_type VARCHAR(100) NOT NULL,
    parent_location_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_locations_parent
        FOREIGN KEY (parent_location_id)
        REFERENCES locations (location_id)
        ON DELETE RESTRICT
);

CREATE INDEX idx_locations_parent
    ON locations (parent_location_id);

CREATE INDEX idx_locations_name
    ON locations (name);

-- ============================================================
-- 6. CATEGORIES
-- Hierarchical promise classification.
-- ============================================================

CREATE TABLE categories (
    category_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    parent_category_id UUID,
    description TEXT,

    CONSTRAINT fk_categories_parent
        FOREIGN KEY (parent_category_id)
        REFERENCES categories (category_id)
        ON DELETE RESTRICT,

    CONSTRAINT uq_categories_name_parent
        UNIQUE (name, parent_category_id)
);

CREATE INDEX idx_categories_parent
    ON categories (parent_category_id);

-- ============================================================
-- 7. PROMISES
-- Central VoteYourWay knowledge entity.
-- ============================================================

CREATE TABLE promises (
    promise_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    section_id UUID NOT NULL,

    original_text TEXT NOT NULL,
    atomic_promise TEXT NOT NULL,

    action VARCHAR(255),

    target_value NUMERIC,
    target_unit VARCHAR(100),
    target_description TEXT,

    timeline_start DATE,
    timeline_end DATE,
    timeline_text TEXT,

    geography_id UUID,
    beneficiary TEXT,
    responsible_department VARCHAR(255),

    extraction_confidence NUMERIC(4,3),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_promises_section
        FOREIGN KEY (section_id)
        REFERENCES sections (section_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_promises_geography
        FOREIGN KEY (geography_id)
        REFERENCES locations (location_id)
        ON DELETE RESTRICT,

    CONSTRAINT chk_promises_confidence
        CHECK (
            extraction_confidence IS NULL
            OR (
                extraction_confidence >= 0
                AND extraction_confidence <= 1
            )
        ),

    CONSTRAINT chk_promises_timeline
        CHECK (
            timeline_start IS NULL
            OR timeline_end IS NULL
            OR timeline_end >= timeline_start
        )
);

CREATE INDEX idx_promises_section
    ON promises (section_id);

CREATE INDEX idx_promises_geography
    ON promises (geography_id);

CREATE INDEX idx_promises_action
    ON promises (action);

-- ============================================================
-- 8. PROMISE_CATEGORIES
-- Many-to-many relationship between promises and categories.
-- ============================================================

CREATE TABLE promise_categories (
    promise_id UUID NOT NULL,
    category_id UUID NOT NULL,

    PRIMARY KEY (promise_id, category_id),

    CONSTRAINT fk_promise_categories_promise
        FOREIGN KEY (promise_id)
        REFERENCES promises (promise_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_promise_categories_category
        FOREIGN KEY (category_id)
        REFERENCES categories (category_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_promise_categories_category
    ON promise_categories (category_id);

-- ============================================================
-- 9. PROMISE_SOURCES
-- Exact source location/provenance within a manifesto.
-- A promise may originate from multiple passages.
-- ============================================================

CREATE TABLE promise_sources (
    promise_source_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    promise_id UUID NOT NULL,
    manifesto_id UUID NOT NULL,
    section_id UUID,
    page_number INTEGER,
    source_text TEXT NOT NULL,
    source_order INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_promise_sources_promise
        FOREIGN KEY (promise_id)
        REFERENCES promises (promise_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_promise_sources_manifesto
        FOREIGN KEY (manifesto_id)
        REFERENCES manifestos (manifesto_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_promise_sources_section
        FOREIGN KEY (section_id)
        REFERENCES sections (section_id)
        ON DELETE SET NULL,

    CONSTRAINT chk_promise_sources_page
        CHECK (
            page_number IS NULL
            OR page_number >= 1
        ),

    CONSTRAINT chk_promise_sources_order
        CHECK (source_order >= 1)
);

CREATE INDEX idx_promise_sources_promise
    ON promise_sources (promise_id);

CREATE INDEX idx_promise_sources_manifesto
    ON promise_sources (manifesto_id);

CREATE INDEX idx_promise_sources_page
    ON promise_sources (manifesto_id, page_number);

-- ============================================================
-- 10. SOURCE_TYPES
-- Controlled vocabulary for evidence sources.
-- ============================================================

CREATE TABLE source_types (
    source_type_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT
);

-- ============================================================
-- 11. EVIDENCE
-- Evidence/source items that can be reused across promises.
-- ============================================================

CREATE TABLE evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type_id UUID NOT NULL,

    source_name VARCHAR(255),
    title TEXT,
    source_url TEXT,

    publication_date DATE,
    relevant_text TEXT,
    evidence_claim TEXT,

    source_reliability NUMERIC(4,3),

    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_evidence_source_type
        FOREIGN KEY (source_type_id)
        REFERENCES source_types (source_type_id)
        ON DELETE RESTRICT,

    CONSTRAINT chk_evidence_reliability
        CHECK (
            source_reliability IS NULL
            OR (
                source_reliability >= 0
                AND source_reliability <= 1
            )
        )
);

CREATE INDEX idx_evidence_source_type
    ON evidence (source_type_id);

CREATE INDEX idx_evidence_publication_date
    ON evidence (publication_date);

-- ============================================================
-- 12. PROMISE_EVIDENCE
-- Many-to-many relationship between promises and evidence.
-- ============================================================

CREATE TABLE promise_evidence (
    promise_id UUID NOT NULL,
    evidence_id UUID NOT NULL,
    relationship VARCHAR(50) NOT NULL DEFAULT 'context',

    PRIMARY KEY (promise_id, evidence_id),

    CONSTRAINT fk_promise_evidence_promise
        FOREIGN KEY (promise_id)
        REFERENCES promises (promise_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_promise_evidence_evidence
        FOREIGN KEY (evidence_id)
        REFERENCES evidence (evidence_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_promise_evidence_relationship
        CHECK (relationship IN ('supports', 'contradicts', 'context'))
);

CREATE INDEX idx_promise_evidence_evidence
    ON promise_evidence (evidence_id);

-- ============================================================
-- 13. PROMISE_STATUSES
-- Controlled vocabulary for verification status.
-- ============================================================

CREATE TABLE promise_statuses (
    status_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT
);

-- ============================================================
-- 14. VERIFICATION_EVENTS
-- Time-versioned verification history.
-- Never overwrite previous verification events.
-- ============================================================

CREATE TABLE verification_events (
    verification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    promise_id UUID NOT NULL,
    status_id UUID NOT NULL,

    confidence NUMERIC(4,3),
    reasoning TEXT,

    verification_date TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_verification_events_promise
        FOREIGN KEY (promise_id)
        REFERENCES promises (promise_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_verification_events_status
        FOREIGN KEY (status_id)
        REFERENCES promise_statuses (status_id)
        ON DELETE RESTRICT,

    CONSTRAINT chk_verification_confidence
        CHECK (
            confidence IS NULL
            OR (
                confidence >= 0
                AND confidence <= 1
            )
        )
);

CREATE INDEX idx_verification_events_promise
    ON verification_events (promise_id);

CREATE INDEX idx_verification_events_date
    ON verification_events (verification_date);

CREATE INDEX idx_verification_events_status
    ON verification_events (status_id);

-- ============================================================
-- 15. VERIFICATION_EVIDENCE
-- Evidence specifically used for a verification decision.
-- ============================================================

CREATE TABLE verification_evidence (
    verification_id UUID NOT NULL,
    evidence_id UUID NOT NULL,
    relationship VARCHAR(50) NOT NULL DEFAULT 'context',

    PRIMARY KEY (verification_id, evidence_id),

    CONSTRAINT fk_verification_evidence_verification
        FOREIGN KEY (verification_id)
        REFERENCES verification_events (verification_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_verification_evidence_evidence
        FOREIGN KEY (evidence_id)
        REFERENCES evidence (evidence_id)
        ON DELETE CASCADE,

    CONSTRAINT chk_verification_evidence_relationship
        CHECK (relationship IN ('supports', 'contradicts', 'context'))
);

CREATE INDEX idx_verification_evidence_evidence
    ON verification_evidence (evidence_id);

-- ============================================================
-- SEED CONTROLLED VOCABULARIES
-- ============================================================

INSERT INTO promise_statuses (name, description) VALUES
    ('Not Started', 'No meaningful implementation has begun.'),
    ('In Progress', 'Implementation is actively underway.'),
    ('Partially Fulfilled', 'Some, but not all, of the commitment has been fulfilled.'),
    ('Fulfilled', 'The commitment has been fulfilled according to available evidence.'),
    ('Delayed', 'Implementation is delayed relative to the stated timeline.'),
    ('Not Fulfilled', 'The commitment has not been fulfilled.'),
    ('Unclear', 'Available evidence is insufficient to determine the status.')
ON CONFLICT (name) DO NOTHING;

INSERT INTO source_types (name, description) VALUES
    ('government_report', 'Official report published by a government body.'),
    ('official_announcement', 'Official announcement by a government or other authoritative institution.'),
    ('legislative_document', 'Bill, act, legislative record, or related document.'),
    ('budget_document', 'Official budget or expenditure document.'),
    ('statistics', 'Official statistical data or dataset.'),
    ('news', 'Journalistic/news source.'),
    ('court_document', 'Judicial order, judgment, or court-related document.'),
    ('other', 'Other relevant evidence source.')
ON CONFLICT (name) DO NOTHING;

-- ============================================================
-- OPTIONAL: updated_at helper for promises
-- ============================================================

CREATE OR REPLACE FUNCTION update_promises_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_promises_updated_at
BEFORE UPDATE ON promises
FOR EACH ROW
EXECUTE FUNCTION update_promises_updated_at();

-- ============================================================
-- END OF VOTEYOURWAY V1 SCHEMA
-- ============================================================
