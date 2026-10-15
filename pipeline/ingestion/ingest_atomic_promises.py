#!/usr/bin/env python3
"""Idempotent VoteYourWay atomic-promise ingestion.

Default mode is DRY RUN. Pass --execute to write to PostgreSQL.
Run from the repository root:
    python pipeline/ingestion/ingest_atomic_promises.py
    python pipeline/ingestion/ingest_atomic_promises.py --execute

Environment:
    DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/VoteYourWay
or set PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD.
The script also loads variables from the repository-root .env file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "aap_2020_atomic_promises_clean.json"
EXPECTED_MANIFESTO_ID = "30398bb3-18f7-4332-adfd-fbadbdcbfb12"


def connect_db():
    load_dotenv(ROOT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return psycopg.connect(database_url, row_factory=dict_row)

    required = ("PGDATABASE", "PGUSER", "PGPASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Database configuration missing: "
            + ", ".join(missing)
            + ". Add DATABASE_URL or PGDATABASE, PGUSER, and PGPASSWORD to .env."
        )

    return psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        row_factory=dict_row,
    )


def load_records() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Validated JSON not found: {DATA_PATH}\n"
            "Make sure aap_2020_atomic_promises_clean.json is in data/processed/."
        )

    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Expected the JSON root to be a list of source records.")

    total_promises = 0
    for index, record in enumerate(data, start=1):
        if not isinstance(record, dict) or not isinstance(record.get("source"), dict):
            raise ValueError(f"Record {index} is missing its source object.")
        if not isinstance(record.get("atomic_promises"), list):
            raise ValueError(f"Record {index} is missing its atomic_promises list.")
        source = record["source"]
        if source.get("manifesto_id") != EXPECTED_MANIFESTO_ID:
            raise ValueError(
                f"Record {index} has unexpected manifesto_id "
                f"{source.get('manifesto_id')!r}; stopping to avoid wrong-manifesto ingestion."
            )
        for promise in record["atomic_promises"]:
            if not promise.get("original_text") or not promise.get("atomic_promise"):
                raise ValueError(
                    f"Record {index} contains a promise missing original_text or atomic_promise."
                )
            total_promises += 1

    if len(data) != 38 or total_promises != 55:
        raise ValueError(
            f"Expected 38 source records and 55 promises, found {len(data)} "
            f"source records and {total_promises} promises. Review the input before ingestion."
        )

    return data


def resolve_section(cur, manifesto_id: str, section_name: str) -> str:
    cur.execute(
        """
        SELECT section_id
        FROM sections
        WHERE manifesto_id = %s AND title = %s
        """,
        (manifesto_id, section_name),
    )
    rows = cur.fetchall()

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one section titled {section_name!r} for manifesto "
            f"{manifesto_id}, found {len(rows)}. Inspect the sections table; "
            "the script will not guess which section to use."
        )
    return rows[0]["section_id"]


def ingest(cur, records: list[dict[str, Any]], execute: bool) -> tuple[int, int, int]:
    inserted_promises = 0
    reused_promises = 0
    inserted_sources = 0
    section_cache: dict[tuple[str, str], str] = {}

    for record_index, record in enumerate(records, start=1):
        source = record["source"]
        manifesto_id = source["manifesto_id"]
        section_name = source["section_name"]
        section_key = (manifesto_id, section_name)

        if section_key not in section_cache:
            section_cache[section_key] = resolve_section(cur, manifesto_id, section_name)
        section_id = section_cache[section_key]

        for promise_index, promise in enumerate(record["atomic_promises"], start=1):
            # Reuse a promise only when its section and both key texts match.
            cur.execute(
                """
                SELECT promise_id
                FROM promises
                WHERE section_id = %s
                  AND original_text = %s
                  AND atomic_promise = %s
                ORDER BY created_at, promise_id
                """,
                (
                    section_id,
                    promise["original_text"],
                    promise["atomic_promise"],
                ),
            )
            matches = cur.fetchall()

            if len(matches) > 1:
                raise RuntimeError(
                    f"Ambiguous existing promise match at record {record_index}, "
                    f"promise {promise_index}. Found {len(matches)} matching database rows."
                )

            if matches:
                promise_id = matches[0]["promise_id"]
                reused_promises += 1
            else:
                promise_id = None
                if execute:
                    cur.execute(
                        """
                        INSERT INTO promises (
                            section_id,
                            original_text,
                            atomic_promise,
                            action,
                            target_value,
                            target_unit,
                            target_description,
                            timeline_start,
                            timeline_end,
                            timeline_text,
                            geography_id,
                            beneficiary,
                            responsible_department,
                            extraction_confidence
                        )
                        VALUES (
                            %(section_id)s,
                            %(original_text)s,
                            %(atomic_promise)s,
                            %(action)s,
                            %(target_value)s,
                            %(target_unit)s,
                            %(target_description)s,
                            %(timeline_start)s,
                            %(timeline_end)s,
                            %(timeline_text)s,
                            %(geography_id)s,
                            %(beneficiary)s,
                            %(responsible_department)s,
                            %(extraction_confidence)s
                        )
                        RETURNING promise_id
                        """,
                        {
                            "section_id": section_id,
                            "original_text": promise.get("original_text"),
                            "atomic_promise": promise.get("atomic_promise"),
                            "action": promise.get("action"),
                            "target_value": promise.get("target_value"),
                            "target_unit": promise.get("target_unit"),
                            "target_description": promise.get("target_description"),
                            "timeline_start": promise.get("timeline_start"),
                            "timeline_end": promise.get("timeline_end"),
                            "timeline_text": promise.get("timeline_text"),
                            # Keep unresolved geography NULL rather than guessing/creating it.
                            "geography_id": None,
                            "beneficiary": promise.get("beneficiary"),
                            "responsible_department": promise.get("responsible_department"),
                            "extraction_confidence": promise.get("extraction_confidence"),
                        },
                    )
                    promise_id = cur.fetchone()["promise_id"]
                inserted_promises += 1

            # The source passage remains separately traceable, even when multiple
            # atomic promises originate from the same passage.
            cur.execute(
                """
                SELECT promise_source_id
                FROM promise_sources
                WHERE promise_id = %s
                  AND manifesto_id = %s
                  AND section_id = %s
                  AND page_number IS NOT DISTINCT FROM %s
                  AND source_text = %s
                  AND source_order = %s
                """,
                (
                    promise_id,
                    manifesto_id,
                    section_id,
                    source.get("page_number"),
                    source.get("source_text"),
                    promise.get("source_order"),
                ),
            )
            source_link_exists = cur.fetchone() is not None

            if not source_link_exists:
                if execute:
                    cur.execute(
                        """
                        INSERT INTO promise_sources (
                            promise_id,
                            manifesto_id,
                            section_id,
                            page_number,
                            source_text,
                            source_order
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            promise_id,
                            manifesto_id,
                            section_id,
                            source.get("page_number"),
                            source.get("source_text"),
                            promise.get("source_order"),
                        ),
                    )
                inserted_sources += 1

    return inserted_promises, reused_promises, inserted_sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Commit inserts to PostgreSQL. Without this flag, run a dry run and roll back.",
    )
    args = parser.parse_args()

    try:
        records = load_records()
        total_promises = sum(len(record["atomic_promises"]) for record in records)
        print(f"Input: {DATA_PATH.relative_to(ROOT_DIR)}")
        print(f"Source records: {len(records)}")
        print(f"Atomic promises: {total_promises}")
        print(f"Mode: {'WRITE TO DATABASE' if args.execute else 'DRY RUN (no database changes)'}")

        with connect_db() as conn:
            with conn.cursor() as cur:
                # Validate all required section mappings before doing any inserts.
                section_pairs = {
                    (record["source"]["manifesto_id"], record["source"]["section_name"])
                    for record in records
                }
                for manifesto_id, section_name in sorted(section_pairs):
                    resolve_section(cur, manifesto_id, section_name)

                inserted, reused, sources = ingest(cur, records, args.execute)

                if args.execute:
                    conn.commit()
                else:
                    conn.rollback()

        print("\nResult")
        print(f"  New promise rows planned/inserted: {inserted}")
        print(f"  Existing promise rows reused:      {reused}")
        print(f"  Source links planned/inserted:     {sources}")
        if args.execute:
            print("\nSUCCESS: transaction committed.")
        else:
            print("\nDRY RUN COMPLETE: no changes were committed.")
            print("If the counts and section mappings look correct, rerun with --execute.")
        return 0

    except Exception as exc:
        print(f"\nINGESTION STOPPED: {exc}", file=sys.stderr)
        print("No transaction was committed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
