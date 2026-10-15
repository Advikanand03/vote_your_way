#!/usr/bin/env python3
"""Validate and perform only safe, deterministic cleanup of extracted promises.

Place this file at: pipeline/processing/validate_atomic_promises.py
Run from the repository root:
    python pipeline/processing/validate_atomic_promises.py

Reads:  data/processed/aap_2020_atomic_promises.json
Writes: data/processed/aap_2020_atomic_promises_clean.json
        data/processed/aap_2020_atomic_promises_validation_report.txt

The input file is never modified. Ambiguous semantic issues are reported for
manual review rather than silently deleting or rewriting promises.
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "aap_2020_atomic_promises_corrected.json"
CLEAN_PATH = ROOT / "data" / "processed" / "aap_2020_atomic_promises_clean.json"
REPORT_PATH = ROOT / "data" / "processed" / "aap_2020_atomic_promises_validation_report.txt"

REQUIRED_PROMISE_FIELDS = (
    "original_text",
    "atomic_promise",
    "action",
    "target_value",
    "target_unit",
    "target_description",
    "timeline_start",
    "timeline_end",
    "timeline_text",
    "geography_id",
    "beneficiary",
    "responsible_department",
    "extraction_confidence",
    "source_order",
)


def normalize_for_comparison(value):
    """Normalize whitespace and common Unicode dash variants for comparisons."""
    value = str(value or "").replace("\u00a0", " ")
    value = re.sub(r"[‐‑‒–—−]", "-", value)
    return re.sub(r"\s+", " ", value).strip()


def add_issue(issues, severity, source_label, message):
    issues.append(f"[{severity}] {source_label}: {message}")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Make sure aap_2020_atomic_promises.json is in data/processed/."
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError("Expected the input JSON root to be a list of source records.")

    cleaned_records = json.loads(json.dumps(records, ensure_ascii=False))
    issues = []
    stats = Counter()
    seen_originals = set()

    for record_index, record in enumerate(cleaned_records, start=1):
        source = record.get("source") or {}
        promises = record.get("atomic_promises")
        source_type = source.get("source_type", "unknown")
        source_number = source.get("source_number", "?")
        source_title = str(source.get("source_title") or "").strip()
        source_text = str(source.get("source_text") or "")
        label = f"record {record_index} ({source_type} {source_number}: {source_title})"

        if not source_text.strip():
            add_issue(issues, "ERROR", label, "source_text is missing or empty.")
        if not isinstance(promises, list):
            add_issue(issues, "ERROR", label, "atomic_promises must be a list.")
            continue
        if not promises:
            add_issue(issues, "WARNING", label, "no atomic promises were extracted.")

        for promise_index, promise in enumerate(promises, start=1):
            stats["promises_checked"] += 1
            promise_label = f"{label}, promise {promise_index}"

            if not isinstance(promise, dict):
                add_issue(issues, "ERROR", promise_label, "promise entry is not an object.")
                continue

            for field in REQUIRED_PROMISE_FIELDS:
                if field not in promise:
                    add_issue(issues, "ERROR", promise_label, f"missing required field '{field}'.")

            original = str(promise.get("original_text") or "").strip()
            atomic = str(promise.get("atomic_promise") or "").strip()
            if not original:
                add_issue(issues, "ERROR", promise_label, "original_text is empty.")
            if not atomic:
                add_issue(issues, "ERROR", promise_label, "atomic_promise is empty.")

            # Safe cleanup: remove a source heading accidentally prepended to a promise.
            # The source_text and atomic_promise are left untouched.
            prefix = source_title + ":"
            if source_title and original.casefold().startswith(prefix.casefold()):
                remainder = original[len(prefix):].strip()
                if remainder and normalize_for_comparison(remainder) in normalize_for_comparison(source_text):
                    promise["original_text"] = remainder
                    original = remainder
                    stats["safe_title_prefix_fixes"] += 1
                    add_issue(
                        issues, "FIXED", promise_label,
                        "removed source_title prefix from original_text; the cleaned text remains traceable to source_text.",
                    )

            # Strict exact-substring check, plus a normalized check to explain formatting-only mismatches.
            if original and original not in source_text:
                if normalize_for_comparison(original) in normalize_for_comparison(source_text):
                    add_issue(
                        issues, "WARNING", promise_label,
                        "original_text is not an exact source_text substring but matches after whitespace/dash normalization; review source fidelity.",
                    )
                else:
                    add_issue(
                        issues, "ERROR", promise_label,
                        "original_text could not be found in source_text; check for paraphrasing, added text, or an incorrect source passage.",
                    )

            confidence = promise.get("extraction_confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
                add_issue(issues, "ERROR", promise_label, "extraction_confidence must be a number from 0 to 1.")

            order = promise.get("source_order")
            if not isinstance(order, int) or isinstance(order, bool) or order < 1:
                add_issue(issues, "ERROR", promise_label, "source_order must be a positive integer.")

            key = (source_type, source_number, normalize_for_comparison(original).casefold())
            if original and key in seen_originals:
                add_issue(issues, "WARNING", promise_label, "duplicate original_text detected for this source type/number.")
            elif original:
                seen_originals.add(key)

            # Semantic review flags: do not auto-edit because these require judgment.
            original_norm = normalize_for_comparison(original).casefold()
            atomic_norm = normalize_for_comparison(atomic).casefold()

            if "over 2 crore trees" in original_norm and "over" not in atomic_norm:
                add_issue(
                    issues, "REVIEW", promise_label,
                    "source says 'Over 2 crore trees'; ensure atomic_promise preserves 'over' (numeric target_value alone cannot represent this qualifier).",
                )
            if "government or private" in original_norm and "government or private" not in atomic_norm:
                add_issue(issues, "REVIEW", promise_label, "atomic_promise may have dropped the 'government or private' qualifier.")
            if "as per their wishes" in original_norm and "as per their wishes" not in atomic_norm:
                add_issue(issues, "REVIEW", promise_label, "atomic_promise may have dropped the purpose 'as per their wishes'.")
            if "using its social, political and moral authority" in original_norm and "using its social, political and moral authority" not in atomic_norm:
                add_issue(issues, "REVIEW", promise_label, "atomic_promise may have dropped 'using its social, political and moral authority'.")

            # This is a known contextual/descriptive sentence in Initiative 28, not clearly a commitment.
            if source_type == "initiative" and str(source_number) == "28":
                if "the rest of the areas will transition to a full state" in atomic_norm:
                    add_issue(
                        issues, "REVIEW", promise_label,
                        "likely contextual description of the proposed bill rather than a direct AAP commitment; manually review whether to exclude it.",
                    )

    # Check source_order sequences independently for each source record.
    for record_index, record in enumerate(cleaned_records, start=1):
        source = record.get("source") or {}
        promises = record.get("atomic_promises")
        if not isinstance(promises, list):
            continue
        orders = [p.get("source_order") for p in promises if isinstance(p, dict) and isinstance(p.get("source_order"), int)]
        expected = list(range(1, len(promises) + 1))
        if orders != expected:
            label = f"record {record_index} ({source.get('source_type', 'unknown')} {source.get('source_number', '?')}: {source.get('source_title', '')})"
            add_issue(issues, "WARNING", label, f"source_order values are {orders}; expected sequential values {expected}.")

    # Keep output separate so the original extraction is preserved.
    with CLEAN_PATH.open("w", encoding="utf-8") as file:
        json.dump(cleaned_records, file, ensure_ascii=False, indent=2)
        file.write("\n")

    error_count = sum(line.startswith("[ERROR]") for line in issues)
    warning_count = sum(line.startswith("[WARNING]") for line in issues)
    review_count = sum(line.startswith("[REVIEW]") for line in issues)
    fixed_count = sum(line.startswith("[FIXED]") for line in issues)

    report_lines = [
        "VOTEYOURWAY ATOMIC PROMISE VALIDATION REPORT",
        "=" * 44,
        f"Input:  {INPUT_PATH.relative_to(ROOT)}",
        f"Cleaned output: {CLEAN_PATH.relative_to(ROOT)}",
        f"Sources checked: {len(cleaned_records)}",
        f"Promises checked: {stats['promises_checked']}",
        f"Safe title-prefix fixes: {stats['safe_title_prefix_fixes']}",
        f"Errors: {error_count}",
        f"Warnings: {warning_count}",
        f"Manual semantic reviews: {review_count}",
        "",
    ]
    report_lines.extend(issues if issues else ["No issues found."])
    report_lines += [
        "",
        "Interpretation:",
        "- FIXED: safe deterministic cleanup was applied to the separate cleaned JSON.",
        "- ERROR: fix before database ingestion.",
        "- WARNING/REVIEW: inspect manually; this script deliberately does not guess or delete promises.",
        "- The original input JSON has not been modified.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Validation finished.")
    print(f"Sources checked: {len(cleaned_records)}")
    print(f"Promises checked: {stats['promises_checked']}")
    print(f"Safe title-prefix fixes: {stats['safe_title_prefix_fixes']}")
    print(f"Errors: {error_count} | Warnings: {warning_count} | Manual reviews: {review_count}")
    print(f"Cleaned JSON: {CLEAN_PATH.relative_to(ROOT)}")
    print(f"Report:       {REPORT_PATH.relative_to(ROOT)}")
    print("Open the report and resolve ERROR items and relevant REVIEW items before database ingestion.")


if __name__ == "__main__":
    main()
