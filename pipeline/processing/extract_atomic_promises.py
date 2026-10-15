import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

GUARANTEES_PATH = (
    BASE_DIR / "data" / "processed" / "aap_2020_guarantees.json"
)

INITIATIVES_PATH = (
    BASE_DIR / "data" / "processed" / "aap_2020_initiatives.json"
)

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "aap_2020_atomic_promises.json"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL")

if not API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")

if not MODEL:
    raise ValueError("GROQ_MODEL is not set in .env")

client = Groq(api_key=API_KEY)

MANIFESTO_ID = "30398bb3-18f7-4332-adfd-fbadbdcbfb12"


# ---------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------

ATOMIC_PROMISE_SCHEMA = {
    "type": "object",
    "properties": {
        "atomic_promises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original_text": {
                        "type": "string"
                    },
                    "atomic_promise": {
                        "type": "string"
                    },
                    "action": {
                        "type": ["string", "null"]
                    },
                    "target_value": {
                        "type": ["number", "null"]
                    },
                    "target_unit": {
                        "type": ["string", "null"]
                    },
                    "target_description": {
                        "type": ["string", "null"]
                    },
                    "timeline_start": {
                        "type": ["string", "null"]
                    },
                    "timeline_end": {
                        "type": ["string", "null"]
                    },
                    "timeline_text": {
                        "type": ["string", "null"]
                    },
                    "geography_name": {
                        "type": ["string", "null"]
                    },
                    "beneficiary": {
                        "type": ["string", "null"]
                    },
                    "responsible_department": {
                        "type": ["string", "null"]
                    },
                    "extraction_confidence": {
                        "type": "number"
                    }
                },
                "required": [
                    "original_text",
                    "atomic_promise",
                    "action",
                    "target_value",
                    "target_unit",
                    "target_description",
                    "timeline_start",
                    "timeline_end",
                    "timeline_text",
                    "geography_name",
                    "beneficiary",
                    "responsible_department",
                    "extraction_confidence"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "atomic_promises"
    ],
    "additionalProperties": False
}


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are the atomic-promise extraction component of VoteYourWay.

Your task is to extract independently verifiable political promises
from manifesto text.

IMPORTANT RULES:

1. Extract promises ONLY from the supplied source text.
2. Do NOT add facts, numbers, departments, dates, locations,
   beneficiaries, or timelines that are not supported by the source.
3. Do NOT evaluate whether a promise is realistic, fulfilled,
   legal, feasible, or truthful.
4. Do NOT verify the promise.
5. Do NOT use outside knowledge.
6. Preserve the meaning of the source.
7. Do not turn background/contextual statements into promises.
8. A promise is atomic when it can be independently evaluated as
   fulfilled, partially fulfilled, delayed, or not fulfilled.
9. If one sentence contains multiple independently verifiable
   commitments, split them.
10. Do not split a promise merely because it contains descriptive
    wording.
11. Preserve important qualifiers such as:
    - "within 6 months"
    - "over the next 5 years"
    - "pilot basis"
    - "continue"
    - "pursue"
    - "ensure"
    - "all"
    - "every"
    - numerical targets
12. If a field is not explicitly supported by the source,
    return null.
13. Never invent a PostgreSQL UUID.
14. extraction_confidence means confidence that the extracted
    atomic promise faithfully represents the source text.
15. It does NOT mean confidence that the political promise will
    actually be fulfilled.
16. Do not strengthen, weaken, or reinterpret the source.
17. Preserve the original actor, object, scope, and purpose of the
    commitment.
18. Do not convert a future-tense promise into a completed or
    differently framed claim.
19. Do not remove meaningful qualifiers, entities, or conditions
    from the source.
20. The atomic_promise should remain as close as reasonably possible
    to the wording of the source while still being atomic.
21. original_text must be copied verbatim from source_text.
"""


def build_user_prompt(source):
    return f"""
Extract the atomic promises from this manifesto source.

Source type:
{source["source_type"]}

Source number:
{source["source_number"]}

Source title:
{source["source_title"]}

Page:
{source["page_number"]}

Source text:
{source["source_text"]}

Return only the required structured JSON.

For each atomic promise:

- original_text MUST be copied verbatim from source_text.
  It must be a contiguous passage from source_text whenever possible.
  Do not paraphrase, normalize, shorten, or rewrite it.

- atomic_promise must be a concise but source-faithful formulation
  of the commitment. Preserve the actor, action, object, scope,
  conditions, qualifiers, entities, and purpose stated in the source.
  Do not add implications or conclusions that are not explicitly stated.

- action should describe the promised action when explicitly identifiable.

- target_value should contain a numeric target only when the source
  explicitly provides one.

- target_unit should describe the unit of that numeric target.

- target_description should be used for non-numeric targets or
  qualitative target conditions.

- timeline_start and timeline_end should be ISO dates only when
  explicit dates are available.

- timeline_text should preserve expressions such as "within 6 months"
  or "over the next 5 years".

- geography_name should only be populated when geography is
  explicitly supported by the source.

- beneficiary should only be populated when explicitly supported.

- responsible_department should only be populated when explicitly
  supported.
"""


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_retryable_error(error):
    """
    Return True for transient Groq/API errors worth retrying.
    """

    status_code = getattr(error, "status_code", None)

    if status_code in {429, 500, 502, 503, 504}:
        return True

    error_text = str(error).lower()

    return any(
        phrase in error_text
        for phrase in [
            "rate limit",
            "over capacity",
            "service unavailable",
            "temporarily unavailable",
            "internal server error",
            "timeout",
            "timed out"
        ]
    )


def extract_from_groq(source, max_attempts=6):

    for attempt in range(1, max_attempts + 1):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": build_user_prompt(source)
                    }
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "atomic_promise_extraction",
                        "strict": True,
                        "schema": ATOMIC_PROMISE_SCHEMA
                    }
                }
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError(
                    f"Groq returned empty response for "
                    f"{source['source_type']} {source['source_number']}"
                )

            return json.loads(content)

        except Exception as error:

            # Do not retry permanent errors.
            if not is_retryable_error(error):
                raise

            # Maximum number of attempts reached.
            if attempt == max_attempts:
                raise

            # 2, 4, 8, 16, 32 seconds.
            wait_seconds = 2 ** attempt

            print(
                f"Transient Groq error on "
                f"{source['source_type']} "
                f"{source['source_number']} "
                f"(attempt {attempt}/{max_attempts}):"
            )

            print(f"  {error}")
            print(f"Retrying in {wait_seconds} seconds...")

            time.sleep(wait_seconds)


def validate_result(source, result):

    promises = result.get("atomic_promises")

    if not isinstance(promises, list):
        raise ValueError(
            f"Invalid atomic_promises for "
            f"{source['source_type']} {source['source_number']}"
        )

    for promise in promises:

        confidence = promise["extraction_confidence"]

        if not 0 <= confidence <= 1:
            raise ValueError(
                f"Invalid confidence {confidence} for "
                f"{source['source_type']} "
                f"{source['source_number']}"
            )

        if not promise["atomic_promise"].strip():
            raise ValueError(
                f"Empty atomic promise for "
                f"{source['source_type']} "
                f"{source['source_number']}"
            )

        if not promise["original_text"].strip():
            raise ValueError(
                f"Empty original_text for "
                f"{source['source_type']} "
                f"{source['source_number']}"
            )


def source_key(source):
    """
    Create a unique identifier for a manifesto source.
    """

    return f"{source['source_type']}:{source['source_number']}"


def load_checkpoint():
    """
    Load previously completed extraction results if available.
    """

    if not OUTPUT_PATH.exists():
        return []

    try:

        data = load_json(OUTPUT_PATH)

    except (json.JSONDecodeError, OSError) as error:

        print(
            f"WARNING: Could not load existing checkpoint: {error}"
        )

        print("Starting with an empty checkpoint.")

        return []

    if not isinstance(data, list):

        print(
            "WARNING: Existing output is not a list. "
            "Starting fresh."
        )

        return []

    valid_results = []

    for item in data:

        if not isinstance(item, dict):
            continue

        source = item.get("source", {})

        if (
            source.get("source_type")
            and source.get("source_number") is not None
            and isinstance(item.get("atomic_promises"), list)
        ):
            valid_results.append(item)

    return valid_results


def save_checkpoint(results):
    """
    Persist all completed results immediately.

    A temporary file is written first so that an interrupted write
    does not corrupt the existing checkpoint.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = OUTPUT_PATH.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    temporary_path.replace(OUTPUT_PATH)


def process_source(source):

    print(
        f"Extracting: "
        f"{source['source_type']} "
        f"{source['source_number']} - "
        f"{source['source_title']}"
    )

    result = extract_from_groq(source)

    validate_result(source, result)

    atomic_promises = []

    for order, promise in enumerate(
        result["atomic_promises"],
        start=1
    ):

        promise["source_order"] = order
        promise["geography_id"] = None

        atomic_promises.append(promise)

    return {
        "source": {
            "source_type": source["source_type"],
            "source_number": source["source_number"],
            "source_title": source["source_title"],
            "manifesto_id": MANIFESTO_ID,
            "section_name": source["section_name"],
            "section_id": None,
            "page_number": source["page_number"],
            "source_text": source["source_text"]
        },
        "atomic_promises": atomic_promises
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    guarantees = load_json(GUARANTEES_PATH)
    initiatives = load_json(INITIATIVES_PATH)

    sources = []

    # -----------------------------
    # Guarantees
    # -----------------------------

    for guarantee in guarantees:

        source_text = " ".join(
            guarantee["promises"]
        )

        sources.append(
            {
                "source_type": "guarantee",
                "source_number": guarantee["guarantee_number"],
                "source_title": guarantee["title"],
                "section_name": "10 Guarantees",
                "page_number": 3,
                "source_text": source_text
            }
        )

    # -----------------------------
    # Initiatives
    # -----------------------------

    for initiative in initiatives:

        sources.append(
            {
                "source_type": "initiative",
                "source_number": initiative["initiative_number"],
                "source_title": initiative["title"],
                "section_name": "Landmark Initiatives",
                "page_number": initiative["page_number"],
                "source_text": initiative["original_text"]
            }
        )

    print(f"Total sources: {len(sources)}")
    print()

    # -----------------------------
    # Load checkpoint
    # -----------------------------

    results = load_checkpoint()

    completed_keys = {
        source_key(item["source"])
        for item in results
    }

    if results:

        print(
            f"Loaded checkpoint with "
            f"{len(results)} completed sources."
        )

        print()

    skipped_count = 0
    failed_sources = []

    # -----------------------------
    # Process sources
    # -----------------------------

    for source in sources:

        key = source_key(source)

        # Skip sources already successfully processed.
        if key in completed_keys:

            print(
                f"Skipping already completed: "
                f"{source['source_type']} "
                f"{source['source_number']} - "
                f"{source['source_title']}"
            )

            skipped_count += 1

            continue

        try:

            result = process_source(source)

            results.append(result)
            completed_keys.add(key)

            # IMPORTANT:
            # Save immediately after every successful source.
            save_checkpoint(results)

            print(
                f"Saved checkpoint: "
                f"{len(results)}/{len(sources)} "
                f"sources complete."
            )

            print()

        except Exception as error:

            print(
                f"ERROR processing "
                f"{source['source_type']} "
                f"{source['source_number']}: {error}"
            )

            print()
            print(
                "Progress has been saved. "
                "Run the script again later to resume."
            )

            failed_sources.append(
                {
                    "source_type": source["source_type"],
                    "source_number": source["source_number"],
                    "source_title": source["source_title"],
                    "error": str(error)
                }
            )

            # Stop at the failed source.
            # Previously completed sources remain saved.
            break

        # Avoid hitting API rate limits.
        time.sleep(2.2)

    # -----------------------------
    # Final summary
    # -----------------------------

    total_promises = sum(
        len(item["atomic_promises"])
        for item in results
    )

    print()
    print("=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)

    print(
        f"Total sources:     {len(sources)}"
    )

    print(
        f"Completed sources: {len(results)}"
    )

    print(
        f"Skipped sources:   {skipped_count}"
    )

    print(
        f"Failed sources:    {len(failed_sources)}"
    )

    print(
        f"Atomic promises:   {total_promises}"
    )

    print(
        f"Saved to:          {OUTPUT_PATH}"
    )

    if failed_sources:

        print()
        print("Failed source:")

        for failed in failed_sources:

            print(
                f"- {failed['source_type']} "
                f"{failed['source_number']} - "
                f"{failed['source_title']}"
            )

    if len(results) == len(sources):

        print()
        print(
            "All manifesto sources processed successfully."
        )

    else:

        print()
        print(
            "Run the script again later to resume "
            "from the checkpoint."
        )


if __name__ == "__main__":
    main()