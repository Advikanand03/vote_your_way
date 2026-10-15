import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

if not MODEL:
    raise ValueError("GROQ_MODEL not found in .env")


# ---------------------------------------------------------
# Groq client
# ---------------------------------------------------------

client = Groq(api_key=API_KEY)


# ---------------------------------------------------------
# Test source: Initiative 27
# ---------------------------------------------------------

source = {
    "source_type": "initiative",
    "source_number": 27,
    "source_title": "LEGAL PROTECTION FOR STREET VENDORS",
    "page_number": 7,
    "source_text": (
        "LEGAL PROTECTION FOR STREET VENDORS: "
        "Delhi will become the first state in India to give legal "
        "protection to street vendors and hawkers. "
        "We will issue the certificate of vending within 6 months "
        "and set up international quality kiosks to prevent the "
        "harassment of street vendors at the hands of MCD and "
        "Delhi Police."
    )
}


# ---------------------------------------------------------
# JSON schema
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

Extract independently verifiable political promises from the
supplied manifesto text.

Rules:

1. Extract ONLY from the supplied source text.
2. Do not use outside knowledge.
3. Do not invent facts, numbers, departments, dates, beneficiaries,
   locations, or timelines.
4. Do not evaluate whether a promise is realistic, legal, fulfilled,
   or feasible.
5. Preserve the meaning of the source.
6. Ignore background/contextual statements that are not promises.
7. Split multiple independently verifiable commitments into separate
   atomic promises.
8. Preserve important qualifiers such as "within 6 months",
   "all", "every", "continue", and numerical targets.
9. If a field is not supported by the source, return null.
10. extraction_confidence measures confidence that the extraction
    faithfully represents the source. It is NOT verification confidence.
11. Do not strengthen, weaken, or reinterpret the source.
12. Preserve the original actor, object, scope, and purpose of the
    commitment.
13. Do not convert a future-tense promise into a completed or
    differently framed claim.
14. Do not remove meaningful qualifiers, entities, or conditions
    from the source.
15. The atomic_promise should remain as close as reasonably possible
    to the wording of the source while still being atomic.
16. original_text must be an exact contiguous passage from the
    supplied source text whenever possible. Do not paraphrase it.
"""


USER_PROMPT = f"""
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

For each atomic promise:

- original_text: the source portion supporting the promise
- atomic_promise: a concise but source-faithful formulation of the commitment. Preserve the actor, action, object, scope, conditions, qualifiers, and purpose stated in the source. Do not add implications or conclusions that are not explicitly stated.
- action: promised action, if identifiable
- target_value: numeric target only when explicitly stated
- target_unit: unit of the numeric target
- target_description: non-numeric target when appropriate
- timeline_start/end: explicit dates only
- timeline_text: wording such as "within 6 months"
- geography_name: only if explicitly supported
- beneficiary: only if explicitly supported
- responsible_department: only if explicitly supported
"""


# ---------------------------------------------------------
# API call
# ---------------------------------------------------------

print(f"Testing model: {MODEL}")
print("Testing Initiative 27...")
print()

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": USER_PROMPT
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


# ---------------------------------------------------------
# Parse response
# ---------------------------------------------------------

content = response.choices[0].message.content

if not content:
    raise ValueError("Groq returned an empty response.")

result = json.loads(content)


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------

print("SUCCESS: Groq API call completed.")
print()
print("Atomic promise extraction result:")
print(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )
)