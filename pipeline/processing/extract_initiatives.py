import json
import re
from pathlib import Path

INPUT_PATH = Path("data/processed/aap_2020_reconstructed.txt")
OUTPUT_PATH = Path("data/processed/aap_2020_initiatives.json")


def get_initiative_pages(text):
    """Return the text belonging to pages 4-7."""

    page4 = re.search(
        r"={60}\s*PAGE 4\s*={60}\s*(.*?)(?=={60}\s*PAGE 8\s*={60})",
        text,
        re.DOTALL,
    )

    if not page4:
        raise ValueError("Could not find pages 4-7 in reconstructed text.")

    return page4.group(1).strip()


def find_headers(text):
    """
    Find initiative numbers at the beginning of lines.

    The title itself is not assumed to be on one line.
    """

    pattern = re.compile(
        r"(?m)^\s*(\d{1,2})\.\s+"
    )

    matches = list(pattern.finditer(text))

    # Keep only numbers 1-28.
    matches = [
        match
        for match in matches
        if 1 <= int(match.group(1)) <= 28
    ]

    return matches


def clean_text(text):
    """Normalize PDF line wrapping without changing wording."""

    text = text.strip()

    # Replace repeated spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Join lines caused by PDF text wrapping.
    text = re.sub(r"\s*\n\s*", " ", text)

    return text.strip()


def extract_title_and_body(block):
    """
    Split an initiative block into title and body.

    The title ends at the first colon.
    """

    block = clean_text(block)

    colon_position = block.find(":")

    if colon_position == -1:
        raise ValueError(
            f"Could not find title/body separator in:\n{block[:200]}"
        )

    title = block[:colon_position].strip()
    body = block[colon_position + 1:].strip()

    return title, body


def determine_page(text, position):
    """Determine whether a position belongs to page 4, 5, 6 or 7."""

    before = text[:position]

    pages = re.findall(
        r"={60}\s*PAGE\s+(\d+)\s*={60}",
        before
    )

    if not pages:
        return None

    return int(pages[-1])


def extract_initiatives(text):
    initiative_text = get_initiative_pages(text)

    headers = find_headers(initiative_text)

    print(f"Detected numbered headers: {len(headers)}")

    if len(headers) != 28:
        detected = [int(match.group(1)) for match in headers]

        raise ValueError(
            f"Expected 28 initiatives, but found {len(headers)}.\n"
            f"Detected numbers: {detected}"
        )

    initiatives = []

    for index, header in enumerate(headers):

        number = int(header.group(1))

        start = header.end()

        if index + 1 < len(headers):
            end = headers[index + 1].start()
        else:
            end = len(initiative_text)

        block = initiative_text[start:end].strip()

        title, body = extract_title_and_body(block)

        page_number = determine_page(
            initiative_text,
            header.start()
        )

        initiatives.append(
            {
                "initiative_number": number,
                "title": title,
                "page_number": page_number,
                "original_text": f"{title}: {body}",
            }
        )

    numbers = [
        item["initiative_number"]
        for item in initiatives
    ]

    expected = list(range(1, 29))

    if numbers != expected:
        raise ValueError(
            f"Initiative numbering is incorrect.\n"
            f"Expected: {expected}\n"
            f"Found: {numbers}"
        )

    return initiatives


def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    text = INPUT_PATH.read_text(
        encoding="utf-8"
    )

    initiatives = extract_initiatives(text)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            initiatives,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"Extracted {len(initiatives)} initiatives.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()