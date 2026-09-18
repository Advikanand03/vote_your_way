import re
import unicodedata
from pathlib import Path

import fitz


PDF_PATH = Path("data/raw/manifestos/c3138e80-2650-4dbf-b31c-8dafffa60de0.pdf")
OUTPUT_PATH = Path("data/processed/aap_2020_reconstructed.txt")
MULTI_COLUMN_PAGES = {4, 5, 6, 7}
INITIATIVE_HEADING = re.compile(r"^\d+\.\s")

# These glyphs are consistently used as substitutions in this PDF's embedded
# fonts. NFKC below handles standard ligature code points such as ﬁ and ﬀ.
CHARACTER_REPLACEMENTS = {
    "Ɵ": "ti",
    "Ō": "ft",
    "Ʃ": "tt",
    "\ufffe": "",
}


def clean_text(text: str) -> str:
    """Fix character-level extraction artifacts without changing wording."""
    text = unicodedata.normalize("NFKC", text)
    for malformed, replacement in CHARACTER_REPLACEMENTS.items():
        text = text.replace(malformed, replacement)
    return text.strip()


def ordered_multicolumn_blocks(page: fitz.Page) -> list[str]:
    """Return pages 4-7 in their visual left-column, then right-column order."""
    page_midpoint = page.rect.width / 2
    full_width_blocks = []
    left_column = []
    right_column = []

    for block in page.get_text("blocks"):
        x0, y0, x1, _y1, text, *_ = block
        if not text.strip():
            continue

        # The manifesto's introductory sentence spans both columns. The
        # initiatives themselves occupy fixed left and right columns.
        if x0 < page_midpoint < x1:
            full_width_blocks.append((y0, text))
        elif (x0 + x1) / 2 < page_midpoint:
            left_column.append((y0, text))
        else:
            right_column.append((y0, text))

    ordered = sorted(full_width_blocks)
    ordered.extend(sorted(left_column))
    ordered.extend(sorted(right_column))
    return [text for _y0, text in ordered]


def join_manifesto_blocks(blocks: list[str]) -> str:
    """Keep blocks distinct, except for a block continuing an initiative."""
    reconstructed = []

    for block in blocks:
        text = clean_text(block)
        if not text:
            continue

        is_continuation = (
            reconstructed
            and not INITIATIVE_HEADING.match(text)
            and not reconstructed[-1].rstrip().endswith((".", "!", "?"))
        )
        if is_continuation:
            reconstructed[-1] = f"{reconstructed[-1]} {text}"
        else:
            reconstructed.append(text)

    return "\n\n".join(reconstructed)


def reconstruct_page(page: fitz.Page, page_no: int) -> str:
    """Extract a page while applying layout reconstruction only where needed."""
    if page_no not in MULTI_COLUMN_PAGES:
        return clean_text(page.get_text("text"))

    return join_manifesto_blocks(ordered_multicolumn_blocks(page))


def reconstruct_manifesto() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(PDF_PATH)
    page_count = len(doc)

    try:
        with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
            for page_no, page in enumerate(doc, start=1):
                text = reconstruct_page(page, page_no)
                output_file.write(f"{'=' * 60}\n")
                output_file.write(f"PAGE {page_no}\n")
                output_file.write(f"{'=' * 60}\n")
                output_file.write(f"{text}\n\n")
    finally:
        doc.close()

    print(f"Pages processed: {page_count}")
    print(f"Output path: {OUTPUT_PATH}")
    print("Manifesto reconstruction completed successfully.")


if __name__ == "__main__":
    reconstruct_manifesto()
