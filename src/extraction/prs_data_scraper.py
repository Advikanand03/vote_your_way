"""
PRS India – Karnataka Bills & Acts Scraper (Robust Version)
==========================================================

Improvements:
- Safe pagination (no infinite loops)
- Deduplication across pages
- Faster scraping
- Covers BOTH bills and acts for 2023–2026

Output:
  data/prs_datasets/prs_karnataka_bills.csv
  data/prs_datasets/prs_karnataka_acts.csv
"""

import csv
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ── Configuration ────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; research-scraper/1.0; "
        "+https://github.com/your-repo)"
    )
}

BASE_BILLS = "https://prsindia.org/bills/states"
BASE_ACTS  = "https://prsindia.org/acts/states"
PRS_OUTPUT_DIR = "data/prs_datasets"

YEARS = list(range(2023, 2027))  # 2023–2026

STATE      = "Karnataka"
DELAY_SEC  = 0.5   # polite delay between requests

def _clean_text(s: str) -> str:
    return " ".join((s or "").split())

# ── Core helpers ─────────────────────────────────────────────────────────────

def fetch_page(base_url: str, state: str, year: int, page: int = 1) -> BeautifulSoup:
    """Fetch one results page and return a BeautifulSoup object. Note: page numbering starts at 1."""
    params = {"title": "", "state": state, "year": year, "page": page}
    last_err = None
    for _ in range(3):
        try:
            resp = requests.get(base_url, params=params, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise last_err


def parse_items(soup: BeautifulSoup, year: int, state: str, base_url: str) -> list[dict]:
    """
    Parse bill/act entries from the page.

    PRS renders each item as an <h3> tag containing an <a> with:
      • text  → title
      • href  → PDF link
    """
    records = []
    seen = set()

    # All result cards live inside the main content area
    # Each item: <h3><a href="...pdf" title="...">Title</a></h3>
    for h3 in soup.select("h3 a[href]"):
        raw_title = h3.get_text(strip=True)
        title = _clean_text(raw_title)
        pdf_url = urljoin("https://prsindia.org", h3["href"])
        key = (title, pdf_url)
        if not title or key in seen:
            continue
        seen.add(key)

        parent = h3.find_parent("h3")
        state_label = state
        if parent:
            sibling = parent.find_next_sibling()
            if sibling:
                candidate = _clean_text(sibling.get_text(strip=True))
                # PRS sometimes shows a button-like state label; keep short tokens
                if candidate and len(candidate) < 50:
                    state_label = candidate

        records.append({
            "year":    year,
            "state":   state_label,
            "title":   title,
            "pdf_url": pdf_url,
            "source":  "prs",
        })

    return records


# ── Public API ────────────────────────────────────────────────────────────────

def scrape_bills(state: str = STATE, years: list[int] = None, delay: float = DELAY_SEC) -> list[dict]:
    years = years or YEARS
    all_rows = []

    for year in years:
        print(f"  [bills] {state} {year} …", end=" ", flush=True)

        page = 1
        max_pages = 20
        total = 0
        seen_titles = set()

        while page <= max_pages:
            soup = fetch_page(BASE_BILLS, state, year, page)
            rows = parse_items(soup, year, state, BASE_BILLS)

            if not rows:
                break

            new_rows = []
            for r in rows:
                if r["title"] not in seen_titles:
                    seen_titles.add(r["title"])
                    new_rows.append(r)

            if not new_rows:
                break

            total += len(new_rows)
            all_rows.extend(new_rows)

            page += 1
            time.sleep(delay)

        print(f"{total} found")

    return all_rows


def scrape_acts(state: str = STATE, years: list[int] = None, delay: float = DELAY_SEC) -> list[dict]:
    years = years or YEARS
    all_rows = []

    for year in years:
        print(f"  [acts]  {state} {year} …", end=" ", flush=True)

        page = 1
        max_pages = 20
        total = 0
        seen_titles = set()

        while page <= max_pages:
            soup = fetch_page(BASE_ACTS, state, year, page)
            rows = parse_items(soup, year, state, BASE_ACTS)

            if not rows:
                break

            new_rows = []
            for r in rows:
                if r["title"] not in seen_titles:
                    seen_titles.add(r["title"])
                    new_rows.append(r)

            if not new_rows:
                break

            total += len(new_rows)
            all_rows.extend(new_rows)

            page += 1
            time.sleep(delay)

        print(f"{total} found")

    return all_rows


def save_csv(rows: list[dict], filepath: str) -> None:
    """Write rows to a CSV file."""
    if not rows:
        print(f"  ⚠  No data to write → {filepath}")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Deduplicate across all rows
    uniq = {}
    for r in rows:
        k = (r.get("title"), r.get("pdf_url"))
        if k not in uniq:
            uniq[k] = r
    rows = list(uniq.values())

    fieldnames = ["year", "state", "title", "pdf_url", "source"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✓  Saved {len(rows)} rows → {filepath}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PRS India Scraper – Karnataka ===\n")

    print("Scraping Bills …")
    bills = scrape_bills()
    save_csv(bills, os.path.join(PRS_OUTPUT_DIR, "prs_karnataka_bills.csv"))

    print("\nScraping Acts …")
    acts = scrape_acts()
    save_csv(acts, os.path.join(PRS_OUTPUT_DIR, "prs_karnataka_acts.csv"))

    print(f"\nSummary: {len(bills)} bills, {len(acts)} acts (before de-duplication in save).")
    print("\nDone.")