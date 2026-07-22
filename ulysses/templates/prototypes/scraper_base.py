"""Demo scraper skeleton -- TODO markers show where job-specific logic goes.

Run with: python demo.py
"""

from __future__ import annotations

import csv

import requests
from bs4 import BeautifulSoup

# TODO: set this to the real page being scraped for this job.
TARGET_URL = "https://example.com"


def fetch_page(url: str) -> str:
    """Fetch a page's HTML, raising on any non-2xx response."""
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.text


def parse_listings(html: str) -> list[dict[str, str]]:
    """Parse the page HTML into a list of structured records.

    TODO: replace the selectors below with real ones for the target site --
    inspect the page's HTML and adjust `.select(...)`/`.select_one(...)` calls.
    """
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, str]] = []
    for item in soup.select(".listing"):
        title_el = item.select_one(".title")
        records.append({"title": title_el.get_text(strip=True) if title_el else ""})
    return records


def save_to_csv(records: list[dict[str, str]], path: str = "output.csv") -> None:
    """Write records to a CSV file, deduplicated by their full field set."""
    if not records:
        print("No records found -- nothing to save.")
        return

    seen: set[tuple[str, ...]] = set()
    deduped = []
    for record in records:
        key = tuple(record.values())
        if key not in seen:
            seen.add(key)
            deduped.append(record)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(deduped[0].keys()))
        writer.writeheader()
        writer.writerows(deduped)
    print(f"Saved {len(deduped)} record(s) to {path}")


def main() -> None:
    html = fetch_page(TARGET_URL)
    records = parse_listings(html)
    save_to_csv(records)


if __name__ == "__main__":
    main()
