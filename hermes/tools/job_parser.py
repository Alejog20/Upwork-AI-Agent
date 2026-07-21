"""Extracts a structured `JobPost` from a raw Upwork notification email body.

Upwork doesn't publish a stable schema for its notification emails, so this
parser works off heuristics (regex patterns, common text markers) rather than
a fixed DOM structure. It degrades gracefully: everything except the job URL
and title is optional, falling back to sane defaults (see `JobPost` field
docs) when a signal is missing from the email.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta

from bs4 import BeautifulSoup

from hermes.models import BudgetRange, BudgetType, JobPost

__all__ = ["JobParseError", "parse_job_email"]

_JOB_LINK_RE = re.compile(r"/jobs/")
_JOB_ID_RE = re.compile(r"~([0-9a-zA-Z]+)")

_BUDGET_RANGE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*-\s*\$\s?([\d,]+(?:\.\d+)?)")
_BUDGET_HOURLY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*/\s*(?:hr|hour)", re.IGNORECASE)
_BUDGET_FIXED_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:fixed price|fixed)", re.IGNORECASE)

_HIRES_RE = re.compile(r"(\d+)\s+hires?", re.IGNORECASE)
_RATING_RE = re.compile(r"(\d(?:\.\d+)?)\s+of\s+5\s+stars?", re.IGNORECASE)
_PROPOSALS_LESS_THAN_RE = re.compile(r"less than\s+(\d+)\s+proposals?", re.IGNORECASE)
_PROPOSALS_COUNT_RE = re.compile(r"(\d+)\s+to\s+(\d+)\s+proposals?", re.IGNORECASE)
_PROPOSALS_EXACT_RE = re.compile(r"(\d+)\s+proposals?", re.IGNORECASE)

_POSTED_AGO_RE = re.compile(r"posted\s+(\d+)\s+(minute|hour|day|second)s?\s+ago", re.IGNORECASE)

_PAYMENT_VERIFIED_PHRASE = "payment method verified"
_SKILLS_HEADING_RE = re.compile(r"skills?\s*:?", re.IGNORECASE)


class JobParseError(Exception):
    """Raised when a job posting cannot be extracted from an email body."""


def parse_job_email(html_body: str) -> tuple[JobPost, None] | tuple[None, JobParseError]:
    """Parse a raw Upwork notification email into a structured `JobPost`.

    Args:
        html_body: The HTML body of the notification email.

    Returns:
        `(job_post, None)` on success, or `(None, error)` if the email doesn't
        contain the minimum required signal (a job link) to build a `JobPost`.
    """
    try:
        soup = BeautifulSoup(html_body, "html.parser")
        text = soup.get_text(" ", strip=True)

        link = soup.find("a", href=_JOB_LINK_RE)
        if link is None or not link.get("href"):
            raise JobParseError("No job link (`/jobs/...`) found in email body")
        url = str(link["href"])
        title = link.get_text(strip=True) or "Untitled job"

        return (
            JobPost(
                id=_extract_job_id(url),
                title=title,
                description=_extract_description(soup, link),
                budget=_extract_budget(text),
                skills_required=_extract_skills(soup, text),
                client_hires=_extract_client_hires(text),
                client_rating=_extract_client_rating(text),
                payment_verified=_PAYMENT_VERIFIED_PHRASE in text.lower(),
                proposals_count=_extract_proposals_count(text),
                posted_at=_extract_posted_at(text),
                url=url,
            ),
            None,
        )
    except JobParseError as exc:
        return None, exc


def _extract_job_id(url: str) -> str:
    match = _JOB_ID_RE.search(url)
    if match:
        return match.group(1)
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _extract_description(soup: BeautifulSoup, link: object) -> str:
    for paragraph in soup.find_all("p"):
        candidate = paragraph.get_text(" ", strip=True)
        if len(candidate) >= 40:
            return candidate
    fallback = soup.get_text(" ", strip=True)
    return fallback[:500] if fallback else ""


def _extract_budget(text: str) -> BudgetRange:
    range_match = _BUDGET_RANGE_RE.search(text)
    if range_match:
        return BudgetRange(
            type=BudgetType.FIXED,
            min_amount=_to_float(range_match.group(1)),
            max_amount=_to_float(range_match.group(2)),
        )

    hourly_match = _BUDGET_HOURLY_RE.search(text)
    if hourly_match:
        amount = _to_float(hourly_match.group(1))
        return BudgetRange(type=BudgetType.HOURLY, min_amount=amount, max_amount=amount)

    fixed_match = _BUDGET_FIXED_RE.search(text)
    if fixed_match:
        amount = _to_float(fixed_match.group(1))
        return BudgetRange(type=BudgetType.FIXED, min_amount=amount, max_amount=amount)

    return BudgetRange()


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _extract_skills(soup: BeautifulSoup, text: str) -> list[str]:
    heading = soup.find(string=_SKILLS_HEADING_RE)
    if heading is None:
        return []

    container = heading.find_parent()
    if container is None:
        return []

    items = [li.get_text(strip=True) for li in container.find_all_next("li", limit=15)]
    if items:
        return [item for item in items if item]

    # Fall back to a comma-separated skills line following the heading text.
    remainder = text[text.lower().find("skill") :]
    line_match = re.search(r"skills?\s*:?\s*([^.]{3,200})", remainder, re.IGNORECASE)
    if not line_match:
        return []
    return [skill.strip() for skill in line_match.group(1).split(",") if skill.strip()]


def _extract_client_hires(text: str) -> int:
    match = _HIRES_RE.search(text)
    return int(match.group(1)) if match else 0


def _extract_client_rating(text: str) -> float | None:
    match = _RATING_RE.search(text)
    return float(match.group(1)) if match else None


def _extract_proposals_count(text: str) -> int | None:
    less_than = _PROPOSALS_LESS_THAN_RE.search(text)
    if less_than:
        return int(less_than.group(1))

    range_match = _PROPOSALS_COUNT_RE.search(text)
    if range_match:
        return int(range_match.group(2))

    exact = _PROPOSALS_EXACT_RE.search(text)
    return int(exact.group(1)) if exact else None


def _extract_posted_at(text: str) -> datetime:
    match = _POSTED_AGO_RE.search(text)
    now = datetime.now(UTC)
    if not match:
        return now

    amount = int(match.group(1))
    unit = match.group(2).lower()
    delta = {
        "second": timedelta(seconds=amount),
        "minute": timedelta(minutes=amount),
        "hour": timedelta(hours=amount),
        "day": timedelta(days=amount),
    }[unit]
    return now - delta
