"""Extracts a structured `JobPost` from a raw Upwork notification email body.

Upwork doesn't publish a stable schema for its notification emails, so this
parser works off heuristics (regex patterns, common text markers) rather than
a fixed DOM structure -- and that template has already changed at least once
underneath this project. Each field is tried against the current template's
shape first, falling back to an older shape so this degrades gracefully
rather than regressing if Upwork reverts or A/B-tests the email body.
Everything except the job URL and title is optional, falling back to sane
defaults (see `JobPost` field docs) when a signal is missing from the email.

Current template, confirmed against live "New job alert: ..." emails
(2026-10): "Posted <title> Fixed-price • $200.00 <description>… more
<skill pill> <skill pill> Payment verified • 4.39 • $770 spent •
United States". Skill pills are `<a href="https://www.upwork.com/nx/search/
jobs/?...">` tags, not a text heading. There is no relative "posted N ago"
phrase in this template at all -- the job is always described as just
posted, so `posted_at` falls back to the *email's own* Date header (passed in
as `received_at`) rather than "now", which matters enormously once an email
sits unread for days or weeks before Ulysses processes it. Client hires and
proposal count aren't present in this template; those stay at their defaults
(0 / `None`) until Upwork's email includes them again.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup

from ulysses.models import BudgetRange, BudgetType, JobPost

__all__ = ["JobParseError", "parse_job_email"]

_JOB_LINK_RE = re.compile(r"/jobs/")
_JOB_ID_RE = re.compile(r"~([0-9a-zA-Z]+)")
_SKILL_PILL_LINK_RE = re.compile(r"/nx/search/jobs/")

_BUDGET_RANGE_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*-\s*\$\s?([\d,]+(?:\.\d+)?)")
_BUDGET_HOURLY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*/\s*(?:hr|hour)", re.IGNORECASE)
_BUDGET_FIXED_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:fixed price|fixed)", re.IGNORECASE)
_BUDGET_FIXED_LABEL_FIRST_RE = re.compile(
    r"Fixed-price\s*•?\s*\$\s?([\d,]+(?:\.\d+)?)", re.IGNORECASE
)
# "Hourly • $20.00 - $35.00" / "Hourly • $25.00/hr" -- inferred by symmetry
# with the confirmed fixed-price label; no live hourly sample was available
# to verify against.
_BUDGET_HOURLY_LABEL_FIRST_RE = re.compile(
    r"Hourly\s*•?\s*\$\s?([\d,]+(?:\.\d+)?)\s*(?:-\s*\$\s?([\d,]+(?:\.\d+)?))?", re.IGNORECASE
)

_HIRES_RE = re.compile(r"(\d+)\s+hires?", re.IGNORECASE)
_RATING_RE = re.compile(r"(\d(?:\.\d+)?)\s+of\s+5\s+stars?", re.IGNORECASE)
_PROPOSALS_LESS_THAN_RE = re.compile(r"less than\s+(\d+)\s+proposals?", re.IGNORECASE)
_PROPOSALS_COUNT_RE = re.compile(r"(\d+)\s+to\s+(\d+)\s+proposals?", re.IGNORECASE)
_PROPOSALS_EXACT_RE = re.compile(r"(\d+)\s+proposals?", re.IGNORECASE)

_POSTED_AGO_RE = re.compile(r"posted\s+(\d+)\s+(minute|hour|day|second)s?\s+ago", re.IGNORECASE)

_PAYMENT_VERIFIED_PHRASES = ("payment method verified", "payment verified")
_SKILLS_HEADING_RE = re.compile(r"skills?\s*:?", re.IGNORECASE)
_JOB_DESCRIPTION_LABEL_RE = re.compile(r"job description\s*:?", re.IGNORECASE)
_TRAILING_MORE_LINK_RE = re.compile(r"\s*more\s*$", re.IGNORECASE)


class JobParseError(Exception):
    """Raised when a job posting cannot be extracted from an email body."""


def parse_job_email(
    html_body: str, received_at: datetime | None = None
) -> tuple[JobPost, None] | tuple[None, JobParseError]:
    """Parse a raw Upwork notification email into a structured `JobPost`.

    Args:
        html_body: The HTML body of the notification email.
        received_at: The email's own Date header, parsed. Used as the
            fallback reference point for `posted_at` when the body has no
            explicit "posted N ago" phrase (the current template never
            does) -- without this, an email sitting unread for weeks would
            otherwise look freshly posted *today*. Defaults to `None`,
            which falls back further to the moment this function runs
            (matches old behavior, and keeps existing callers/tests valid).

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
        url = _strip_tracking_params(str(link["href"]))
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
                payment_verified=any(
                    phrase in text.lower() for phrase in _PAYMENT_VERIFIED_PHRASES
                ),
                proposals_count=_extract_proposals_count(text),
                posted_at=_extract_posted_at(text, received_at),
                url=url,
            ),
            None,
        )
    except JobParseError as exc:
        return None, exc


def _strip_tracking_params(url: str) -> str:
    """Drop the query string and fragment from a job URL.

    Upwork's email links carry mailgun/utm tracking params (raw `&`
    separators) that have no bearing on the actual job page -- stripping
    them gives a clean, stable URL that's safe to interpolate into an
    HTML-parse-mode Telegram message (an unescaped `&` there risks the
    message being rejected as an invalid HTML entity) and that stays
    identical across repeat sends of the same job, unlike the tracking
    params, which can vary per email.
    """
    scheme, netloc, path, _query, _fragment = urlsplit(url)
    return urlunsplit((scheme, netloc, path, "", ""))


def _extract_job_id(url: str) -> str:
    match = _JOB_ID_RE.search(url)
    if match:
        return match.group(1)
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _extract_description(soup: BeautifulSoup, link: object) -> str:
    # Current template: the description sits in a plain, unclassed <div>,
    # truncated with an inline "more" link (to the full job page) right
    # after it -- the only reliable landmark since there's no heading or
    # class to select on. The block's own text is prefixed with the job
    # title run directly into a literal "Job Description:" label with no
    # separator, so that prefix is stripped when present.
    more_link = soup.find("a", string=lambda s: bool(s) and s.strip().lower() == "more")
    if more_link is not None:
        parent = more_link.find_parent()
        if parent is not None:
            block_text = _TRAILING_MORE_LINK_RE.sub("", parent.get_text(" ", strip=True))
            label_match = _JOB_DESCRIPTION_LABEL_RE.search(block_text)
            if label_match:
                block_text = block_text[label_match.end() :].strip()
            if block_text:
                return block_text

    for paragraph in soup.find_all("p"):
        candidate = paragraph.get_text(" ", strip=True)
        if len(candidate) >= 40:
            return candidate

    fallback = soup.get_text(" ", strip=True)
    return fallback[:500] if fallback else ""


def _extract_budget(text: str) -> BudgetRange:
    # Checked before the generic "$X - $Y" range match below: a bare range
    # pattern would otherwise misclassify "Hourly • $20.00 - $35.00" as a
    # FIXED budget, since the dollar-range shape alone doesn't say which.
    hourly_label_match = _BUDGET_HOURLY_LABEL_FIRST_RE.search(text)
    if hourly_label_match:
        min_amount = _to_float(hourly_label_match.group(1))
        max_amount = (
            _to_float(hourly_label_match.group(2)) if hourly_label_match.group(2) else min_amount
        )
        return BudgetRange(type=BudgetType.HOURLY, min_amount=min_amount, max_amount=max_amount)

    # Current template: "Fixed-price • $200.00" (label, then amount) --
    # confirmed against live emails.
    label_first_match = _BUDGET_FIXED_LABEL_FIRST_RE.search(text)
    if label_first_match:
        amount = _to_float(label_first_match.group(1))
        return BudgetRange(type=BudgetType.FIXED, min_amount=amount, max_amount=amount)

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
    # Current template: skills are rendered as pill-styled links to a skill
    # search page (`/nx/search/jobs/?...`), not a text heading at all --
    # confirmed against live emails. A trailing "+N" pill (Upwork's own
    # "N more skills, truncated" indicator) is not itself a skill.
    pills = [pill.get_text(strip=True) for pill in soup.find_all("a", href=_SKILL_PILL_LINK_RE)]
    pills = [pill for pill in pills if pill and not re.fullmatch(r"\+\d+", pill)]
    if pills:
        return pills

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


def _extract_posted_at(text: str, received_at: datetime | None) -> datetime:
    """Resolve `posted_at`, anchored to the email's own date, not processing time.

    `received_at` (the email's Date header) is the reference point whenever
    it's available -- both for the fallback (current template: no relative
    phrase at all, so this *is* the only signal) and for an explicit
    "posted N ago" phrase, which was itself written relative to when the
    email was sent, not whenever Ulysses happens to process it. Without
    this, an email sitting unread for days or weeks looks freshly posted
    the moment it's finally read.
    """
    reference = received_at or datetime.now(UTC)
    match = _POSTED_AGO_RE.search(text)
    if not match:
        return reference

    amount = int(match.group(1))
    unit = match.group(2).lower()
    delta = {
        "second": timedelta(seconds=amount),
        "minute": timedelta(minutes=amount),
        "hour": timedelta(hours=amount),
        "day": timedelta(days=amount),
    }[unit]
    return reference - delta
