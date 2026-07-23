"""Demo email-bot skeleton -- TODO markers show where job-specific logic goes.

Run with: python demo.py
"""

from __future__ import annotations

import email
import imaplib
import os

IMAP_HOST = "imap.gmail.com"
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD", "")


def fetch_unread_subjects() -> list[str]:
    """Connect and list unread email subjects.

    TODO: replace with the real per-job logic (filter by sender, extract
    attachments, forward, auto-reply, etc). This skeleton just lists subjects.
    """
    subjects = []
    with imaplib.IMAP4_SSL(IMAP_HOST) as conn:
        conn.login(EMAIL_USER, EMAIL_APP_PASSWORD)
        conn.select("INBOX")
        _, data = conn.search(None, "UNSEEN")
        for uid in data[0].split():
            _, msg_data = conn.fetch(uid, "(BODY.PEEK[HEADER])")
            message = email.message_from_bytes(msg_data[0][1])
            subjects.append(message.get("Subject", ""))
    return subjects


def main() -> None:
    if not EMAIL_USER or not EMAIL_APP_PASSWORD:
        print("Set EMAIL_USER and EMAIL_APP_PASSWORD (see config.example.env) and re-run.")
        return
    subjects = fetch_unread_subjects()
    print(f"Found {len(subjects)} unread email(s):")
    for subject in subjects:
        print(f" - {subject}")


if __name__ == "__main__":
    main()
