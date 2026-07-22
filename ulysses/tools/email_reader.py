"""Async IMAP watcher for new Upwork notification emails.

`imaplib2` (like the stdlib `imaplib`) is a blocking, synchronous client with
no native asyncio support. To honor the "async-first" I/O rule without
reimplementing IMAP, the blocking connect/search/fetch sequence runs inside
`asyncio.to_thread`, keeping the event loop free.
"""

from __future__ import annotations

import asyncio
import email
from dataclasses import dataclass
from email.message import Message

import imaplib2
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

__all__ = ["EmailReader", "RawEmail"]

_imap_retry = retry(
    retry=retry_if_exception_type((OSError, imaplib2.IMAP4.error)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)


@dataclass(slots=True)
class RawEmail:
    """A single fetched email, ready for `job_parser` to extract structure from."""

    uid: str
    subject: str
    html_body: str
    received_at_header: str


class EmailReader:
    """Polls an IMAP mailbox for unread Upwork job notification emails."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        app_password: str,
        mailbox: str = "INBOX",
    ) -> None:
        """Create a reader bound to one mailbox.

        Args:
            host: IMAP host (e.g. `imap.gmail.com` or `imap.mail.me.com`).
            port: IMAP SSL port, normally 993.
            user: Full email address to log in with.
            app_password: An app-specific password — never the account password.
            mailbox: Mailbox to search, defaults to `INBOX`.
        """
        self._host = host
        self._port = port
        self._user = user
        self._app_password = app_password
        self._mailbox = mailbox

    async def fetch_new_upwork_emails(self) -> list[RawEmail]:
        """Connect, search for unseen Upwork emails, and return their HTML bodies."""
        return await asyncio.to_thread(self._fetch_new_upwork_emails_sync)

    @_imap_retry
    def _fetch_new_upwork_emails_sync(self) -> list[RawEmail]:
        emails: list[RawEmail] = []
        conn = imaplib2.IMAP4_SSL(self._host, self._port)
        try:
            conn.login(self._user, self._app_password)
            conn.select(self._mailbox)
            status, data = conn.search(None, '(UNSEEN FROM "upwork.com")')
            if status != "OK":
                logger.warning("IMAP search returned non-OK status: {}", status)
                return emails

            uids = data[0].split() if data and data[0] else []
            for uid in uids:
                raw_email = self._fetch_one(conn, uid)
                if raw_email is not None:
                    emails.append(raw_email)
        finally:
            conn.logout()
        return emails

    def _fetch_one(self, conn: imaplib2.IMAP4_SSL, uid: bytes) -> RawEmail | None:
        # BODY.PEEK[] fetches the full message without setting the \Seen flag,
        # so checking for new jobs never silently marks the user's real inbox
        # as read out from under them.
        status, msg_data = conn.fetch(uid, "(BODY.PEEK[])")
        if status != "OK" or not msg_data or msg_data[0] is None:
            logger.warning("Failed to fetch IMAP message uid={}", uid)
            return None

        raw_bytes = msg_data[0][1]
        message = email.message_from_bytes(raw_bytes)
        html_body = _extract_html_body(message)
        if html_body is None:
            logger.warning("No HTML body found in message uid={}", uid)
            return None

        return RawEmail(
            uid=uid.decode(),
            subject=message.get("Subject", ""),
            html_body=html_body,
            received_at_header=message.get("Date", ""),
        )


def _extract_html_body(message: Message) -> str | None:
    """Walk a (possibly multipart) email message and return its HTML part, if any."""
    parts = message.walk() if message.is_multipart() else [message]
    for part in parts:
        if part.get_content_type() != "text/html":
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    return None
