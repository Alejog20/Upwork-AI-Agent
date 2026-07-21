"""Tests for `hermes.tools.email_reader`, with `imaplib2` fully mocked out."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes.tools.email_reader import EmailReader

MULTIPART_HTML_EMAIL = (
    b"Subject: Python scraper job\r\n"
    b"Date: Mon, 1 Jan 2024 00:00:00 +0000\r\n"
    b'Content-Type: multipart/alternative; boundary="BOUNDARY"\r\n'
    b"\r\n"
    b"--BOUNDARY\r\n"
    b"Content-Type: text/plain\r\n\r\n"
    b"plain text body\r\n"
    b"--BOUNDARY\r\n"
    b"Content-Type: text/html; charset=utf-8\r\n\r\n"
    b"<html><body><a href='https://www.upwork.com/jobs/~01'>Job</a></body></html>\r\n"
    b"--BOUNDARY--\r\n"
)

PLAIN_TEXT_ONLY_EMAIL = b"Subject: No HTML\r\nContent-Type: text/plain\r\n\r\njust plain text\r\n"


@pytest.fixture
def reader() -> EmailReader:
    return EmailReader(host="imap.gmail.com", port=993, user="me@gmail.com", app_password="secret")


class TestFetchNewUpworkEmails:
    async def test_returns_parsed_html_bodies(self, reader: EmailReader) -> None:
        mock_conn = MagicMock()
        mock_conn.search.return_value = ("OK", [b"1"])
        mock_conn.fetch.return_value = ("OK", [(b"1 (RFC822 {123}", MULTIPART_HTML_EMAIL)])

        with patch("hermes.tools.email_reader.imaplib2.IMAP4_SSL", return_value=mock_conn):
            emails = await reader.fetch_new_upwork_emails()

        assert len(emails) == 1
        assert "upwork.com/jobs" in emails[0].html_body
        assert emails[0].subject == "Python scraper job"
        mock_conn.login.assert_called_once_with("me@gmail.com", "secret")
        mock_conn.select.assert_called_once_with("INBOX")
        mock_conn.logout.assert_called_once()

    async def test_returns_empty_list_on_search_failure(self, reader: EmailReader) -> None:
        mock_conn = MagicMock()
        mock_conn.search.return_value = ("NO", [None])

        with patch("hermes.tools.email_reader.imaplib2.IMAP4_SSL", return_value=mock_conn):
            emails = await reader.fetch_new_upwork_emails()

        assert emails == []
        mock_conn.logout.assert_called_once()

    async def test_skips_messages_with_no_html_part(self, reader: EmailReader) -> None:
        mock_conn = MagicMock()
        mock_conn.search.return_value = ("OK", [b"1"])
        mock_conn.fetch.return_value = ("OK", [(b"1 (RFC822 {50}", PLAIN_TEXT_ONLY_EMAIL)])

        with patch("hermes.tools.email_reader.imaplib2.IMAP4_SSL", return_value=mock_conn):
            emails = await reader.fetch_new_upwork_emails()

        assert emails == []

    async def test_skips_failed_fetch(self, reader: EmailReader) -> None:
        mock_conn = MagicMock()
        mock_conn.search.return_value = ("OK", [b"1"])
        mock_conn.fetch.return_value = ("NO", None)

        with patch("hermes.tools.email_reader.imaplib2.IMAP4_SSL", return_value=mock_conn):
            emails = await reader.fetch_new_upwork_emails()

        assert emails == []

    async def test_no_unread_messages_returns_empty_list(self, reader: EmailReader) -> None:
        mock_conn = MagicMock()
        mock_conn.search.return_value = ("OK", [b""])

        with patch("hermes.tools.email_reader.imaplib2.IMAP4_SSL", return_value=mock_conn):
            emails = await reader.fetch_new_upwork_emails()

        assert emails == []
