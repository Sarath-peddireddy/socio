"""
Untrusted email content for LLM prompts.

Everything a sender controls (body, subject, names in headers, attachment
names) is sanitized and placed inside <untrusted_email_body> tags, so the
model can tell data from instructions. '<' and '>' inside the content are
escaped, so an email cannot close the tag early or fake a new one.

This lowers the risk of prompt injection; it does not remove it. The hard
limits are elsewhere: the narrow email-mode toolset and owner approval for
every reply and calendar change (app.agent.graph).
"""

from __future__ import annotations

import html
import re


TAG = "untrusted_email_body"

# Zero-width characters, bidirectional overrides, and soft hyphens: invisible
# to a human reading the email, but read by the model.
_INVISIBLE = re.compile("[­​-‏‪-‮⁠-⁤⁦-⁩﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LOOKS_LIKE_HTML = re.compile(r"(?i)<(html|body|div|p|br|table|span|font)\b")
_HTML_HIDDEN = re.compile(r"(?is)<(script|style|head|title)\b.*?</\1\s*>|<!--.*?-->")
_HTML_TAG = re.compile(r"(?s)<[^>]+>")
_BLANK_LINES = re.compile(r"\n[ \t]*(\n[ \t]*){2,}")


def sanitize_text(text: str | None) -> str:
    """Plain text with HTML markup, invisible characters, and control characters removed."""

    text = text or ""

    # The parser falls back to the HTML part when an email has no plain-text part.
    if _LOOKS_LIKE_HTML.search(text):
        text = _HTML_HIDDEN.sub(" ", text)
        text = _HTML_TAG.sub(" ", text)
        text = html.unescape(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _escape(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def _attribute(value) -> str:
    return html.escape(" ".join(sanitize_text(str(value or "")).split()), quote=True)


def inline_untrusted(value) -> str:
    """A sender-controlled header value (name, address) on one line, with no tags and no quotes."""

    return _attribute(value)


def wrap_untrusted_email(email: dict, *, body: str | None = None, max_chars: int | None = None) -> str:
    """
    One email as an <untrusted_email_body> block.

    Headers become escaped attributes; body defaults to email["body"]
    (pass body= for a preview). max_chars truncates the sanitized body.
    """

    text = sanitize_text(email.get("body") if body is None else body)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n[... truncated ...]"

    attributes = {
        "message_id": email.get("id"),
        "sender": email.get("from"),
        "reply_to": email.get("reply_to"),
        "to": email.get("to"),
        "cc": email.get("cc"),
        "date": email.get("date"),
        "subject": email.get("subject"),
        "attachments": ", ".join(a.get("filename", "") for a in email.get("attachments") or []),
    }
    attribute_text = " ".join(f'{name}="{_attribute(value)}"' for name, value in attributes.items() if value)

    return f"<{TAG} {attribute_text}>\n{_escape(text)}\n</{TAG}>"


def untrusted_email_view(email: dict, *, body: str | None = None, max_chars: int | None = None) -> dict:
    """An email as a tool result: trusted ids for tool calls, everything else wrapped."""

    return {
        "id": email.get("id"),
        "thread_id": email.get("thread_id"),
        "sent_by_owner": "SENT" in (email.get("label_ids") or []),
        "email": wrap_untrusted_email(email, body=body, max_chars=max_chars),
    }
