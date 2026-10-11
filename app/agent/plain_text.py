"""
Last clean-up of Socio's own words before they reach the owner (Telegram, --ask).

The prompts already ask for plain language; models still slip. This removes what
must never reach the chat:
- raw JSON (a worker's status object, a tool result pasted back);
- internal ids: approval numbers (#12), Gmail message/thread ids, Drive "drive:<id>"
  paths, and other long machine ids; links and email addresses are kept intact;
- "the owner" / "the user" where it means the person reading ("for the owner" ->
  "for you", "the owner's" -> "your").

Only for the agent's free text. Approval cards are never passed through this: they
must show exactly what will be sent or written.
"""

from __future__ import annotations

import re


_CODE_FENCE_JSON = re.compile(r"```(?:json)?\s*[\[{].*?[\]}]\s*```", re.DOTALL | re.IGNORECASE)
# A flat {...} with at least one "key": inside; applied until nothing is left (nested objects).
_JSON_OBJECT = re.compile(r"\{[^{}]*\"[\w\s-]+\"\s*:[^{}]*\}")

# Kept as they are: links and email addresses (a Sheets link contains the file id on purpose).
_PROTECTED = re.compile(r"(https?://\S+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+)")

_APPROVAL_NUMBER = re.compile(r"\b(approval request|approval|request)\s*#\d+", re.IGNORECASE)
_ASIDE_NUMBER = re.compile(r"\s*\((?:approval request|approval|request)?\s*#\d+\)", re.IGNORECASE)
_BARE_NUMBER = re.compile(r"\s*#\d+\b")
_DRIVE_PATH = re.compile(r"`?drive:[A-Za-z0-9_-]+`?")
_ID_LABEL = r"(message|thread|event|task|file|document|doc|spreadsheet|draft|email)[ _]id\b\s*[:=]?\s*`?[A-Za-z0-9_-]{6,}`?"
_ASIDE_ID = re.compile(rf"\s*\(\s*{_ID_LABEL}\s*\)", re.IGNORECASE)   # "(message id 18c...)" -> gone
_LABELLED_ID = re.compile(rf"\b{_ID_LABEL}", re.IGNORECASE)           # "Task id: MTA3..." -> "Task"
_NAMED_GMAIL_ID = re.compile(r"\s*\b(?:(?:in|on|from)\s+)?(?:the\s+)?(?:message|thread|email)\s+[0-9a-f]{16}\b", re.IGNORECASE)
_GMAIL_ID = re.compile(r"(?<![\w-])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{16}(?![\w-])")
_LONG_ID = re.compile(r"(?<![\w/.-])(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{20,}(?![\w.-])")

_PERSON_OBJECT = re.compile(r"\b(for|to|with|by|from) the (?:owner|user)\b(?!'s)", re.IGNORECASE)
_PERSON_POSSESSIVE = re.compile(r"\b(t)he (?:owner|user)'s\b", re.IGNORECASE)

_EMPTY_BRACKETS = re.compile(r"\s*\(\s*\)|\s*\[\s*\]")
# "You have one open task: - **Pay rent**": a list dash glued into a sentence.
_INLINE_DASH = re.compile(r"(?<=[:,;])[ \t]+[-*•][ \t]+(?=\S)")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:!?])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def without_internal_ids(text: str) -> str:
    """Approval numbers out of owner-facing text ("Approval request #3" -> "The approval request")."""

    text = _ASIDE_NUMBER.sub("", text or "")
    text = _APPROVAL_NUMBER.sub(lambda m: "The approval request" if m.group(0)[0].isupper() else "the approval request", text)
    return _BARE_NUMBER.sub("", text)


def _without_json(text: str) -> str:
    text = _CODE_FENCE_JSON.sub("", text)
    while True:
        cleaned = _JSON_OBJECT.sub("", text)
        if cleaned == text:
            return text
        text = cleaned


def _machine_id(match: re.Match) -> str:
    """Drop a long token only if it looks generated: a name like Q3_Budget_Report_2026 stays."""

    pieces = [piece for piece in re.split(r"[_-]", match.group(0)) if piece]
    readable = len(pieces) > 1 and all(piece.isalpha() or piece.isdigit() or len(piece) <= 4 for piece in pieces)
    return match.group(0) if readable else ""


def _clean_words(part: str) -> str:
    part = without_internal_ids(part)
    part = _DRIVE_PATH.sub("", part)
    part = _ASIDE_ID.sub("", part)
    part = _LABELLED_ID.sub(lambda m: m.group(1), part)
    part = _NAMED_GMAIL_ID.sub("", part)
    part = _GMAIL_ID.sub("", part)
    part = _LONG_ID.sub(_machine_id, part)
    part = _PERSON_OBJECT.sub(lambda m: f"{m.group(1)} you", part)
    return _PERSON_POSSESSIVE.sub(lambda m: "Your" if m.group(1) == "T" else "your", part)


# "The email has been sent successfully." Socio never sends inside a chat run (every email
# goes out only when the owner taps Approve & Send), so this in an answer is always a repeat
# of what the card already shows (developer request 2026-10-11).
# Within one sentence: a '.' inside a word (gmail.com, 15.55) does not end it.
_IN_SENTENCE = r"(?:[^.!?\n]|[.!?](?=\S))*"
_SENT_SENTENCE = re.compile(
    _IN_SENTENCE
    + r"(?:\b(?:has|have)\s+been\s+(?:successfully\s+)?sent\b|\b(?:was|were|got)\s+(?:successfully\s+)?sent\b"
    r"|\bsent\s+successfully\b|\bwent\s+out\b|\b(?:has|have)\s+been\s+delivered\b)"
    + _IN_SENTENCE + r"[.!]?",
    re.IGNORECASE,
)
# Unless the owner asked exactly that ("Did the email go out?", "Have you sent it?").
_ASKED_IF_SENT = re.compile(
    r"\b(did|has|have|was|were|is)\b.*\b(sent|send|e-?mail(ed)?|mail(ed)?|go(ne)?\s+out|went|deliver\w*)\b",
    re.IGNORECASE,
)
# Only sentences about an email ("Power went out across Spain" in a news briefing stays).
_ABOUT_EMAIL = re.compile(r"\b(e-?mails?|mail|message|reply)\b", re.IGNORECASE)
NOTHING_MORE = "Nothing else is needed from you. Any reply will show up in your inbox."


def without_sent_repeat(reply: str, owner_text: str) -> str:
    """Drop "the email has been sent" sentences unless the owner asked whether it was sent."""

    if _ASKED_IF_SENT.search(owner_text or "") or not _SENT_SENTENCE.search(reply or ""):
        return reply
    drop = lambda match: "" if _ABOUT_EMAIL.search(match.group(0)) else match.group(0)
    lines = [_MULTI_SPACE.sub(" ", _SENT_SENTENCE.sub(drop, line)).strip() for line in reply.splitlines()]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned or NOTHING_MORE


def plain_reply(text: str) -> str:
    """The agent's text, safe and plain for the owner's chat."""

    text = _without_json(text or "")
    parts = _PROTECTED.split(text)
    # split() with one group: even indexes are ordinary text, odd ones are links / addresses.
    text = "".join(part if index % 2 else _clean_words(part) for index, part in enumerate(parts))

    lines = []
    for line in text.splitlines():
        line = _EMPTY_BRACKETS.sub("", line)
        line = _INLINE_DASH.sub(" ", line)
        line = _SPACE_BEFORE_PUNCT.sub(r"\1", line)
        line = _MULTI_SPACE.sub(" ", line).rstrip()
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
