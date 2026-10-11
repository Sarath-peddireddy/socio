"""
Memory ingestion: only what the owner explicitly asks Socio to remember.

Approving, rejecting, or editing a request teaches Socio nothing by itself (a
single decision is not a rule). Memories come from the owner's own words:

1. /remember TEXT (Telegram) or --remember TEXT (CLI): a standing preference.
2. Asking Socio in chat ("remember: ...", "make it a rule ..."): the agent's
   remember_rule tool passes the owner's words here, and they are classified first. A STANDING_POLICY ("never book calls before noon") goes to
   long-term memory; a SITUATIONAL note ("already talked to him", "I'm tired
   today") is kept for 24 hours only (app.memory.episodic).

Gemini only restructures and classifies the owner's text; if it is unavailable,
the note is stored as written (and classified by keywords). Context about the
request comes from structured fields only (type, counterpart address, time, file
or document name), never from email text, so an email cannot write itself into
memory.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from email.utils import getaddresses
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.state import AgentStateStore
from app.agent.untrusted import inline_untrusted
from app.memory.semantic import SemanticMemory, memory_enabled


logger = logging.getLogger(__name__)

DECISION_WORDS = {
    "approved": "approved",
    "rejected": "rejected",
    "edited": "rewrote in Gmail (did not send as drafted)",
}


def _short(value: Any, width: int = 60) -> str:
    text = inline_untrusted(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def _first_address(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    addresses = [address.lower() for _, address in getaddresses([str(value or "")]) if "@" in address]
    return addresses[0] if addresses else ""


def counterpart(approval: dict[str, Any]) -> str:
    """Who or what a decision was about: an email address, a document, or 'owner'."""

    payload = approval.get("payload") or {}
    kind = approval.get("type")

    if kind in {"send_reply", "send_reply_with_attachment"}:
        address = _first_address((payload.get("reply") or {}).get("to"))
    elif kind == "book_meeting":
        address = _first_address(payload.get("attendee_email"))
    elif kind in {"move_event", "cancel_event"}:
        address = _first_address(payload.get("other_attendees"))
    elif kind == "trash_email":
        address = _first_address(payload.get("sender"))
    elif kind == "send_email":
        address = _first_address(payload.get("to"))
    elif kind in {"create_document", "create_spreadsheet"}:
        return f"document '{_short(payload.get('title'), 50)}'"
    elif kind in {"append_document", "append_sheet_row"}:
        return f"document '{_short(payload.get('document_title') or payload.get('document'), 50)}'"
    elif kind == "enable_skill_script":
        return f"skill '{_short(payload.get('name'), 50)}'"
    else:
        address = ""

    return address or "owner"


def _decision_subject(approval: dict[str, Any]) -> str:
    from app.agent.approvals import format_time

    payload = approval.get("payload") or {}
    kind = approval.get("type")
    files = ", ".join(_short(a.get("name"), 40) for a in payload.get("attachments", []))

    subjects = {
        "send_reply": "a drafted reply",
        "send_reply_with_attachment": f"a reply with {files or 'a file'}",
        "send_email": f"a new email (subject '{_short(payload.get('subject'))}')" + (f" with {files}" if files else ""),
        "book_meeting": f"booking '{_short(payload.get('title'))}' at {format_time(payload.get('new_start'))}",
        "move_event": f"moving '{_short(payload.get('title'))}' to {format_time(payload.get('new_start'))}",
        "cancel_event": f"cancelling '{_short(payload.get('title'))}' at {format_time(payload.get('old_start'))}",
        "trash_email": "moving an email to Trash",
        "create_document": "creating a Google Doc",
        "append_document": "appending text to a Google Doc",
        "create_spreadsheet": "creating a Google Sheet",
        "append_sheet_row": "adding a row to a Google Sheet",
        "enable_skill_script": "trusting a skill script",
    }
    return subjects.get(kind, kind or "an action")


def decision_tuple(approval: dict[str, Any], decision: str) -> tuple[str, str, str]:
    """(entity, preference, context) for one approval decision, from structured fields only."""

    word = DECISION_WORDS.get(decision, decision)
    preference = f"Owner {word} {_decision_subject(approval)}"
    context = f"approval #{approval.get('id')}, {date.today().isoformat()}"
    return counterpart(approval), preference, context


class PreferenceTuple(BaseModel):
    entity: str = Field(description="Who or what it applies to: an email address, a domain, a document name, or 'owner' for general habits.")
    preference: str = Field(description="The rule in a short imperative sentence, in the owner's terms.")
    context: str = Field(default="", description="When it applies or why, briefly.")


class ExtractedPreferences(BaseModel):
    preferences: list[PreferenceTuple] = Field(description="1 to 3 standing preferences stated by the owner.")


EXTRACTION_PROMPT = """Turn the owner's note into 1 to 3 standing preferences for their email assistant.
Only restate what the owner said. Do not add rules they did not state, and keep their meaning exactly.
- entity: who or what it applies to (an email address, a domain, a document name, or "owner" for general habits)
- preference: one short imperative sentence ("No meetings before 12:00 IST")
- context: when it applies or why, briefly ("" if not stated)
{facts}
Owner's note:
\"\"\"{note}\"\"\""""


def _extract(note: str, approval: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    from app.agent.llm import get_llm, invoke_llm

    facts = ""
    if approval:
        decision = {"resolved": "edited"}.get(approval.get("status"), approval.get("status") or "decided")
        entity, preference, _ = decision_tuple(approval, decision)
        facts = f"The note explains this decision: {preference} (about {entity}).\n"

    llm = get_llm().with_structured_output(ExtractedPreferences)
    result = invoke_llm(llm, EXTRACTION_PROMPT.format(facts=facts, note=note))
    return [(p.entity, p.preference, p.context) for p in (result.preferences if result else [])[:3] if p.entity and p.preference]


def learn_from_owner_note(note: str, approval: dict[str, Any] | None = None, store: AgentStateStore | None = None) -> list[dict[str, Any]]:
    """
    Store a standing preference the owner stated (/remember) as memories.

    approval is the request the note is about, if any (adds context for the extraction).
    Returns [{"id", "entity", "preference", "created"}]; [] when memory is off.
    """

    note = " ".join((note or "").split())[:1000]
    if not note or not memory_enabled():
        return []

    try:
        tuples = _extract(note, approval)
    except Exception as exc:
        logger.warning("Preference extraction failed, storing the note as written: %s", exc)
        tuples = []

    if not tuples:
        context = f"about approval #{approval['id']}" if approval else "told by the owner"
        tuples = [(counterpart(approval) if approval else "owner", note, context)]

    memory = SemanticMemory(store)
    source = "owner_correction" if approval else "owner_note"
    saved = []
    for entity, preference, context in tuples:
        memory_id, created = memory.upsert(entity, preference, context, source=source)
        saved.append({"id": memory_id, "entity": entity, "preference": preference, "created": created})
    return saved


# ----- owner rules from chat (remember_rule): standing policy or situational note --------

STANDING_POLICY = "STANDING_POLICY"
SITUATIONAL = "SITUATIONAL"


class RuleClassification(BaseModel):
    kind: Literal["STANDING_POLICY", "SITUATIONAL"] = Field(
        description="STANDING_POLICY for a lasting rule about the future; SITUATIONAL for this case or today only."
    )
    summary: str = Field(description="The note restated in one short sentence, keeping the owner's meaning exactly.")
    preferences: list[PreferenceTuple] = Field(
        default_factory=list, description="For STANDING_POLICY: 1 to 3 rules the owner stated. Empty for SITUATIONAL."
    )


CLASSIFY_PROMPT = """The owner turned down a request from their email assistant and then wrote the note below.
Classify the note:
- STANDING_POLICY: a lasting rule for the future, for example "never book calls before noon",
  "always cc my manager", "no meetings after 5pm", "only send my resume as PDF".
- SITUATIONAL: about this case or today only, for example "I'm tired today", "already talked to him",
  "not this week".
Then give a one-sentence summary. For STANDING_POLICY also list 1 to 3 preferences:
- entity: who or what it applies to (an email address, a domain, a document name, or "owner" for general habits)
- preference: one short imperative sentence ("No meetings before 12:00 IST")
- context: when it applies or why, briefly ("" if not stated)
Only restate what the owner said. Do not add rules they did not state.
{facts}
Owner's note:
\"\"\"{note}\"\"\""""

# Keyword fallback for when Gemini is unavailable.
_STANDING_WORDS = re.compile(
    r"\b(never|always|only|every|any ?time|from now on|in (the )?future|don'?t ever|no \w+ (before|after)|(before|after) \d)",
    re.IGNORECASE,
)
_SITUATIONAL_WORDS = re.compile(
    r"\b(today|tonight|tomorrow|this (time|week|one|morning|afternoon|evening)|right now|at the moment|already|tired|for now)\b",
    re.IGNORECASE,
)


def guess_rule_kind(note: str) -> str:
    """Keyword classification; when unsure, a note the owner asked to keep is kept (standing)."""

    if _STANDING_WORDS.search(note):
        return STANDING_POLICY
    if _SITUATIONAL_WORDS.search(note):
        return SITUATIONAL
    return STANDING_POLICY


def _classify(note: str, approval: dict[str, Any] | None) -> RuleClassification:
    from app.agent.llm import get_llm, invoke_llm

    facts = ""
    if approval:
        decision = {"resolved": "edited"}.get(approval.get("status"), approval.get("status") or "decided")
        entity, preference, _ = decision_tuple(approval, decision)
        facts = f"The note is about this decision: {preference} (about {entity}).\n"

    llm = get_llm().with_structured_output(RuleClassification)
    result = invoke_llm(llm, CLASSIFY_PROMPT.format(facts=facts, note=note))
    if result is None or result.kind not in {STANDING_POLICY, SITUATIONAL}:
        raise ValueError("No classification returned.")
    return result


def promote_owner_rule(note: str, approval: dict[str, Any] | None = None, store: AgentStateStore | None = None) -> dict[str, Any]:
    """
    Classify the owner's note and keep it accordingly.

    Returns {"kind", "summary", "saved": [...]} for a standing policy (long-term
    memory) or {"kind", "summary", "note_id"} for a situational note (24 hours).
    """

    from app.memory import episodic

    note = " ".join((note or "").split())[:1000]
    if not note:
        raise ValueError("Empty note.")

    entity = counterpart(approval) if approval else "owner"
    try:
        classified = _classify(note, approval)
        kind, summary = classified.kind, " ".join(classified.summary.split()) or note
        tuples = [(p.entity, p.preference, p.context) for p in classified.preferences[:3] if p.entity and p.preference]
    except Exception as exc:
        logger.warning("Rule classification failed, using keywords: %s", exc)
        kind, summary, tuples = guess_rule_kind(note), note, []

    if kind == SITUATIONAL:
        note_id = episodic.add_note(note, entity=entity, approval_id=approval.get("id") if approval else None, store=store)
        return {"kind": SITUATIONAL, "summary": summary, "note_id": note_id}

    if not tuples:
        context = f"about approval #{approval['id']}" if approval else "told by the owner"
        tuples = [(entity, note, context)]

    memory = SemanticMemory(store)
    saved = []
    for item_entity, preference, context in tuples:
        memory_id, created = memory.upsert(item_entity, preference, context, source="owner_rule")
        saved.append({"id": memory_id, "entity": item_entity, "preference": preference, "created": created})
    summary = "; ".join(
        item["preference"] + ("" if item["entity"].lower() == "owner" else f" ({item['entity']})") for item in saved
    )
    return {"kind": STANDING_POLICY, "summary": summary, "saved": saved}
