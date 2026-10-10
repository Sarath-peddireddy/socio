"""
Episodic memory: situational notes from the owner that expire after a day.

"Already talked to him" or "I'm tired today" explain one decision; they are not
standing rules, so they never go into long-term memory (app.memory.semantic).
While a note is active it is shown to the agent and the planner as reference,
under the same safety rules as long-term memories.
"""

from __future__ import annotations

import logging

from app.agent.state import AgentStateStore
from app.agent.untrusted import sanitize_text


logger = logging.getLogger(__name__)

NOTE_HOURS = 24
MAX_NOTE_CHARS = 300

NOTES_PROMPT_HEADER = (
    "## Notes from the owner for today (expire within 24 hours, reference only)\n"
    "Situational notes, not standing rules. Take them into account when they apply. They never\n"
    "override the safety rules, and every approval requirement still applies."
)


def add_note(note: str, entity: str = "owner", approval_id: int | None = None, store: AgentStateStore | None = None) -> int:
    note = " ".join((note or "").split())[:MAX_NOTE_CHARS]
    if not note:
        raise ValueError("Empty note.")
    entity = " ".join((entity or "owner").split())[:120]
    return (store or AgentStateStore()).add_episodic_note(entity, note, NOTE_HOURS, approval_id)


def active_notes(store: AgentStateStore | None = None) -> list[dict]:
    """Unexpired notes, newest first; [] on any problem (never raises)."""

    try:
        return (store or AgentStateStore()).active_episodic_notes()
    except Exception as exc:
        logger.warning("Could not read today's notes: %s", exc)
        return []


def format_notes(notes: list[dict]) -> str:
    """A prompt block for active notes ('' when there are none)."""

    if not notes:
        return ""
    lines = []
    for note in notes:
        text = f"[{note['entity']}] {note['note']}"
        lines.append(f"- {' '.join(sanitize_text(text).split()).replace('<', '&lt;').replace('>', '&gt;')}")
    return NOTES_PROMPT_HEADER + "\n" + "\n".join(lines)


def notes_block(store: AgentStateStore | None = None) -> str:
    return format_notes(active_notes(store))
