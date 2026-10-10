"""
Local soft holds on calendar slots.

A slot proposed for a booking or a move is not in Google Calendar until the
owner approves it. Until then (released when the approval is decided, or
after SLOT_HOLD_HOURS) it is treated as busy, so the inbox loop does not
offer or propose the same slot to two senders.

Holds live in the agent's SQLite state (app.agent.state, table slot_holds).
"""

from __future__ import annotations

from datetime import datetime

from app.agent.state import AgentStateStore


def held_periods(store: AgentStateStore | None = None) -> list[dict]:
    """Active holds with parsed (timezone-aware) start/end datetimes."""

    store = store or AgentStateStore()

    return [
        {**hold, "start": datetime.fromisoformat(hold["start_time"]), "end": datetime.fromisoformat(hold["end_time"])}
        for hold in store.active_slot_holds()
    ]


def overlapping_holds(start: datetime, end: datetime, ignore_message_id: str | None = None, store: AgentStateStore | None = None) -> list[dict]:
    """
    Holds that overlap [start, end).

    ignore_message_id skips holds made for that email, so a retried run can
    propose its own slot again.
    """

    return [
        hold
        for hold in held_periods(store)
        if hold["start"] < end and start < hold["end"]
        and not (ignore_message_id and hold["message_id"] == ignore_message_id)
    ]
