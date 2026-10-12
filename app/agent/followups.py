"""
Optional automatic Google Tasks for emails that need the owner. OFF by default.

Developer decision (2026-10-11): Socio creates a task only when the owner asks
(and approves the card). With AGENT_AUTO_TASKS=1, an email that ends up waiting
on the owner (draft to review, approval pending, something to check) gets a
task, completed when the owner approves/rejects/resolves it. With it off,
Socio neither creates nor completes these tasks.
A failure here never affects how the email itself was handled.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.agent.state import AgentStateStore


logger = logging.getLogger(__name__)

AUTO_TASK_TYPE = "auto_task"
AUTO_TASK_LABELS = {
    "needs_review": "Review reply draft",
    "needs_attention": "Check",
    "awaiting_approval": "Approve or reject",
}


def auto_tasks_enabled() -> bool:
    return os.getenv("AGENT_AUTO_TASKS", "0").strip().lower() in {"1", "true", "yes", "on"}


def create_followup_task(store: AgentStateStore, message_id: str, thread_id: str, status: str, subject: str, sender: str, details: str) -> dict[str, Any] | None:
    """
    Offer one follow-up task per email (idempotent): a "New Task" approval card, never a task
    created behind the owner's back (audit M9; every task creation needs a tap). Returns
    {"approval_id"} or None.
    """

    label = AUTO_TASK_LABELS.get(status)
    if not label or not auto_tasks_enabled():
        return None

    asked = {action["type"] for action in store.actions_for_message(message_id)}
    if AUTO_TASK_TYPE in asked:   # offered (or, before this change, created) once already
        return None

    title = f"Socio: {label} - {subject or '(no subject)'}"[:200]
    notes = "\n".join(
        part for part in [
            f"From: {sender}" if sender else "",
            (details or "")[:600],
            f"Gmail: https://mail.google.com/mail/u/0/#all/{thread_id}" if thread_id else "",
        ] if part
    )
    payload = {"title": title, "notes": notes, "due": "", "reason": "A reminder for an email that needs you.", "followup": True}

    from app.agent.approvals import describe

    approval_id = store.add_pending_approval(
        "create_task", describe({"type": "create_task", "payload": payload}), payload, message_id=message_id, thread_id=thread_id,
    )
    store.add_action(AUTO_TASK_TYPE, title, {"status": status}, "proposed", message_id=message_id, result={"approval_id": approval_id})
    return {"approval_id": approval_id}


def complete_followup_task(store: AgentStateStore, message_id: str | None) -> None:
    """
    The email was dealt with: a follow-up card still waiting for the owner is closed quietly.
    A task the owner already added stays (closing it is the owner's call, through the task
    tracker's "Is this done?" card): nothing in Google Tasks changes without a tap.
    """

    if not message_id or not auto_tasks_enabled():
        return

    for approval in store.open_approvals_of_type("create_task"):
        if approval.get("message_id") == message_id and (approval.get("payload") or {}).get("followup"):
            store.resolve_pending_approval(approval["id"], "The email was dealt with")
