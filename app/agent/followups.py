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
    """Create one follow-up task per email (idempotent). Returns the task or None."""

    label = AUTO_TASK_LABELS.get(status)
    if not label or not auto_tasks_enabled():
        return None

    if any(action["type"] == AUTO_TASK_TYPE for action in store.actions_for_message(message_id, status="completed")):
        return None

    from app.tasks.tasks import create_task

    title = f"Socio: {label} - {subject or '(no subject)'}"[:200]
    notes = "\n".join(
        part for part in [
            f"From: {sender}" if sender else "",
            (details or "")[:600],
            f"Gmail: https://mail.google.com/mail/u/0/#all/{thread_id}" if thread_id else "",
        ] if part
    )

    try:
        task = create_task(title, notes)
    except Exception as exc:
        logger.warning("Could not create follow-up task for %s: %s", message_id, exc)
        return None

    store.add_action(AUTO_TASK_TYPE, title, {"status": status}, "completed", message_id=message_id, result=task)
    return task


def complete_followup_task(store: AgentStateStore, message_id: str | None) -> None:
    """Complete the follow-up task for an email, if Socio created one."""

    if not message_id or not auto_tasks_enabled():
        return

    from app.tasks.tasks import complete_task

    for action in store.actions_for_message(message_id, status="completed"):
        task_id = (action.get("result") or {}).get("task_id")
        if action["type"] != AUTO_TASK_TYPE or not task_id:
            continue
        try:
            complete_task(task_id)
            store.add_action("auto_task_completed", task_id, {}, "completed", message_id=message_id)
        except Exception as exc:
            logger.warning("Could not complete follow-up task %s: %s", task_id, exc)
