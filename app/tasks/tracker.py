"""
Keeps track of the owner's open Google Tasks and asks before closing one.

Developer decision (2026-10-11): Socio tracks every task in the default list
and asks "Can I close this task?" (an approval card, never a direct change) when
it learns the work is done:

- Due date passed: the day after a task's due date, Socio asks once
  ("Is this done?" Yes, close it / Not yet), between 09:00 and 21:00 IST. A new
  due date gets a new question; "Not yet" is not asked again for the same date.
- The owner says so in chat ("I sent the invoice"): the owner-mode agent sees
  the open tasks (open_tasks_block) and proposes closing the matching one.
- Ticked or deleted in the Tasks app: Socio just stops tracking it (no
  question) and closes any card still asking about it.
- Old automatic reminders ("Socio: Check - Interview", from the follow-up
  feature that is now off): one card offers to close them all at once, so the
  owner never has to delete them by hand. Asked once per batch.

check_tasks() runs on the bot's schedule. It costs one Google Tasks call and no
AI call.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from app.agent.state import AgentStateStore
from app.agent.untrusted import inline_untrusted
from app.calendar.events import INDIA_TZ


logger = logging.getLogger(__name__)

LIST_LIMIT = 100  # one page of the Tasks API
ASK_FROM_HOUR, ASK_UNTIL_HOUR = 9, 21  # IST: no "is this done?" in the middle of the night
MAX_QUESTIONS_PER_CHECK = 3  # never flood the chat after a long break
PROMPT_TASKS = 15
OLD_REMINDER_PREFIX = "Socio: "  # title format of app.agent.followups tasks


def _pretty_date(value: str) -> str:
    try:
        return date.fromisoformat(value[:10]).strftime("%a %d %b")
    except ValueError:
        return value


def sync_tasks(store: AgentStateStore) -> dict[str, int]:
    """Bring tracked_tasks up to date with Google Tasks. Returns counts of new and closed tasks."""

    from app.tasks.tasks import list_tasks

    tasks = [t for t in list_tasks(include_completed=False, max_results=LIST_LIMIT) if t.get("task_id")]
    open_ids = {t["task_id"] for t in tasks if t.get("status") != "completed"}

    new = 0
    for task in tasks:
        if task["task_id"] in open_ids and (task.get("title") or "").strip():
            new += store.upsert_tracked_task(task["task_id"], task["title"].strip(), task.get("due") or "")

    closed = 0
    if len(tasks) < LIST_LIMIT:  # a full page may have cut some open tasks off: close nothing then
        for row in store.tracked_tasks("open"):
            if row["task_id"] not in open_ids:
                closed += close_tracked(store, row["task_id"], "completed_or_removed_in_tasks_app")
    return {"new": new, "closed": closed}


def close_tracked(store: AgentStateStore, task_id: str, how: str) -> bool:
    """Stop tracking a task and close any card still asking about it."""

    closed = store.close_tracked_task(task_id, how)
    for approval in store.open_approvals_of_type("complete_task"):
        if (approval.get("payload") or {}).get("task_id") == task_id:
            store.resolve_pending_approval(approval["id"], "The task was already closed in Google Tasks")
    return closed


def due_questions(store: AgentStateStore, now: datetime | None = None) -> list[int]:
    """Ask once about each open task whose due date has passed. Returns the new approval ids."""

    now = (now or datetime.now(INDIA_TZ)).astimezone(INDIA_TZ)
    if not ASK_FROM_HOUR <= now.hour < ASK_UNTIL_HOUR:
        return []

    asking = {(a.get("payload") or {}).get("task_id") for a in store.open_approvals_of_type("complete_task")}
    created: list[int] = []
    for row in store.tracked_tasks("open"):
        if len(created) >= MAX_QUESTIONS_PER_CHECK:
            break
        due = row.get("due") or ""
        try:
            overdue = bool(due) and date.fromisoformat(due[:10]) < now.date()
        except ValueError:
            overdue = False
        if not overdue or row.get("due_asked_for") == due or row["task_id"] in asking:
            continue
        approval_id = store.add_pending_approval(
            "complete_task",
            f"Mark Google Task '{row['title']}' as completed",
            {
                "task_id": row["task_id"],
                "title": row["title"],
                "due": due,
                "reason": f"It was due on {_pretty_date(due)}.",
                "trigger": "due",
            },
        )
        store.mark_due_asked(row["task_id"], due)
        created.append(approval_id)
    return created


def cleanup_question(store: AgentStateStore, now: datetime | None = None) -> list[int]:
    """One card offering to close all old automatic reminder tasks (asked once per batch)."""

    now = (now or datetime.now(INDIA_TZ)).astimezone(INDIA_TZ)
    if not ASK_FROM_HOUR <= now.hour < ASK_UNTIL_HOUR or store.open_approvals_of_type("close_old_reminders"):
        return []
    old = [
        row for row in store.tracked_tasks("open")
        if row["title"].startswith(OLD_REMINDER_PREFIX) and not row.get("cleanup_asked")
    ]
    if not old:
        return []
    approval_id = store.add_pending_approval(
        "close_old_reminders",
        f"Close {len(old)} old reminder tasks",
        {
            "task_ids": [row["task_id"] for row in old],
            "examples": [row["title"][len(OLD_REMINDER_PREFIX):] for row in old[:5]],
            "count": len(old),
        },
    )
    store.mark_cleanup_asked([row["task_id"] for row in old])
    return [approval_id]


def check_tasks(store: AgentStateStore | None = None, now: datetime | None = None) -> list[int]:
    """The scheduled check: sync with Google Tasks, then ask about overdue tasks and old reminders."""

    store = store or AgentStateStore()
    counts = sync_tasks(store)
    if counts["new"] or counts["closed"]:
        logger.info("Tasks: %s new, %s closed in the Tasks app", counts["new"], counts["closed"])
    return cleanup_question(store, now) + due_questions(store, now)


def open_tasks_block(store: AgentStateStore | None = None, limit: int = PROMPT_TASKS) -> str:
    """The owner's open tasks for the owner-mode prompt (from the local copy: no API call)."""

    rows = [row for row in (store or AgentStateStore()).tracked_tasks("open") if not row["title"].startswith(OLD_REMINDER_PREFIX)][:limit]
    if not rows:
        return ""
    lines = [
        "## The owner's open Google Tasks (reference only; use task_id in tool calls, never show it to the owner)"
    ]
    for row in rows:
        due = f" (due {row['due']})" if row.get("due") else ""
        lines.append(f"- task_id={row['task_id']}: {inline_untrusted(row['title'])}{due}")
    return "\n".join(lines)
