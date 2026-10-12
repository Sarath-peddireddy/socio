"""
Data retention: delete old operational records so the SQLite file and its WAL stay small.

Three lifetimes, in days, from .env (0 = keep forever):

- RETENTION_DAYS_OPERATIONAL (30): finished supervisor runs and their worker tasks, decided
  approval cards, the action audit log, logged failures, released slot holds, expired 24 h
  notes, and failure cases that never got a written correction.
- RETENTION_DAYS_MESSAGES (90): Telegram conversation turns.
- RETENTION_DAYS_ANCHORS (1): conversation anchors (already ignored after 24 hours).

Never deleted here: open / executing approvals, approvals whose outcome is 'unknown' (the owner
still has to check Gmail), processed_messages (it stops an old email from being handled twice),
memories, owner-written corrections, scheduled routines, tracked tasks, and skills.

Every timestamp compared here is SQLite's own UTC text ("YYYY-MM-DD HH:MM:SS", from
CURRENT_TIMESTAMP / datetime('now')), so datetime('now', '-N days') compares correctly.
Runs from the Telegram bot's daily maintenance (at startup, then every 24 hours) and is
followed by a WAL checkpoint.
"""

from __future__ import annotations

import logging
import os

from app.agent.state import AgentStateStore


logger = logging.getLogger(__name__)

DEFAULT_DAYS = {"operational": 30, "messages": 90, "anchors": 1}

# Approval statuses that are final: the card was decided and nothing more will happen.
DECIDED_APPROVALS = ("approved", "rejected", "resolved", "stale", "rechecked", "failed")
# Supervisor runs that are over (running / planning ones are left alone).
FINISHED_RUNS = ("done", "failed", "partial", "dry_run", "interrupted")


def retention_days(kind: str) -> int:
    """Days to keep one kind of record (RETENTION_DAYS_<KIND>); 0 or less = keep forever."""

    raw = os.getenv(f"RETENTION_DAYS_{kind.upper()}", "").strip()
    try:
        return int(raw) if raw else DEFAULT_DAYS[kind]
    except ValueError:
        logger.warning("RETENTION_DAYS_%s=%r is not a whole number; using %d.", kind.upper(), raw, DEFAULT_DAYS[kind])
        return DEFAULT_DAYS[kind]


def _statements(operational: int, messages: int, anchors: int) -> list[tuple[str, str, tuple]]:
    """(label, SQL, params) for every purge whose lifetime is switched on."""

    decided = ", ".join("?" for _ in DECIDED_APPROVALS)
    finished = ", ".join("?" for _ in FINISHED_RUNS)
    cutoff = f"-{operational} days"
    statements = []

    if operational > 0:
        statements += [
            # Worker tasks first, while their run still says it is finished.
            ("worker_tasks",
             f"DELETE FROM worker_tasks WHERE run_id IN (SELECT id FROM supervisor_runs "
             f"WHERE status IN ({finished}) AND updated_at < datetime('now', ?))",
             (*FINISHED_RUNS, cutoff)),
            ("supervisor_runs",
             f"DELETE FROM supervisor_runs WHERE status IN ({finished}) AND updated_at < datetime('now', ?)",
             (*FINISHED_RUNS, cutoff)),
            ("pending_approvals",
             f"DELETE FROM pending_approvals WHERE status IN ({decided}) "
             "AND COALESCE(decided_at, created_at) < datetime('now', ?)",
             (*DECIDED_APPROVALS, cutoff)),
            ("slot_holds",
             "DELETE FROM slot_holds WHERE (status != 'active' OR expires_at < CURRENT_TIMESTAMP) "
             "AND created_at < datetime('now', ?)",
             (cutoff,)),
            ("actions", "DELETE FROM actions WHERE created_at < datetime('now', ?)", (cutoff,)),
            ("failures", "DELETE FROM failures WHERE created_at < datetime('now', ?)", (cutoff,)),
            ("feedback_requests", "DELETE FROM feedback_requests WHERE created_at < datetime('now', ?)", (cutoff,)),
            ("episodic_notes",
             "DELETE FROM episodic_notes WHERE expires_at < CURRENT_TIMESTAMP AND created_at < datetime('now', ?)",
             (cutoff,)),
            # A case the owner corrected is a lesson for the planner: kept until they remove it.
            ("failure_cases",
             "DELETE FROM failure_cases WHERE status != 'corrected' AND timestamp < datetime('now', ?)",
             (cutoff,)),
        ]
    if messages > 0:
        statements.append(
            ("conversation_turns", "DELETE FROM conversation_turns WHERE created_at < datetime('now', ?)",
             (f"-{messages} days",))
        )
    if anchors > 0:
        statements.append(
            ("conversation_anchors", "DELETE FROM conversation_anchors WHERE updated_at < datetime('now', ?)",
             (f"-{anchors} days",))
        )
    return statements


def purge_expired_records(store: AgentStateStore | None = None, *, checkpoint: bool = True) -> dict[str, object]:
    """
    Delete records past their lifetime in one transaction, then fold the WAL into the
    database (PRAGMA wal_checkpoint(TRUNCATE)).

    Returns {"deleted": {table: rows}, "checkpoint": {...} or None}.
    """

    store = store or AgentStateStore()
    statements = _statements(retention_days("operational"), retention_days("messages"), retention_days("anchors"))

    deleted: dict[str, int] = {}
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")  # one writer for the whole purge
        for label, sql, params in statements:
            deleted[label] = deleted.get(label, 0) + conn.execute(sql, params).rowcount

    removed = {table: rows for table, rows in deleted.items() if rows}
    if removed:
        logger.info("Retention purge removed %s", removed)
    return {"deleted": deleted, "checkpoint": store.checkpoint() if checkpoint else None}
