from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "agent_state.sqlite3"

# Final outcomes: a message with one of these statuses is never processed again.
HANDLED_STATUSES = (
    "processed",           # legacy value from earlier versions
    "replied",             # reply sent and confirmed in the thread
    "replied_unverified",  # Gmail accepted the reply but it was not found in the thread
    "no_action",           # decided that nothing needed to be done
    "needs_review",        # left for the owner (draft created or decision too uncertain)
    "needs_attention",     # partial or uncertain side effect; owner must check
    "awaiting_approval",   # a change (move/cancel event) waits for the owner's approval
    "superseded",          # a newer message in the same thread was handled instead
    "failed_permanently",  # failed too many times; will not be retried
)

# Non-final: picked up again on the next run.
RETRYABLE_STATUSES = ("in_progress", "failed")


class AgentStateStore:
    """Persist agent progress so a restarted run does not repeat completed work."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or os.getenv("AGENT_STATE_DB") or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        """Add a column to a table created by an older version of this file."""

        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    status TEXT NOT NULL,
                    summary TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    target TEXT,
                    payload TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_type TEXT NOT NULL,
                    target TEXT,
                    payload TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shared_files (
                    path TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            self._ensure_column(conn, "processed_messages", "attempts", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "processed_messages", "updated_at", "TEXT")
            self._ensure_column(conn, "actions", "message_id", "TEXT")
            self._ensure_column(conn, "actions", "result", "TEXT")
            self._ensure_column(conn, "pending_approvals", "message_id", "TEXT")
            self._ensure_column(conn, "pending_approvals", "thread_id", "TEXT")
            self._ensure_column(conn, "pending_approvals", "status", "TEXT NOT NULL DEFAULT 'open'")
            self._ensure_column(conn, "pending_approvals", "resolution", "TEXT")
            self._ensure_column(conn, "pending_approvals", "result", "TEXT")
            self._ensure_column(conn, "pending_approvals", "notified", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "pending_approvals", "decided_at", "TEXT")
            self._ensure_column(conn, "failures", "message_id", "TEXT")

    # ----- processed messages -------------------------------------------------

    def mark_processed(self, message_id: str, thread_id: str | None = None, status: str = "processed", summary: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_messages (message_id, thread_id, status, summary, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(message_id) DO UPDATE SET
                    thread_id = COALESCE(excluded.thread_id, processed_messages.thread_id),
                    status = excluded.status,
                    summary = excluded.summary,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    message_id,
                    thread_id,
                    status,
                    summary,
                    json.dumps(metadata or {}),
                ),
            )

    def start_attempt(self, message_id: str, thread_id: str | None = None) -> int:
        """Mark a message in progress and return how many attempts have now been made."""

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_messages (message_id, thread_id, status, attempts, updated_at)
                VALUES (?, ?, 'in_progress', 1, CURRENT_TIMESTAMP)
                ON CONFLICT(message_id) DO UPDATE SET
                    thread_id = COALESCE(excluded.thread_id, processed_messages.thread_id),
                    status = 'in_progress',
                    attempts = processed_messages.attempts + 1,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (message_id, thread_id),
            )
            row = conn.execute(
                "SELECT attempts FROM processed_messages WHERE message_id = ?", (message_id,)
            ).fetchone()
            return int(row["attempts"])

    def abort_attempt(self, message_id: str, reason: str = "") -> None:
        """Undo an attempt that failed because of an outage, so it does not count toward the retry limit."""

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE processed_messages
                SET status = 'failed',
                    attempts = MAX(attempts - 1, 0),
                    summary = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE message_id = ?
                """,
                (reason[:2000], message_id),
            )

    def get_message_record(self, message_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM processed_messages WHERE message_id = ?", (message_id,)
            ).fetchone()
            return dict(row) if row else None

    def is_processed(self, message_id: str) -> bool:
        """True when the message reached a final outcome and must not be handled again."""

        placeholders = ", ".join("?" for _ in HANDLED_STATUSES)
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT 1 FROM processed_messages WHERE message_id = ? AND status IN ({placeholders})",
                (message_id, *HANDLED_STATUSES),
            ).fetchone()
            return row is not None

    def interrupted_messages(self) -> list[dict[str, Any]]:
        """Messages left 'in_progress' by a run that crashed or was stopped."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM processed_messages WHERE status = 'in_progress' ORDER BY updated_at"
            ).fetchall()
            return [dict(row) for row in rows]

    def status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM processed_messages GROUP BY status"
            ).fetchall()
            return {row["status"]: row["n"] for row in rows}

    # ----- actions ------------------------------------------------------------

    def add_action(self, action_type: str, target: str | None, payload: dict[str, Any] | None = None, status: str = "completed", message_id: str | None = None, result: Any = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO actions (type, target, payload, status, message_id, result) VALUES (?, ?, ?, ?, ?, ?)",
                (action_type, target, json.dumps(payload or {}), status, message_id, json.dumps(result, default=str)),
            )

    def actions_for_message(self, message_id: str, status: str | None = None) -> list[dict[str, Any]]:
        """Actions recorded for one email, oldest first, with payload/result decoded."""

        query = "SELECT type, target, payload, status, result, created_at FROM actions WHERE message_id = ?"
        params: tuple = (message_id,)

        if status:
            query += " AND status = ?"
            params += (status,)

        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY id", params).fetchall()

        actions = []
        for row in rows:
            action = dict(row)
            action["args"] = json.loads(action.pop("payload") or "{}")
            action["result"] = json.loads(action["result"]) if action["result"] else None
            actions.append(action)
        return actions

    def retryable_messages(self, max_attempts: int) -> list[dict[str, Any]]:
        """Messages left unfinished (interrupted or failed) that still have attempts left."""

        placeholders = ", ".join("?" for _ in RETRYABLE_STATUSES)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT message_id, thread_id, status, attempts FROM processed_messages "
                f"WHERE status IN ({placeholders}) AND attempts < ? ORDER BY updated_at",
                (*RETRYABLE_STATUSES, max_attempts),
            ).fetchall()
            return [dict(row) for row in rows]

    def completed_actions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT type, target, payload, status, message_id, created_at FROM actions ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    # ----- pending approvals ----------------------------------------------------

    def add_pending_approval(self, action_type: str, target: str | None, payload: dict[str, Any] | None = None, message_id: str | None = None, thread_id: str | None = None) -> int:
        """Record an approval request and return its id (an identical open request is reused)."""

        with self._connect() as conn:
            if message_id:
                existing = conn.execute(
                    "SELECT id FROM pending_approvals WHERE message_id = ? AND action_type = ? AND status = 'open'",
                    (message_id, action_type),
                ).fetchone()
                if existing:
                    return int(existing["id"])

            cursor = conn.execute(
                "INSERT INTO pending_approvals (action_type, target, payload, message_id, thread_id) VALUES (?, ?, ?, ?, ?)",
                (action_type, target, json.dumps(payload or {}, default=str), message_id, thread_id),
            )
            return int(cursor.lastrowid)

    def get_approval(self, approval_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM pending_approvals WHERE id = ?", (approval_id,)).fetchone()

        if not row:
            return None

        approval = dict(row)
        approval["type"] = approval["action_type"]
        approval["payload"] = json.loads(approval["payload"] or "{}")
        approval["result"] = json.loads(approval["result"]) if approval.get("result") else None
        return approval

    def claim_approval(self, approval_id: int) -> bool:
        """Atomically move an open approval to 'executing'. False if it was already decided."""

        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE pending_approvals SET status = 'executing' WHERE id = ? AND status = 'open'",
                (approval_id,),
            )
            return cursor.rowcount > 0

    def finish_approval(self, approval_id: int, status: str, resolution: str, result: Any = None) -> None:
        """Record the final decision: 'approved', 'rejected', 'failed', or 'resolved'."""

        with self._connect() as conn:
            conn.execute(
                "UPDATE pending_approvals SET status = ?, resolution = ?, result = ?, decided_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, resolution[:2000], json.dumps(result, default=str), approval_id),
            )

    def resolve_pending_approval(self, approval_id: int, resolution: str = "resolved") -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE pending_approvals SET status = 'resolved', resolution = ?, decided_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND status = 'open'",
                (resolution, approval_id),
            )
            return cursor.rowcount > 0

    def pending_approvals(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, action_type AS type, target, payload, message_id, thread_id, created_at "
                "FROM pending_approvals WHERE status = 'open' ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def unnotified_approvals(self) -> list[dict[str, Any]]:
        """Open approvals the owner has not been shown yet (e.g. on Telegram)."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM pending_approvals WHERE status = 'open' AND notified = 0 ORDER BY id"
            ).fetchall()
        return [self.get_approval(row["id"]) for row in rows]

    def mark_approval_notified(self, approval_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE pending_approvals SET notified = 1 WHERE id = ?", (approval_id,))

    # ----- files the owner allowed Socio to send without asking -------------------

    def is_file_shareable(self, path: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM shared_files WHERE path = ?", (path,)).fetchone() is not None

    def allow_shared_file(self, path: str, name: str = "") -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO shared_files (path, name) VALUES (?, ?)", (path, name))

    def shared_files(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT path, name, created_at FROM shared_files ORDER BY name")]

    # ----- failures -------------------------------------------------------------

    def add_failure(self, source: str, message: str, message_id: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO failures (source, message, message_id) VALUES (?, ?, ?)",
                (source, message, message_id),
            )

    def failures(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT source, message, message_id, created_at FROM failures ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]
