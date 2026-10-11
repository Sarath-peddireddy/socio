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
    "awaiting_approval",   # a reply or calendar change waits for the owner's approval
    "superseded",          # a newer message in the same thread was handled instead
    "failed_permanently",  # failed too many times; will not be retried
)

# Non-final: picked up again on the next run.
RETRYABLE_STATUSES = ("in_progress", "failed")

# How long a slot proposed for a booking/move stays held while the owner decides.
SLOT_HOLD_HOURS = 24

# How long a process waits for another one's write before giving up.
BUSY_TIMEOUT_SECONDS = 30

# Lease on a run lock; renewed while the run makes progress, so a crashed
# process cannot block later runs for longer than this.
RUN_LOCK_MINUTES = 30


class AgentStateStore:
    """Persist agent progress so a restarted run does not repeat completed work."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or os.getenv("AGENT_STATE_DB") or DEFAULT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        # The bot and the CLI can run at the same time: wait for a writer instead of
        # failing with "database is locked".
        conn = sqlite3.connect(self.db_path, timeout=BUSY_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        """Add a column to a table created by an older version of this file."""

        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _initialize(self) -> None:
        """
        Apply schema migrations in order (tracked with PRAGMA user_version).

        Each migration runs once, inside a write transaction, so two processes
        starting at the same time cannot both apply it.
        """

        conn = sqlite3.connect(self.db_path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            # Write-ahead logging: readers never block the writer (persists in the file).
            conn.execute("PRAGMA journal_mode=WAL")
            if conn.execute("PRAGMA user_version").fetchone()[0] >= SCHEMA_VERSION:
                return

            conn.execute("BEGIN IMMEDIATE")
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                for number, migrate in MIGRATIONS:
                    if number > version:
                        migrate(conn)
                        conn.execute(f"PRAGMA user_version = {int(number)}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    @classmethod
    def _migrate_v1_baseline(cls, conn: sqlite3.Connection) -> None:
        """Tables from before versioned migrations (idempotent, so it also upgrades old files)."""

        conn.execute(
            "CREATE TABLE IF NOT EXISTS run_locks (name TEXT PRIMARY KEY, holder TEXT NOT NULL, expires_at TEXT NOT NULL)"
        )
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
            CREATE TABLE IF NOT EXISTS slot_holds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                approval_id INTEGER NOT NULL,
                message_id TEXT,
                label TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cls._ensure_column(conn, "processed_messages", "attempts", "INTEGER NOT NULL DEFAULT 0")
        cls._ensure_column(conn, "processed_messages", "updated_at", "TEXT")
        cls._ensure_column(conn, "actions", "message_id", "TEXT")
        cls._ensure_column(conn, "actions", "result", "TEXT")
        cls._ensure_column(conn, "pending_approvals", "message_id", "TEXT")
        cls._ensure_column(conn, "pending_approvals", "thread_id", "TEXT")
        cls._ensure_column(conn, "pending_approvals", "status", "TEXT NOT NULL DEFAULT 'open'")
        cls._ensure_column(conn, "pending_approvals", "resolution", "TEXT")
        cls._ensure_column(conn, "pending_approvals", "result", "TEXT")
        cls._ensure_column(conn, "pending_approvals", "notified", "INTEGER NOT NULL DEFAULT 0")
        cls._ensure_column(conn, "pending_approvals", "decided_at", "TEXT")
        cls._ensure_column(conn, "failures", "message_id", "TEXT")

    @staticmethod
    def _migrate_v2_memory_and_runs(conn: sqlite3.Connection) -> None:
        """Long-term memory, supervisor run checkpoints, and trusted skill scripts."""

        # Semantic memory (app.memory.semantic): one (entity, preference, context) tuple
        # per row with its embedding (float32 bytes, L2-normalized).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT NOT NULL,
                preference TEXT NOT NULL,
                context TEXT,
                source TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                hits INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Supervisor runs (app.agent.supervisor): the plan and every worker's fan-in payload.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS supervisor_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request TEXT NOT NULL,
                plan TEXT,
                status TEXT NOT NULL DEFAULT 'planning',
                report TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS worker_tasks (
                run_id INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                depends_on TEXT NOT NULL DEFAULT '[]',
                tools_allowed TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'pending',
                payload TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, task_id)
            )
            """
        )
        # Python skill scripts the owner approved, pinned to the exact file contents.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trusted_skill_scripts (
                name TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                approval_id INTEGER,
                approved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (name, sha256)
            )
            """
        )
        # Owner corrections asked for after a Reject / Edit in Gmail (Telegram reply prompt).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_requests (
                chat_message_id INTEGER PRIMARY KEY,
                approval_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    @classmethod
    def _migrate_v3_notes_and_workflows(cls, conn: sqlite3.Connection) -> None:
        """Short-lived owner notes, workflow success counts, and which runs replayed a skill."""

        # Situational notes from the owner ("already talked to him"): expire after a day.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT NOT NULL,
                note TEXT NOT NULL,
                approval_id INTEGER,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Successful multi-step runs per tool-graph shape, for the "save as a skill?" hint.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_stats (
                signature TEXT PRIMARY KEY,
                successes INTEGER NOT NULL DEFAULT 0,
                last_run_id INTEGER,
                suggested_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cls._ensure_column(conn, "supervisor_runs", "skill", "TEXT")

    @classmethod
    def _migrate_v4_tracked_tasks(cls, conn: sqlite3.Connection) -> None:
        """The owner's Google Tasks as Socio last saw them (app.tasks.tracker)."""

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracked_tasks (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                due TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                closed_how TEXT,
                due_asked_for TEXT,
                cleanup_asked INTEGER NOT NULL DEFAULT 0,
                first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    @classmethod
    def _migrate_v5_conversation_context(cls, conn: sqlite3.Connection) -> None:
        """Telegram conversation context: the active entities and recent turns per chat (app.memory.context_*)."""

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_anchors (
                chat_id INTEGER PRIMARY KEY,
                active_thread_id TEXT,
                active_contact TEXT,
                active_document TEXT,
                metadata_json TEXT DEFAULT '{}',
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                owner_text TEXT NOT NULL,
                socio_text TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_turns_chat ON conversation_turns (chat_id, archived, id)")

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
        self.release_slot_holds(approval_id)

    def resolve_pending_approval(self, approval_id: int, resolution: str = "resolved") -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE pending_approvals SET status = 'resolved', resolution = ?, decided_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND status = 'open'",
                (resolution, approval_id),
            )
            resolved = cursor.rowcount > 0
        if resolved:
            self.release_slot_holds(approval_id)
        return resolved

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

    def update_unshown_approval(self, approval_id: int, target: str, payload: dict[str, Any]) -> bool:
        """Change an open approval the owner has not been shown yet. False if it was shown or decided."""

        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE pending_approvals SET target = ?, payload = ? WHERE id = ? AND status = 'open' AND notified = 0",
                (target, json.dumps(payload, default=str), approval_id),
            )
            return cursor.rowcount > 0

    def mark_approval_notified(self, approval_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE pending_approvals SET notified = 1 WHERE id = ?", (approval_id,))

    # ----- soft holds on calendar slots (see app.calendar.holds) -----------------

    def add_slot_hold(self, approval_id: int, start_time: str, end_time: str, label: str = "", message_id: str | None = None, hours: int = SLOT_HOLD_HOURS) -> None:
        """Hold a proposed slot for an approval (one hold per approval; a repeat is ignored)."""

        with self._connect() as conn:
            if conn.execute(
                "SELECT 1 FROM slot_holds WHERE approval_id = ? AND status = 'active'", (approval_id,)
            ).fetchone():
                return
            conn.execute(
                "INSERT INTO slot_holds (approval_id, message_id, label, start_time, end_time, expires_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now', ?))",
                (approval_id, message_id, label, start_time, end_time, f"+{int(hours)} hours"),
            )

    def active_slot_holds(self) -> list[dict[str, Any]]:
        """Holds that are neither released nor expired."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT approval_id, message_id, label, start_time, end_time, expires_at FROM slot_holds "
                "WHERE status = 'active' AND expires_at > CURRENT_TIMESTAMP ORDER BY start_time"
            ).fetchall()
            return [dict(row) for row in rows]

    def release_slot_holds(self, approval_id: int) -> None:
        """Free an approval's slot once it is decided (approved, rejected, failed, or resolved)."""

        with self._connect() as conn:
            conn.execute("UPDATE slot_holds SET status = 'released' WHERE approval_id = ? AND status = 'active'", (approval_id,))

    # ----- run locks (one inbox run at a time across processes) -----------------

    def acquire_lock(self, name: str, holder: str, minutes: int = RUN_LOCK_MINUTES) -> bool:
        """Take (or renew) a named lease. False if another holder has a lease that has not expired."""

        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO run_locks (name, holder, expires_at) VALUES (?, '', datetime('now', '-1 second'))",
                (name,),
            )
            cursor = conn.execute(
                "UPDATE run_locks SET holder = ?, expires_at = datetime('now', ?) "
                "WHERE name = ? AND (holder = ? OR expires_at <= CURRENT_TIMESTAMP)",
                (holder, f"+{int(minutes)} minutes", name, holder),
            )
            return cursor.rowcount > 0

    def release_lock(self, name: str, holder: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE run_locks SET holder = '', expires_at = datetime('now', '-1 second') WHERE name = ? AND holder = ?",
                (name, holder),
            )

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

    # ----- supervisor run checkpoints (see app.agent.supervisor) -----------------

    def start_supervisor_run(self, request: str) -> int:
        with self._connect() as conn:
            return int(conn.execute("INSERT INTO supervisor_runs (request) VALUES (?)", (request[:4000],)).lastrowid)

    def save_supervisor_plan(self, run_id: int, plan: list[dict[str, Any]], skill: str = "") -> None:
        """Store the task DAG (and the skill it came from, if any) and one 'pending' checkpoint row per worker task."""

        with self._connect() as conn:
            conn.execute(
                "UPDATE supervisor_runs SET plan = ?, skill = ?, status = 'running', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(plan), skill or None, run_id),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO worker_tasks (run_id, task_id, goal, depends_on, tools_allowed) VALUES (?, ?, ?, ?, ?)",
                [
                    (run_id, task["task_id"], task["goal_contract"], json.dumps(task.get("depends_on", [])), json.dumps(task.get("tools_allowed", [])))
                    for task in plan
                ],
            )

    def update_worker_task(self, run_id: int, task_id: str, status: str, payload: dict[str, Any] | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE worker_tasks SET status = ?, payload = COALESCE(?, payload), updated_at = CURRENT_TIMESTAMP "
                "WHERE run_id = ? AND task_id = ?",
                (status, json.dumps(payload, default=str) if payload is not None else None, run_id, task_id),
            )

    def worker_tasks(self, run_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM worker_tasks WHERE run_id = ? ORDER BY rowid", (run_id,)).fetchall()
        tasks = []
        for row in rows:
            task = dict(row)
            task["depends_on"] = json.loads(task["depends_on"])
            task["tools_allowed"] = json.loads(task["tools_allowed"])
            task["payload"] = json.loads(task["payload"]) if task["payload"] else None
            tasks.append(task)
        return tasks

    def finish_supervisor_run(self, run_id: int, status: str, report: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE supervisor_runs SET status = ?, report = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, report[:4000], run_id),
            )

    def get_supervisor_run(self, run_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM supervisor_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        run = dict(row)
        run["plan"] = json.loads(run["plan"]) if run["plan"] else None
        return run

    def last_successful_plan_run(self, min_tasks: int = 2) -> dict[str, Any] | None:
        """The newest run whose tasks all succeeded, with at least min_tasks tasks, not replayed from a skill."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, request, plan FROM supervisor_runs WHERE status = 'done' AND skill IS NULL "
                "AND plan IS NOT NULL ORDER BY id DESC LIMIT 20"
            ).fetchall()
        for row in rows:
            plan = json.loads(row["plan"])
            if len(plan) >= min_tasks:
                return {"id": row["id"], "request": row["request"], "plan": plan}
        return None

    def record_workflow_success(self, signature: str, run_id: int) -> dict[str, Any]:
        """Count one more success for a tool-graph shape. Returns {successes, suggested}."""

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO workflow_stats (signature, successes, last_run_id) VALUES (?, 1, ?) "
                "ON CONFLICT(signature) DO UPDATE SET successes = successes + 1, last_run_id = excluded.last_run_id, "
                "updated_at = CURRENT_TIMESTAMP",
                (signature, run_id),
            )
            row = conn.execute("SELECT successes, suggested_at FROM workflow_stats WHERE signature = ?", (signature,)).fetchone()
        return {"successes": int(row["successes"]), "suggested": row["suggested_at"] is not None}

    def mark_workflow_suggested(self, signature: str) -> bool:
        """Remember that the owner was offered /save_skill for this shape. False if already offered."""

        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE workflow_stats SET suggested_at = CURRENT_TIMESTAMP WHERE signature = ? AND suggested_at IS NULL",
                (signature,),
            )
            return cursor.rowcount > 0

    def close_stale_supervisor_runs(self, minutes: int = RUN_LOCK_MINUTES) -> int:
        """Mark runs left 'planning'/'running' by a stopped process as interrupted (not re-run)."""

        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE supervisor_runs SET status = 'interrupted', updated_at = CURRENT_TIMESTAMP "
                "WHERE status IN ('planning', 'running') AND updated_at <= datetime('now', ?)",
                (f"-{int(minutes)} minutes",),
            )
            conn.execute(
                "UPDATE worker_tasks SET status = 'interrupted', updated_at = CURRENT_TIMESTAMP "
                "WHERE status IN ('pending', 'running') AND run_id IN (SELECT id FROM supervisor_runs WHERE status = 'interrupted')"
            )
            return cursor.rowcount

    def interrupted_supervisor_runs(self) -> list[dict[str, Any]]:
        """Recent runs a stopped process left unfinished (never re-run automatically: they may have sent approvals)."""

        self.close_stale_supervisor_runs()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, request, status, created_at FROM supervisor_runs "
                "WHERE status = 'interrupted' AND updated_at > datetime('now', '-7 days') ORDER BY id"
            ).fetchall()
            return [dict(row) for row in rows]

    # ----- trusted skill scripts (see app.memory.skills) --------------------------

    def trust_skill_script(self, name: str, sha256: str, approval_id: int | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trusted_skill_scripts (name, sha256, approval_id) VALUES (?, ?, ?)",
                (name, sha256, approval_id),
            )

    def is_skill_script_trusted(self, name: str, sha256: str) -> bool:
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM trusted_skill_scripts WHERE name = ? AND sha256 = ?", (name, sha256)
            ).fetchone() is not None

    # ----- situational notes (expire; see app.memory.episodic) ----------------------

    def add_episodic_note(self, entity: str, note: str, hours: int, approval_id: int | None = None) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO episodic_notes (entity, note, approval_id, expires_at) VALUES (?, ?, ?, datetime('now', ?))",
                (entity, note, approval_id, f"{int(hours):+d} hours"),
            )
            return int(cursor.lastrowid)

    def active_episodic_notes(self, limit: int = 5) -> list[dict[str, Any]]:
        """Notes that have not expired, newest first."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, entity, note, approval_id, expires_at, created_at FROM episodic_notes "
                "WHERE expires_at > CURRENT_TIMESTAMP ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    # ----- tracked Google Tasks (app.tasks.tracker) ---------------------------------

    def upsert_tracked_task(self, task_id: str, title: str, due: str = "") -> bool:
        """Record an open task (new, renamed, re-dated, or reopened). Returns True if it is new."""

        with self._connect() as conn:
            known = conn.execute("SELECT 1 FROM tracked_tasks WHERE task_id = ?", (task_id,)).fetchone()
            conn.execute(
                """
                INSERT INTO tracked_tasks (task_id, title, due) VALUES (?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    title = excluded.title, due = excluded.due, status = 'open', closed_how = NULL,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (task_id, title, due or None),
            )
            return known is None

    def tracked_tasks(self, status: str = "open") -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT task_id, title, due, status, closed_how, due_asked_for, cleanup_asked, first_seen FROM tracked_tasks "
                "WHERE status = ? ORDER BY COALESCE(due, '9999-12-31'), first_seen",
                (status,),
            ).fetchall()
            return [dict(row) for row in rows]

    def close_tracked_task(self, task_id: str, how: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE tracked_tasks SET status = 'closed', closed_how = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE task_id = ? AND status = 'open'",
                (how, task_id),
            )
            return cursor.rowcount > 0

    def mark_due_asked(self, task_id: str, due: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE tracked_tasks SET due_asked_for = ?, updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (due, task_id),
            )

    def mark_cleanup_asked(self, task_ids: list[str]) -> None:
        with self._connect() as conn:
            conn.executemany("UPDATE tracked_tasks SET cleanup_asked = 1 WHERE task_id = ?", [(i,) for i in task_ids])

    def open_approvals_of_type(self, action_type: str) -> list[dict[str, Any]]:
        """Open approval requests of one type, with their payloads."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM pending_approvals WHERE status = 'open' AND action_type = ? ORDER BY id", (action_type,)
            ).fetchall()
        return [self.get_approval(row["id"]) for row in rows]

    # ----- Telegram rule prompts (no longer used: the "+ Add Permanent Rule" button was removed)

    def add_feedback_request(self, chat_message_id: int, approval_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO feedback_requests (chat_message_id, approval_id) VALUES (?, ?)",
                (chat_message_id, approval_id),
            )

    def pop_feedback_request(self, chat_message_id: int) -> int | None:
        """The approval a Telegram prompt asked about; each prompt is answered once."""

        with self._connect() as conn:
            row = conn.execute(
                "DELETE FROM feedback_requests WHERE chat_message_id = ? RETURNING approval_id", (chat_message_id,)
            ).fetchone()
            return int(row["approval_id"]) if row else None

    # ----- Telegram conversation context (app.memory.context_anchor / context_assembler)

    def upsert_conversation_anchor(self, chat_id: int, fields: dict[str, str], metadata: dict[str, Any] | None = None) -> None:
        """Set the given anchor fields (others keep their value) and merge metadata; refreshes updated_at."""

        allowed = {"active_thread_id", "active_contact", "active_document"}
        fields = {name: value for name, value in fields.items() if name in allowed}
        with self._connect() as conn:
            row = conn.execute("SELECT metadata_json FROM conversation_anchors WHERE chat_id = ?", (chat_id,)).fetchone()
            merged = {**json.loads((row["metadata_json"] if row else None) or "{}"), **(metadata or {})}
            conn.execute("INSERT OR IGNORE INTO conversation_anchors (chat_id) VALUES (?)", (chat_id,))
            assignments = "".join(f"{name} = ?, " for name in fields)
            conn.execute(
                f"UPDATE conversation_anchors SET {assignments}metadata_json = ?, updated_at = datetime('now') WHERE chat_id = ?",
                (*fields.values(), json.dumps(merged, default=str), chat_id),
            )

    def conversation_anchor(self, chat_id: int, max_age_hours: int) -> dict[str, Any] | None:
        """The anchor if it was updated within max_age_hours, else None."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT active_thread_id, active_contact, active_document, metadata_json, updated_at "
                "FROM conversation_anchors WHERE chat_id = ? AND updated_at >= datetime('now', ?)",
                (chat_id, f"-{int(max_age_hours)} hours"),
            ).fetchone()
        if not row:
            return None
        anchor = dict(row)
        anchor["metadata"] = json.loads(anchor.pop("metadata_json") or "{}")
        return anchor

    def clear_conversation_anchor(self, chat_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_anchors WHERE chat_id = ?", (chat_id,))

    def add_conversation_turn(self, chat_id: int, owner_text: str, socio_text: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO conversation_turns (chat_id, owner_text, socio_text) VALUES (?, ?, ?)",
                (chat_id, owner_text, socio_text),
            )
            return int(cursor.lastrowid)

    def recent_conversation_turns(self, chat_id: int, limit: int, max_age_hours: int) -> list[dict[str, Any]]:
        """The newest unarchived turns within max_age_hours, oldest first."""

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, owner_text, socio_text, created_at FROM conversation_turns "
                "WHERE chat_id = ? AND archived = 0 AND created_at >= datetime('now', ?) ORDER BY id DESC LIMIT ?",
                (chat_id, f"-{int(max_age_hours)} hours", limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def append_to_last_conversation_turn(self, chat_id: int, text: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE conversation_turns SET socio_text = socio_text || ? WHERE id = "
                "(SELECT MAX(id) FROM conversation_turns WHERE chat_id = ? AND archived = 0)",
                (text, chat_id),
            )
            return cursor.rowcount > 0

    def archive_conversation_turns(self, chat_id: int) -> int:
        """Stop loading this chat's earlier turns (/new, /clear, /reset). The rows are kept."""

        with self._connect() as conn:
            cursor = conn.execute("UPDATE conversation_turns SET archived = 1 WHERE chat_id = ? AND archived = 0", (chat_id,))
            return cursor.rowcount


# Ordered schema migrations; PRAGMA user_version records the last one applied.
# Append new ones at the end; never edit or reorder a released migration.
MIGRATIONS = [
    (1, AgentStateStore._migrate_v1_baseline),
    (2, AgentStateStore._migrate_v2_memory_and_runs),
    (3, AgentStateStore._migrate_v3_notes_and_workflows),
    (4, AgentStateStore._migrate_v4_tracked_tasks),
    (5, AgentStateStore._migrate_v5_conversation_context),
]
SCHEMA_VERSION = MIGRATIONS[-1][0]
