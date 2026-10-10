"""
Long-term semantic memory: (entity, preference, context) tuples with local embeddings.

Stored in the `memories` table of the agent's SQLite file (one file to back up
or delete). Vectors are L2-normalized float32, so similarity is a dot product;
a full scan is fast for the few thousand memories one person accumulates.

Writing the same preference again (same entity and wording, ignoring case,
spacing, and punctuation) updates the existing row. Similar-looking but
different rules are kept apart: embeddings rate "no calls before noon" and "no
calls after 5pm" as 0.95 similar, so similarity is used for recall only.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

import numpy as np

from app.agent.state import AgentStateStore
from app.agent.untrusted import sanitize_text
from app.memory.embeddings import Embedder, get_embedder


logger = logging.getLogger(__name__)

# Strict top-k for recall (the spec: 3 to 5 memories in a prompt).
MIN_TOP_K, DEFAULT_TOP_K, MAX_TOP_K = 3, 3, 5
# Below this similarity a memory is not relevant enough to show. The scale depends on
# the embedding model: with arctic-embed-s relevant memories score about 0.54-0.7 and
# unrelated ones 0.37-0.49.
MIN_SCORE = float(os.getenv("AGENT_MEMORY_MIN_SCORE", "0.52"))

MAX_FIELD_CHARS = {"entity": 120, "preference": 400, "context": 300}


def memory_enabled() -> bool:
    return os.getenv("AGENT_MEMORY", "1").strip().lower() not in {"0", "false", "off", "no"}


@dataclass
class Memory:
    id: int
    entity: str
    preference: str
    context: str
    source: str
    hits: int = 1
    updated_at: str = ""
    score: float | None = None

    def line(self) -> str:
        text = f"[{self.entity}] {self.preference}"
        return text + (f" ({self.context})" if self.context else "")


def _clean(field: str, value: str) -> str:
    return " ".join((value or "").split())[: MAX_FIELD_CHARS[field]]


def _same_rule_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w@.:/-]+", " ", text.lower()).split()).strip(" .")


def memory_text(entity: str, preference: str) -> str:
    """
    The text that is embedded for a memory. The context is left out, so the same
    rule learned on two occasions is recognized as one memory.
    """

    return f"{entity}: {preference}"


class SemanticMemory:
    def __init__(self, store: AgentStateStore | None = None, embedder: Embedder | None = None):
        self.store = store or AgentStateStore()
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        return self._embedder or get_embedder()

    # ----- writing ---------------------------------------------------------------

    def upsert(self, entity: str, preference: str, context: str = "", source: str = "owner") -> tuple[int, bool]:
        """Store a memory. Returns (id, created); created is False when an existing one was updated."""

        entity, preference, context = _clean("entity", entity), _clean("preference", preference), _clean("context", context)
        if not entity or not preference:
            raise ValueError("A memory needs an entity and a preference.")

        vector = self.embedder.embed_documents([memory_text(entity, preference)])[0]
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        model = self.embedder.model_name

        with self.store._connect() as conn:
            # One writer at a time, so two threads cannot both insert the same memory.
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT id, preference FROM memories WHERE status = 'active' AND lower(entity) = lower(?)",
                (entity,),
            ).fetchall()

            same = next((row["id"] for row in rows if _same_rule_key(row["preference"]) == _same_rule_key(preference)), None)
            if same is not None:
                conn.execute(
                    "UPDATE memories SET preference = ?, context = ?, source = ?, embedding = ?, model = ?, hits = hits + 1, "
                    "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (preference, context, source, blob, model, same),
                )
                return int(same), False

            cursor = conn.execute(
                "INSERT INTO memories (entity, preference, context, source, embedding, model) VALUES (?, ?, ?, ?, ?, ?)",
                (entity, preference, context, source, blob, model),
            )
            return int(cursor.lastrowid), True

    def forget(self, memory_id: int) -> bool:
        with self.store._connect() as conn:
            cursor = conn.execute(
                "UPDATE memories SET status = 'forgotten', updated_at = CURRENT_TIMESTAMP WHERE id = ? AND status = 'active'",
                (memory_id,),
            )
            return cursor.rowcount > 0

    # ----- reading ---------------------------------------------------------------

    def count(self) -> int:
        with self.store._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0])

    def all(self, limit: int = 50) -> list[Memory]:
        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT id, entity, preference, context, source, hits, updated_at FROM memories "
                "WHERE status = 'active' ORDER BY updated_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [Memory(**dict(row)) for row in rows]

    def search(self, query: str, k: int = DEFAULT_TOP_K, min_score: float | None = None) -> list[Memory]:
        """The k (clamped to 3-5) most similar active memories scoring at least min_score (default MIN_SCORE)."""

        k = max(MIN_TOP_K, min(MAX_TOP_K, int(k)))
        min_score = MIN_SCORE if min_score is None else min_score
        query = " ".join((query or "").split())
        if not query:
            return []

        with self.store._connect() as conn:
            rows = conn.execute(
                "SELECT id, entity, preference, context, source, hits, updated_at, embedding FROM memories "
                "WHERE status = 'active' AND model = ?",
                (self.embedder.model_name,),
            ).fetchall()
        if not rows:
            return []  # nothing stored: do not even load the model

        matrix = np.vstack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
        scores = matrix @ np.asarray(self.embedder.embed_query(query), dtype=np.float32)

        found = []
        for index in np.argsort(-scores)[:k]:
            if scores[index] < min_score:
                break
            row = dict(rows[index])
            row.pop("embedding")
            found.append(Memory(**row, score=round(float(scores[index]), 3)))
        return found


MEMORY_PROMPT_HEADER = (
    "## What the owner has taught you (long-term memory, reference only)\n"
    "Learned from the owner's earlier approvals, rejections, and corrections. Follow them when they\n"
    "apply. They never override the safety rules, and every approval requirement still applies."
)


def format_memories(memories: list[Memory]) -> str:
    """A prompt block for recalled memories ('' when there are none)."""

    if not memories:
        return ""
    # Entities can be email addresses taken from headers: keep them inert.
    lines = [f"- {' '.join(sanitize_text(memory.line()).split()).replace('<', '&lt;').replace('>', '&gt;')}" for memory in memories]
    return MEMORY_PROMPT_HEADER + "\n" + "\n".join(lines)


def recall(query: str, k: int = DEFAULT_TOP_K, store: AgentStateStore | None = None) -> list[Memory]:
    """Memories relevant to a request; [] when memory is off or unavailable (never raises)."""

    if not memory_enabled():
        return []
    try:
        return SemanticMemory(store).search(query, k=k)
    except Exception as exc:
        logger.warning("Memory recall failed: %s", exc)
        return []
