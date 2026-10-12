"""
Rule conflicts: a new standing rule is checked against the owner's active rules before it is saved.

Two contradicting rules in one prompt ("never reply to recruiters" and "draft replies to
recruiters about AI roles") leave the model to guess. So before a standing rule is stored
(/remember, or the remember_rule tool in chat), one supervisor-chain call compares it with the
active rules. On a conflict nothing is saved: a "rule_conflict" approval card asks the owner to

- Replace Old Rule: the old rule becomes 'replaced', the new one is saved;
- Merge into One: the suggested merged rule is saved, the old one becomes 'replaced';
- Keep Both with Priority: both stay; the new one is saved noting it takes priority;
- Cancel: nothing changes.

The rules live in the memories table (app.memory.semantic); there is no second rule store.
If the model cannot be reached, the rule is saved as before (unchecked) and a warning is logged.
24-hour situational notes are not checked.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agent.state import AgentStateStore
from app.agent.untrusted import sanitize_text
from app.memory.semantic import SemanticMemory


logger = logging.getLogger(__name__)

# Rules compared per check (newest first); one person has a handful, this only bounds the prompt.
MAX_RULES_CHECKED = 40
CHOICES = ("replace", "merge", "both")
PRIORITY_NOTE = "Takes priority over: {old}"


class ConflictCheck(BaseModel):
    has_conflict: bool = Field(description="True only if following the new rule would break an existing rule in some case.")
    conflicting_rule_id: int | None = Field(default=None, description="The number of the existing rule it contradicts.")
    conflict_explanation: str = Field(default="", description="One short sentence, addressed to the owner as 'you'.")
    suggested_merged_rule: str = Field(default="", description="One imperative sentence that keeps both intentions.")


CONFLICT_PROMPT = """You check an email assistant's rules for contradictions.

Existing rules (number. [who it applies to] rule):
{rules}

New rule:
{candidate}

Does the new rule contradict one existing rule, i.e. is there a situation where following one
means breaking the other? Examples:
- "Never reply to recruiter emails" vs "Draft replies to recruiter emails mentioning AI roles": conflict.
- "No meetings on Friday afternoons" vs "Book team sync every Friday at 3 PM": conflict.
- "No meetings before noon" vs "Always tag receipt emails with #expenses": no conflict (unrelated).
- The same rule reworded, or a rule that only adds detail without contradicting: no conflict.
Rules about different people or documents do not conflict.

If there is a conflict, give the existing rule's number, a one-sentence explanation addressed to
the owner as "you", and one merged rule that keeps both intentions (the new rule as the
exception). Otherwise has_conflict=false."""


def _rule_line(entity: str, preference: str) -> str:
    return " ".join(sanitize_text(f"[{entity}] {preference}").split())


def check_conflict(candidates: list[tuple[str, str, str]], store: AgentStateStore | None = None) -> dict[str, Any] | None:
    """
    The conflict between new rules (entity, preference, context) and the active rules, or None.

    Returns {"existing_id", "existing_entity", "existing_text", "explanation", "merged"}.
    None also when there are no active rules (no model call) or the check could not run.
    """

    from app.agent.llm import get_llm, invoke_llm

    existing = SemanticMemory(store).all(limit=MAX_RULES_CHECKED)
    if not existing or not candidates:
        return None

    rules = "\n".join(f"{memory.id}. {_rule_line(memory.entity, memory.preference)}" for memory in existing)
    candidate = "\n".join(f"- {_rule_line(entity, preference)}" for entity, preference, _ in candidates)
    try:
        llm = get_llm().with_structured_output(ConflictCheck)
        result = invoke_llm(llm, CONFLICT_PROMPT.format(rules=rules, candidate=candidate))
        if result is not None and not isinstance(result, ConflictCheck):
            result = ConflictCheck.model_validate(result if isinstance(result, dict) else result.model_dump())
    except Exception as exc:
        logger.warning("Rule conflict check failed; saving the rule unchecked: %s", exc)
        return None

    if result is None or not result.has_conflict:
        return None
    by_id = {memory.id: memory for memory in existing}
    old = by_id.get(result.conflicting_rule_id or -1)
    if old is None:
        logger.warning("Rule conflict check named an unknown rule (%r); saving the rule.", result.conflicting_rule_id)
        return None

    # A rule identical to the old one is an update (SemanticMemory.upsert), never a conflict.
    if any(entity.lower() == old.entity.lower() and preference.strip().lower() == old.preference.strip().lower()
           for entity, preference, _ in candidates):
        return None

    return {
        "existing_id": old.id,
        "existing_entity": old.entity,
        "existing_text": old.preference,
        "explanation": " ".join(result.conflict_explanation.split())[:400],
        "merged": " ".join(result.suggested_merged_rule.split())[:400],
    }


def conflict_proposal(note: str, candidates: list[tuple[str, str, str]], conflict: dict[str, Any], source: str) -> dict[str, Any]:
    """The rule_conflict approval payload (with "type", as tools return proposals)."""

    return {
        "type": "rule_conflict",
        "note": note,
        "new_text": "; ".join(preference for _, preference, _ in candidates),
        "candidates": [list(item) for item in candidates],
        "source": source,
        **conflict,
    }


def describe_conflict(payload: dict[str, Any]) -> str:
    return f"New rule \"{payload.get('new_text')}\" conflicts with \"{payload.get('existing_text')}\""


def save_candidates(candidates: list[list[str]] | list[tuple[str, str, str]], source: str, store: AgentStateStore | None = None,
                    extra_context: str = "") -> list[dict[str, Any]]:
    memory = SemanticMemory(store)
    saved = []
    for entity, preference, context in candidates:
        context = "; ".join(part for part in (context, extra_context) if part)
        memory_id, created = memory.upsert(entity, preference, context, source=source)
        saved.append({"id": memory_id, "entity": entity, "preference": preference, "created": created})
    return saved


def resolve_conflict(payload: dict[str, Any], choice: str, store: AgentStateStore | None = None) -> dict[str, Any]:
    """Apply the owner's choice on a rule_conflict card. Returns {"message", "saved", "retired"}."""

    if choice not in CHOICES:
        raise ValueError(f"Unknown choice {choice!r}.")
    memory = SemanticMemory(store)
    source = payload.get("source") or "owner_rule"
    candidates = payload.get("candidates") or []
    old_id = payload.get("existing_id")

    if choice == "merge":
        merged = (payload.get("merged") or "").strip()
        if not merged:
            raise ValueError("There is no merged version of these rules. Choose Replace or Keep Both instead.")
        saved = save_candidates([(payload.get("existing_entity") or "owner", merged, "merged from two rules")], source, store)
        retired = bool(old_id) and memory.retire(int(old_id))
        return {"message": f"Merged into one rule: {merged}", "saved": saved, "retired": retired}

    if choice == "replace":
        retired = bool(old_id) and memory.retire(int(old_id))
        saved = save_candidates(candidates, source, store)
        return {"message": f"Replaced. New rule: {payload.get('new_text')}", "saved": saved, "retired": retired}

    saved = save_candidates(candidates, source, store, extra_context=PRIORITY_NOTE.format(old=payload.get("existing_text", "")))
    return {"message": "Kept both. The new rule wins when they clash.", "saved": saved, "retired": False}
