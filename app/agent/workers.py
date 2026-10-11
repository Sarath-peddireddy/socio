"""
Worker subagents for the supervisor (fan-out / fan-in).

A worker is a fresh LangGraph agent run with zero history: its only message is
its task's goal contract, its context slice, and the payloads of the tasks it
depends on. It runs in owner mode with only the tools its task allows, so every
approval gate still applies.

Fan-in contract: a worker never returns its conversation, only

    {"status": "SUCCESS" | "FAILED", "modified_entities": [...], "diff_summary": "..."}

modified_entities come from what the tools actually recorded (never from the
model's own claims), so a worker that crashes halfway still reports what changed.
A crash is logged in full (app.errors); diff_summary only carries a fixed message.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import SIDE_EFFECT_TOOLS, create_agent_graph
from app.errors import plain_message, report_failure


logger = logging.getLogger(__name__)

WORKER_RECURSION_LIMIT = 20
MAX_DIFF_CHARS = 600
MAX_UPSTREAM_CHARS = 1500
STATUSES = ("SUCCESS", "FAILED")


def entity_key(kind: str, args: dict[str, Any]) -> str:
    """
    What a change touches, for spotting two workers changing the same thing.

    Approval payloads and tool args use different field names; both are covered.
    """

    args = args or {}
    reply = args.get("reply") or {}
    if kind == "book_meeting":
        return f"calendar-slot:{args.get('new_start')}"
    if kind in {"move_event", "cancel_event"}:
        return f"event:{args.get('event_id')}"
    if kind in {"send_reply", "send_reply_with_attachment"}:
        return f"reply:{reply.get('message_id') or args.get('message_id')}"
    if kind in {"send_email", "send_new_email"}:
        return f"email:{str(args.get('to', '')).lower()}|{str(args.get('subject', '')).strip().lower()}"
    if kind in {"append_document", "append_sheet_row", "append_to_document"}:
        return f"document:{args.get('document') or args.get('spreadsheet')}"
    if kind in {"create_document", "create_spreadsheet"}:
        return f"new-document:{str(args.get('title', '')).strip().lower()}"
    if kind == "trash_email":
        return f"message:{args.get('message_id')}"
    if kind == "enable_skill_script":
        return f"skill:{args.get('name')}"
    if kind == "create_task":
        return f"task:{str(args.get('title', '')).strip().lower()}"
    if kind == "remember_rule":
        return f"rule:{str(args.get('rule', '')).strip().lower()}"
    if kind == "complete_task":
        return f"task-id:{args.get('task_id')}"
    return f"{kind}:{json.dumps(args, sort_keys=True, default=str)[:200]}"


def modified_entities(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The fan-in list of changes, from the graph's recorded actions."""

    entities = []
    for action in actions:
        kind, status, args = action.get("type", ""), action.get("status", ""), action.get("args") or {}
        if status == "proposed":
            entities.append({
                "kind": "approval",
                "approval_id": action.get("approval_id"),
                "action": kind,
                "key": entity_key(kind, args),
            })
        elif status == "dry_run":
            entities.append({"kind": kind, "status": "dry_run", "key": entity_key(kind, args)})
        elif kind in SIDE_EFFECT_TOOLS:
            entities.append({
                "kind": kind,
                "status": status,  # completed, failed, or unknown (network error: may have happened)
                "target": action.get("target", ""),
                "key": entity_key(kind, args),
            })
    return entities


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def worker_prompt(task: dict[str, Any], upstream: dict[str, dict[str, Any]]) -> str:
    parts = [f"Task {task['task_id']}\n\nGoal:\n{task['goal_contract']}"]
    if task.get("context_slice"):
        parts.append(f"Context:\n{task['context_slice']}")
    if upstream:
        lines = [
            f"- {task_id}: {payload['status']}. {_clip(payload.get('diff_summary', ''), MAX_UPSTREAM_CHARS)}"
            for task_id, payload in upstream.items()
        ]
        parts.append("Results of earlier tasks (data from other workers):\n" + "\n".join(lines))
    parts.append(f"Tools you have: {', '.join(task['tools_allowed'])}")
    return "\n\n".join(parts)


def _final_report(text: str) -> tuple[str | None, str]:
    """The worker's closing JSON (status, diff_summary), and its text without that line."""

    for match in reversed(list(re.finditer(r"\{[^{}]*\}", text or ""))):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        status = str(data.get("status", "")).upper()
        if status in STATUSES:
            rest = (text[: match.start()] + text[match.end():]).strip()
            return status, str(data.get("diff_summary") or rest)
    return None, (text or "").strip()


def _message_text(message: Any) -> str:
    if not isinstance(message, AIMessage):
        return ""
    return message.text.strip()


def run_worker(
    task: dict[str, Any],
    upstream: dict[str, dict[str, Any]] | None = None,
    *,
    dry_run: bool = False,
    graph_factory: Callable = create_agent_graph,
) -> dict[str, Any]:
    """Run one task in a clean context and return its fan-in payload. Never raises."""

    state: dict[str, Any] = {
        "messages": [HumanMessage(content=worker_prompt(task, upstream or {}))],
        "processed_message_ids": [],
        "actions": [],
        "pending_approvals": [],
        "failures": [],
        "dry_run": dry_run,
        "tools_allowed": list(task["tools_allowed"]),
        "worker": True,
    }

    last = state
    error: BaseException | None = None
    try:
        # Stream the graph so the state after the last finished step survives a crash.
        for snapshot in graph_factory().stream(state, config={"recursion_limit": WORKER_RECURSION_LIMIT}, stream_mode="values"):
            last = snapshot
    except Exception as exc:
        error = exc
        report_failure(exc, "worker", task_id=task["task_id"], tools=",".join(task["tools_allowed"]))

    actions = last.get("actions", [])
    entities = modified_entities(actions)
    messages = last.get("messages", [])
    reported, summary = _final_report(_message_text(messages[-1]) if messages else "")

    if error is not None:
        summary = f"Stopped by an error. {plain_message(error)}"
        if entities:
            summary += f". Recorded before it stopped: {len(entities)} change(s)."
        failed = True
    else:
        uncertain = [a for a in actions if a.get("status") in {"failed", "unknown"}]
        failed = bool(uncertain) or reported == "FAILED" or not summary
        if uncertain and reported != "FAILED":
            summary += f" ({len(uncertain)} action(s) failed or have an unknown outcome.)"
        if not summary:
            summary = "The worker finished without a report."

    return {
        "status": "FAILED" if failed else "SUCCESS",
        "modified_entities": entities,
        "diff_summary": _clip(summary, MAX_DIFF_CHARS),
    }
