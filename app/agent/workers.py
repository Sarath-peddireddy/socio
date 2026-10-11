"""
Worker subagents for the supervisor (fan-out / fan-in).

A worker is a fresh LangGraph agent run with zero history: its only message is
its task's goal contract, its context slice, and the payloads of the tasks it
depends on. It runs in owner mode with only the tools its task allows, so every
approval gate still applies.

Fan-in contract: a worker never returns its conversation, only

    {"status": "SUCCESS" | "FAILED" | "NEEDS_INPUT", "modified_entities": [...], "diff_summary": "..."}

modified_entities come from what the tools actually recorded (never from the
model's own claims), so a worker that crashes halfway still reports what changed.
A crash is logged in full (app.errors); diff_summary only carries a fixed message.

Time budget: every model call (60 s) and Google call (30 s) already has its own
socket timeout, and a timed-out model call fails over to the next provider
(app.agent.llm_router). On top of that a worker has a wall-clock budget
(AGENT_WORKER_TIMEOUT, default 180 s), checked after every graph step: past it,
the worker stops before its next model call or tool round and reports FAILED.
Python threads cannot be killed, so a step already running finishes first (each
step is bounded by the timeouts above). Independent workers carry on; tasks that
depend on a stopped one are not started (supervisor.run_plan).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import SIDE_EFFECT_TOOLS, create_agent_graph
from app.errors import plain_message, report_failure


logger = logging.getLogger(__name__)

WORKER_RECURSION_LIMIT = 20
# Wall-clock budget per worker, in seconds (see the module docstring).
WORKER_TIMEOUT_SECONDS = float(os.getenv("AGENT_WORKER_TIMEOUT", "180"))
TIMEOUT_TEXT = "It took too long, so I stopped it. Please try this part again."
MAX_DIFF_CHARS = 600
MAX_UPSTREAM_CHARS = 1500
# NEEDS_INPUT: the worker could not go on without the person's answer; diff_summary is
# the question. It blocks dependent tasks like a failure but is shown as a question.
NEEDS_INPUT = "NEEDS_INPUT"
STATUSES = ("SUCCESS", "FAILED", NEEDS_INPUT)
# Statuses models write instead of SUCCESS when their change is waiting for approval:
# proposing it was the job, so the task succeeded.
WAITING_STATUSES = {"WAITING", "PENDING", "AWAITING_APPROVAL", "WAITING_FOR_APPROVAL", "PENDING_APPROVAL", "PROPOSED", "DONE", "COMPLETED"}


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
    if kind in {"create_document", "create_spreadsheet", "create_and_share", "create_and_share_file"}:
        return f"new-document:{str(args.get('title', '')).strip().lower()}"
    if kind in {"share_file", "share_drive_file"}:
        return f"share:{args.get('file_id') or args.get('file')}|{str(args.get('recipient') or args.get('email') or '').lower()}"
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
    """
    The worker's closing JSON (status, diff_summary), and its text without that line.

    Any status object is removed from the text, whatever status it names: the summary
    is read by the owner and must never carry raw JSON.
    """

    for match in reversed(list(re.finditer(r"\{[^{}]*\}", text or ""))):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or not ({"status", "diff_summary"} & set(data)):
            continue
        status = str(data.get("status", "")).strip().upper().replace(" ", "_")
        status = "SUCCESS" if status in WAITING_STATUSES else status
        status = NEEDS_INPUT if status in {"QUESTION", "NEEDS_CLARIFICATION", "NEED_INPUT", "NEEDS_ANSWER", "BLOCKED"} else status
        rest = (text[: match.start()] + text[match.end():]).strip()
        summary = str(data.get("diff_summary") or rest)
        return (status if status in STATUSES else None), summary
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
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run one task in a clean context and return its fan-in payload. Never raises."""

    budget = WORKER_TIMEOUT_SECONDS if timeout is None else timeout
    deadline = time.monotonic() + budget

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
    timed_out = False
    try:
        # Stream the graph so the state after the last finished step survives a crash.
        steps = graph_factory().stream(state, config={"recursion_limit": WORKER_RECURSION_LIMIT}, stream_mode="values")
        for snapshot in steps:
            last = snapshot
            if time.monotonic() > deadline:
                timed_out = True
                break
        if timed_out:
            close = getattr(steps, "close", None)
            if close:
                close()  # no further model call or tool round
            logger.warning("Worker %s stopped after its %.0f s budget.", task["task_id"], budget)
    except Exception as exc:
        error = exc
        report_failure(exc, "worker", task_id=task["task_id"], tools=",".join(task["tools_allowed"]))

    actions = last.get("actions", [])
    entities = modified_entities(actions)
    messages = last.get("messages", [])
    reported, summary = _final_report(_message_text(messages[-1]) if messages else "")

    if timed_out and error is None:
        summary = TIMEOUT_TEXT
        if entities:
            summary += " Part of it was done before it stopped."
        failed = True
    elif error is not None:
        summary = f"It stopped because of an error. {plain_message(error)}"
        if entities:
            summary += " Part of it was done before it stopped."
        failed = True
    else:
        uncertain = [a for a in actions if a.get("status") in {"failed", "unknown"}]
        failed = bool(uncertain) or reported == "FAILED" or not summary
        if uncertain and reported != "FAILED":
            summary += " (Something didn't go through, or I can't tell whether it happened.)"
        if not summary:
            summary = "It finished without saying what happened."

    asking = error is None and not failed and reported == NEEDS_INPUT
    return {
        "status": NEEDS_INPUT if asking else ("FAILED" if failed else "SUCCESS"),
        "modified_entities": entities,
        "diff_summary": _clip(summary, MAX_DIFF_CHARS),
    }
