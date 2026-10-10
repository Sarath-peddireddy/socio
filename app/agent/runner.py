"""
Per-email processing loop.

    fetch email -> retrieve full thread -> decide whether action is needed
    -> agent chooses and executes tools -> verify results -> record outcome

Deterministic steps (fetching, deciding, recording) are plain Python; only
"choose tools and act" is delegated to the LangGraph agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from email.utils import parseaddr
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.decision import decide_reply_needed
from app.agent.followups import complete_followup_task, create_followup_task
from app.agent.graph import create_agent_graph
from app.agent.llm import LLMUnavailableError
from app.agent.state import AgentStateStore
from app.agent.tools import thread_participants
from app.gmail.messages import search_messages
from app.gmail.tools import get_thread_details
from app.google_api import GoogleApiUnavailableError


logger = logging.getLogger(__name__)

DEFAULT_QUERY = "in:inbox -from:me newer_than:7d"
MAX_BODY_CHARS = 4000
MAX_ATTEMPTS = 3
AGENT_RECURSION_LIMIT = 30


@dataclass
class MessageOutcome:
    message_id: str
    thread_id: str = ""
    sender: str = ""
    subject: str = ""
    status: str = ""
    intent: str = ""
    reason: str = ""
    summary: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)


def _latest_inbound(thread: list[dict]) -> dict | None:
    """Most recent message in the thread that the owner did not send."""

    for email in reversed(thread):
        if "SENT" not in email.get("label_ids", []):
            return email
    return None


def _thread_transcript(thread: list[dict]) -> str:
    parts = []

    for index, email in enumerate(thread, start=1):
        body = (email.get("body") or "").strip()
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + "\n[... truncated ...]"

        sent_by_owner = "SENT" in email.get("label_ids", [])
        attachments = ", ".join(a.get("filename", "") for a in email.get("attachments", [])) or "none"

        parts.append(
            f"--- Message {index} (id={email.get('id')}){' [sent by the owner]' if sent_by_owner else ''}\n"
            f"From: {email.get('from', '')}\n"
            f"Reply-To: {email.get('reply_to', '')}\n"
            f"To: {email.get('to', '')}\n"
            f"Cc: {email.get('cc', '')}\n"
            f"Date: {email.get('date', '')}\n"
            f"Subject: {email.get('subject', '')}\n"
            f"Attachments: {attachments}\n"
            f"<<<EMAIL BODY (untrusted)>>>\n{body}\n<<<END EMAIL BODY>>>"
        )

    return "\n\n".join(parts)


def _greeting_instruction(target: dict) -> str:
    """Tell the agent exactly whom to greet, so it never guesses a name."""

    sender_name = parseaddr(target.get("from", ""))[0].strip().strip('"')

    if sender_name and "@" not in sender_name:
        return f"Sender's name (from the From header): {sender_name}. Greet them by their first name."

    return (
        "The sender's name is unknown. Greet them neutrally (\"Hi,\") and do not guess "
        "a name from the email address."
    )


def _task_prompt(target: dict, thread: list[dict], decision, earlier_actions: list[dict] | None = None) -> str:
    reply_to = target.get("reply_to") or target.get("from", "")

    recovery = ""
    if earlier_actions:
        done = "\n".join(f"- {a['type']}: {a['result']}" for a in earlier_actions)
        recovery = (
            "\nAn earlier attempt at this email was interrupted after completing these actions. "
            "Do NOT repeat them; reuse their results and finish the remaining steps "
            f"(for example, reply with the time of the event already created):\n{done}\n"
        )

    return (
        "Handle the following email.\n\n"
        f"message_id to reply to: {target.get('id')}\n"
        f"thread_id: {target.get('thread_id')}\n"
        f"Reply recipient (from Reply-To/From): {reply_to}\n"
        f"{_greeting_instruction(target)}\n"
        f"Triage: intent={getattr(decision, 'intent', 'unknown')}, "
        f"confidence={decision.confidence}, reason={decision.reason}\n"
        f"{recovery}\n"
        "Full thread, oldest first:\n\n"
        f"{_thread_transcript(thread)}"
    )


def _final_text(result: dict) -> str:
    messages = result.get("messages", [])
    return messages[-1].text.strip() if messages else ""


def _outcome_status(actions: list[dict], failures: list[str]) -> str:
    """Map what the agent actually did (per tool results) to a final status."""

    by_type: dict[str, list[dict]] = {}
    for action in actions:
        by_type.setdefault(action["type"], []).append(action)

    if any(a["status"] == "dry_run" for a in actions):
        return "dry_run"

    if any(a["status"] == "unknown" for a in actions):
        return "needs_attention"

    sends = [a for a in by_type.get("send_gmail_reply", []) if a["status"] == "completed"]
    if sends:
        verified = all((a.get("result") or {}).get("verified") for a in sends)
        return "replied" if verified else "replied_unverified"

    if any(a["status"] == "proposed" for a in actions):
        return "awaiting_approval"

    if any(a["status"] == "completed" for a in by_type.get("create_reply_draft", [])):
        return "needs_review"

    if failures:
        return "failed"

    return "no_action"


def handle_message(message_id: str, thread_id: str, *, dry_run: bool = False, store: AgentStateStore | None = None) -> MessageOutcome:
    """Run the full loop for one email."""

    store = store or AgentStateStore()
    outcome = MessageOutcome(message_id=message_id, thread_id=thread_id)

    if store.is_processed(message_id):
        outcome.status = "skipped_already_handled"
        return outcome

    attempts = 0 if dry_run else store.start_attempt(message_id, thread_id)

    if attempts > MAX_ATTEMPTS:
        outcome.status = "failed_permanently"
        outcome.reason = f"Gave up after {MAX_ATTEMPTS} failed attempts."
        store.mark_processed(message_id, thread_id, status=outcome.status, summary=outcome.reason)
        return outcome

    # 1-2. Retrieve the full thread.
    thread = get_thread_details(thread_id)
    target = _latest_inbound(thread)

    if target is None:
        outcome.status = "no_action"
        outcome.reason = "Every message in this thread was sent by the owner."
    else:
        outcome.sender = parseaddr(target.get("from", ""))[1]
        outcome.subject = target.get("subject", "")

        # 3. Decide whether action is needed.
        decision = decide_reply_needed(thread)
        outcome.intent = getattr(decision, "intent", "")
        outcome.reason = decision.reason

        if not decision.reply_needed:
            outcome.status = "no_action"
        elif decision.confidence == "low":
            # Too uncertain to act autonomously: leave it for the owner.
            outcome.status = "needs_review"
            if not dry_run:
                store.add_pending_approval(
                    "review_decision",
                    f"Unsure whether to reply: {decision.reason}"[:500],
                    {"intent": outcome.intent, "reason": decision.reason, "subject": outcome.subject},
                    message_id=message_id,
                    thread_id=thread_id,
                )
        else:
            # Work completed by an interrupted earlier attempt (restart recovery).
            earlier_actions = [] if dry_run else [
                action for action in store.actions_for_message(message_id, status="completed")
                if not action["type"].startswith("auto_task")
            ]

            # 4-5. Let the agent choose and execute tools; tools verify their own results.
            agent = create_agent_graph()
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=_task_prompt(target, thread, decision, earlier_actions))],
                    "processed_message_ids": [],
                    "actions": [],
                    "pending_approvals": [],
                    "failures": [],
                    "target_message_id": target["id"],
                    "target_thread_id": thread_id,
                    "reply_recipient": target.get("reply_to") or target.get("from", ""),
                    "participants": sorted(thread_participants(thread)),
                    "dry_run": dry_run,
                },
                config={"recursion_limit": AGENT_RECURSION_LIMIT},
            )
            outcome.actions = earlier_actions + result.get("actions", [])
            outcome.summary = _final_text(result)
            outcome.status = _outcome_status(outcome.actions, result.get("failures", []))

    # 6. Record the outcome (never in dry-run mode, so a later real run still handles it).
    if not dry_run:
        store.mark_processed(
            message_id,
            thread_id,
            status=outcome.status,
            summary=(outcome.summary or outcome.reason)[:2000],
            metadata={"intent": outcome.intent, "reason": outcome.reason, "attempts": attempts},
        )

        # Put anything waiting on the owner into their Google Tasks.
        create_followup_task(
            store, message_id, thread_id, outcome.status, outcome.subject, outcome.sender,
            outcome.summary or outcome.reason,
        )

    return outcome


MAX_APPROVALS_TO_RECONCILE = 25


def reconcile_pending_approvals(store: AgentStateStore) -> int:
    """
    Close open approvals the owner has already dealt with.

    An approval is created when the latest message needed a reply; if the
    latest message in that thread is now one the owner sent, it is done.
    Returns the number of approvals closed.
    """

    closed = 0

    for approval in store.pending_approvals()[:MAX_APPROVALS_TO_RECONCILE]:
        if not approval.get("thread_id"):
            continue

        try:
            thread = get_thread_details(approval["thread_id"])
        except GoogleApiUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Could not check approval #%s: %s", approval["id"], exc)
            continue

        if thread and "SENT" in thread[-1].get("label_ids", []):
            if store.resolve_pending_approval(approval["id"], "Owner replied in the thread"):
                closed += 1
                complete_followup_task(store, approval.get("message_id"))

    return closed


def process_inbox(query: str = DEFAULT_QUERY, max_messages: int = 10, dry_run: bool = False) -> list[MessageOutcome]:
    """
    Process unhandled emails matching a Gmail search query.

    Only the newest listed message of each thread is handled; the thread
    context covers the older ones. Stops early if Google is unreachable.
    """

    store = AgentStateStore()
    newest_per_thread: dict[str, str] = {}
    older_in_thread: dict[str, list[str]] = {}

    # Recovery first: emails an earlier run left interrupted or failed, even if
    # they have since dropped out of the search window.
    if not dry_run:
        for record in store.retryable_messages(MAX_ATTEMPTS):
            if record["thread_id"]:
                newest_per_thread.setdefault(record["thread_id"], record["message_id"])

    try:
        if not dry_run:
            reconcile_pending_approvals(store)
        listed = search_messages(query=query, max_results=max_messages)
    except GoogleApiUnavailableError as exc:
        logger.error("Stopping run before processing: %s", exc)
        if not dry_run:
            store.add_failure("process_inbox", str(exc))
        return [MessageOutcome("-", status="failed", reason=str(exc))]

    for item in listed:  # Gmail lists newest first
        if item["threadId"] not in newest_per_thread:
            newest_per_thread[item["threadId"]] = item["id"]
        elif newest_per_thread[item["threadId"]] != item["id"]:
            older_in_thread.setdefault(item["threadId"], []).append(item["id"])

    outcomes: list[MessageOutcome] = []

    for thread_id, message_id in newest_per_thread.items():
        try:
            outcome = handle_message(message_id, thread_id, dry_run=dry_run, store=store)
        except (GoogleApiUnavailableError, LLMUnavailableError) as exc:
            # An outage is not this email's fault: stop the batch and do not
            # count the attempt, so the email is retried normally next run.
            logger.error("Stopping run: %s", exc)
            if not dry_run:
                store.add_failure("process_inbox", str(exc), message_id=message_id)
                store.abort_attempt(message_id, reason=str(exc))
            outcomes.append(MessageOutcome(message_id, thread_id, status="failed", reason=str(exc)))
            break
        except Exception as exc:
            logger.exception("Failed to process message %s", message_id)
            if not dry_run:
                store.add_failure("handle_message", f"{type(exc).__name__}: {exc}", message_id=message_id)
                store.mark_processed(message_id, thread_id, status="failed", summary=str(exc))
            outcomes.append(MessageOutcome(message_id, thread_id, status="failed", reason=f"{type(exc).__name__}: {exc}"))
            continue

        outcomes.append(outcome)

        if not dry_run and outcome.status not in {"failed", "skipped_already_handled"}:
            for older_id in older_in_thread.get(thread_id, []):
                if not store.is_processed(older_id):
                    store.mark_processed(older_id, thread_id, status="superseded", summary=f"Handled via newer message {message_id}")

    return outcomes


@dataclass
class OwnerCommandResult:
    reply: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    approval_ids: list[int] = field(default_factory=list)


def handle_owner_command(text: str, *, dry_run: bool = False, history: list | None = None) -> OwnerCommandResult:
    """
    Run the agent on a direct instruction from the owner (e.g. a Telegram message).

    The owner is trusted, so the full toolset is available; trashing email and
    moving/cancelling events still only create approval requests.
    history is optional earlier HumanMessage/AIMessage turns for context.
    """

    agent = create_agent_graph()
    result = agent.invoke(
        {
            "messages": list(history or []) + [HumanMessage(content=f"Instruction from the owner:\n{text}")],
            "processed_message_ids": [],
            "actions": [],
            "pending_approvals": [],
            "failures": [],
            "dry_run": dry_run,
        },
        config={"recursion_limit": AGENT_RECURSION_LIMIT},
    )

    return OwnerCommandResult(
        reply=_final_text(result) or "Done.",
        actions=result.get("actions", []),
        approval_ids=[approval["id"] for approval in result.get("pending_approvals", []) if "id" in approval],
    )
