"""
Human-in-the-loop approvals.

High-blast-radius actions are never done by the agent directly: booking,
moving, or cancelling meetings, trashing email, and sending attachments. The
agent records an approval request; only approve() below executes it, after the
owner says yes (Telegram button or `python -m app.main --approve ID`).
Everything is re-validated at execution time.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from app.agent.followups import complete_followup_task
from app.agent.state import AgentStateStore
from app.calendar.events import cancel_event, move_event, parse_ist
from app.calendar.tools import create_calendar_event
from app.files.local import resolve_shareable_file
from app.gmail.messages import send_reply, trash_message


logger = logging.getLogger(__name__)

# Approvals that perform an action when approved.
EXECUTABLE_TYPES = {"trash_email", "book_meeting", "move_event", "cancel_event", "send_reply_with_attachment"}
# Approvals that only ask the owner to look at something; approving closes them.
REVIEW_TYPES = {"review_draft", "review_decision"}


def format_time(value: str | None) -> str:
    """'2026-10-16T11:00:00+05:30' -> 'Fri 16 Oct, 11:00 IST'."""

    if not value:
        return "?"
    try:
        return parse_ist(value).strftime("%a %d %b, %H:%M IST")
    except ValueError:
        return value


def describe(approval: dict[str, Any]) -> str:
    """One human-readable description of what approving will do."""

    payload = approval.get("payload") or {}
    kind = approval.get("type") or approval.get("action_type")

    if kind == "trash_email":
        scope = "the whole conversation" if payload.get("whole_thread") else "the email"
        text = f"Move {scope} to Trash (recoverable for 30 days):\n'{payload.get('subject', '')}' from {payload.get('sender', '')}"
    elif kind == "book_meeting":
        text = (
            f"Book '{payload.get('title')}'\n"
            f"{format_time(payload.get('new_start'))} - {format_time(payload.get('new_end'))}\n"
            f"Invite: {payload.get('attendee_email') or 'nobody'}"
        )
        if payload.get("meeting_link"):
            text += f"\nLink: {payload['meeting_link']}"
    elif kind == "move_event":
        text = (
            f"Move calendar event '{payload.get('title')}'\n"
            f"from {format_time(payload.get('old_start'))}\n"
            f"to   {format_time(payload.get('new_start'))} - {format_time(payload.get('new_end'))}\n"
            f"Attendees are notified by Google: {', '.join(payload.get('other_attendees', [])) or 'none'}"
        )
    elif kind == "cancel_event":
        text = (
            f"Cancel calendar event '{payload.get('title')}' at {format_time(payload.get('old_start'))}\n"
            f"Attendees are notified by Google: {', '.join(payload.get('other_attendees', [])) or 'none'}"
        )
    elif kind == "send_reply_with_attachment":
        names = ", ".join(a["name"] for a in payload.get("attachments", []))
        to = (payload.get("reply") or {}).get("to", "the sender")
        text = f"Reply to {to} with: {names}\nMessage:\n\"{(payload.get('reply') or {}).get('body', '')}\""
    elif kind == "review_draft":
        text = f"A reply draft is waiting in Gmail Drafts.\nWhat to check: {payload.get('target') or approval.get('target', '')}"
    elif kind == "review_decision":
        text = f"Unsure whether to reply to '{payload.get('subject', '')}'.\n{payload.get('reason', '')}"
    else:
        text = approval.get("target") or kind

    if payload.get("reason") and kind in EXECUTABLE_TYPES:
        text += f"\nWhy: {payload['reason']}"

    reply = payload.get("reply")
    if reply and kind != "send_reply_with_attachment":
        text += f"\nThen reply to {reply.get('to')}:\n\"{reply.get('body')}\""

    return text


def _send_follow_up_reply(payload: dict[str, Any]) -> dict[str, Any] | None:
    reply = payload.get("reply")
    if not reply:
        return None

    return send_reply(
        thread_id=reply["thread_id"],
        message_id=reply["message_id"],
        to=reply["to"],
        subject="",
        body=reply["body"],
    )


def _execute(approval: dict[str, Any]) -> dict[str, Any]:
    payload = approval["payload"]
    kind = approval["type"]

    if kind == "trash_email":
        return {"trash": trash_message(payload["message_id"], whole_thread=bool(payload.get("whole_thread")))}

    if kind == "send_reply_with_attachment":
        reply = payload["reply"]
        attachments = [str(resolve_shareable_file(a["path"])) for a in payload["attachments"]]
        sent = send_reply(
            thread_id=reply["thread_id"], message_id=reply["message_id"], to=reply["to"],
            subject="", body=reply["body"], attachment_paths=attachments,
        )
        return {"reply": sent, "attachments": [a["name"] for a in payload["attachments"]]}

    if kind == "book_meeting":
        # create_calendar_event re-checks that the slot is still free.
        event = create_calendar_event.invoke({
            "title": payload["title"],
            "start_time": payload["new_start"],
            "end_time": payload["new_end"],
            "description": payload.get("description", ""),
            "attendee_email": payload.get("attendee_email", ""),
            "meeting_link": payload.get("meeting_link", ""),
        })
        if not event.get("verified"):
            raise RuntimeError(f"Google did not confirm the event: {event}")
        result: dict[str, Any] = {"event": event}
    elif kind == "move_event":
        result = {
            "event": move_event(payload["event_id"], parse_ist(payload["new_start"]), parse_ist(payload["new_end"]))
        }
    elif kind == "cancel_event":
        result = {"event": cancel_event(payload["event_id"])}
    else:
        raise ValueError(f"Unknown approval type: {kind}")

    # The calendar change succeeded; a failed reply must not hide that.
    try:
        result["reply"] = _send_follow_up_reply(payload)
    except Exception as exc:
        result["reply_error"] = f"{type(exc).__name__}: {exc}"

    return result


def approve(approval_id: int, store: AgentStateStore | None = None, always_allow: bool = False) -> dict[str, Any]:
    """
    Execute an approval. Safe to call twice: only the first call acts.

    always_allow (file shares only): also remember the attached files as
    shareable, so future email requests for them are sent without asking.
    """

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)

    if approval is None:
        return {"ok": False, "message": f"Approval #{approval_id} not found."}

    if approval["status"] != "open":
        return {"ok": False, "message": f"Approval #{approval_id} was already {approval['status']}."}

    if approval["type"] in REVIEW_TYPES:
        store.resolve_pending_approval(approval_id, "Marked done by owner")
        complete_followup_task(store, approval.get("message_id"))
        return {"ok": True, "message": f"#{approval_id} marked done."}

    if approval["type"] not in EXECUTABLE_TYPES:
        return {"ok": False, "message": f"Approval #{approval_id} has unknown type {approval['type']}."}

    if not store.claim_approval(approval_id):
        return {"ok": False, "message": f"Approval #{approval_id} is already being handled."}

    message_id, thread_id = approval.get("message_id"), approval.get("thread_id")

    if always_allow and approval["type"] == "send_reply_with_attachment":
        for attachment in approval["payload"].get("attachments", []):
            store.allow_shared_file(attachment["path"], attachment["name"])

    try:
        result = _execute(approval)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("Approval #%s failed: %s", approval_id, error)
        store.finish_approval(approval_id, "failed", error)
        store.add_failure(f"approval:{approval['type']}", error, message_id=message_id)
        if message_id:
            store.mark_processed(message_id, thread_id, status="needs_attention", summary=f"Approved change failed: {error}")
        return {"ok": False, "message": f"#{approval_id} failed, nothing was changed: {error}"}

    store.finish_approval(approval_id, "approved", "Approved and executed", result)
    complete_followup_task(store, message_id)
    store.add_action(approval["type"], approval.get("target"), approval["payload"], "completed", message_id=message_id, result=result)

    message = f"#{approval_id} done."
    if result.get("reply"):
        message += " Reply sent in the thread."
    if result.get("reply_error"):
        message += f" The calendar was updated, but the reply failed: {result['reply_error']}"

    if message_id and approval["type"] != "trash_email":
        status = "needs_attention" if result.get("reply_error") else ("replied" if result.get("reply") else "no_action")
        store.mark_processed(message_id, thread_id, status=status, summary=message)

    return {"ok": True, "message": message, "result": result}


def reject(approval_id: int, store: AgentStateStore | None = None) -> dict[str, Any]:
    """Reject an approval; nothing is changed."""

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)

    if approval is None:
        return {"ok": False, "message": f"Approval #{approval_id} not found."}

    if not store.claim_approval(approval_id):
        return {"ok": False, "message": f"Approval #{approval_id} was already {approval['status']}."}

    store.finish_approval(approval_id, "rejected", f"Rejected by owner at {datetime.now().isoformat(timespec='seconds')}")
    complete_followup_task(store, approval.get("message_id"))

    if approval.get("message_id") and approval["type"] in {"book_meeting", "move_event", "cancel_event", "send_reply_with_attachment"}:
        store.mark_processed(
            approval["message_id"], approval.get("thread_id"), status="needs_review",
            summary="Owner rejected the proposed change; handle this email manually.",
        )

    return {"ok": True, "message": f"#{approval_id} rejected. Nothing was changed."}
