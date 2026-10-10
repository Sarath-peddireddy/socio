"""
Human-in-the-loop approvals.

Actions that reach other people or cannot be undone are never done by the
agent directly: every email (replies to incoming email, and in owner mode
replies and new emails too), Google Docs/Sheets writes, booking, moving, or
cancelling meetings, trashing email, and trusting a skill script. The agent
records an approval request; only approve() below executes it, after the owner
says yes (Telegram button or `python -m app.main --approve ID`). A held email
can instead be moved to Gmail Drafts for editing (to_draft). Everything is
re-validated at execution time.

Every decision (approve, reject, Edit in Gmail) is also recorded in long-term
memory (app.memory.learning), so later runs can follow the owner's preferences.
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
from app.gmail.drafts import create_draft, create_reply_draft
from app.gmail.messages import send_new_email, send_reply, trash_message
from app.memory.learning import learn_from_decision


logger = logging.getLogger(__name__)

# Approvals that send an email reply.
REPLY_TYPES = {"send_reply", "send_reply_with_attachment"}
# Approvals that send any email; these can also be moved to Gmail Drafts (Edit in Gmail).
EMAIL_TYPES = REPLY_TYPES | {"send_email"}
# Owner-mode Google Docs / Sheets writes.
DOCUMENT_TYPES = {"create_document", "append_document", "create_spreadsheet", "append_sheet_row"}
# Approvals that perform an action when approved.
EXECUTABLE_TYPES = (
    {"trash_email", "book_meeting", "move_event", "cancel_event", "enable_skill_script"}
    | EMAIL_TYPES
    | DOCUMENT_TYPES
)
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
    elif kind == "send_reply":
        reply = payload.get("reply") or {}
        text = f"Reply to {reply.get('to', 'the sender')}:\n\"{reply.get('body', '')}\""
    elif kind == "send_reply_with_attachment":
        names = ", ".join(a["name"] for a in payload.get("attachments", []))
        to = (payload.get("reply") or {}).get("to", "the sender")
        text = f"Reply to {to} with: {names}\nMessage:\n\"{(payload.get('reply') or {}).get('body', '')}\""
    elif kind == "send_email":
        text = f"New email to {payload.get('to')}\nSubject: {payload.get('subject') or '(none)'}\n\"{payload.get('body', '')}\""
        if payload.get("attachments"):
            text += "\nFiles: " + ", ".join(a["name"] for a in payload["attachments"])
    elif kind == "create_document":
        text = f"Create Google Doc '{payload.get('title')}' in the Socio folder:\n{_diff_lines(payload.get('content', ''))}"
    elif kind == "append_document":
        text = f"Append to Google Doc '{payload.get('document_title') or payload.get('document')}':\n{_diff_lines(payload.get('text', ''))}"
    elif kind == "create_spreadsheet":
        headers = " | ".join(payload.get("headers") or []) or "(no header row)"
        text = f"Create Google Sheet '{payload.get('title')}' in the Socio folder:\n+ {headers}"
    elif kind == "append_sheet_row":
        header = payload.get("header") or []
        cells = [
            f"+ {header[i] if i < len(header) and header[i] else f'column {i + 1}'}: {value}"
            for i, value in enumerate(payload.get("values") or [])
        ]
        tab = f" (tab {payload['tab']})" if payload.get("tab") else ""
        text = f"Add a row to Google Sheet '{payload.get('document_title') or payload.get('document')}'{tab}:\n" + "\n".join(cells)
    elif kind == "enable_skill_script":
        text = (
            f"Trust skill script skills/{payload.get('name')}.py (sha256 {str(payload.get('sha256'))[:12]}…)?\n"
            "It will run with your user's permissions on files from your documents folder. "
            "Only approve code you wrote or reviewed.\n"
            f"First lines:\n{payload.get('preview', '')}"
        )
    elif kind == "review_draft":
        text = f"A reply draft is waiting in Gmail Drafts.\nWhat to check: {payload.get('target') or approval.get('target', '')}"
    elif kind == "review_decision":
        text = f"Unsure whether to reply to '{payload.get('subject', '')}'.\n{payload.get('reason', '')}"
    else:
        text = approval.get("target") or kind

    if payload.get("reason") and kind in EXECUTABLE_TYPES:
        text += f"\nWhy: {payload['reason']}"

    reply = payload.get("reply")
    if reply and kind not in REPLY_TYPES:
        text += f"\nThen reply to {reply.get('to')}:\n\"{reply.get('body')}\""

    return text


def _diff_lines(text: str, limit: int = 700) -> str:
    """Text to be added, as '+ ' lines (a short state diff for the approval card)."""

    text = text if len(text) <= limit else text[:limit] + "…"
    return "\n".join(f"+ {line}" for line in text.splitlines()) or "+ (empty)"


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

    if kind in REPLY_TYPES:
        reply = payload["reply"]
        attachments = [str(resolve_shareable_file(a["path"])) for a in payload.get("attachments", [])]
        sent = send_reply(
            thread_id=reply["thread_id"], message_id=reply["message_id"], to=reply["to"],
            subject="", body=reply["body"], attachment_paths=attachments,
        )
        return {"reply": sent, "attachments": [a["name"] for a in payload.get("attachments", [])]}

    if kind == "send_email":
        attachments = [str(resolve_shareable_file(a["path"])) for a in payload.get("attachments", [])]
        sent = send_new_email(to=payload["to"], subject=payload.get("subject", ""), body=payload["body"], attachment_paths=attachments)
        return {"email": sent, "attachments": [a["name"] for a in payload.get("attachments", [])]}

    if kind in DOCUMENT_TYPES:
        from app.workspace import docs, sheets

        if kind == "create_document":
            return {"document": docs.create_document(payload["title"], payload.get("content", ""))}
        if kind == "append_document":
            return {"document": docs.append_to_document(payload["document"], payload["text"])}
        if kind == "create_spreadsheet":
            return {"spreadsheet": sheets.create_spreadsheet(payload["title"], payload.get("headers") or None)}
        return {"spreadsheet": sheets.append_sheet_row(payload["document"], payload["values"], payload.get("tab", ""))}

    if kind == "enable_skill_script":
        from app.memory.skills import trust_script

        return {"skill": trust_script(payload["name"], payload["sha256"], approval_id=approval.get("id"))}

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


def approve(approval_id: int, store: AgentStateStore | None = None) -> dict[str, Any]:
    """Execute an approval. Safe to call twice: only the first call acts."""

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
    learn_from_decision(approval, "approved", store)
    store.add_action(approval["type"], approval.get("target"), approval["payload"], "completed", message_id=message_id, result=result)

    message = f"#{approval_id} done."
    if result.get("reply"):
        message += " Reply sent in the thread."
    if result.get("email"):
        message += " Email sent."
    for key in ("document", "spreadsheet"):
        if (result.get(key) or {}).get("url"):
            message += f" {result[key]['url']}"
    if result.get("skill"):
        message += " The script can run now."
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
    # A rejection is a correction: remember it (the owner can add why, see telegram_bot).
    learn_from_decision(approval, "rejected", store)

    if approval.get("message_id") and approval["type"] in {"book_meeting", "move_event", "cancel_event"} | REPLY_TYPES:
        store.mark_processed(
            approval["message_id"], approval.get("thread_id"), status="needs_review",
            summary="Owner rejected the proposed change; handle this email manually.",
        )

    return {"ok": True, "message": f"#{approval_id} rejected. Nothing was changed."}


def to_draft(approval_id: int, store: AgentStateStore | None = None) -> dict[str, Any]:
    """
    Move a held email to Gmail Drafts so the owner can edit and send it themselves.

    Nothing is sent. An incoming email is left as needs_review (its follow-up task stays open).
    """

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)

    if approval is None:
        return {"ok": False, "message": f"Approval #{approval_id} not found."}

    if approval["type"] not in EMAIL_TYPES:
        return {"ok": False, "message": f"Approval #{approval_id} is not an email, so it cannot be moved to Drafts."}

    if not store.claim_approval(approval_id):
        return {"ok": False, "message": f"Approval #{approval_id} was already {approval['status']}."}

    payload = approval["payload"]
    attachments = payload.get("attachments", [])

    try:
        # A Gmail draft made here carries at most one file; with several, the owner attaches them.
        attachment = str(resolve_shareable_file(attachments[0]["path"])) if len(attachments) == 1 else None
        if approval["type"] == "send_email":
            draft = create_draft(to=payload["to"], subject=payload.get("subject", ""), body=payload["body"], attachment_path=attachment)
        else:
            reply = payload["reply"]
            draft = create_reply_draft(message_id=reply["message_id"], body=reply["body"], attachment_path=attachment)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("Approval #%s: draft failed: %s", approval_id, error)
        store.finish_approval(approval_id, "failed", error)
        store.add_failure("approval:to_draft", error, message_id=approval.get("message_id"))
        return {"ok": False, "message": f"#{approval_id}: could not create the draft, nothing was sent: {error}"}

    store.finish_approval(approval_id, "resolved", "Moved to Gmail Drafts by owner", {"draft_id": draft.get("id")})
    learn_from_decision(approval, "edited", store)

    message = f"#{approval_id} saved to Gmail Drafts. Nothing was sent; edit and send it from Gmail."
    if len(attachments) > 1:
        message += " Attach the files there yourself."

    if approval.get("message_id"):
        store.mark_processed(approval["message_id"], approval.get("thread_id"), status="needs_review", summary=message)

    return {"ok": True, "message": message, "draft_id": draft.get("id")}
