"""
Human-in-the-loop approvals.

Actions that reach other people or cannot be undone are never done by the
agent directly: every email (replies to incoming email, and in owner mode
replies and new emails too), Google Docs/Sheets writes, sharing a file Socio
created (and "create, share, and email the link" as one approval), booking,
moving, or cancelling meetings, trashing email, and trusting a skill script. The agent
records an approval request; only approve() below executes it, after the owner
says yes (Telegram button or `python -m app.main --approve ID`). A held email
can instead be moved to Gmail Drafts for editing (to_draft). Everything is
re-validated at execution time.

A decision teaches Socio nothing by itself; the owner can turn a rejection or an
edit into a rule explicitly (by asking Socio to remember it, see app.memory.learning).
Failures are logged in full (app.errors) and reported with a short fixed message.
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
from app.errors import report_failure, safe_error_text


logger = logging.getLogger(__name__)

# Approvals that send an email reply.
REPLY_TYPES = {"send_reply", "send_reply_with_attachment"}
# Approvals that send any email; these can also be moved to Gmail Drafts (Edit in Gmail).
EMAIL_TYPES = REPLY_TYPES | {"send_email"}
# Owner-mode Google Docs / Sheets writes.
DOCUMENT_TYPES = {"create_document", "append_document", "create_spreadsheet", "append_sheet_row"}
# Giving someone access to a file Socio created (and, for create_and_share, creating it
# first and emailing them the link).
SHARE_TYPES = {"create_and_share", "share_file"}
# Approvals that perform an action when approved.
EXECUTABLE_TYPES = (
    {"trash_email", "book_meeting", "move_event", "cancel_event", "enable_skill_script", "create_task", "complete_task",
     "close_old_reminders"}
    | EMAIL_TYPES
    | DOCUMENT_TYPES
    | SHARE_TYPES
)
# Approvals that only ask the owner to look at something; approving closes them.
REVIEW_TYPES = {"review_draft", "review_decision"}

# How a decided request is described to the owner ("This was already ...").
STATUS_WORDS = {
    "approved": "approved", "rejected": "rejected", "resolved": "handled",
    "failed": "tried and it failed", "executing": "being handled",
}

# What the owner is told after approving (no internal numbers; the CLI adds them).
DONE_TEXT = {
    "trash_email": "Moved to Trash. You can restore it from Gmail Trash for 30 days.",
    "send_reply": "Reply sent.",
    "send_reply_with_attachment": "Reply sent with the file.",
    "send_email": "Email sent.",
    "book_meeting": "Meeting booked.",
    "move_event": "Meeting moved.",
    "cancel_event": "Meeting cancelled.",
    "create_document": "Document created.",
    "append_document": "Text added to the document.",
    "create_spreadsheet": "Spreadsheet created.",
    "append_sheet_row": "Row added to the sheet.",
    "enable_skill_script": "Script approved. It can run now.",
    "create_task": "Task added to Google Tasks.",
    "complete_task": "Task marked as completed.",
}


def _already(approval: dict[str, Any]) -> str:
    return f"This request was already {STATUS_WORDS.get(approval['status'], approval['status'])}."


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
    elif kind == "create_and_share":
        what = "Google Sheet" if payload.get("kind") == "spreadsheet" else "Google Doc"
        text = (
            f"Create {what} '{payload.get('title')}', give {payload.get('recipient')} {payload.get('role', 'editor')} access, "
            f"and email them the link\nSubject: {payload.get('subject') or '(none)'}\n\"{payload.get('body', '')}\""
        )
        if payload.get("headers"):
            text += "\nColumns: " + " | ".join(payload["headers"])
    elif kind == "share_file":
        text = f"Give {payload.get('recipient')} {payload.get('role', 'editor')} access to '{payload.get('title')}'"
    elif kind == "create_task":
        due = f" (due {payload['due']})" if payload.get("due") else ""
        text = f"Add Google Task '{payload.get('title', '')}'{due}"
        if payload.get("notes"):
            text += f"\nNotes: {payload['notes'][:300]}"
    elif kind == "close_old_reminders":
        text = (
            f"Close {payload.get('count', 0)} old reminder tasks Socio added automatically "
            f"(e.g. {', '.join(repr(e) for e in payload.get('examples', [])[:3])})"
        )
    elif kind == "complete_task":
        due = f" (due {payload['due']})" if payload.get("due") else ""
        text = f"Mark Google Task '{payload.get('title', '')}'{due} as completed"
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

    if kind == "create_and_share":
        from app.workspace.sharing import create_and_share

        return create_and_share(payload)

    if kind == "share_file":
        from app.workspace.sharing import share_file, shareable_file

        file = shareable_file(payload["file_id"])  # still Socio's and not trashed
        return {"share": share_file(file["id"], payload["recipient"], payload.get("role", "editor"), notify=True)}

    if kind == "create_task":
        from app.tasks import tasks

        # Not retried inside create_task: a timed-out insert may still have created it.
        task = tasks.create_task(payload["title"], payload.get("notes", ""), payload.get("due", ""))
        if not task.get("verified"):
            raise RuntimeError(f"Google did not confirm the task was created: {task}")
        return {"task": task}

    if kind == "close_old_reminders":
        from app.tasks import tasks

        closed, gone = 0, 0
        for task_id in payload.get("task_ids", []):
            try:
                if tasks.complete_task(task_id).get("verified"):
                    closed += 1
            except Exception as exc:  # one missing task must not stop the rest
                gone += 1
                logger.warning("Could not close old reminder %s: %s", task_id, exc)
        return {"closed": closed, "not_closed": gone}

    if kind == "complete_task":
        from googleapiclient.errors import HttpError

        from app.tasks import tasks

        try:
            current = tasks.get_task(payload["task_id"])
        except HttpError as exc:
            if getattr(getattr(exc, "resp", None), "status", None) in (404, "404"):
                raise ValueError("That task is no longer in Google Tasks.") from exc
            raise
        if current.get("status") == "completed":  # already ticked in the app: nothing to do
            return {"task": {**current, "verified": True}}
        task = tasks.complete_task(payload["task_id"])
        if not task.get("verified"):
            raise RuntimeError(f"Google did not confirm the task was completed: {task}")
        return {"task": task}

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
        report_failure(exc, "approval.follow_up_reply", approval_id=approval.get("id"), type=kind)
        result["reply_error"] = f"{type(exc).__name__}: {exc}"
        result["reply_error_text"] = safe_error_text(exc)

    return result


def share_outcome(kind: str, payload: dict[str, Any], result: dict[str, Any]) -> str:
    """What happened after a sharing approval, step by step and truthfully."""

    title, recipient = payload.get("title", ""), payload.get("recipient", "")
    if kind == "share_file":
        return f"Shared \"{title}\" with {recipient}. Google emailed them the link."

    url = (result.get("file") or {}).get("url", "")
    if result.get("share_error"):
        return f"Created \"{title}\", but I couldn't share it with {recipient}, so no email was sent. {result['share_error']} {url}".strip()
    if result.get("email_error"):
        return (
            f"Created \"{title}\" and gave {recipient} access, but the email didn't go through. "
            f"{result['email_error']} You can send them this link: {url}"
        ).strip()
    return f"Created \"{title}\" and sent edit link to {recipient}. {url}".strip()


def approve(approval_id: int, store: AgentStateStore | None = None) -> dict[str, Any]:
    """Execute an approval. Safe to call twice: only the first call acts."""

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)

    if approval is None:
        return {"ok": False, "message": "That request no longer exists."}

    if approval["status"] != "open":
        return {"ok": False, "message": _already(approval)}

    if approval["type"] in REVIEW_TYPES:
        store.resolve_pending_approval(approval_id, "Marked done by owner")
        complete_followup_task(store, approval.get("message_id"))
        return {"ok": True, "message": "Marked as done."}

    if approval["type"] not in EXECUTABLE_TYPES:
        return {"ok": False, "message": "Socio doesn't know how to carry out this request."}

    if not store.claim_approval(approval_id):
        return {"ok": False, "message": "This request is already being handled."}

    message_id, thread_id = approval.get("message_id"), approval.get("thread_id")

    try:
        result = _execute(approval)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report_failure(exc, "approval.execute", approval_id=approval_id, type=approval["type"])
        store.finish_approval(approval_id, "failed", error)
        store.add_failure(f"approval:{approval['type']}", error, message_id=message_id)
        if message_id:
            store.mark_processed(message_id, thread_id, status="needs_attention", summary=f"Approved change failed: {error}")
        return {"ok": False, "message": f"That didn't go through, so nothing was changed. {safe_error_text(exc)}", "error": error}

    store.finish_approval(approval_id, "approved", "Approved and executed", result)
    complete_followup_task(store, message_id)
    store.add_action(approval["type"], approval.get("target"), approval["payload"], "completed", message_id=message_id, result=result)
    task = result.get("task") or {}
    if approval["type"] == "create_task" and task.get("task_id"):
        store.upsert_tracked_task(task["task_id"], task.get("title") or approval["payload"]["title"], task.get("due") or "")
    elif approval["type"] == "complete_task":
        store.close_tracked_task(approval["payload"]["task_id"], "closed_by_socio")

    message = DONE_TEXT.get(approval["type"], "Done.")
    if approval["type"] in SHARE_TYPES:
        message = share_outcome(approval["type"], approval["payload"], result)
    if approval["type"] == "close_old_reminders":
        message = f"Closed {result['closed']} old reminder tasks. You can reopen any of them in Google Tasks."
        if result["not_closed"]:
            message += f" {result['not_closed']} were already gone."
        for task_id in approval["payload"].get("task_ids", []):
            store.close_tracked_task(task_id, "closed_by_socio_cleanup")
    if result.get("reply") and approval["type"] not in REPLY_TYPES:
        message += " The reply was sent in the conversation."
    for key in ("document", "spreadsheet"):
        if (result.get(key) or {}).get("url"):
            message += f" {result[key]['url']}"
    if result.get("reply_error"):
        message += f" The calendar was updated, but the reply failed. {result.get('reply_error_text', '')}".rstrip()

    if message_id and approval["type"] != "trash_email":
        status = "needs_attention" if result.get("reply_error") else ("replied" if result.get("reply") else "no_action")
        store.mark_processed(message_id, thread_id, status=status, summary=message)

    # Part of it happened (e.g. the sheet was created but the email failed): not a full success.
    partial = bool(result.get("reply_error") or result.get("share_error") or result.get("email_error"))
    if result.get("share_error") or result.get("email_error"):
        store.add_failure(f"approval:{approval['type']}", result.get("share_error") or result.get("email_error"), message_id=message_id)

    return {"ok": True, "partial": partial, "message": message, "result": result}


def reject(approval_id: int, store: AgentStateStore | None = None) -> dict[str, Any]:
    """Reject an approval; nothing is changed."""

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)

    if approval is None:
        return {"ok": False, "message": "That request no longer exists."}

    if not store.claim_approval(approval_id):
        return {"ok": False, "message": _already(approval)}

    store.finish_approval(approval_id, "rejected", f"Rejected by owner at {datetime.now().isoformat(timespec='seconds')}")
    complete_followup_task(store, approval.get("message_id"))

    if approval.get("message_id") and approval["type"] in {"book_meeting", "move_event", "cancel_event"} | REPLY_TYPES:
        store.mark_processed(
            approval["message_id"], approval.get("thread_id"), status="needs_review",
            summary="Owner rejected the proposed change; handle this email manually.",
        )

    return {"ok": True, "message": "Rejected. No changes were made."}


def to_draft(approval_id: int, store: AgentStateStore | None = None) -> dict[str, Any]:
    """
    Move a held email to Gmail Drafts so the owner can edit and send it themselves.

    Nothing is sent. An incoming email is left as needs_review (its follow-up task stays open).
    """

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)

    if approval is None:
        return {"ok": False, "message": "That request no longer exists."}

    if approval["type"] not in EMAIL_TYPES:
        return {"ok": False, "message": "This is not an email, so it cannot be moved to Gmail Drafts."}

    if not store.claim_approval(approval_id):
        return {"ok": False, "message": _already(approval)}

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
        report_failure(exc, "approval.to_draft", approval_id=approval_id, type=approval["type"])
        store.finish_approval(approval_id, "failed", error)
        store.add_failure("approval:to_draft", error, message_id=approval.get("message_id"))
        return {"ok": False, "message": f"The draft could not be created, and nothing was sent. {safe_error_text(exc)}", "error": error}

    store.finish_approval(approval_id, "resolved", "Moved to Gmail Drafts by owner", {"draft_id": draft.get("id")})

    message = "Saved to Gmail Drafts. Nothing was sent; edit and send it from Gmail."
    if len(attachments) > 1:
        message += " Attach the files there yourself."

    if approval.get("message_id"):
        store.mark_processed(approval["message_id"], approval.get("thread_id"), status="needs_review", summary=message)

    return {"ok": True, "message": message, "draft_id": draft.get("id")}
