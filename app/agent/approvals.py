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

A tap claims the request atomically (open -> executing), so a double tap or a
repeated Telegram update runs it once. Hours can pass between the card and the
tap: before acting, check_still_current() re-reads the world. If the email
thread got a new message, or the proposed slot is no longer free, nothing is
done and the request becomes 'stale' (the owner can re-check the email).

If the connection drops while Google is carrying it out (a timed-out send), nobody knows
whether it happened: the request becomes 'unknown' (never retried), and the owner is told
to check Sent / Calendar / Drive, never "nothing was changed". Writes already done before a
later step failed (a created doc, access given) are listed (app.google_api.completed_write).

A decision teaches Socio nothing by itself; the owner can turn a rejection or an
edit into a rule explicitly (by asking Socio to remember it, see app.memory.learning).
A rejected or stale proposal is logged as a failure case (app.agent.eval_feedback); only a
correction the owner writes in reply to that card becomes a lesson for the planner.
A rule_conflict card (app.memory.rule_engine) is decided with decide_rule_conflict().
Failures are logged in full (app.errors) and reported with a short fixed message.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from email.utils import parseaddr
from typing import Any

from app.agent.eval_feedback import capture_approval_failure
from app.agent.followups import complete_followup_task
from app.agent.state import AgentStateStore
from app.calendar.events import cancel_event, find_conflicts, get_event, move_event, parse_ist, summarize_event
from app.calendar.tools import INDIA_TZ, create_calendar_event
from app.files.local import resolve_shareable_file
from app.gmail.drafts import create_draft, create_reply_draft
from app.gmail.messages import send_new_email, send_reply, trash_message
from app.gmail.tools import get_thread_details
from app.errors import report_failure, safe_error_text
from app.google_api import outcome_unknown, recording_writes


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
# What a recipient can do with each access level ("they can view it").
ROLE_VERB = {"editor": "edit", "commenter": "comment on", "viewer": "view"}
# Approvals that only ask the owner to look at something; approving closes them.
REVIEW_TYPES = {"review_draft", "review_decision"}

# How a decided request is described to the owner ("This was already ...").
STATUS_WORDS = {
    "approved": "approved", "rejected": "rejected", "resolved": "handled",
    "failed": "tried and it failed", "executing": "being handled",
    "stale": "stopped because things changed", "rechecked": "re-checked",
    "unknown": "tried, but I couldn't tell whether it went through",
}
# When the connection drops mid-action (outcome unknown): what may have happened, and where
# the owner can see whether it did. Never "nothing was changed": it may have been.
UNKNOWN_WHAT = {
    "send_reply": ("the reply went out", "your Gmail Sent folder"),
    "send_reply_with_attachment": ("the reply went out", "your Gmail Sent folder"),
    "send_email": ("the email went out", "your Gmail Sent folder"),
    "book_meeting": ("the meeting was booked", "Google Calendar"),
    "move_event": ("the meeting was moved", "Google Calendar"),
    "cancel_event": ("the meeting was cancelled", "Google Calendar"),
    "trash_email": ("it was moved to Trash", "Gmail Trash"),
    "create_document": ("the document was created", "your Socio folder in Google Drive"),
    "create_spreadsheet": ("the spreadsheet was created", "your Socio folder in Google Drive"),
    "create_and_share": ("the file was created", "your Socio folder in Google Drive"),
    "append_document": ("the text was added", "the document"),
    "append_sheet_row": ("the row was added", "the sheet"),
    "share_file": ("access was given", "the file's sharing settings in Google Drive"),
    "create_task": ("the task was added", "Google Tasks"),
    "complete_task": ("the task was closed", "Google Tasks"),
    "close_old_reminders": ("the reminders were closed", "Google Tasks"),
}
UNKNOWN_REPLY = "I can't tell whether the reply went out (the connection to Google dropped). Check your Gmail Sent folder before sending it again."
# Approvals that change a calendar slot; the slot must still be free when the owner taps.
SLOT_TYPES = {"book_meeting", "move_event"}
# Something only the owner knows, left in a reply by the agent: "[OWNER: your notice period]".
GAP_MARKER = re.compile(r"\[OWNER:[^\]]*\]", re.IGNORECASE)
GAP_REFUSAL = "This reply still has a gap only you can fill. Reply to the card with the answer, or use Edit in Gmail."


def reply_gaps(approval: dict[str, Any]) -> list[str]:
    """The [OWNER: ...] gaps still in an email approval's text (it can't be sent while any remain)."""

    if approval.get("type") not in EMAIL_TYPES:
        return []
    payload = approval.get("payload") or {}
    body = payload.get("body") if approval.get("type") == "send_email" else (payload.get("reply") or {}).get("body")
    return GAP_MARKER.findall(body or "")


def _for_gmail(body: str) -> str:
    """A draft the owner finishes in Gmail: each gap reads "[fill in: ...]", not Socio's marker."""

    return GAP_MARKER.sub(lambda match: f"[fill in: {match.group(0)[7:-1].strip()}]", body)


def fill_reply(approval_id: int, body: str, question: str = "", store: AgentStateStore | None = None) -> int | None:
    """
    Replace a gap reply with the version the owner completed in chat: the old request is closed
    and a new one (same recipient, thread, files, snapshot) carries the new text. Returns the new
    request's id, or None if the old one was decided meanwhile. question: what is still missing.
    """

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)
    body = (body or "").strip()
    if not approval or approval["type"] not in EMAIL_TYPES or not body:
        return None
    if not store.resolve_pending_approval(approval_id, "Filled in by the owner in chat"):
        return None

    payload = {key: value for key, value in approval["payload"].items() if key != "question"}
    if approval["type"] == "send_email":
        payload["body"] = body
    else:
        payload["reply"] = {**payload["reply"], "body": body}
    if question and GAP_MARKER.search(body):
        payload["question"] = question
    new = {"type": approval["type"], "payload": payload}
    return store.add_pending_approval(
        approval["type"], describe(new), payload, message_id=approval.get("message_id"), thread_id=approval.get("thread_id"),
    )

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


def _who(email: dict[str, Any]) -> str:
    name, address = parseaddr(email.get("from", ""))
    return name or address or "Someone"


def check_still_current(approval: dict[str, Any]) -> str | None:
    """
    Why this approved change no longer fits the world, in plain words; None if it still does.

    - An email reply (also the reply after a booking): the thread must not have a message
      newer than the one it was written against (whoever wrote it, the owner included).
      Gmail's historyId is not used: reading or labelling a message changes it too.
    - A booking or a move: the slot must still be in the future and free in Calendar.
    Requests made before this check existed carry no snapshot and skip the thread part.
    Google errors propagate (the caller reports them; nothing is done).
    """

    payload, kind = approval["payload"], approval["type"]
    reply = payload.get("reply") or {}

    if reply.get("thread_id") and reply.get("last_seen_message_id"):
        thread = get_thread_details(reply["thread_id"])
        ids = [email.get("id") for email in thread]
        if reply.get("message_id") and reply["message_id"] not in ids:
            return "The email this answers is no longer in the conversation."
        seen = reply["last_seen_message_id"]
        if seen not in ids:  # the newest message then was deleted
            return "The conversation changed after I prepared this."
        newer = thread[ids.index(seen) + 1:]
        if newer:
            latest = newer[-1]
            if "SENT" in latest.get("label_ids", []):
                return "You wrote in this conversation yourself after I prepared this."
            return f"{_who(latest)} wrote again in this conversation after I prepared this."

    # Moving or cancelling: the meeting must still be the one on the card (audit M1). Only its
    # status and times are compared: Google's "updated" also changes when someone RSVPs.
    if kind in {"move_event", "cancel_event"} and payload.get("event_id"):
        event = summarize_event(get_event(payload["event_id"]))
        if event.get("status") == "cancelled":
            return "That meeting has already been cancelled."
        if payload.get("old_start") and not _same_time(event.get("start"), payload["old_start"]):
            return f"The meeting was moved meanwhile (it is now {format_time(event.get('start'))})."
        if payload.get("old_end") and not _same_time(event.get("end"), payload["old_end"]):
            return "The meeting's length was changed meanwhile."

    # Trashing a whole conversation: nothing may have arrived since (it would go to Trash unseen).
    if kind == "trash_email" and payload.get("whole_thread") and payload.get("thread_id") and payload.get("last_seen_message_id"):
        ids = [email.get("id") for email in get_thread_details(payload["thread_id"])]
        if payload["last_seen_message_id"] in ids and ids[-1] != payload["last_seen_message_id"]:
            return "A new message arrived in this conversation after I prepared this."

    if kind in SLOT_TYPES:
        start, end = parse_ist(payload["new_start"]), parse_ist(payload["new_end"])
        if start <= datetime.now(INDIA_TZ):
            return "That time has already passed."
        conflicts = find_conflicts(start, end, ignore_event_id=payload.get("event_id") if kind == "move_event" else None)
        if conflicts:
            first = conflicts[0]
            return f"That time is no longer free: it overlaps with '{first['title']}' ({format_time(first.get('start'))})."

    return None


def _same_time(a: str | None, b: str | None) -> bool:
    """Two event times as Google / the card wrote them ('...T15:00:00+05:30' or an all-day date)."""

    try:
        return parse_ist(a or "") == parse_ist(b or "")
    except ValueError:
        return (a or "") == (b or "")


def clock_12(moment: datetime) -> str:
    """12-hour clock for people: 15:00 -> '3:00 PM' (no leading zero, works on Windows too)."""

    return f"{moment.hour % 12 or 12}:{moment:%M} {'AM' if moment.hour < 12 else 'PM'}"


def format_time(value: str | None) -> str:
    """'2026-10-16T15:00:00+05:30' -> 'Fri 16 Oct, 3:00 PM IST'."""

    if not value:
        return "?"
    try:
        moment = parse_ist(value)
    except ValueError:
        return value
    return f"{moment:%a %d %b}, {clock_12(moment)} IST"


def invitees(payload: dict[str, Any]) -> list[str]:
    """Everyone a booking invites (older approvals only have attendee_email)."""

    return list(payload.get("attendees") or ([payload["attendee_email"]] if payload.get("attendee_email") else []))


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
            f"Invite: {', '.join(invitees(payload)) or 'nobody'}"
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
            f"Create {what} '{payload.get('title')}', give {payload.get('recipient')} {access_words(payload)}, "
            f"and email them the link\nSubject: {payload.get('subject') or '(none)'}\n\"{payload.get('body', '')}\""
        )
        if payload.get("headers"):
            text += "\nColumns: " + " | ".join(payload["headers"])
    elif kind == "share_file":
        text = f"Give {payload.get('recipient')} {access_words(payload)} to '{payload.get('title')}'"
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
        if payload.get("body"):
            text += f"\nDraft:\n\"{payload['body']}\""
    elif kind == "review_decision":
        text = f"Unsure whether to reply to '{payload.get('subject', '')}'.\n{payload.get('reason', '')}"
    elif kind == "rule_conflict":
        from app.memory.rule_engine import describe_conflict

        text = describe_conflict(payload)
    else:
        text = approval.get("target") or kind

    if payload.get("links") and kind in EMAIL_TYPES:
        text += "\n" + links_summary(payload)

    if payload.get("reason") and kind in EXECUTABLE_TYPES:
        text += f"\nWhy: {payload['reason']}"

    reply = payload.get("reply")
    if reply and kind not in REPLY_TYPES:
        text += f"\nThen reply to {reply.get('to')}:\n\"{reply.get('body')}\""

    return text


def access_words(payload: dict[str, Any]) -> str:
    return f"{payload.get('role') or 'editor'} access"


def links_summary(payload: dict[str, Any]) -> str:
    """What an email's Google file links come with: the live link, and who gets what access."""

    lines = []
    for link in payload.get("links") or []:
        if link.get("can_share"):
            access = f"recipients get {access_words(payload)}"
        else:
            access = "access unchanged (not a file Socio created; they may need to request access)"
        lines.append(f"Live link (not a copy) to {link.get('type', 'Google file')} '{link.get('name')}': {access}")
    return "\n".join(lines)


def email_recipients(approval: dict[str, Any]) -> list[str]:
    from email.utils import getaddresses

    payload = approval.get("payload") or {}
    to = payload.get("to") if approval.get("type") == "send_email" else (payload.get("reply") or {}).get("to", "")
    return [address.lower() for _, address in getaddresses([to or ""]) if address]


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

    if kind in EMAIL_TYPES:
        from app.workspace.sharing import share_links, with_links

        links = payload.get("links") or []
        result: dict[str, Any] = {"attachments": [a["name"] for a in payload.get("attachments", [])]}
        # Access first: an email whose link doesn't open for the recipient is not sent.
        if any(link.get("can_share") for link in links):
            result["shared"] = []
            try:
                result["shared"] = share_links(links, email_recipients(approval), payload.get("role") or "editor")
            except Exception as exc:
                report_failure(exc, "approval.share_links", approval_id=approval.get("id"))
                result["share_error"] = safe_error_text(exc)
                return result
        attachments = [str(resolve_shareable_file(a["path"])) for a in payload.get("attachments", [])]
        if kind in REPLY_TYPES:
            reply = payload["reply"]
            result["reply"] = send_reply(
                thread_id=reply["thread_id"], message_id=reply["message_id"], to=reply["to"],
                subject="", body=with_links(reply["body"], links), attachment_paths=attachments,
            )
        else:
            result["email"] = send_new_email(
                to=payload["to"], subject=payload.get("subject", ""), body=with_links(payload["body"], links),
                attachment_paths=attachments,
            )
        return result

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
        from app.workspace.sharing import LINK_PLACEHOLDER, share_file, shareable_file

        file = shareable_file(payload["file_id"])  # still Socio's and not trashed
        # Asked in an email: our reply in the thread tells them (no separate Google email).
        result = {"share": share_file(file["id"], payload["recipient"], payload.get("role") or "editor",
                                      notify=payload.get("notify", True))}
        if payload.get("reply"):
            reply = dict(payload["reply"])
            reply["body"] = LINK_PLACEHOLDER.sub(lambda _: file["url"] or payload.get("url", ""), reply["body"])
            try:
                result["reply"] = _send_follow_up_reply({"reply": reply})
            except Exception as exc:  # access was given; a failed reply must not hide that
                report_failure(exc, "approval.follow_up_reply", approval_id=approval.get("id"), type=kind)
                result["reply_error"] = f"{type(exc).__name__}: {exc}"
                result["reply_error_text"] = safe_error_text(exc)
                result["reply_unknown"] = outcome_unknown(exc)
        return result

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
            "attendee_emails": payload.get("attendees") or [],
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
        result["reply_unknown"] = outcome_unknown(exc)

    return result


def _email_key(payload: dict[str, Any]) -> tuple:
    """Two new emails are the same message when subject, text, and files match (recipients aside)."""

    return (
        " ".join(str(payload.get("subject") or "").split()).lower(),
        " ".join(str(payload.get("body") or "").split()),
        tuple(sorted(str(a.get("path")) for a in payload.get("attachments") or [])),
        tuple(sorted(str(link.get("file_id")) for link in payload.get("links") or [])),
        payload.get("role") or "",
    )


def merge_same_emails(approval_ids: list[int], store: AgentStateStore) -> list[int]:
    """
    One instruction that produced the same new email for several people becomes ONE
    email to all of them (the way a person writes it), before the owner sees any card.

    Only open, not-yet-shown owner-mode `send_email` approvals from this instruction
    are merged, and only when subject, text, and files are identical: a personalised
    message ("Hi Sam" / "Hi Raj") stays separate. Recipients keep their order, each
    once. The merged-away approvals are closed (never sent). Returns the ids still open.
    """

    kept: dict[tuple, dict[str, Any]] = {}
    remaining: list[int] = []
    for approval_id in approval_ids:
        approval = store.get_approval(approval_id)
        mergeable = (
            approval is not None and approval["type"] == "send_email" and approval["status"] == "open"
            and not approval.get("notified") and not approval.get("message_id")
        )
        if not mergeable:
            remaining.append(approval_id)
            continue
        key = _email_key(approval["payload"])
        first = kept.get(key)
        if first is None:
            kept[key] = approval
            remaining.append(approval_id)
            continue
        recipients = [a.strip() for a in f"{first['payload']['to']}, {approval['payload']['to']}".split(",") if a.strip()]
        merged = {**first, "payload": {**first["payload"], "to": ", ".join(dict.fromkeys(r.lower() for r in recipients))}}
        if store.update_unshown_approval(first["id"], describe(merged), merged["payload"]):
            kept[key] = merged
            store.resolve_pending_approval(approval_id, f"Merged into #{first['id']}: the same email to several people")
        else:  # the first card was shown meanwhile: leave both as they are
            remaining.append(approval_id)
    return remaining


def share_outcome(kind: str, payload: dict[str, Any], result: dict[str, Any]) -> str:
    """What happened after a sharing approval, step by step and truthfully."""

    title, recipient = payload.get("title", ""), payload.get("recipient", "")
    verb = ROLE_VERB.get(payload.get("role") or "editor", "edit")
    if kind == "share_file":
        told = " Google emailed them the link." if payload.get("notify", True) else ""
        return f"Shared \"{title}\" with {recipient} (they can {verb} it).{told}"

    url = (result.get("file") or {}).get("url", "")
    if result.get("share_error"):
        return f"Created \"{title}\", but I couldn't share it with {recipient}, so no email was sent. {result['share_error']} {url}".strip()
    if result.get("email_unknown"):
        return (
            f"Created \"{title}\" and gave {recipient} access, but I can't tell whether the email went out "
            f"(the connection to Google dropped). Check your Gmail Sent folder before sending it again. {url}"
        ).strip()
    if result.get("email_error"):
        return (
            f"Created \"{title}\" and gave {recipient} access, but the email didn't go through. "
            f"{result['email_error']} You can send them this link: {url}"
        ).strip()
    return f"Created \"{title}\" and emailed {recipient} the link (they can {verb} it). {url}".strip()


def email_link_outcome(approval: dict[str, Any], result: dict[str, Any]) -> str:
    """After an email with Google file links: what was shared and sent, truthfully."""

    payload = approval["payload"]
    links = payload.get("links") or []
    names = ", ".join(f"\"{link.get('name')}\"" for link in links)
    who = ", ".join(email_recipients(approval)) or "them"
    if result.get("share_error"):
        given = " Some access was already given." if result.get("shared") else ""
        return f"I couldn't give {who} access to {names}, so the email wasn't sent.{given} {result['share_error']}".strip()
    sent = "Reply sent" if approval["type"] in REPLY_TYPES else "Email sent"
    message = f"{sent} with the live link to {names}."
    if any(link.get("can_share") for link in links):
        message += f" {who} can now {ROLE_VERB.get(payload.get('role') or 'editor', 'open')} it."
    if any(not link.get("can_share") for link in links):
        message += " Access unchanged: if they can't open it, share it from Google Drive."
    return message


def approve(approval_id: int, store: AgentStateStore | None = None) -> dict[str, Any]:
    """Execute an approval. Safe to call twice: only the first call acts."""

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)

    if approval is None:
        return {"ok": False, "message": "That request no longer exists."}

    # duplicate: a second tap on a card that is already decided or running; the card
    # already shows the first tap's result, so the bot must not overwrite it.
    if approval["status"] != "open":
        return {"ok": False, "duplicate": True, "message": _already(approval)}

    if approval["type"] == "rule_conflict":  # the CLI's --approve: replace the old rule
        return decide_rule_conflict(approval_id, "replace", store)

    if approval["type"] in REVIEW_TYPES:
        if not store.resolve_pending_approval(approval_id, "Marked done by owner"):
            return {"ok": False, "duplicate": True, "message": "This request was already handled."}
        complete_followup_task(store, approval.get("message_id"))
        return {"ok": True, "message": "Marked as done."}

    if approval["type"] not in EXECUTABLE_TYPES:
        return {"ok": False, "message": "Socio doesn't know how to carry out this request."}

    if reply_gaps(approval):  # never send "[OWNER: ...]" to anyone; the request stays open
        return {"ok": False, "message": GAP_REFUSAL}

    if not store.claim_approval(approval_id):
        return {"ok": False, "duplicate": True, "message": "This request is already being handled."}

    message_id, thread_id = approval.get("message_id"), approval.get("thread_id")
    acting = False  # True once _execute starts: from then on Google may have changed something

    with recording_writes() as written:
        try:
            stale = check_still_current(approval)
            if stale:
                store.finish_approval(approval_id, "stale", stale)
                if message_id:
                    store.mark_processed(message_id, thread_id, status="needs_review", summary=f"Not carried out: {stale}")
                capture_approval_failure(approval, "stale", store)
                return {"ok": False, "stale": True, "message": f"{stale} Nothing was sent or changed."}
            acting = True
            result = _execute(approval)
        except Exception as exc:
            if not acting:
                return _precheck_failed(approval, exc, store)
            return _execution_failed(approval, exc, unknown=acting and outcome_unknown(exc), done=list(written), store=store)

    if approval["type"] in EMAIL_TYPES and result.get("share_error"):
        return _share_failed(approval, result, store)

    store.finish_approval(approval_id, "approved", "Approved and executed", result)
    complete_followup_task(store, message_id)
    store.add_action(approval["type"], approval.get("target"), approval["payload"], "completed", message_id=message_id, result=result)
    task = result.get("task") or {}
    if approval["type"] == "create_task" and task.get("task_id"):
        store.upsert_tracked_task(task["task_id"], task.get("title") or approval["payload"]["title"], task.get("due") or "")
    elif approval["type"] == "complete_task":
        store.close_tracked_task(approval["payload"]["task_id"], "closed_by_socio")

    message = DONE_TEXT.get(approval["type"], "Done.")
    if approval["type"] in EMAIL_TYPES and approval["payload"].get("links"):
        message = email_link_outcome(approval, result)
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
        done = "Access was given" if approval["type"] == "share_file" else "The calendar was updated"
        if result.get("reply_unknown"):
            message += f" {done}. {UNKNOWN_REPLY}"
        else:
            message += f" {done}, but the reply failed. {result.get('reply_error_text', '')}".rstrip()

    if message_id and approval["type"] != "trash_email":
        status = "needs_attention" if result.get("reply_error") else ("replied" if result.get("reply") else "no_action")
        store.mark_processed(message_id, thread_id, status=status, summary=message)

    # Part of it happened (e.g. the sheet was created but the email failed): not a full success.
    partial = bool(result.get("reply_error") or result.get("share_error") or result.get("email_error"))
    if result.get("share_error") or result.get("email_error"):
        store.add_failure(f"approval:{approval['type']}", result.get("share_error") or result.get("email_error"), message_id=message_id)

    return {"ok": True, "partial": partial, "message": message, "result": result}


PRECHECK_FAILED = "I couldn't check that this is still OK to do, so I did nothing. Tap Approve again in a few minutes."


def _precheck_failed(approval: dict[str, Any], exc: Exception, store: AgentStateStore) -> dict[str, Any]:
    """
    Reading the world before acting failed (Google busy, sign-in expired): nothing was done,
    so the request goes back to open and the card keeps its buttons (audit M2: it used to
    become 'failed' for good and the owner had to ask again from scratch).
    """

    report_failure(exc, "approval.precheck", approval_id=approval["id"], type=approval["type"])
    store.add_failure(f"approval:{approval['type']}", f"Pre-check failed: {type(exc).__name__}: {exc}", message_id=approval.get("message_id"))
    store.release_claim(approval["id"])
    return {"ok": False, "message": f"{PRECHECK_FAILED} {safe_error_text(exc)}".strip()}


def _share_failed(approval: dict[str, Any], result: dict[str, Any], store: AgentStateStore) -> dict[str, Any]:
    """
    An email with Google file links whose access could not be given: the email was not sent
    (a link that doesn't open must not go out). Recorded as failed and the incoming email as
    needing the owner (audit M3: it used to be 'approved' and the email 'no_action').
    """

    approval_id, message_id = approval["id"], approval.get("message_id")
    message = email_link_outcome(approval, result)
    store.finish_approval(approval_id, "failed", f"Access not given, email not sent: {result['share_error']}", result)
    store.add_failure(f"approval:{approval['type']}", result["share_error"], message_id=message_id)
    if result.get("shared"):  # some access was given before the failure
        store.add_action(approval["type"], approval.get("target"), approval["payload"], "partial", message_id=message_id, result=result)
    if message_id:
        store.mark_processed(message_id, approval.get("thread_id"), status="needs_attention", summary=message)
    return {"ok": False, "partial": bool(result.get("shared")), "message": message, "result": result}


def _execution_failed(
    approval: dict[str, Any], exc: Exception, *, unknown: bool, done: list[str], store: AgentStateStore,
) -> dict[str, Any]:
    """
    A failed approval, reported truthfully.

    unknown: the connection dropped (or Google answered 5xx) while it was acting, so it may
    have happened. It is never retried and never reported as "nothing was changed"; the
    request becomes 'unknown' and the owner is told where to check.
    done: writes that did happen before the failure (a created doc, access given).
    """

    approval_id, kind = approval["id"], approval["type"]
    message_id, thread_id = approval.get("message_id"), approval.get("thread_id")
    error = f"{type(exc).__name__}: {exc}"
    report_failure(exc, "approval.execute", approval_id=approval_id, type=kind, outcome="unknown" if unknown else "failed")

    already = f" Already done: {'; '.join(done)}." if done else ""
    if unknown:
        what, where = UNKNOWN_WHAT.get(kind, ("it went through", "Gmail, Google Calendar, and Google Drive"))
        if done and kind in {"create_document", "create_spreadsheet", "create_and_share"}:
            what, where = "the rest of it went through", "that file"  # the file exists; a later step is in doubt
        message = f"I can't tell whether {what}: the connection to Google dropped while I was doing it.{already} Check {where} before trying again."
        summary = f"Approved change may or may not have happened (connection dropped): {error}"
    elif done:
        message = f"That only partly went through.{already} The rest wasn't done. {safe_error_text(exc)}"
        summary = f"Approved change partly failed: {error}"
    else:
        message = f"That didn't go through, so nothing was changed. {safe_error_text(exc)}"
        summary = f"Approved change failed: {error}"

    store.finish_approval(approval_id, "unknown" if unknown else "failed", error, {"done": done} if done else None)
    store.add_failure(f"approval:{kind}", error, message_id=message_id)
    if unknown or done:
        # Like the agent's own uncertain actions (graph.execute_tools): recorded, never as completed.
        store.add_action(kind, approval.get("target"), approval["payload"], "unknown" if unknown else "partial",
                         message_id=message_id, result={"done": done, "error": error})
    if message_id:
        store.mark_processed(message_id, thread_id, status="needs_attention", summary=summary)
    return {"ok": False, "unknown": unknown, "message": message.strip(), "error": error}


def reject(approval_id: int, store: AgentStateStore | None = None) -> dict[str, Any]:
    """Reject an approval; nothing is changed."""

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)

    if approval is None:
        return {"ok": False, "message": "That request no longer exists."}

    if not store.claim_approval(approval_id):
        return {"ok": False, "duplicate": True, "message": _already(store.get_approval(approval_id) or approval)}

    store.finish_approval(approval_id, "rejected", f"Rejected by owner at {datetime.now().isoformat(timespec='seconds')}")
    complete_followup_task(store, approval.get("message_id"))
    if approval["type"] == "rule_conflict":
        return {"ok": True, "message": "Okay, I didn't save the new rule. Your rules are unchanged."}
    capture_approval_failure(approval, "rejected", store)

    if approval.get("message_id") and approval["type"] in {"book_meeting", "move_event", "cancel_event"} | REPLY_TYPES:
        store.mark_processed(
            approval["message_id"], approval.get("thread_id"), status="needs_review",
            summary="Owner rejected the proposed change; handle this email manually.",
        )

    return {"ok": True, "message": "Rejected. No changes were made."}


def decide_rule_conflict(approval_id: int, choice: str, store: AgentStateStore | None = None) -> dict[str, Any]:
    """
    The owner's answer on a rule_conflict card: "replace", "merge", or "both" (Cancel = reject()).
    Claimed like any approval, so a double tap applies it once.
    """

    from app.memory.rule_engine import resolve_conflict

    store = store or AgentStateStore()
    approval = store.get_approval(approval_id)
    if approval is None:
        return {"ok": False, "message": "That request no longer exists."}
    if approval["type"] != "rule_conflict":
        return {"ok": False, "message": "This is not a rule question."}
    if approval["status"] != "open" or not store.claim_approval(approval_id):
        return {"ok": False, "duplicate": True, "message": _already(store.get_approval(approval_id) or approval)}

    try:
        result = resolve_conflict(approval["payload"], choice, store)
    except Exception as exc:
        report_failure(exc, "approval.rule_conflict", approval_id=approval_id, choice=choice)
        store.finish_approval(approval_id, "failed", f"{type(exc).__name__}: {exc}")
        return {"ok": False, "message": f"I couldn't save the rule, so your rules are unchanged. {safe_error_text(exc)}"}

    store.finish_approval(approval_id, "approved", f"Rule conflict: {choice}", result)
    return {"ok": True, "message": result["message"], "choice": choice}


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
        return {"ok": False, "duplicate": True, "message": _already(store.get_approval(approval_id) or approval)}

    from app.workspace.sharing import with_links

    payload = approval["payload"]
    attachments = payload.get("attachments", [])
    links = payload.get("links") or []

    try:
        # A Gmail draft made here carries at most one file; with several, the owner attaches them.
        attachment = str(resolve_shareable_file(attachments[0]["path"])) if len(attachments) == 1 else None
        if approval["type"] == "send_email":
            draft = create_draft(
                to=payload["to"], subject=payload.get("subject", ""), body=_for_gmail(with_links(payload["body"], links)),
                attachment_path=attachment,
            )
        else:
            reply = payload["reply"]
            draft = create_reply_draft(message_id=reply["message_id"], body=_for_gmail(with_links(reply["body"], links)), attachment_path=attachment)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report_failure(exc, "approval.to_draft", approval_id=approval_id, type=approval["type"])
        store.finish_approval(approval_id, "failed", error)
        store.add_failure("approval:to_draft", error, message_id=approval.get("message_id"))
        return {"ok": False, "message": f"The draft could not be created, and nothing was sent. {safe_error_text(exc)}", "error": error}

    store.finish_approval(approval_id, "resolved", "Moved to Gmail Drafts by owner", {"draft_id": draft.get("id")})

    message = "Saved to Gmail Drafts. Nothing was sent; edit and send it from Gmail."
    if any(link.get("can_share") for link in links):
        message += " Nobody's access changed: share the file from Google Drive if they need it."
    if len(attachments) > 1:
        message += " Attach the files there yourself."

    if approval.get("message_id"):
        store.mark_processed(approval["message_id"], approval.get("thread_id"), status="needs_review", summary=message)

    return {"ok": True, "message": message, "draft_id": draft.get("id")}
