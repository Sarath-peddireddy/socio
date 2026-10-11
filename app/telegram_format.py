"""
Telegram HTML formatting for Socio.

Every message the bot sends uses parse_mode="HTML" (Telegram's small HTML
subset: b, i, u, s, code, pre, blockquote, a; no <br>, newlines are literal).

- esc() escapes any dynamic value (subjects, senders, file names, model text)
  before it goes into a template.
- md_to_html() turns the agent's free text into safe HTML: it escapes first,
  then converts the Markdown the model tends to write (**bold**, `code`,
  "* " bullets, "# " headings), so no raw ** ever reaches the chat.
- clip_html() shortens a message without cutting a tag or entity in half.
- approval_card() renders one approval request as a single card with its buttons.
"""

from __future__ import annotations

import html
import re
from datetime import date
from typing import Any

from app.agent.approvals import (
    DOCUMENT_TYPES,
    EMAIL_TYPES,
    EXECUTABLE_TYPES,
    SHARE_TYPES,
    format_time,
    invitees,
)
from app.calendar.events import parse_ist
from app.workspace.sharing import with_link, with_links


MAX_MESSAGE_CHARS = 4000  # Telegram allows 4096 characters per message
BODY_CHARS = 2500  # longest email body / diff shown on a card
EXPANDABLE_AFTER = 600  # longer quotes start collapsed

# Every command, in menu order: (command, argument shown on /help, short description).
# The /help card and Telegram's "/" suggestion menu (setMyCommands) both come from this list.
BOT_COMMANDS = [
    ("run", "", "Trigger immediate inbox check"),
    ("status", "", "Inspect active holds, pending approvals & system health"),
    ("approvals", "", "Review pending decision cards"),
    ("memories", "", "View your saved rules and standing preferences"),
    ("schedules", "", "See or cancel your scheduled routines"),
    ("remember", "TEXT", "Save a rule, e.g. no calls before 12:00"),
    ("forget", "ID", "Delete a saved rule (numbers from /memories)"),
    ("new", "", "Start a fresh conversation"),
    ("reset", "", "Clear the conversation context (like /new)"),
    ("dryrun", "", "Check the inbox without sending or changing anything"),
    ("skills", "", "List your saved skills (reusable workflows)"),
    ("save_skill", "NAME", "Save your last multi-step request as a skill"),
    ("forget_skill", "NAME", "Archive a saved skill"),
    ("help", "", "Show what I can do and these commands"),
]
# Other names for the same command: shown next to it on /help, and in the "/" menu.
COMMAND_ALIASES = {"schedules": ["routines"], "reset": ["clear"], "help": ["start"]}


def commands_html() -> str:
    lines = []
    for command, argument, description in BOT_COMMANDS:
        names = " · ".join(f"<code>/{name}</code>" for name in COMMAND_ALIASES.get(command, []))
        usage = f"<code>{html.escape(f'/{command} {argument}'.strip())}</code>" + (f" · {names}" if names else "")
        lines.append(f"• {usage} — {html.escape(description, quote=False)}")
    return "\n".join(lines)


def menu_commands() -> list[dict[str, str]]:
    """The "/" suggestion menu for setMyCommands (plain text, no arguments)."""

    menu = []
    for command, _, description in BOT_COMMANDS:
        menu.append({"command": command, "description": description})
        menu += [{"command": alias, "description": f"Same as /{command}"} for alias in COMMAND_ALIASES.get(command, [])]
    return menu


WELCOME_HTML = """👋 <b>Welcome to Socio</b>
Your autonomous Workspace Chief of Staff.

I triage your inbox, coordinate your calendar, and manage your documents with strict Human-in-the-Loop governance.

━━━━━━━━━━━━━━━━━━━━
⚡ <b>Core Capabilities:</b>

📬 <b>Email Triage &amp; Drafting</b>
• Scans new threads in the background.
• Filters noise, newsletters, and automated no-replies.
• Prepares professional context-aware replies for your 1-tap review.

📅 <b>Calendar &amp; Meeting Coordination</b>
• Identifies scheduling requests and locates free windows (11:00–16:00 IST).
• Places local 24-hr soft-holds on proposed slots to prevent double-booking.
• Books, reschedules, or moves events only after explicit authorization.

📁 <b>Workspace &amp; File Actions</b>
• Locates attachments and references across Google Drive.
• Creates notes and appends structured rows to Google Docs and Sheets.
• Adds Google Tasks for action items needing your personal follow-up.

⏰ <b>Routines &amp; News</b>
• Sends you reports on a schedule, e.g. "Send me the top AI news every day at 9 AM".
• Reads the latest headlines from public tech news sites.

🛡️ <b>Security &amp; Memory</b>
• Inbound content is strictly quarantined inside untrusted XML boundaries.
• Zero-touch sends and file deletions are disabled—nothing mutates without approval.
• Learns your preferences over time via <code>/remember</code>.

━━━━━━━━━━━━━━━━━━━━
💡 <b>Quick Commands:</b>
{commands}

<i>You can also message me naturally (e.g., "What's on my calendar tomorrow?" or "Draft an email to Alex").</i>""".replace("{commands}", commands_html())

# "what can you do?", "how can you help", "features" ... (the whole message, not a mention inside a request).
CAPABILITY_QUESTION = re.compile(
    r"^\s*(?:(?:hi|hello|hey)\b[\s,!.]*)?(?:socio\b[\s,!.]*)?"
    r"(?:what (?:else )?(?:can|do) you do(?: for me)?"
    r"|how (?:can|do|could) you help(?: me)?"
    r"|what (?:are|is) (?:your|socio'?s) (?:features|capabilities|commands)"
    r"|(?:show me |list )?(?:your )?(?:features|capabilities|commands)"
    r"|help|what are you|who are you)"
    r"[\s?!.]*$",
    re.IGNORECASE,
)


def is_capability_question(text: str) -> bool:
    return bool(CAPABILITY_QUESTION.match(text or ""))


# A message that only closes the conversation ("Okay", "Thanks!", "hmm", "ok cool 👍"):
# the bot reacts 👍 instead of replying (developer request 2026-10-11). "yes", "sure",
# "done" are not here: they answer a question or say a task is finished.
_THANKS = r"thanks?(?: (?:a lot|so much|very much))?|thank you(?: (?:so|very) much)?|thx|thnx|tq|ty|cheers|bye"
_OK = (
    r"ok(?:a+y|ey|ie|i)?|o?k+|h+m+|m+h*m+|cool|great|nice|good|got it|gotcha|alright|all right|fine|noted"
    r"|perfect|awesome|understood|sounds good|no worries|np"
)
_EMOJI = "[\U0001F44C\U0001F44D\U0001F44F\U0001F64F\U0001F600-\U0001F64F\U0001F3FB-\U0001F3FF❤✅☺️]+"
ACKNOWLEDGEMENT = re.compile(rf"^\s*(?:(?:(?:{_THANKS}|{_OK})\b|{_EMOJI})[\s,.!~]*){{1,4}}$", re.IGNORECASE)
THANKS = re.compile(rf"\b(?:{_THANKS})\b", re.IGNORECASE)


def is_acknowledgement(text: str, last_reply: str = "") -> bool:
    """
    True when the message needs no answer. "Okay" right after a question of ours is an
    answer ("Shall I use 30 minutes?" "ok"), so it goes to the agent; thanks never does.
    """

    text = (text or "").strip()
    if len(text) > 40 or not ACKNOWLEDGEMENT.match(text):
        return False
    if THANKS.search(text):
        return True
    # Only Socio's own words count, not the notes saved after it ("(Shown to the owner as ...").
    asked = "?" in (last_reply or "").split("\n(")[0]
    return not asked


# ----- escaping and conversion ----------------------------------------------------------


def esc(value: Any) -> str:
    """Escape a dynamic value for Telegram HTML."""

    return html.escape("" if value is None else str(value), quote=False)


def strip_html(text: str) -> str:
    """Plain text from our HTML (tags dropped, entities decoded)."""

    return html.unescape(re.sub(r"<[^>]+>", "", text or ""))


_BULLET = re.compile(r"^(\s*)[*\-+]\s+(?=\S)")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$")
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])")


def md_to_html(text: str) -> str:
    """The agent's (possibly Markdown) text as safe Telegram HTML."""

    lines = []
    for line in (text or "").splitlines():
        heading = _HEADING.match(line)
        if heading:
            line = f"**{heading.group(1)}**"
        line = _BULLET.sub(r"\1• ", line)
        lines.append(line)

    parts = re.split(r"(`[^`\n]+`)", "\n".join(lines))
    out = []
    for part in parts:
        if len(part) > 1 and part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{esc(part[1:-1])}</code>")
            continue
        part = esc(part)
        part = _BOLD.sub(r"<b>\1</b>", part)
        part = _ITALIC.sub(r"<i>\1</i>", part)
        out.append(part.replace("**", ""))
    return "".join(out)


_TAG = re.compile(r"<(/?)([a-z\-]+)[^>]*>", re.IGNORECASE)


def clip_html(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Shorten HTML to the limit, never splitting a tag or an entity, and close open tags."""

    if len(text) <= limit:
        return text
    cut = text[: limit - 120]
    if cut.rfind("<") > cut.rfind(">"):
        cut = cut[: cut.rfind("<")]
    if cut.rfind("&") > cut.rfind(";"):
        cut = cut[: cut.rfind("&")]

    open_tags: list[str] = []
    for closing, name in _TAG.findall(cut):
        name = name.lower()
        if not closing:
            open_tags.append(name)
        elif name in open_tags:
            del open_tags[len(open_tags) - 1 - open_tags[::-1].index(name)]
    return cut + "".join(f"</{name}>" for name in reversed(open_tags)) + "\n… (truncated)"


def _short(text: Any, limit: int = BODY_CHARS) -> str:
    text = "" if text is None else str(text).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def quote(text: Any) -> str:
    """An email body (or other long text) as a blockquote; long ones start collapsed."""

    body = _short(text)
    tag = "blockquote expandable" if len(body) > EXPANDABLE_AFTER or body.count("\n") > 10 else "blockquote"
    return f"<{tag}>{esc(body) or '(empty)'}</blockquote>"


def _pre(lines: list[str]) -> str:
    return f"<pre>{esc(_short(chr(10).join(lines)))}</pre>"


def _pretty_date(value: str) -> str:
    """'2026-10-16' -> 'Fri 16 Oct'."""

    try:
        return date.fromisoformat(value[:10]).strftime("%a %d %b")
    except ValueError:
        return value


def slot_text(start: str | None, end: str | None) -> str:
    """'Fri 16 Oct, 11:00–11:30 (IST)' (both dates shown if the slot spans midnight)."""

    try:
        first, last = parse_ist(start or ""), parse_ist(end or "")
    except ValueError:
        return f"{format_time(start)} – {format_time(end)}"
    if first.date() == last.date():
        return f"{first:%a %d %b, %H:%M}–{last:%H:%M} (IST)"
    return f"{first:%a %d %b, %H:%M} – {last:%a %d %b, %H:%M} (IST)"


def _field(label: str, value: Any) -> str:
    return f"<b>{label}:</b> {esc(value)}"


# ----- replies split by intent ---------------------------------------------------------------
# One message per kind of content, so an answer, a problem, and a question never share one text
# box. Kinds as in app.agent.supervisor (answer / done / problem / question).

REPLY_HEADERS = {
    "answer": "📋 <b>Here's what I found</b>",
    "done": "✅ <b>Done</b>",
    "problem": "⚠️ <b>Couldn't finish</b>",
    "question": "❓ <b>I need your answer</b>",
}
REPLY_ORDER = ("answer", "done", "problem", "question")
SINGLE_PREFIX = {"done": "✅ ", "problem": "⚠️ "}


def reply_messages(parts: list) -> tuple[list[str], list[str]]:
    """
    The reply parts as Telegram HTML messages: (before the approval cards, after them).

    A reply of one part is one plain message (no header: a simple answer reads like a
    person). Several parts: one message per kind, with a short header and one bullet per
    item. Questions come last, after the cards, so the last thing on screen is what needs
    an answer.
    """

    parts = [part for part in parts if (part.text or "").strip()]
    if not parts:
        return [], []
    if len(parts) == 1:
        part = parts[0]
        text = md_to_html(SINGLE_PREFIX.get(part.kind, "") + part.text.strip())
        return ([], [text]) if part.kind == "question" else ([text], [])

    before, after = [], []
    for kind in REPLY_ORDER:
        items = [part.text.strip() for part in parts if part.kind == kind]
        if not items:
            continue
        body = md_to_html(items[0]) if len(items) == 1 else "\n".join(f"• {md_to_html(item)}" for item in items)
        (after if kind == "question" else before).append(f"{REPLY_HEADERS[kind]}\n{body}")
    return before, after


# ----- scheduled routines -------------------------------------------------------------------

ROUTINE_HEADER = "⏰ <b>Scheduled Briefing:</b> {title}"
NO_ROUTINES_HTML = (
    "⏰ You have no scheduled routines.\n"
    "<i>Just ask, e.g. \"Send me the top AI news every day at 9 AM\".</i>"
)


def routine_header(title: str) -> str:
    return ROUTINE_HEADER.format(title=esc(title))


def schedules_html(jobs: list[dict], describe, next_run) -> tuple[str, list]:
    """The /schedules message: one line per active routine and one Cancel button each (no ids shown)."""

    if not jobs:
        return NO_ROUTINES_HTML, []
    lines = ["⏰ <b>Active Scheduled Routines:</b>", ""]
    buttons = []
    for number, job in enumerate(jobs, 1):
        lines.append(f"{number}. <b>{esc(job['title'])}</b> ({esc(describe(job))})")
        lines.append(f"   <i>Next: {esc(next_run(job))}</i>")
        buttons.append([{"text": f"Cancel {number}. {job['title'][:40]}", "callback_data": f"unsched:{job['id']}"}])
    return "\n".join(lines), buttons


# ----- approval cards -----------------------------------------------------------------------


def _buttons(approval: dict[str, Any]) -> list:
    approval_id = approval["id"]
    kind = approval["type"]

    def button(text: str, action: str) -> dict:
        return {"text": text, "callback_data": f"{action}:{approval_id}"}

    if kind in EMAIL_TYPES:
        return [[button("✅ Approve & Send", "approve"), button("✏️ Edit in Gmail", "draft")], [button("❌ Reject", "reject")]]
    if kind not in EXECUTABLE_TYPES:
        return [[button("✔️ Mark done", "approve")]]
    approve_label = {
        "trash_email": "🗑️ Approve & Trash",
        "book_meeting": "✅ Approve & Book",
        "move_event": "✅ Approve & Move",
        "cancel_event": "✅ Approve & Cancel",
        "enable_skill_script": "✅ Trust Script",
        "create_task": "✅ Approve & Add",
    }.get(kind, "✅ Approve")
    if kind == "close_old_reminders":
        return [[button("✅ Close all", "approve"), button("Keep them", "reject")]]
    if kind in SHARE_TYPES:
        return [[button("✅ Approve & Send" if kind == "create_and_share" else "✅ Approve & Share", "approve"),
                 button("❌ Cancel", "reject")]]
    if kind == "complete_task":
        due_question = (approval.get("payload") or {}).get("trigger") == "due"
        return [[button("✅ Yes, close it" if due_question else "✅ Close task", "approve"),
                 button("Not yet" if due_question else "Keep open", "reject")]]
    return [[button(approve_label, "approve"), button("❌ Reject", "reject")]]


def _link_lines(payload: dict[str, Any]) -> list[str]:
    """One line per Google file linked in an email: what it is and who can open it."""

    role = str(payload.get("role") or "editor").capitalize()
    return [
        _field(link.get("type", "Google file"), f"{link.get('name')} (link in the message; "
               + (f"they get {role} access)" if link.get("can_share") else "access unchanged, they may need to request it)"))
        for link in payload.get("links") or []
    ]


def _follow_up(payload: dict[str, Any]) -> list[str]:
    reply = payload.get("reply")
    if not reply:
        return []
    return ["", _field("Then reply to", reply.get("to")), quote(reply.get("body"))]


def _reason(payload: dict[str, Any]) -> list[str]:
    return [_field("Reason", payload["reason"])] if payload.get("reason") else []


def card_text(approval: dict[str, Any], hold_active: bool = False) -> str:
    """One approval request as an HTML card (without buttons)."""

    payload = approval.get("payload") or {}
    kind = approval.get("type")

    if kind == "trash_email":
        title = "🗑️ <b>Trash Conversation</b>" if payload.get("whole_thread") else "🗑️ <b>Trash Email</b>"
        lines = [
            f"{title}", "",
            _field("Subject", payload.get("subject") or "(no subject)"),
            _field("From", payload.get("sender")),
            *_reason(payload), "",
            "<i>Note: Recoverable from Gmail Trash for 30 days.</i>",
        ]
    elif kind in EMAIL_TYPES:
        reply = payload.get("reply") or {}
        to = payload.get("to") if kind == "send_email" else reply.get("to")
        subject = payload.get("subject") if kind == "send_email" else reply.get("subject")
        body = payload.get("body") if kind == "send_email" else reply.get("body")
        lines = ["✉️ <b>Outgoing Email Proposal</b>", "", _field("To", to or "the sender")]
        if subject or kind == "send_email":
            lines.append(_field("Subject", subject or "(no subject)"))
        lines += _link_lines(payload)
        if payload.get("attachments"):
            lines.append(_field("Files", ", ".join(a["name"] for a in payload["attachments"])))
        # The message exactly as it will be sent: a Google file's real link where it goes.
        lines += ["", "<b>Draft:</b>", quote(with_links(body, payload.get("links") or []))]
    elif kind == "book_meeting":
        lines = [
            "📅 <b>Meeting Proposal</b>", "",
            _field("Title", payload.get("title")),
            _field("With", ", ".join(invitees(payload)) or "nobody (no invite)"),
            _field("Proposed Slot", slot_text(payload.get("new_start"), payload.get("new_end"))),
        ]
        if payload.get("meeting_link"):
            lines.append(_field("Link", payload["meeting_link"]))
        if hold_active:
            lines.append("<b>Hold Status:</b> Local 24h soft-hold active")
        lines += _reason(payload) + _follow_up(payload)
    elif kind == "move_event":
        lines = [
            "🔁 <b>Move Meeting</b>", "",
            _field("Title", payload.get("title")),
            _field("From", format_time(payload.get("old_start"))),
            _field("To", slot_text(payload.get("new_start"), payload.get("new_end"))),
            _field("Attendees notified", ", ".join(payload.get("other_attendees", [])) or "none"),
        ]
        if hold_active:
            lines.append("<b>Hold Status:</b> Local 24h soft-hold active")
        lines += _reason(payload) + _follow_up(payload)
    elif kind == "cancel_event":
        lines = [
            "🚫 <b>Cancel Meeting</b>", "",
            _field("Title", payload.get("title")),
            _field("When", format_time(payload.get("old_start"))),
            _field("Attendees notified", ", ".join(payload.get("other_attendees", [])) or "none"),
            *_reason(payload), *_follow_up(payload),
        ]
    elif kind in DOCUMENT_TYPES:
        lines = ["📝 <b>Workspace Update</b>", ""]
        if kind == "create_document":
            lines += [_field("Target", f"New Google Doc '{payload.get('title')}' (Socio folder)"), _field("Action", "Create document")]
            diff = [f"+ {line}" for line in str(payload.get("content") or "").splitlines()] or ["+ (empty)"]
        elif kind == "append_document":
            lines += [_field("Target", payload.get("document_title") or payload.get("document")), _field("Action", "Append text")]
            diff = [f"+ {line}" for line in str(payload.get("text") or "").splitlines()] or ["+ (empty)"]
        elif kind == "create_spreadsheet":
            lines += [_field("Target", f"New Google Sheet '{payload.get('title')}' (Socio folder)"), _field("Action", "Create spreadsheet")]
            diff = ["+ " + (" | ".join(payload.get("headers") or []) or "(no header row)")]
        else:
            target = payload.get("document_title") or payload.get("document")
            if payload.get("tab"):
                target = f"{target} (tab {payload['tab']})"
            header = payload.get("header") or []
            lines += [_field("Target", target), _field("Action", "Append row")]
            diff = [
                f"+ {header[i] if i < len(header) and header[i] else f'column {i + 1}'}: {value}"
                for i, value in enumerate(payload.get("values") or [])
            ]
        lines += [_pre(diff), *_reason(payload)]
    elif kind == "create_and_share":
        sheet = payload.get("kind") == "spreadsheet"
        lines = [
            "📊 <b>Create &amp; Share Spreadsheet</b>" if sheet else "📄 <b>Create &amp; Share Document</b>", "",
            f"• {_field('Title', payload.get('title'))}",
            f"• {_field('Recipient', payload.get('recipient'))} ({esc(str(payload.get('role') or 'editor').capitalize())})",
        ]
        if sheet and payload.get("headers"):
            lines.append(f"• {_field('Columns', ' | '.join(payload['headers']))}")
        if not sheet and payload.get("content"):
            lines.append(f"• <b>Content:</b>\n{quote(payload['content'])}")
        link = "🔗 Sheet link" if sheet else "🔗 Doc link"
        lines += [
            f"• {_field('Subject', payload.get('subject') or '(no subject)')}",
            f"• <b>Message:</b>\n{quote(with_link(payload.get('body') or '', link))}",
            f"<i>{link} becomes the real link once it's created.</i>",
        ]
    elif kind == "share_file":
        lines = [
            "🔗 <b>Share File</b>", "",
            f"• {_field('File', payload.get('title'))}",
            f"• {_field('Recipient', payload.get('recipient'))} ({esc(str(payload.get('role') or 'editor').capitalize())})",
            "<i>Google emails them the link.</i>",
        ]
    elif kind == "create_task":
        lines = [
            "📌 <b>New Task</b>", "",
            _field("Task", payload.get("title") or "(untitled)"),
        ]
        if payload.get("due"):
            lines.append(_field("Due", payload["due"]))
        if payload.get("notes"):
            lines.append(_field("Notes", _short(payload["notes"], 600)))
        lines += _reason(payload)
    elif kind == "close_old_reminders":
        lines = [
            "🧹 <b>Tidy up old reminders?</b>", "",
            f"I found <b>{int(payload.get('count') or 0)}</b> reminder tasks I added automatically for emails "
            "(that feature is off now). For example:",
            *[f"• {esc(example)}" for example in payload.get("examples", [])[:5]],
            "", "<i>Closing marks them completed; you can reopen any of them in Google Tasks.</i>",
        ]
    elif kind == "complete_task" and payload.get("trigger") == "due":
        lines = [
            "🗓️ <b>Is this done?</b>", "",
            _field("Task", payload.get("title") or "(untitled)"),
            _field("Was due", _pretty_date(payload.get("due") or "")),
            "", "<i>If it's done, I'll close it in Google Tasks for you.</i>",
        ]
    elif kind == "complete_task":
        lines = [
            "☑️ <b>Can I close this task?</b>", "",
            _field("Task", payload.get("title") or "(untitled)"),
        ]
        if payload.get("due"):
            lines.append(_field("Due", _pretty_date(payload["due"])))
        lines += _reason(payload) + ["", "<i>You can reopen it in Google Tasks later.</i>"]
    elif kind == "enable_skill_script":
        lines = [
            "🔐 <b>Trust Skill Script</b>", "",
            _field("Script", f"skills/{payload.get('name')}.py"),
            _field("SHA-256", f"{str(payload.get('sha256'))[:12]}…"),
            "It will run with your user's permissions on files from your documents folder. Only approve code you wrote or reviewed.",
            "<b>First lines:</b>", _pre(str(payload.get("preview") or "").splitlines()),
        ]
    elif kind == "review_draft":
        lines = [
            "📝 <b>Draft Ready in Gmail</b>", "",
            _field("What to check", payload.get("target") or approval.get("target")),
            "<i>Edit and send it from Gmail Drafts, then tap Mark done.</i>",
        ]
    elif kind == "review_decision":
        lines = [
            "🤔 <b>Needs Your Decision</b>", "",
            _field("Subject", payload.get("subject") or "(no subject)"),
            _field("Why", payload.get("reason")),
        ]
    else:
        lines = ["📌 <b>Approval needed</b>", "", esc(approval.get("target") or kind)]

    return "\n".join(lines)


def approval_card(approval: dict[str, Any], hold_active: bool = False) -> tuple[str, list]:
    """The card text and its inline keyboard for an open approval request."""

    return card_text(approval, hold_active), _buttons(approval)
