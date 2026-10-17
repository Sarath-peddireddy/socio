"""
Telegram front end for Socio.

    python -m app.telegram_bot

- Only the owner's chat (TELEGRAM_OWNER_CHAT_ID) can use the bot. Until it is
  set, the bot only replies with the sender's chat id so it can be configured.
- Every message is Telegram HTML (parse_mode="HTML", app.telegram_format):
  dynamic values are escaped, the agent's Markdown is converted, never raw.
- /start, /help, and "what can you do?" get the capabilities card (no model call).
- Plain messages are instructions for the agent (owner mode); multi-part ones
  are split across worker subagents (app.agent.supervisor).
- Each approval request is one card with its buttons (Approve / Reject; emails
  also get Edit in Gmail, which saves them as a Gmail draft instead). When an
  instruction only produced approval requests, the cards are all that is sent.
- Reject / Edit in Gmail end quietly: the card is updated in place and nothing
  else is sent (a rejected proposal's card ends with an optional "reply to this
  card" line). Nothing is learned from a tap; rules come only from the owner
  asking ("remember: never book calls before noon", the remember_rule tool) or
  /remember. A rule that contradicts an existing one becomes a Rule Conflict card
  (Replace / Merge / Keep Both / Cancel). A Telegram reply to a rejected card that says what
  was wrong is kept as a lesson for next time (app.agent.eval_feedback; /forget_lesson).
- Once a day (and at startup) records past their retention period are deleted
  (app.retention_manager), then the WAL is checkpointed.
- Owner-facing text never shows internal numbers (approval, message, task ids).
- The follow-up context (last few exchanges) includes what each card proposed
  and how the owner decided, so "create a task for this" refers to it. It is
  stored in SQLite (app.memory.context_assembler: whole turns, 2,000-token cap,
  24 h), with the thread / contact / document the chat is about
  (app.memory.context_anchor), so both survive a restart. /new, /clear, /reset
  start over.
- /save_skill NAME saves the last successful multi-step instruction as a skill.
- Routines ("send me the top AI news every day at 9 AM", app.scheduler) are checked once a
  minute; each result arrives as "⏰ Scheduled Briefing". /schedules lists them with Cancel buttons.
- The inbox is checked only when the owner asks: /check (or /run), or a message like "check my emails"
  (developer decision 2026-10-17: Socio never scans the inbox on its own). The same check refreshes the
  Google Tasks tracker. AGENT_POLL_MINUTES > 0 would bring back a timer; the default is 0 (off).
- Every handler runs behind the error gateway (app.errors): details go to
  logs/agent.log, the chat only ever gets a short predefined status card.

Uses the Telegram Bot API over HTTPS long polling (no webhook, nothing exposed).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from app import observability
from app.agent import clock
from app.agent.approvals import STATUS_WORDS, approve, decide_rule_conflict, describe, fill_reply, reject, reply_gaps, to_draft
from app.agent.eval_feedback import correctable_case, forget_lesson, is_proposal, list_lessons, read_correction, record_correction
from app.agent.gap_fill import fill_gaps
from app.agent.llm import LLMUnavailableError
from app.agent.runner import (
    BUSY_STATUS,
    DEFAULT_QUERY,
    RECHECKED_ALREADY,
    handle_owner_command,
    process_inbox,
    recheck_stale_approval,
)
from app.agent.state import AgentStateStore
from app.agent.supervisor import save_last_plan
from app.calendar.tools import INDIA_TZ
from app.errors import MESSAGES as ERROR_MESSAGES, report_failure, scrub, setup_logging
from app.google_api import GoogleApiUnavailableError
from app.memory import skills as skill_store
from app.memory.embeddings import get_embedder
from app.memory.episodic import active_notes
from app.memory.context_anchor import (
    clear_context_anchor,
    entities_from_approval,
    entities_from_run,
    format_anchor,
    get_context_anchor,
    update_context_anchor,
)
from app.memory.context_assembler import archive_history, load_history, note_on_last_turn, save_turn
from app.memory.learning import learn_from_owner_note
from app.memory.semantic import SemanticMemory, memory_enabled
from app.retention_manager import purge_expired_records
from app.scheduler.jobs import describe_schedule, next_run_text
from app.scheduler.service import TICK_SECONDS, RoutineRun, run_due_jobs
from app.tasks.tracker import check_tasks
from app.telegram_format import (
    MAX_MESSAGE_CHARS,
    WELCOME_HTML,
    approval_card,
    card_text,
    clip_html,
    esc,
    is_acknowledgement,
    is_capability_question,
    is_inbox_check_request,
    md_to_html,
    menu_commands,
    reply_messages,
    gap_question,
    routine_header,
    schedules_html,
    stale_notice,
    strip_html,
)


load_dotenv()
logger = logging.getLogger("telegram_bot")

HELP_TEXT = WELCOME_HTML

REJECTED_TEXT = "❌ Proposal rejected. No changes were made."
# A reply to a rejected proposal card that says what was wrong (app.agent.eval_feedback).
CORRECTION_HINT = "<i>If you like, reply to this card to tell me what to do differently next time.</i>"
CORRECTION_SAVED = "📝 Got it. Next time I'll keep this in mind:\n<i>{lesson}</i>"
# Rule conflict card buttons -> app.agent.approvals.decide_rule_conflict choice.
RULE_CHOICES = {"rule_replace": "replace", "rule_merge": "merge", "rule_both": "both"}
KEPT_OPEN_TEXT = "👍 Okay, I'll leave it open."
KEPT_ALL_TEXT = "👍 Okay, I'll leave them as they are."
DRAFTED_TEXT = "✏️ Saved to Gmail Drafts. Nothing was sent."
RESET_TEXT = "🧹 <b>Context Reset</b>\nActive conversational memory cleared. Ready for a new task."
OLD_RULE_BUTTON_TEXT = "That button is no longer used. To add a rule, just tell me, e.g. \"remember: never book calls before noon\"."
DECISION_WORDS = {"approve": "approved", "reject": "rejected", "draft": "moved to Gmail Drafts (not sent)"}
# What a Re-check found, by the email's new outcome (any new card arrives right below).
RECHECK_OUTCOMES = {
    "awaiting_approval": "🔄 Re-checked. Anything that needs you is below.",
    "no_action": "🔄 Re-checked: nothing needs doing in this conversation now.",
    "needs_review": "🔄 Re-checked: this one needs you. Anything I prepared is below or in Gmail Drafts.",
    "needs_attention": "🔄 Re-checked, but something didn't go through. Please look at it in Gmail.",
    "failed": "🔄 Re-checked, but I couldn't finish. I'll try again the next time you ask me to check the inbox.",
}
# Shown the instant a card's button is tapped (toast, and on the card until the result).
WORKING_TOAST = {"approve": "Working on it…", "draft": "Saving it to Gmail Drafts…", "reject": "Rejected."}
# Inbox summary lines for emails that got no card (fixed wording; never the agent's own notes).
INBOX_NOTES = {
    "failed": "I couldn't finish this one. I'll try again the next time you ask me to check the inbox.",
    "failed_permanently": "I tried 3 times and couldn't finish it. Please handle it in Gmail.",
    "needs_attention": "Something didn't go through. Please check it in Gmail.",
    "needs_review": "This one needs you. Please look at it in Gmail.",
    "replied_unverified": "The reply went out, but I couldn't confirm it in the conversation.",
}


def _gateway_text(reason: str) -> bool:
    """True when an outcome's reason is one of the error gateway's fixed sentences (app.errors)."""

    return any((reason or "").startswith(text.removeprefix("⚠️").strip()) for text in ERROR_MESSAGES.values())


def already_text(approval: dict) -> str:
    """What a tap on a card that is no longer open says (plain words, no numbers)."""

    return f"This was already {STATUS_WORDS.get(approval['status'], 'handled')}."


# A message read this long after it was sent is anchored to the real time (sent_at).
BACKLOG_MINUTES = 10
# A card Telegram refused is sent again cut to this many characters (emoji count double there).
SHORT_CARD_CHARS = 1500
# A card update lost to a network drop is retried on this many polls before giving up.
MAX_EDIT_TRIES = 5
# Database maintenance (WAL checkpoint): at startup, then once a day.
MAINTENANCE_SECONDS = 24 * 60 * 60

INTERESTING_STATUSES = {
    "replied": "✅ Replied",
    "replied_unverified": "⚠️ Replied (not verified)",
    "needs_review": "📝 Needs your review",
    "needs_attention": "⚠️ Needs attention",
    "awaiting_approval": "⏳ Waiting for approval",
    "failed": "❌ Failed",
    "failed_permanently": "❌ Gave up",
    "dry_run": "🧪 Dry run",
}


class TelegramError(RuntimeError):
    pass


class TelegramAPI:
    def __init__(self, token: str):
        self._base = f"https://api.telegram.org/bot{token}/"

    def call(self, method: str, **params):
        timeout = params.get("timeout", 0) + 15
        response = requests.post(self._base + method, json=params, timeout=timeout)
        data = response.json()
        if not data.get("ok"):
            raise TelegramError(f"{method}: {data.get('description', 'unknown error')}")
        return data.get("result")


def _clip(text: str) -> str:
    return text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 20] + "\n… (truncated)"


def sent_at(message: dict) -> datetime | None:
    """
    When the owner sent a Telegram message (its own Unix timestamp), in IST; None (= the real
    time) for a message read long after it was sent, e.g. queued while the bot was off: "in 2
    hours" or "today" must not resolve to a time that has already passed (audit M10).
    """

    from datetime import timedelta

    try:
        moment = datetime.fromtimestamp(int(message["date"]), tz=timezone.utc).astimezone(INDIA_TZ)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return None
    if _now() - moment > timedelta(minutes=BACKLOG_MINUTES):
        return None
    return moment


def _now() -> datetime:
    return datetime.now(INDIA_TZ)


def _trace_metadata(message: dict) -> dict:
    """The frozen ingress time and the dates every prompt of this request will use (app.agent.clock)."""

    at = sent_at(message) or datetime.now(INDIA_TZ)
    return {"ingress_utc": at.astimezone(timezone.utc).isoformat(), "temporal_anchor": clock.temporal_context(at)}


class Bot:
    def __init__(self, api, owner_chat_id: int | None, poll_minutes: float = 0, store: AgentStateStore | None = None):
        self.api = api
        self.owner_chat_id = owner_chat_id
        self.poll_seconds = max(0.0, poll_minutes) * 60
        self.store = store or AgentStateStore()
        self.offset: int | None = None
        self.next_inbox_run = time.monotonic()
        self.next_task_check = time.monotonic()
        self.next_routine_check = time.monotonic()
        self.next_maintenance = time.monotonic()
        # Decisions taken on cards since the last instruction, for the follow-up context.
        self.decisions: list[str] = []
        # Card updates Telegram didn't receive (network drop): retried each poll, newest wins.
        self.pending_edits: dict[tuple, dict] = {}
        # The last card sent, if it is a reply waiting for the owner's answer in chat.
        self.last_gap_card: int | None = None
        # Question messages opened by 💬 Answer -> the card they belong to.
        self.answer_prompts: dict[int, int] = {}
        # The "Google Sign-In Needed" card was sent for the current outage: scheduled checks then
        # stay quiet until a check succeeds again (audit H4: it was resent every 10 minutes).
        self.signin_notice_sent = False

    @property
    def history(self) -> list:
        """The follow-up context for the next instruction (stored turns, trimmed to the budget)."""

        return load_history(self.owner_chat_id, self.store)

    def reset_context(self) -> None:
        """Forget the conversation: earlier turns are archived and the anchor is cleared."""

        archive_history(self.owner_chat_id, self.store)
        clear_context_anchor(self.owner_chat_id, self.store)
        self.decisions = []

    # ----- sending -------------------------------------------------------------

    def _call_html(self, method: str, params: dict):
        """Send HTML; if Telegram cannot parse it (a formatting bug), resend as plain text."""

        params = {**params, "text": clip_html(scrub(params["text"])), "parse_mode": "HTML"}
        try:
            sent = self.api.call(method, **params)
        except TelegramError as exc:
            if "parse entities" not in str(exc).lower():
                raise
            logger.warning("Telegram rejected the HTML of a %s (%s); sending it as plain text.", method, exc)
            params.pop("parse_mode")
            params["text"] = _clip(strip_html(params["text"]))
            sent = self.api.call(method, **params)
        if method == "sendMessage":
            observability.record_reply(params["text"])  # the trace's output: what the owner got
        return sent

    def send(self, text: str, buttons: list | None = None):
        """Send HTML text (callers escape dynamic values with esc / md_to_html)."""

        params = {"chat_id": self.owner_chat_id, "text": text, "disable_web_page_preview": True}
        if buttons:
            params["reply_markup"] = {"inline_keyboard": buttons}
        return self._call_html("sendMessage", params)

    def notify_approvals(self, resend_open: bool = False) -> int:
        """Send approval requests the owner has not seen yet (or all open ones), one card each."""

        approvals = (
            [self.store.get_approval(a["id"]) for a in self.store.pending_approvals()][::-1]
            if resend_open
            else self.store.unnotified_approvals()
        )
        for approval in approvals:
            text, buttons = approval_card(approval)
            sent = self._send_card(approval, text, buttons)
            card_message = sent.get("message_id") if isinstance(sent, dict) else None
            self.store.mark_approval_notified(approval["id"], card_message)
            # The newest card decides what a plain next message may answer (see _gap_card_for).
            self.last_gap_card = approval["id"] if reply_gaps(approval) and card_message else None

        return len(approvals)

    def _send_card(self, approval: dict, text: str, buttons: list) -> dict | None:
        """
        One card, never blocking the others (audit M8): if Telegram refuses it (too long, a
        character it can't take...), it goes again shortened, then as its title with the
        buttons. If even that is refused, the owner is told once and the card is not retried
        every poll. A network error still propagates: nothing is marked, the next poll retries.
        """

        attempts = [text, clip_html(text, SHORT_CARD_CHARS) + "\n<i>(Shortened to fit Telegram.)</i>",
                    text.split("\n", 1)[0] + "\n<i>(The details couldn't be shown here.)</i>"]
        last_error: Exception | None = None
        for attempt in attempts:
            try:
                return self.send(attempt, buttons)
            except TelegramError as exc:
                last_error = exc
                logger.warning("Telegram refused an approval card (%s); trying a shorter version.", exc)
        report_failure(last_error, "telegram.card", approval_id=approval["id"], type=approval.get("type"))
        self._report_error(
            "⚠️ <b>A request couldn't be shown here</b>\nNothing was done with it. "
            "See it with <code>python -m app.main --status</code> on the computer running Socio."
        )
        return None

    # ----- inbox -----------------------------------------------------------------

    def run_inbox(self, dry_run: bool = False, announce: bool = False) -> None:
        query = os.getenv("AGENT_INBOX_QUERY", DEFAULT_QUERY)
        outcomes = process_inbox(query=query, dry_run=dry_run)

        if outcomes and outcomes[0].status == BUSY_STATUS:
            # Another process (e.g. the CLI) is processing the inbox; the scheduled check stays quiet.
            if announce:
                self.send(f"⏳ {esc(outcomes[0].reason)}")
            return

        # An email with a card is shown by its card alone (developer: never the same thing
        # twice). The rest get one line each, in fixed wording, never the agent's own notes.
        carded = {a.get("message_id") for a in self.store.pending_approvals() if a.get("message_id")}
        lines = []
        for outcome in outcomes:
            label = INTERESTING_STATUSES.get(outcome.status)
            if not label or (outcome.message_id in carded and not dry_run):
                continue
            line = f"{label}: <b>{esc(outcome.subject or '(no subject)')}</b>"
            if outcome.sender:
                line += f"\n   from {esc(outcome.sender)}"
            if outcome.status == "failed" and outcome.message_id == "-":
                line = f"⚠️ {esc(outcome.reason)}"  # the check itself failed (e.g. Google unreachable)
            elif outcome.status == "failed" and _gateway_text(outcome.reason):
                line += f"\n   <i>{esc(outcome.reason)}</i>"  # a predefined outage sentence, safe to show
            elif outcome.status in INBOX_NOTES:
                line += f"\n   <i>{esc(INBOX_NOTES[outcome.status])}</i>"
            lines.append(line)

        prefix = "🧪 Dry run - " if dry_run else ""
        if lines:
            self.send(f"{prefix}📬 <b>Inbox check</b>\n\n" + "\n\n".join(lines))
        elif announce:
            waiting = any(o.message_id in carded for o in outcomes)
            self.send(f"{prefix}📬 Inbox checked. " + ("What needs you is in the cards." if waiting else "Nothing needs you right now."))

    def check_now(self) -> None:
        """
        The owner asked for an inbox check (/check, /run, "check my emails"): the only way the inbox is
        read unless AGENT_POLL_MINUTES is set. Then the Google Tasks tracker is refreshed too ("Is this
        done?" cards, and the open tasks the agent knows about), quietly: a Tasks problem never hides
        the inbox result.
        """

        self.send("📬 Checking the inbox…")
        self.run_inbox(announce=True)
        try:
            check_tasks(self.store)
        except Exception as exc:
            report_failure(exc, "telegram.task_check")

    # ----- incoming updates ------------------------------------------------------

    def handle_update(self, update: dict) -> None:
        if "callback_query" in update:
            self._handle_button(update["callback_query"])
        elif "message" in update:
            self._handle_message(update["message"])

    def _is_owner(self, chat_id) -> bool:
        return self.owner_chat_id is not None and int(chat_id) == int(self.owner_chat_id)

    def _handle_button(self, query: dict) -> None:
        chat_id = query.get("message", {}).get("chat", {}).get("id")

        if not self._is_owner(chat_id):
            self.api.call("answerCallbackQuery", callback_query_id=query["id"], text="Not allowed.")
            return

        action, _, raw_id = (query.get("data") or "").partition(":")
        if action == "unsched":
            try:
                self._cancel_routine(raw_id, query)
            except Exception as exc:
                text = report_failure(exc, "telegram.cancel_routine", html=True, job_id=raw_id)
                self._answer(query, strip_html(text))
                self._report_error(text)
            return
        if action not in {"approve", "draft", "reject", "rule", "recheck", "answer", *RULE_CHOICES} or not raw_id.isdigit():
            self._answer(query, "Unknown button.")
            return

        try:
            if action == "rule":  # cards sent before the button was removed
                self._answer(query, OLD_RULE_BUTTON_TEXT)
            elif action in RULE_CHOICES:
                self._decide_rule(RULE_CHOICES[action], int(raw_id), query)
            elif action == "answer":
                self._ask_for_answer(int(raw_id), query)
            elif action == "recheck":
                self._recheck(int(raw_id), query)
            else:
                self._decide(action, int(raw_id), query)
        except Exception as exc:
            text = report_failure(exc, "telegram.callback", html=True, action=action, approval_id=raw_id)
            self._answer(query, strip_html(text))
            self._report_error(text)

    def _answer(self, query: dict, text: str, alert: bool = False) -> None:
        """
        Close the button's loading spinner (best effort: an old query cannot be answered). Plain
        text; alert=True shows it as a pop-up the owner has to dismiss.
        """

        params = {"callback_query_id": query["id"], "text": scrub(text)[:190]}
        if alert:
            params["show_alert"] = True
        try:
            self.api.call("answerCallbackQuery", **params)
        except Exception as exc:
            logger.warning("Could not answer a button press: %s", exc)

    def _edit_card(self, query: dict, text: str, buttons: list | None = None) -> None:
        self._edit_message(query["message"]["chat"]["id"], query["message"]["message_id"], text, buttons)

    def _edit_message(self, chat_id, message_id, text: str, buttons: list | None = None) -> None:
        params = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if buttons:
            params["reply_markup"] = {"inline_keyboard": buttons}
        key = (params["chat_id"], params["message_id"])
        self.pending_edits.pop(key, None)  # this edit supersedes any queued older one
        try:
            self._call_html("editMessageText", params)
        except requests.RequestException as exc:
            # Updating the card is only the display: whatever the tap did has happened (or not)
            # already. A dropped connection must not turn into an error card; retry it instead.
            logger.warning("Could not update a card (%s); retrying on the next poll.", type(exc).__name__)
            self.pending_edits[key] = {"params": params, "tries": 1}
        except TelegramError as exc:
            if "message is not modified" not in str(exc).lower():
                raise

    def _retry_card_edits(self) -> None:
        """Send card updates that failed earlier (a few tries, then give up quietly)."""

        for key, item in list(self.pending_edits.items()):
            try:
                self._call_html("editMessageText", item["params"])
            except (requests.RequestException, TelegramError) as exc:
                unchanged = "message is not modified" in str(exc).lower()
                if not unchanged and item["tries"] < MAX_EDIT_TRIES:
                    item["tries"] += 1
                    continue
                if not unchanged:
                    logger.warning("Gave up updating a card after %d tries: %s", item["tries"], type(exc).__name__)
            self.pending_edits.pop(key, None)

    def _decide(self, action: str, approval_id: int, query: dict) -> None:
        current = self.store.get_approval(approval_id)
        if current and current["status"] != "open":
            # A double tap or a repeated update: the card already shows (or is about to show)
            # the first tap's result. A short toast only; never overwrite the card.
            self._answer(query, already_text(current))
            return

        # Stop the button's spinner at once (Google calls take seconds), and replace the
        # buttons with a working line so there is nothing to tap twice. The result follows.
        self._answer(query, WORKING_TOAST[action])
        if current and action != "reject":  # rejecting is instant: no in-between state
            self._edit_card(query, f"{card_text(current)}\n\n⏳ <i>{WORKING_TOAST[action]}</i>")

        try:
            if action == "reject":
                outcome = reject(approval_id, self.store)
            elif action == "draft":
                outcome = to_draft(approval_id, self.store)
            else:
                outcome = approve(approval_id, self.store)
        except Exception:
            self._restore_card(query, approval_id)
            raise

        approval = self.store.get_approval(approval_id)
        if outcome.get("duplicate"):
            # Lost the race to another process (the CLI): show what the request is now.
            line = f"ℹ️ {esc(already_text(approval))}" if approval else ""
            self._edit_card(query, f"{card_text(approval)}\n\n{line}" if approval else esc(outcome["message"]))
            return

        if approval and approval["status"] == "open":
            # Nothing happened and the request is still waiting (e.g. a gap left in a reply):
            # the card keeps its buttons, with the reason under it.
            text, buttons = approval_card(approval)
            self._edit_card(query, f"{text}\n\n⚠️ {esc(outcome['message'])}", buttons)
            return

        if outcome.get("stale") and approval:
            # Nothing was done: say why, and for an incoming email offer to look again.
            self._edit_card(query, *self._stale_card(approval, outcome["message"]))
            return

        decided = approval
        if action == "reject" and outcome["ok"] and decided and decided["type"] == "complete_task":
            status_line = KEPT_OPEN_TEXT
        elif action == "reject" and outcome["ok"] and decided and decided["type"] == "close_old_reminders":
            status_line = KEPT_ALL_TEXT
        elif action == "reject" and outcome["ok"] and decided and decided["type"] == "rule_conflict":
            status_line = f"❌ {esc(outcome['message'])}"
        elif action == "reject" and outcome["ok"]:
            # Still no follow-up message; a proposal's card invites an optional correction
            # (developer approved the hint line 2026-10-12), read by _answer_correction.
            status_line = REJECTED_TEXT + (f"\n{CORRECTION_HINT}" if decided and is_proposal(decided) else "")
        elif action == "draft" and outcome["ok"]:
            extra = " Attach the files there yourself." if "Attach the files" in outcome["message"] else ""
            status_line = DRAFTED_TEXT + extra
        else:
            mark = "✅" if outcome["ok"] and not outcome.get("partial") else "⚠️"
            status_line = f"{mark} {esc(outcome['message'])}"

        # The card is rendered again from the stored request (it is decided now, so no hold
        # line and no buttons). Nothing else is sent.
        original = card_text(approval) if approval else esc(query.get("message", {}).get("text", ""))
        self._edit_card(query, f"{original}\n\n{status_line}")
        if outcome["ok"] and approval:
            self.decisions.append(f"The owner {DECISION_WORDS[action]}: {describe(approval)}")
            if action != "reject":
                update_context_anchor(self.owner_chat_id, entities_from_approval(approval), self.store)

    def _decide_rule(self, choice: str, approval_id: int, query: dict) -> None:
        """Replace / Merge / Keep Both on a rule conflict card (Cancel is the usual reject)."""

        current = self.store.get_approval(approval_id)
        if current and current["status"] != "open":
            self._answer(query, already_text(current))
            return
        self._answer(query, "Saving your rules…")
        outcome = decide_rule_conflict(approval_id, choice, self.store)
        if outcome.get("duplicate"):
            return
        approval = self.store.get_approval(approval_id)
        original = card_text(approval) if approval else esc(query.get("message", {}).get("text", ""))
        mark = "✅" if outcome["ok"] else "⚠️"
        self._edit_card(query, f"{original}\n\n{mark} {esc(outcome['message'])}")

    @staticmethod
    def _stale_card(approval: dict, message: str | None = None) -> tuple[str, list | None]:
        """The "things changed" card; cards about an incoming email get the Re-check button."""

        recheck = bool(approval.get("message_id") and approval.get("thread_id"))
        reason = message or f"{approval.get('resolution') or ''} Nothing was sent or changed.".strip()
        text = f"{card_text(approval)}\n\n{stale_notice(reason, recheck)}"
        return text, ([[{"text": "🔄 Re-check", "callback_data": f"recheck:{approval['id']}"}]] if recheck else None)

    def _restore_card(self, query: dict, approval_id: int) -> None:
        """After an unexpected error: show the card as it really is (buttons back if still open)."""

        approval = self.store.get_approval(approval_id)
        if not approval:
            return
        try:
            if approval["status"] == "open":
                text, buttons = approval_card(approval)
                self._edit_card(query, text, buttons)
            else:
                self._edit_card(query, f"{card_text(approval)}\n\nℹ️ {esc(already_text(approval))}")
        except Exception as exc:
            logger.warning("Could not restore a card: %s", exc)

    def _recheck(self, approval_id: int, query: dict) -> None:
        """Re-check on a stale card: read the conversation again and propose afresh."""

        approval = self.store.get_approval(approval_id)
        if not approval:  # removed by the retention purge (decided more than 30 days ago)
            self._answer(query, "This is too old to re-check. Ask me again if you still need it.")
            return
        if approval["status"] != "stale":
            self._answer(query, "This was already re-checked.")
            return

        self._answer(query, "⏳ Reading the conversation again…")
        self._edit_card(query, f"{card_text(approval)}\n\n⏳ <i>Reading the conversation again…</i>")
        try:
            outcome = recheck_stale_approval(approval_id, self.store)
        except Exception:
            self._edit_card(query, *self._stale_card(approval))  # still stale: button back
            raise
        if outcome.status == BUSY_STATUS:
            self._edit_card(query, *self._stale_card(approval))
            self.send(f"⏳ {esc(outcome.reason)}")
            return

        line = RECHECK_OUTCOMES.get(outcome.status, RECHECK_OUTCOMES["awaiting_approval"])
        if outcome.status == RECHECKED_ALREADY:
            line = f"🔄 {esc(outcome.reason)}"
        decided = self.store.get_approval(approval_id) or approval
        self._edit_card(query, f"{card_text(decided)}\n\n⚠️ {esc(approval.get('resolution') or '')}\n{line}")
        self.notify_approvals()

    def _handle_message(self, message: dict) -> None:
        chat_id = message.get("chat", {}).get("id")

        if self.owner_chat_id is None:
            # Setup mode: tell the person their chat id, do nothing else.
            logger.warning("Setup: message from chat id %s", chat_id)
            self.api.call(
                "sendMessage",
                chat_id=chat_id,
                parse_mode="HTML",
                text=(
                    f"Your chat id is <code>{esc(chat_id)}</code>.\n"
                    f"Add <code>TELEGRAM_OWNER_CHAT_ID={esc(chat_id)}</code> to .env and restart the bot."
                ),
            )
            return

        if not self._is_owner(chat_id):
            logger.warning("Ignored message from non-owner chat %s", chat_id)
            return

        text = (message.get("text") or "").strip()
        if not text:
            self.send("Please send a text message.")
            return

        # One Langfuse trace per owner message, from receipt to the last message sent
        # (session = this chat, so a conversation's turns are grouped).
        with observability.interaction(
            "telegram_interaction",
            session_id=str(chat_id),
            user_id=str((message.get("from") or {}).get("id") or chat_id),
            tags=["telegram"],
            input={"text": text},
            metadata=_trace_metadata(message),
        ):
            self._dispatch(message, text)

    def _dispatch(self, message: dict, text: str) -> None:
        command = text.split()[0].split("@")[0].lower()
        argument = text[len(text.split()[0]):].strip()
        if text.startswith("/"):
            observability.annotate_root(intent=f"command {command}")

        if command in {"/start", "/help"} or is_capability_question(text):
            self.send(HELP_TEXT)
        elif command in {"/check", "/run"} or (not text.startswith("/") and is_inbox_check_request(text)):
            observability.annotate_root(intent="inbox_check")
            self.check_now()
        elif command == "/dryrun":
            self.send("🧪 Dry run: checking the inbox without sending or changing anything…")
            self.run_inbox(dry_run=True, announce=True)
        elif command == "/status":
            self.send(self._status_text())
        elif command == "/approvals":
            if not self.notify_approvals(resend_open=True):
                self.send("No open approval requests.")
        elif command == "/new":
            self.reset_context()
            self.send("Okay, starting fresh.")
        elif command in {"/clear", "/reset"}:
            self.reset_context()
            self.send(RESET_TEXT)
        elif command == "/remember":
            if argument:
                self._learn(argument, None)
            else:
                self.send("Tell me what to remember, e.g. <code>/remember no calls before 12:00</code>")
        elif command == "/memories":
            self.send(self._memories_text())
        elif command == "/forget":
            ok = argument.isdigit() and SemanticMemory(self.store).forget(int(argument))
            self.send(f"Forgot memory {esc(argument)}." if ok else "Usage: <code>/forget ID</code> (see /memories).")
        elif command == "/forget_lesson":
            ok = argument.isdigit() and forget_lesson(int(argument), self.store)
            self.send(f"Forgot lesson {esc(argument)}." if ok else "Usage: <code>/forget_lesson ID</code> (see /memories).")
        elif command == "/skills":
            self.send(self._skills_text())
        elif command in {"/schedules", "/routines"}:
            self.send(*self._schedules())
        elif command == "/save_skill":
            self._save_skill(argument)
        elif command == "/forget_skill":
            try:
                ok = bool(argument) and skill_store.archive_skill(argument)
            except skill_store.SkillError:
                ok = False
            self.send(f"Archived skill <b>{esc(argument)}</b> (skills/.archive)." if ok else "Usage: <code>/forget_skill NAME</code> (see /skills).")
        elif not text.startswith("/") and self._answer_gap(message, text):
            observability.annotate_root(intent="gap_answer")
            pass  # it answered a reply card's question: the filled-in card was sent
        elif not text.startswith("/") and self._answer_correction(message, text):
            observability.annotate_root(intent="correction")
        elif is_acknowledgement(text, self._last_reply()):
            observability.annotate_root(intent="acknowledgement")
            self._react(message)
        else:
            observability.annotate_root(intent="instruction")
            self._run_instruction(text, sent_at(message))

    def _ask_for_answer(self, approval_id: int, query: dict) -> None:
        """💬 Answer on a reply card: open Telegram's reply box with the question."""

        approval = self.store.get_approval(approval_id)
        if not approval or approval["status"] != "open" or not reply_gaps(approval):
            self._answer(query, "This reply doesn't need an answer any more.")
            return
        self._answer(query, "Type your answer below.")
        prompt = self._call_html("sendMessage", {
            "chat_id": self.owner_chat_id,
            "text": f"❓ <b>{esc(gap_question(approval))}</b>\nType your answer below.",
            # Opens the reply box on the phone, already pointing at this question.
            "reply_markup": {"force_reply": True, "input_field_placeholder": "Your answer"},
        })
        if isinstance(prompt, dict) and prompt.get("message_id"):
            self.answer_prompts[prompt["message_id"]] = approval_id
        self.last_gap_card = approval_id

    def _gap_card_for(self, message: dict) -> dict | None:
        """
        The open reply card with a gap that this message may answer: the card it replies to
        (Telegram "reply"), else the last card sent if that one is waiting for an answer.
        """

        replied_to = (message.get("reply_to_message") or {}).get("message_id")
        candidates = []
        if replied_to in self.answer_prompts:  # typed into the box opened by 💬 Answer
            candidates.append(self.store.get_approval(self.answer_prompts[replied_to]))
        elif replied_to:
            candidates.append(self.store.approval_for_card(replied_to))
        if not replied_to and self.last_gap_card:
            candidates.append(self.store.get_approval(self.last_gap_card))
        for approval in candidates:
            if approval and approval["status"] == "open" and reply_gaps(approval):
                return approval
        return None

    def _answer_gap(self, message: dict, text: str) -> bool:
        """Fill a reply card's gap from the owner's answer. False if the message wasn't an answer."""

        approval = self._gap_card_for(message)
        if not approval:
            return False
        try:
            self.api.call("sendChatAction", chat_id=self.owner_chat_id, action="typing")
        except Exception as exc:
            logger.info("Could not show typing: %s", exc)

        payload = approval["payload"]
        draft = payload.get("body") if approval["type"] == "send_email" else (payload.get("reply") or {}).get("body", "")
        filled = fill_gaps(draft, payload.get("question", ""), text)
        if not filled.is_answer:
            return False  # an instruction sent after the card: handled as usual

        new_id = fill_reply(approval["id"], filled.body, filled.still_missing, self.store)
        if new_id is None:
            self.send("That reply was already handled, so I left it as it was.")
            return True
        self.last_gap_card = None
        if approval.get("card_message_id"):
            old = self.store.get_approval(approval["id"]) or approval
            self._edit_message(
                self.owner_chat_id, approval["card_message_id"],
                f"{card_text(old)}\n\n✏️ <i>Filled in with your answer. The updated reply is below.</i>",
            )
        self.notify_approvals()
        return True

    def _answer_correction(self, message: dict, text: str) -> bool:
        """
        A Telegram reply to a rejected (or stale) proposal card: if it says what was wrong, keep
        it as a lesson (app.agent.eval_feedback). False when it isn't a correction (a new
        instruction such as "send it to Bob instead"): it is then handled as usual.
        """

        replied_to = (message.get("reply_to_message") or {}).get("message_id")
        approval = self.store.approval_for_card(replied_to) if replied_to else None
        if not approval or approval["status"] not in {"rejected", "stale"}:
            return False
        case = correctable_case(approval["id"], self.store)
        if not case:
            return False
        try:
            correction = read_correction(case, text)
        except Exception as exc:
            # Unread, so not stored; the message goes on as an instruction (which reports any outage).
            report_failure(exc, "telegram.correction", approval_id=approval["id"])
            return False
        if not correction.is_correction:
            return False
        if record_correction(case["case_id"], text, correction.lesson, self.store):
            self.send(CORRECTION_SAVED.format(lesson=esc(correction.lesson)))
        else:
            self.send("I already have your correction for that one.")
        return True

    def _last_reply(self) -> str:
        replies = [m.content for m in self.history if isinstance(m, AIMessage)]
        return str(replies[-1]) if replies else ""

    def _react(self, message: dict) -> None:
        """A 👍 on the owner's "Okay" / "Thanks": no reply, no model call, nothing saved."""

        try:
            self.api.call(
                "setMessageReaction", chat_id=self.owner_chat_id, message_id=message["message_id"],
                reaction=[{"type": "emoji", "emoji": "👍"}],
            )
        except Exception as exc:
            logger.info("Could not react to the message: %s", exc)

    def set_command_menu(self) -> None:
        """The command suggestions Telegram shows when the owner types "/" (owner chat only)."""

        try:
            self.api.call(
                "setMyCommands", commands=menu_commands(),
                scope={"type": "chat", "chat_id": self.owner_chat_id},
            )
        except Exception as exc:
            logger.warning("Could not set the command menu: %s", exc)

    def _run_instruction(self, text: str, received_at: datetime | None = None) -> None:
        # "typing…" is cosmetic: a network drop here once cost the whole instruction (the update
        # was already consumed, so the owner's message was silently lost).
        try:
            self.api.call("sendChatAction", chat_id=self.owner_chat_id, action="typing")
        except Exception as exc:
            logger.warning("Could not show 'typing': %s", exc)
        self._note_decisions()
        context = format_anchor(get_context_anchor(self.owner_chat_id, self.store))
        kwargs = {"received_at": received_at} if received_at else {}
        result = handle_owner_command(text, history=self.history, context=context, **kwargs)

        # One message per intent (answer / done / problem), then the approval cards, then any
        # question. Nothing at all when the cards say it all (no "I have sent an approval request").
        if result.parts:
            before, after = reply_messages(result.parts)
        else:
            before, after = ([md_to_html(result.reply)] if result.reply.strip() else []), []
        for message in before:
            self.send(message)
        if result.approval_ids:
            self.notify_approvals()
        for message in after:
            self.send(message)

        # The context for follow-ups says what each card proposed, not just that cards were sent.
        remembered = result.reply
        cards = [self.store.get_approval(i) for i in result.approval_ids]
        if cards:
            remembered += "\n(Shown to the owner as approval cards, waiting for their decision:\n" + "\n".join(
                f"- {describe(card)}" for card in cards if card
            ) + ")"
        save_turn(self.owner_chat_id, text, remembered, self.store)

        # What the conversation is now about: what the agent read or proposed, and the cards.
        entities = entities_from_run(getattr(result, "tool_calls", None) or [], result.actions or [])
        for card in cards:
            if card:
                entities.update(entities_from_approval(card))
        update_context_anchor(self.owner_chat_id, entities, self.store)

    # ----- scheduled routines --------------------------------------------------------

    def _schedules(self) -> tuple[str, list]:
        return schedules_html(self.store.scheduled_jobs(), describe_schedule, next_run_text)

    def _cancel_routine(self, job_id: str, query: dict) -> None:
        job = self.store.get_scheduled_job(job_id)
        stopped = bool(job) and self.store.cancel_scheduled_job(job_id)
        self._answer(query, f"Stopped: {job['title']}" if stopped else "That routine was already stopped.")
        text, buttons = self._schedules()
        params = {
            "chat_id": query["message"]["chat"]["id"],
            "message_id": query["message"]["message_id"],
            "text": text,
            "disable_web_page_preview": True,
        }
        if buttons:
            params["reply_markup"] = {"inline_keyboard": buttons}
        self._call_html("editMessageText", params)

    def run_routines(self) -> None:
        run_due_jobs(self._run_routine, self._deliver_routine, store=self.store)

    def _run_routine(self, job: dict):
        # No chat history or anchor: a routine stands on its own and must not act on
        # whatever the chat was last about.
        try:
            self.api.call("sendChatAction", chat_id=self.owner_chat_id, action="typing")
        except Exception as exc:  # only the "typing…" hint; the routine still runs
            logger.warning("Could not show typing for a routine: %s", exc)
        with observability.interaction(
            "scheduled_routine",
            session_id=str(self.owner_chat_id),
            tags=["routine"],
            input={"routine": job.get("title"), "prompt": job["prompt_query"]},
            metadata={"temporal_anchor": clock.temporal_context()},
        ) as trace:
            result = handle_owner_command(job["prompt_query"])
            trace.update(output={"reply": result.reply, "approval_cards": len(result.approval_ids)})
            return result

    def _deliver_routine(self, outcome: RoutineRun) -> None:
        job, header = outcome.job, routine_header(outcome.job["title"])
        if outcome.status == "missed":
            later = f" Next time: {esc(next_run_text(job))}." if job.get("is_active") else ""
            self.send(f"⏰ I missed your <b>{esc(job['title'])}</b> routine because I wasn't running at its time.{later}")
            return
        if outcome.status == "failed":
            self.send(f"{header}\n\n{outcome.error_html}")
            return

        result = outcome.result
        if result.parts:
            before, after = reply_messages(result.parts)
        else:
            before, after = ([md_to_html(result.reply)] if result.reply.strip() else []), []
        messages = [f"{header}\n\n{before[0]}" if before else header] + before[1:]
        for message in messages:
            self.send(message)
        if result.approval_ids:
            self.notify_approvals()
        for message in after:
            self.send(message)
        # Kept in the conversation so "tell me more about the second story" works.
        save_turn(self.owner_chat_id, f"(Scheduled routine: {job['title']})", result.reply, self.store)

    def _note_decisions(self) -> None:
        """Add the owner's taps since the last instruction to the last reply in the context."""

        if self.decisions:
            # The owner saw each result on its card ("✅ Email sent"): context for "it" / "that",
            # not news to report again.
            note_on_last_turn(
                self.owner_chat_id,
                "\n(Afterwards: " + " / ".join(self.decisions) + ". The owner already sees the result on the card.)",
                self.store,
            )
        self.decisions = []

    def _learn(self, text: str, approval: dict | None) -> None:
        if not memory_enabled():
            self.send("Memory is turned off (AGENT_MEMORY=0).")
            return
        saved = learn_from_owner_note(text, approval, store=self.store)
        if any(item.get("conflict_approval_id") for item in saved):
            self.notify_approvals()  # the rule conflict card: nothing was saved yet
            return
        lines = [f"🧠 {'Noted' if item['created'] else 'Updated'}: {esc(item['preference'])}" for item in saved]
        self.send("\n".join(lines) or "Nothing to remember.")

    def _save_skill(self, name: str) -> None:
        if not name:
            self.send("Usage: <code>/save_skill NAME</code> - saves your last successful multi-step instruction as a skill.")
            return
        try:
            saved = save_last_plan(name, store=self.store)
        except skill_store.SkillError as exc:
            self.send(f"⚠️ {esc(exc)}")
            return
        text = (
            f"🧩 Saved skill <b>{esc(saved.name)}</b> ({saved.tasks} steps; "
            f"parameters: {esc(', '.join(saved.parameters) or 'none')})."
        )
        if not saved.generalized:
            text += " It was saved exactly as it ran, so it repeats the same names and dates."
        self.send(text + " A matching request will reuse it (/skills to see it).")

    def _memories_text(self) -> str:
        memories = SemanticMemory(self.store).all(limit=20)
        notes = active_notes(self.store)
        lessons = list_lessons(self.store, limit=10)
        if not memories and not notes and not lessons:
            return "I haven't learned any rules yet. Just tell me one, e.g. \"remember: never book calls before noon\"."
        lines = []
        if memories:
            lines.append("🧠 <b>Rules I follow</b> (newest first):")
            lines += [
                f"{m.id}. {esc(m.preference)}" + ("" if m.entity.lower() == "owner" else f" <i>({esc(m.entity)})</i>")
                for m in memories
            ]
            lines.append("<i>To remove one, send /forget and its number.</i>")
        if notes:
            lines.append("🕒 <b>For today only:</b>")
            lines += [f"• {esc(n['note'])}" for n in notes]
        if lessons:
            lines.append("📝 <b>Lessons from your corrections:</b>")
            lines += [f"{lesson['case_id']}. {esc(lesson['lesson'])}" for lesson in lessons]
            lines.append("<i>To remove one, send /forget_lesson and its number.</i>")
        return "\n".join(lines)

    def _skills_text(self) -> str:
        skills = skill_store.list_skills()
        if not skills:
            return "No saved skills yet. After a multi-step instruction succeeds, save it with <code>/save_skill NAME</code>."
        lines = []
        for skill in skills:
            kind = f"script {skill.script}" if skill.script else f"{len(skill.plan)} steps"
            lines.append(f"• <b>{esc(skill.name)}</b> ({esc(kind)}, by {esc(skill.author)}): {esc(skill.description)}")
        return "🧩 <b>Saved skills:</b>\n" + "\n".join(lines)

    def _status_text(self) -> str:
        counts = self.store.status_counts()
        lines = ["📊 <b>Socio status</b>", ""]
        lines.append("<b>Email outcomes:</b> " + (esc(", ".join(f"{k} {v}" for k, v in sorted(counts.items()))) or "none yet"))
        lines.append(f"<b>Open approval requests:</b> {len(self.store.pending_approvals())} (/approvals to show)")
        lines.append(f"<b>Open tasks I'm keeping an eye on:</b> {len(self.store.tracked_tasks('open'))}")
        lines.append(f"<b>Scheduled routines:</b> {len(self.store.scheduled_jobs())} (/schedules)")
        interrupted = self.store.interrupted_supervisor_runs()
        if interrupted:
            lines.append(f"<b>Instructions stopped by a restart</b> (not re-run): {len(interrupted)}")
        stuck = self.store.stuck_approvals()
        if stuck:
            # Never retried: it may have gone through (an email may already be sent).
            lines.append(
                f"<b>Approved actions cut off by a restart:</b> {len(stuck)} "
                "(check Gmail / Calendar: they may or may not have happened)"
            )
        if memory_enabled():
            lines.append(f"<b>Memories:</b> {SemanticMemory(self.store).count()} (/memories); <b>skills:</b> {len(skill_store.list_skills())} (/skills)")
        # Raw failure messages hold internal details: only a count goes to the chat.
        failures = self.store.failures()
        if failures:
            lines.append(f"<b>Problems logged for review:</b> {len(failures)}")
        return "\n".join(lines)

    # ----- main loop ---------------------------------------------------------------

    def poll_once(self, long_poll_seconds: int = 25) -> None:
        updates = self.api.call(
            "getUpdates",
            offset=self.offset,
            timeout=long_poll_seconds,
            allowed_updates=["message", "callback_query"],
        )
        if self.pending_edits:  # Telegram is reachable again: catch up on card updates
            self._retry_card_edits()

        for update in updates:
            self.offset = update["update_id"] + 1
            try:
                self.handle_update(update)
            except Exception as exc:
                kind = "callback_query" if "callback_query" in update else "message"
                self._report_error(report_failure(exc, "telegram.update", html=True, update_id=update.get("update_id"), kind=kind))

        if self.owner_chat_id is not None:
            # Routines the owner scheduled (an explicit instruction) are checked every minute,
            # whether or not an inbox timer is set. The inbox and task checks below run only with
            # AGENT_POLL_MINUTES > 0 (default 0: on request only, see check_now).
            if time.monotonic() >= self.next_routine_check:
                self.next_routine_check = time.monotonic() + TICK_SECONDS
                try:
                    self.run_routines()
                except Exception as exc:
                    # Each routine reports its own failure; this is the due-check itself.
                    report_failure(exc, "telegram.routines")

            if self.poll_seconds and time.monotonic() >= self.next_inbox_run:
                self.next_inbox_run = time.monotonic() + self.poll_seconds
                from app.errors import classify

                try:
                    self.run_inbox()
                    self.signin_notice_sent = False  # signed in again: a later expiry is reported anew
                except (LLMUnavailableError, GoogleApiUnavailableError) as exc:
                    # An outage during a scheduled check stays quiet; the next check retries.
                    report_failure(exc, "telegram.scheduled_inbox")
                except Exception as exc:
                    card = report_failure(exc, "telegram.scheduled_inbox", html=True)
                    if classify(exc) != "google_signin":
                        self._report_error(card)
                    elif not self.signin_notice_sent:
                        # Once per outage: retrying cannot fix it, only signing in again does.
                        self.signin_notice_sent = True
                        self._report_error(card)

            if self.poll_seconds and time.monotonic() >= self.next_task_check:
                self.next_task_check = time.monotonic() + self.poll_seconds
                try:
                    check_tasks(self.store)  # asks "Is this done?" through approval cards
                except Exception as exc:
                    # A background check (e.g. not signed in to Google Tasks) never interrupts the chat.
                    report_failure(exc, "telegram.task_check")

            try:
                self.notify_approvals()
            except Exception as exc:
                self._report_error(report_failure(exc, "telegram.notify_approvals", html=True))

        if time.monotonic() >= self.next_maintenance:
            self.next_maintenance = time.monotonic() + MAINTENANCE_SECONDS
            self.maintain_database()

    def maintain_database(self) -> None:
        """
        Delete records past their retention period (app.retention_manager), then fold the SQLite
        WAL file back into the database. Quiet; failures are only logged.
        """

        try:
            result = purge_expired_records(self.store, checkpoint=False)
            logger.info("Retention purge: %s", {k: v for k, v in result["deleted"].items() if v} or "nothing old")
        except Exception as exc:
            report_failure(exc, "telegram.db_maintenance")
        try:
            result = self.store.checkpoint()
            logger.info("Database checkpoint: %s", result)
        except Exception as exc:
            report_failure(exc, "telegram.db_checkpoint")

    def _report_error(self, text: str) -> None:
        """Send a predefined gateway status card (never raw exception text)."""

        if self.owner_chat_id is None:
            return
        try:
            self.send(text)
        except Exception:
            logger.exception("Could not report error to Telegram")

    def run_forever(self) -> None:
        if self.owner_chat_id is not None:
            self.set_command_menu()
            # A greeting must never stop the bot: a network drop here once made it exit at startup.
            try:
                self.send("🤖 <b>Socio is online.</b> Send /help for commands.")
            except Exception as exc:
                logger.warning("Could not send the 'online' message: %s", exc)

        # Langfuse keys: rejected -> tracing off with one warning; unreachable -> carry on.
        observability.check_connection_in_background()

        # Load the local embedding model in the background so the first recall is fast.
        if memory_enabled():
            threading.Thread(target=lambda: getattr(get_embedder(), "warm_up", lambda: None)(), daemon=True).start()

        while True:
            try:
                self.poll_once()
            except (requests.RequestException, TelegramError) as exc:
                logger.warning("Telegram connection problem: %s. Retrying in 10s.", exc)
                time.sleep(10)
            except Exception as exc:  # never let one bad poll stop the bot
                report_failure(exc, "telegram.poll")
                time.sleep(10)


def main() -> int:
    from app.google_api import disable_browser_login

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    setup_logging()
    # Background process: a missing / expired Google sign-in raises (and the owner is told to
    # run --google-login) instead of opening a browser nobody watches (audit H4).
    disable_browser_login()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in .env (create a bot with @BotFather in Telegram).")
        return 1

    owner = os.getenv("TELEGRAM_OWNER_CHAT_ID", "").strip()
    owner_chat_id = int(owner) if owner else None
    if owner_chat_id is None:
        print("TELEGRAM_OWNER_CHAT_ID is not set: setup mode. Send any message to your bot to get your chat id.")

    poll_minutes = max(0.0, float(os.getenv("AGENT_POLL_MINUTES", "0") or 0))
    checks = f"every {poll_minutes:g} min (AGENT_POLL_MINUTES)" if poll_minutes else "only when you ask (/check)"
    print(f"Bot running. Inbox checks: {checks}. Press Ctrl+C to stop.")

    try:
        Bot(TelegramAPI(token), owner_chat_id, poll_minutes).run_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        observability.shutdown()  # send the last traces (bounded by the exporter's timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
