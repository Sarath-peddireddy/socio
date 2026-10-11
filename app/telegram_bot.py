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
  else is sent. Nothing is learned from a tap; rules come only from the owner
  asking ("remember: never book calls before noon", the remember_rule tool) or
  /remember.
- Owner-facing text never shows internal numbers (approval, message, task ids).
- The follow-up context (last few exchanges) includes what each card proposed
  and how the owner decided, so "create a task for this" refers to it.
- /save_skill NAME saves the last successful multi-step instruction as a skill.
- The inbox is processed every AGENT_POLL_MINUTES (default 10, 0 = off).
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

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.approvals import approve, describe, reject, to_draft
from app.agent.llm import LLMUnavailableError
from app.agent.runner import BUSY_STATUS, DEFAULT_QUERY, handle_owner_command, process_inbox
from app.agent.state import AgentStateStore
from app.agent.supervisor import save_last_plan
from app.errors import report_failure, scrub, setup_logging
from app.google_api import GoogleApiUnavailableError
from app.memory import skills as skill_store
from app.memory.embeddings import get_embedder
from app.memory.episodic import active_notes
from app.memory.learning import learn_from_owner_note
from app.memory.semantic import SemanticMemory, memory_enabled
from app.tasks.tracker import check_tasks
from app.telegram_format import (
    MAX_MESSAGE_CHARS,
    WELCOME_HTML,
    approval_card,
    card_text,
    clip_html,
    esc,
    is_capability_question,
    md_to_html,
    strip_html,
)


load_dotenv()
logger = logging.getLogger("telegram_bot")

HISTORY_TURNS = 3  # owner <-> agent exchanges kept for follow-up questions
HELP_TEXT = WELCOME_HTML

REJECTED_TEXT = "❌ Proposal rejected. No changes were made."
KEPT_OPEN_TEXT = "👍 Okay, I'll leave it open."
KEPT_ALL_TEXT = "👍 Okay, I'll leave them as they are."
DRAFTED_TEXT = "✏️ Saved to Gmail Drafts. Nothing was sent."
OLD_RULE_BUTTON_TEXT = "That button is no longer used. To add a rule, just tell me, e.g. \"remember: never book calls before noon\"."
DECISION_WORDS = {"approve": "approved", "reject": "rejected", "draft": "moved to Gmail Drafts (not sent)"}

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


class Bot:
    def __init__(self, api, owner_chat_id: int | None, poll_minutes: float = 10, store: AgentStateStore | None = None):
        self.api = api
        self.owner_chat_id = owner_chat_id
        self.poll_seconds = max(0.0, poll_minutes) * 60
        self.store = store or AgentStateStore()
        self.history: list = []
        self.offset: int | None = None
        self.next_inbox_run = time.monotonic()
        self.next_task_check = time.monotonic()
        # Decisions taken on cards since the last instruction, for the follow-up context.
        self.decisions: list[str] = []

    # ----- sending -------------------------------------------------------------

    def _call_html(self, method: str, params: dict):
        """Send HTML; if Telegram cannot parse it (a formatting bug), resend as plain text."""

        params = {**params, "text": clip_html(scrub(params["text"])), "parse_mode": "HTML"}
        try:
            return self.api.call(method, **params)
        except TelegramError as exc:
            if "parse entities" not in str(exc).lower():
                raise
            logger.warning("Telegram rejected the HTML of a %s (%s); sending it as plain text.", method, exc)
            params.pop("parse_mode")
            params["text"] = _clip(strip_html(params["text"]))
            return self.api.call(method, **params)

    def send(self, text: str, buttons: list | None = None):
        """Send HTML text (callers escape dynamic values with esc / md_to_html)."""

        params = {"chat_id": self.owner_chat_id, "text": text, "disable_web_page_preview": True}
        if buttons:
            params["reply_markup"] = {"inline_keyboard": buttons}
        return self._call_html("sendMessage", params)

    def _held_approvals(self) -> set[int]:
        return {hold["approval_id"] for hold in self.store.active_slot_holds()}

    def notify_approvals(self, resend_open: bool = False) -> int:
        """Send approval requests the owner has not seen yet (or all open ones), one card each."""

        approvals = (
            [self.store.get_approval(a["id"]) for a in self.store.pending_approvals()][::-1]
            if resend_open
            else self.store.unnotified_approvals()
        )
        held = self._held_approvals() if approvals else set()

        for approval in approvals:
            text, buttons = approval_card(approval, hold_active=approval["id"] in held)
            self.send(text, buttons)
            self.store.mark_approval_notified(approval["id"])

        return len(approvals)

    # ----- inbox -----------------------------------------------------------------

    def run_inbox(self, dry_run: bool = False, announce: bool = False) -> None:
        query = os.getenv("AGENT_INBOX_QUERY", DEFAULT_QUERY)
        outcomes = process_inbox(query=query, dry_run=dry_run)

        if outcomes and outcomes[0].status == BUSY_STATUS:
            # Another process (e.g. the CLI) is processing the inbox; the scheduled check stays quiet.
            if announce:
                self.send(f"⏳ {esc(outcomes[0].reason)}")
            return

        lines = []
        quiet = 0
        for outcome in outcomes:
            label = INTERESTING_STATUSES.get(outcome.status)
            if not label:
                quiet += 1
                continue
            line = f"{label}: <b>{esc(outcome.subject or '(no subject)')}</b>"
            if outcome.sender:
                line += f"\n   from {esc(outcome.sender)}"
            if outcome.status in {"failed", "needs_attention", "needs_review"} and (outcome.reason or outcome.summary):
                line += f"\n   <i>{esc((outcome.summary or outcome.reason)[:300])}</i>"
            lines.append(line)

        if lines or announce:
            header = f"{'🧪 Dry run - ' if dry_run else ''}📬 <b>Inbox check:</b> {len(lines)} handled"
            if quiet:
                header += f", {quiet} needed nothing"
            self.send(header + ("\n\n" + "\n\n".join(lines) if lines else ""))

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
        if action not in {"approve", "draft", "reject", "rule"} or not raw_id.isdigit():
            self._answer(query, "Unknown button.")
            return

        try:
            if action == "rule":  # cards sent before the button was removed
                self._answer(query, OLD_RULE_BUTTON_TEXT)
            else:
                self._decide(action, int(raw_id), query)
        except Exception as exc:
            text = report_failure(exc, "telegram.callback", html=True, action=action, approval_id=raw_id)
            self._answer(query, strip_html(text))
            self._report_error(text)

    def _answer(self, query: dict, text: str) -> None:
        """Close the button's loading spinner (best effort: an old query cannot be answered). Plain text."""

        try:
            self.api.call("answerCallbackQuery", callback_query_id=query["id"], text=scrub(text)[:190])
        except Exception as exc:
            logger.warning("Could not answer a button press: %s", exc)

    def _decide(self, action: str, approval_id: int, query: dict) -> None:
        if action == "reject":
            outcome = reject(approval_id, self.store)
        elif action == "draft":
            outcome = to_draft(approval_id, self.store)
        else:
            outcome = approve(approval_id, self.store)
        self._answer(query, outcome["message"])

        decided = self.store.get_approval(approval_id)
        if action == "reject" and outcome["ok"] and decided and decided["type"] == "complete_task":
            status_line = KEPT_OPEN_TEXT
        elif action == "reject" and outcome["ok"] and decided and decided["type"] == "close_old_reminders":
            status_line = KEPT_ALL_TEXT
        elif action == "reject" and outcome["ok"]:
            status_line = REJECTED_TEXT
        elif action == "draft" and outcome["ok"]:
            extra = " Attach the files there yourself." if "Attach the files" in outcome["message"] else ""
            status_line = DRAFTED_TEXT + extra
        else:
            mark = "✅" if outcome["ok"] else "⚠️"
            status_line = f"{mark} {esc(outcome['message'])}"

        # The card is rendered again from the stored request (it is decided now, so no hold
        # line and no buttons). Nothing else is sent.
        approval = self.store.get_approval(approval_id)
        original = card_text(approval) if approval else esc(query.get("message", {}).get("text", ""))
        self._call_html("editMessageText", {
            "chat_id": query["message"]["chat"]["id"],
            "message_id": query["message"]["message_id"],
            "text": f"{original}\n\n{status_line}",
            "disable_web_page_preview": True,
        })
        if outcome["ok"] and approval:
            self.decisions.append(f"The owner {DECISION_WORDS[action]}: {describe(approval)}")

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

        command = text.split()[0].split("@")[0].lower()
        argument = text[len(text.split()[0]):].strip()

        if command in {"/start", "/help"} or is_capability_question(text):
            self.send(HELP_TEXT)
        elif command == "/run":
            self.send("📬 Checking the inbox…")
            self.run_inbox(announce=True)
        elif command == "/dryrun":
            self.send("🧪 Dry run: checking the inbox without sending or changing anything…")
            self.run_inbox(dry_run=True, announce=True)
        elif command == "/status":
            self.send(self._status_text())
        elif command == "/approvals":
            if not self.notify_approvals(resend_open=True):
                self.send("No open approval requests.")
        elif command == "/new":
            self.history, self.decisions = [], []
            self.send("Okay, starting fresh.")
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
        elif command == "/skills":
            self.send(self._skills_text())
        elif command == "/save_skill":
            self._save_skill(argument)
        elif command == "/forget_skill":
            try:
                ok = bool(argument) and skill_store.archive_skill(argument)
            except skill_store.SkillError:
                ok = False
            self.send(f"Archived skill <b>{esc(argument)}</b> (skills/.archive)." if ok else "Usage: <code>/forget_skill NAME</code> (see /skills).")
        else:
            self._run_instruction(text)

    def _run_instruction(self, text: str) -> None:
        self.api.call("sendChatAction", chat_id=self.owner_chat_id, action="typing")
        self._note_decisions()
        result = handle_owner_command(text, history=self.history)

        # An empty reply means the approval cards say it all: no "I have sent an approval request" preamble.
        if result.reply.strip():
            self.send(md_to_html(result.reply))
        if result.approval_ids:
            self.notify_approvals()

        # The context for follow-ups says what each card proposed, not just that cards were sent.
        remembered = result.reply
        cards = [self.store.get_approval(i) for i in result.approval_ids]
        if cards:
            remembered += "\n(Shown to the owner as approval cards, waiting for their decision:\n" + "\n".join(
                f"- {describe(card)}" for card in cards if card
            ) + ")"
        self.history += [HumanMessage(content=f"Instruction from the owner:\n{text}"), AIMessage(content=remembered.strip() or "Done.")]
        self.history = self.history[-2 * HISTORY_TURNS:]

    def _note_decisions(self) -> None:
        """Add the owner's taps since the last instruction to the last reply in the context."""

        if self.decisions and self.history and isinstance(self.history[-1], AIMessage):
            note = "\n(Afterwards: " + " / ".join(self.decisions) + ")"
            self.history[-1] = AIMessage(content=self.history[-1].content + note)
        self.decisions = []

    def _learn(self, text: str, approval: dict | None) -> None:
        if not memory_enabled():
            self.send("Memory is turned off (AGENT_MEMORY=0).")
            return
        saved = learn_from_owner_note(text, approval, store=self.store)
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
        if not memories and not notes:
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
        lines.append(f"<b>Active calendar holds:</b> {len(self.store.active_slot_holds())}")
        lines.append(f"<b>Open tasks I'm keeping an eye on:</b> {len(self.store.tracked_tasks('open'))}")
        interrupted = self.store.interrupted_supervisor_runs()
        if interrupted:
            lines.append(f"<b>Instructions stopped by a restart</b> (not re-run): {len(interrupted)}")
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

        for update in updates:
            self.offset = update["update_id"] + 1
            try:
                self.handle_update(update)
            except Exception as exc:
                kind = "callback_query" if "callback_query" in update else "message"
                self._report_error(report_failure(exc, "telegram.update", html=True, update_id=update.get("update_id"), kind=kind))

        if self.owner_chat_id is not None:
            if self.poll_seconds and time.monotonic() >= self.next_inbox_run:
                self.next_inbox_run = time.monotonic() + self.poll_seconds
                try:
                    self.run_inbox()
                except (LLMUnavailableError, GoogleApiUnavailableError) as exc:
                    # An outage during a scheduled check stays quiet; the next check retries.
                    report_failure(exc, "telegram.scheduled_inbox")
                except Exception as exc:
                    self._report_error(report_failure(exc, "telegram.scheduled_inbox", html=True))

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
            self.send("🤖 <b>Socio is online.</b> Send /help for commands.")

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
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    setup_logging()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in .env (create a bot with @BotFather in Telegram).")
        return 1

    owner = os.getenv("TELEGRAM_OWNER_CHAT_ID", "").strip()
    owner_chat_id = int(owner) if owner else None
    if owner_chat_id is None:
        print("TELEGRAM_OWNER_CHAT_ID is not set: setup mode. Send any message to your bot to get your chat id.")

    poll_minutes = float(os.getenv("AGENT_POLL_MINUTES", "10"))
    print(f"Bot running. Inbox check every {poll_minutes:g} min (0 = off). Press Ctrl+C to stop.")

    try:
        Bot(TelegramAPI(token), owner_chat_id, poll_minutes).run_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
