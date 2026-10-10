"""
Telegram front end for Socio.

    python -m app.telegram_bot

- Only the owner's chat (TELEGRAM_OWNER_CHAT_ID) can use the bot. Until it is
  set, the bot only replies with the sender's chat id so it can be configured.
- Plain messages are instructions for the agent (owner mode); multi-part ones
  are split across worker subagents (app.agent.supervisor).
- Approval requests arrive with Approve / Reject buttons (emails also get
  Edit in Gmail, which saves them as a Gmail draft instead).
- After a Reject or Edit in Gmail the bot asks why (optional); replying to that
  question teaches Socio a preference (long-term memory). /remember does too.
- The inbox is processed every AGENT_POLL_MINUTES (default 10, 0 = off).

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

from app.agent.approvals import EMAIL_TYPES, EXECUTABLE_TYPES, approve, describe, reject, to_draft
from app.agent.llm import LLMUnavailableError
from app.agent.runner import BUSY_STATUS, DEFAULT_QUERY, handle_owner_command, process_inbox
from app.agent.state import AgentStateStore
from app.google_api import GoogleApiUnavailableError
from app.memory import skills as skill_store
from app.memory.embeddings import get_embedder
from app.memory.learning import learn_from_owner_note
from app.memory.semantic import SemanticMemory, memory_enabled


load_dotenv()
logger = logging.getLogger("telegram_bot")

MAX_MESSAGE_CHARS = 4000
HISTORY_TURNS = 3  # owner <-> agent exchanges kept for follow-up questions

HELP_TEXT = """Socio commands:
/run - check the inbox now
/dryrun - check the inbox without sending or changing anything
/status - outcomes, open approvals, recent failures
/approvals - show open approval requests again
/new - forget the conversation context
/remember TEXT - teach me a preference (e.g. no calls before noon)
/memories - what I have learned; /forget ID removes one
/skills - saved procedures; /forget_skill NAME archives one

Or just type an instruction, e.g.
- send my resume to hr@company.com
- what's on my calendar this week?
- move my 3pm call with Raj to Friday 11am
- trash the Swiggy email
- add Acme, PM, applied today to my applications sheet
- write a cover letter for Acme in a doc
- remind me to send the invoice on Friday
Every email, Docs/Sheets change, booking, moving or cancelling meetings, and
deleting email asks for your approval first. An email can also be moved to
Gmail Drafts (Edit in Gmail) so you can change it before sending. After you
reject something, reply to my "why?" message and I'll remember it."""

FEEDBACK_PROMPT = "💬 Why? Reply to this message and I'll remember it for next time (optional)."

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

    # ----- sending -------------------------------------------------------------

    def send(self, text: str, buttons: list | None = None, force_reply: bool = False):
        params = {"chat_id": self.owner_chat_id, "text": _clip(text), "disable_web_page_preview": True}
        if buttons:
            params["reply_markup"] = {"inline_keyboard": buttons}
        elif force_reply:
            params["reply_markup"] = {"force_reply": True, "input_field_placeholder": "e.g. never book calls before noon"}
        return self.api.call("sendMessage", **params)

    def notify_approvals(self, resend_open: bool = False) -> int:
        """Send approval requests the owner has not seen yet (or all open ones)."""

        approvals = (
            [self.store.get_approval(a["id"]) for a in self.store.pending_approvals()][::-1]
            if resend_open
            else self.store.unnotified_approvals()
        )

        for approval in approvals:
            approval_id = approval["id"]
            if approval["type"] in EMAIL_TYPES:
                title = {
                    "send_reply_with_attachment": f"📎 Send this reply with a file? #{approval_id}",
                    "send_email": f"✉️ Send this email? #{approval_id}",
                }.get(approval["type"], f"✉️ Send this reply? #{approval_id}")
                buttons = [
                    [{"text": "✅ Approve & Send", "callback_data": f"approve:{approval_id}"},
                     {"text": "✏️ Edit in Gmail", "callback_data": f"draft:{approval_id}"}],
                    [{"text": "❌ Reject", "callback_data": f"reject:{approval_id}"}],
                ]
            elif approval["type"] in EXECUTABLE_TYPES:
                title = f"🔐 Approval needed #{approval_id}"
                buttons = [[
                    {"text": "✅ Approve", "callback_data": f"approve:{approval_id}"},
                    {"text": "❌ Reject", "callback_data": f"reject:{approval_id}"},
                ]]
            else:
                title = f"📝 For your review #{approval_id}"
                buttons = [[{"text": "✔️ Mark done", "callback_data": f"approve:{approval_id}"}]]

            self.send(f"{title}\n\n{describe(approval)}", buttons)
            self.store.mark_approval_notified(approval_id)

        return len(approvals)

    # ----- inbox -----------------------------------------------------------------

    def run_inbox(self, dry_run: bool = False, announce: bool = False) -> None:
        query = os.getenv("AGENT_INBOX_QUERY", DEFAULT_QUERY)
        outcomes = process_inbox(query=query, dry_run=dry_run)

        if outcomes and outcomes[0].status == BUSY_STATUS:
            # Another process (e.g. the CLI) is processing the inbox; the scheduled check stays quiet.
            if announce:
                self.send(f"⏳ {outcomes[0].reason}")
            return

        lines = []
        quiet = 0
        for outcome in outcomes:
            label = INTERESTING_STATUSES.get(outcome.status)
            if not label:
                quiet += 1
                continue
            line = f"{label}: {outcome.subject or '(no subject)'}"
            if outcome.sender:
                line += f"\n   from {outcome.sender}"
            if outcome.status in {"failed", "needs_attention", "needs_review"} and (outcome.reason or outcome.summary):
                line += f"\n   {(outcome.summary or outcome.reason)[:300]}"
            lines.append(line)

        if lines or announce:
            header = f"{'🧪 Dry run - ' if dry_run else ''}Inbox check: {len(lines)} handled"
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
        if action not in {"approve", "draft", "reject"} or not raw_id.isdigit():
            self.api.call("answerCallbackQuery", callback_query_id=query["id"], text="Unknown button.")
            return

        if action == "reject":
            outcome = reject(int(raw_id), self.store)
        elif action == "draft":
            outcome = to_draft(int(raw_id), self.store)
        else:
            outcome = approve(int(raw_id), self.store)
        self.api.call("answerCallbackQuery", callback_query_id=query["id"], text=outcome["message"][:190])

        original = query.get("message", {}).get("text", "")
        mark = {"reject": "❌", "draft": "✏️"}.get(action, "✅") if outcome["ok"] else "⚠️"
        self.api.call(
            "editMessageText",
            chat_id=chat_id,
            message_id=query["message"]["message_id"],
            text=_clip(f"{original}\n\n{mark} {outcome['message']}"),
        )

        # A rejection or a rewrite is a correction: ask why, so it can be remembered.
        if outcome["ok"] and action in {"reject", "draft"} and memory_enabled():
            prompt = self.send(FEEDBACK_PROMPT, force_reply=True)
            if isinstance(prompt, dict) and prompt.get("message_id"):
                self.store.add_feedback_request(int(prompt["message_id"]), int(raw_id))

    def _handle_message(self, message: dict) -> None:
        chat_id = message.get("chat", {}).get("id")

        if self.owner_chat_id is None:
            # Setup mode: tell the person their chat id, do nothing else.
            logger.warning("Setup: message from chat id %s", chat_id)
            self.api.call(
                "sendMessage",
                chat_id=chat_id,
                text=f"Your chat id is {chat_id}.\nAdd TELEGRAM_OWNER_CHAT_ID={chat_id} to .env and restart the bot.",
            )
            return

        if not self._is_owner(chat_id):
            logger.warning("Ignored message from non-owner chat %s", chat_id)
            return

        text = (message.get("text") or "").strip()
        if not text:
            self.send("Please send a text message.")
            return

        # The owner answered a "why?" prompt: learn from it instead of running an instruction.
        replied_to = (message.get("reply_to_message") or {}).get("message_id")
        if replied_to is not None:
            approval_id = self.store.pop_feedback_request(int(replied_to))
            if approval_id is not None:
                self._learn(text, self.store.get_approval(approval_id))
                return

        command = text.split()[0].split("@")[0].lower()
        argument = text[len(text.split()[0]):].strip()

        if command in {"/start", "/help"}:
            self.send(HELP_TEXT)
        elif command == "/run":
            self.send("Checking the inbox…")
            self.run_inbox(announce=True)
        elif command == "/dryrun":
            self.send("Dry run: checking the inbox without sending or changing anything…")
            self.run_inbox(dry_run=True, announce=True)
        elif command == "/status":
            self.send(self._status_text())
        elif command == "/approvals":
            if not self.notify_approvals(resend_open=True):
                self.send("No open approval requests.")
        elif command == "/new":
            self.history = []
            self.send("Okay, starting fresh.")
        elif command == "/remember":
            if argument:
                self._learn(argument, None)
            else:
                self.send("Tell me what to remember, e.g. /remember no calls before 12:00")
        elif command == "/memories":
            self.send(self._memories_text())
        elif command == "/forget":
            ok = argument.isdigit() and SemanticMemory(self.store).forget(int(argument))
            self.send(f"Forgot memory {argument}." if ok else "Usage: /forget ID (see /memories).")
        elif command == "/skills":
            self.send(self._skills_text())
        elif command == "/forget_skill":
            try:
                ok = bool(argument) and skill_store.archive_skill(argument)
            except skill_store.SkillError:
                ok = False
            self.send(f"Archived skill {argument} (skills/.archive)." if ok else "Usage: /forget_skill NAME (see /skills).")
        else:
            self._run_instruction(text)

    def _run_instruction(self, text: str) -> None:
        self.api.call("sendChatAction", chat_id=self.owner_chat_id, action="typing")
        result = handle_owner_command(text, history=self.history)
        self.send(result.reply)

        self.history += [HumanMessage(content=f"Instruction from the owner:\n{text}"), AIMessage(content=result.reply)]
        self.history = self.history[-2 * HISTORY_TURNS:]

    def _learn(self, text: str, approval: dict | None) -> None:
        if not memory_enabled():
            self.send("Memory is turned off (AGENT_MEMORY=0).")
            return
        saved = learn_from_owner_note(text, approval, store=self.store)
        lines = [f"{'🧠 Noted' if item['created'] else '🧠 Updated'} [{item['id']}]: {item['entity']}: {item['preference']}" for item in saved]
        self.send("\n".join(lines) or "Nothing to remember.")

    def _memories_text(self) -> str:
        memories = SemanticMemory(self.store).all(limit=20)
        if not memories:
            return "I haven't learned anything yet. Reject or edit a request, or use /remember."
        return "What I've learned (newest first):\n" + "\n".join(f"[{m.id}] {m.line()}" for m in memories)

    def _skills_text(self) -> str:
        skills = skill_store.list_skills()
        if not skills:
            return "No saved skills yet. Multi-step instructions that are likely to recur get saved here."
        lines = []
        for skill in skills:
            kind = f"script {skill.script}" if skill.script else f"{len(skill.plan)} steps"
            lines.append(f"- {skill.name} ({kind}, by {skill.author}): {skill.description}")
        return "Saved skills:\n" + "\n".join(lines)

    def _status_text(self) -> str:
        counts = self.store.status_counts()
        lines = ["Email outcomes: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none yet")]
        lines.append(f"Open approval requests: {len(self.store.pending_approvals())} (/approvals to show)")
        interrupted = self.store.interrupted_supervisor_runs()
        if interrupted:
            lines.append(f"Interrupted instructions (not re-run): {', '.join('#' + str(r['id']) for r in interrupted)}")
        if memory_enabled():
            lines.append(f"Memories: {SemanticMemory(self.store).count()} (/memories); skills: {len(skill_store.list_skills())} (/skills)")
        failures = self.store.failures()[:3]
        if failures:
            lines.append("Recent failures:")
            lines += [f"- {f['source']}: {f['message'][:150]}" for f in failures]
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
            except (LLMUnavailableError, GoogleApiUnavailableError) as exc:
                self._report_error(f"Service unavailable, try again later: {exc}")
            except Exception as exc:
                logger.exception("Update failed")
                self._report_error(f"Something went wrong: {type(exc).__name__}: {exc}")

        if self.owner_chat_id is not None:
            if self.poll_seconds and time.monotonic() >= self.next_inbox_run:
                self.next_inbox_run = time.monotonic() + self.poll_seconds
                try:
                    self.run_inbox()
                except (LLMUnavailableError, GoogleApiUnavailableError) as exc:
                    logger.warning("Inbox check skipped: %s", exc)
                except Exception as exc:
                    logger.exception("Inbox check failed")
                    self._report_error(f"Inbox check failed: {type(exc).__name__}: {exc}")

            self.notify_approvals()

    def _report_error(self, text: str) -> None:
        if self.owner_chat_id is None:
            return
        try:
            self.send(f"⚠️ {text}")
        except Exception:
            logger.exception("Could not report error to Telegram")

    def run_forever(self) -> None:
        if self.owner_chat_id is not None:
            self.send("🤖 Socio is online. Send /help for commands.")

        # Load the local embedding model in the background so the first recall is fast.
        if memory_enabled():
            threading.Thread(target=lambda: getattr(get_embedder(), "warm_up", lambda: None)(), daemon=True).start()

        while True:
            try:
                self.poll_once()
            except (requests.RequestException, TelegramError) as exc:
                logger.warning("Telegram connection problem: %s. Retrying in 10s.", exc)
                time.sleep(10)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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
