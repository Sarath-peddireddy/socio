"""
Telegram front end for Socio.

    python -m app.telegram_bot

- Only the owner's chat (TELEGRAM_OWNER_CHAT_ID) can use the bot. Until it is
  set, the bot only replies with the sender's chat id so it can be configured.
- Plain messages are instructions for the agent (owner mode); multi-part ones
  are split across worker subagents (app.agent.supervisor).
- Approval requests arrive with Approve / Reject buttons (emails also get
  Edit in Gmail, which saves them as a Gmail draft instead).
- Reject ends quietly: the card says the proposal was rejected and offers an
  optional "+ Add Permanent Rule" button (also after Edit in Gmail). Only if the
  owner taps it and types a note is anything learned: a standing policy goes to
  long-term memory, a situational note is kept for 24 hours. /remember too.
- /save_skill NAME saves the last successful multi-step instruction as a skill.
- The inbox is processed every AGENT_POLL_MINUTES (default 10, 0 = off).
- Every handler runs behind the error gateway (app.errors): details go to
  logs/agent.log, the chat only ever gets a short predefined message.

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
from app.agent.supervisor import save_last_plan
from app.errors import log_path, report_failure, scrub, setup_logging
from app.google_api import GoogleApiUnavailableError
from app.memory import skills as skill_store
from app.memory.embeddings import get_embedder
from app.memory.episodic import active_notes
from app.memory.learning import STANDING_POLICY, learn_from_owner_note, promote_owner_rule
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
/save_skill NAME - save your last multi-step instruction as a skill
/skills - saved skills; /forget_skill NAME archives one

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
reject something, tap "+ Add Permanent Rule" if you want me to remember why."""

REJECTED_TEXT = "❌ Proposal rejected. No changes were made."
ADD_RULE_BUTTON = "+ Add Permanent Rule"
RULE_PROMPT = "✍️ Reply to this message with the rule, e.g. \"never book calls before noon\"."

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
        # The approval whose "+ Add Permanent Rule" was tapped last; the next plain message is the rule.
        self.awaiting_rule: int | None = None

    # ----- sending -------------------------------------------------------------

    def send(self, text: str, buttons: list | None = None, force_reply: bool = False):
        params = {"chat_id": self.owner_chat_id, "text": _clip(scrub(text)), "disable_web_page_preview": True}
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
        if action not in {"approve", "draft", "reject", "rule"} or not raw_id.isdigit():
            self._answer(query, "Unknown button.")
            return

        try:
            if action == "rule":
                self._ask_for_rule(int(raw_id), query)
            else:
                self._decide(action, int(raw_id), query)
        except Exception as exc:
            text = report_failure(exc, "telegram.callback", action=action, approval_id=raw_id)
            self._answer(query, text)
            self._report_error(text)

    def _answer(self, query: dict, text: str) -> None:
        """Close the button's loading spinner (best effort: an old query cannot be answered)."""

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

        if action == "reject" and outcome["ok"]:
            status_line = REJECTED_TEXT
        else:
            mark = {"draft": "✏️"}.get(action, "✅") if outcome["ok"] else "⚠️"
            status_line = f"{mark} {outcome['message']}"

        original = query.get("message", {}).get("text", "")
        params = {
            "chat_id": query["message"]["chat"]["id"],
            "message_id": query["message"]["message_id"],
            "text": _clip(scrub(f"{original}\n\n{status_line}")),
        }
        # Nothing else is sent: learning from a rejection or a rewrite is opt-in.
        if outcome["ok"] and action in {"reject", "draft"} and memory_enabled():
            params["reply_markup"] = {"inline_keyboard": [[{"text": ADD_RULE_BUTTON, "callback_data": f"rule:{approval_id}"}]]}
        self.api.call("editMessageText", **params)

    def _ask_for_rule(self, approval_id: int, query: dict) -> None:
        approval = self.store.get_approval(approval_id)
        if approval is None or approval["status"] not in {"rejected", "resolved"} or not memory_enabled():
            self._answer(query, "There is nothing to add a rule for here.")
            return

        self._answer(query, "Type the rule as a reply.")
        prompt = self.send(RULE_PROMPT, force_reply=True)
        if isinstance(prompt, dict) and prompt.get("message_id"):
            self.store.add_feedback_request(int(prompt["message_id"]), approval_id)
        self.awaiting_rule = approval_id

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

        # The owner answered the "+ Add Permanent Rule" prompt: classify and keep it
        # instead of running it as an instruction.
        replied_to = (message.get("reply_to_message") or {}).get("message_id")
        if replied_to is not None:
            approval_id = self.store.pop_feedback_request(int(replied_to))
            if approval_id is not None:
                self.awaiting_rule = None
                self._promote(text, self.store.get_approval(approval_id))
                return
        if self.awaiting_rule is not None:
            approval_id, self.awaiting_rule = self.awaiting_rule, None
            if not text.startswith("/"):  # a command cancels the pending rule
                self._promote(text, self.store.get_approval(approval_id))
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
        elif command == "/save_skill":
            self._save_skill(argument)
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

    def _promote(self, text: str, approval: dict | None) -> None:
        """The owner's note after "+ Add Permanent Rule": a standing policy or a note for today."""

        if not memory_enabled():
            self.send("Memory is turned off (AGENT_MEMORY=0).")
            return
        result = promote_owner_rule(text, approval, store=self.store)
        if result["kind"] == STANDING_POLICY:
            self.send(f"✓ Rule saved: {result['summary']}")
        else:
            self.send("Understood for today.")

    def _save_skill(self, name: str) -> None:
        if not name:
            self.send("Usage: /save_skill NAME - saves your last successful multi-step instruction as a skill.")
            return
        try:
            saved = save_last_plan(name, store=self.store)
        except skill_store.SkillError as exc:
            self.send(f"⚠️ {exc}")
            return
        text = f"🧩 Saved skill '{saved.name}' ({saved.tasks} steps; parameters: {', '.join(saved.parameters) or 'none'})."
        if not saved.generalized:
            text += " It was saved exactly as it ran, so it repeats the same names and dates."
        self.send(text + " A matching request will reuse it (/skills to see it).")

    def _memories_text(self) -> str:
        memories = SemanticMemory(self.store).all(limit=20)
        notes = active_notes(self.store)
        if not memories and not notes:
            return "I haven't learned anything yet. Use /remember, or tap \"+ Add Permanent Rule\" after rejecting something."
        lines = []
        if memories:
            lines.append("What I've learned (newest first):")
            lines += [f"[{m.id}] {m.line()}" for m in memories]
        if notes:
            lines.append("For today only:")
            lines += [f"- [{n['entity']}] {n['note']}" for n in notes]
        return "\n".join(lines)

    def _skills_text(self) -> str:
        skills = skill_store.list_skills()
        if not skills:
            return "No saved skills yet. After a multi-step instruction succeeds, save it with /save_skill NAME."
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
        # Raw failure messages hold internal details: only a count goes to the chat.
        failures = self.store.failures()
        if failures:
            lines.append(f"Logged failures: {len(failures)} (details are in {log_path().parent.name}/{log_path().name} on this computer)")
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
                self._report_error(report_failure(exc, "telegram.update", update_id=update.get("update_id"), kind=kind))

        if self.owner_chat_id is not None:
            if self.poll_seconds and time.monotonic() >= self.next_inbox_run:
                self.next_inbox_run = time.monotonic() + self.poll_seconds
                try:
                    self.run_inbox()
                except (LLMUnavailableError, GoogleApiUnavailableError) as exc:
                    # An outage during a scheduled check stays quiet; the next check retries.
                    report_failure(exc, "telegram.scheduled_inbox")
                except Exception as exc:
                    self._report_error(report_failure(exc, "telegram.scheduled_inbox"))

            try:
                self.notify_approvals()
            except Exception as exc:
                self._report_error(report_failure(exc, "telegram.notify_approvals"))

    def _report_error(self, text: str) -> None:
        """Send a predefined gateway message (never raw exception text)."""

        if self.owner_chat_id is None:
            return
        try:
            self.send(text)
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
