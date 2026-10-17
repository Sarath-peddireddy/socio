"""
Conversation history for owner instructions: stored per Telegram chat, loaded in
whole turns, capped by a token budget.

A turn is one owner message plus everything that answered it (Socio's reply, and
any assistant tool calls with their tool results). History is only ever cut
between turns, never inside one, so an assistant message with tool_calls is never
sent without its tool responses (providers reject that with HTTP 400).

Stored turns hold plain text only (the owner's instruction and Socio's final reply
with the cards it showed); raw tool output never goes into history. Turns older
than HISTORY_HOURS, or archived by /new, are not loaded.

Tokens are estimated as characters / 4 (no tokenizer dependency); the estimate is
used only for the history budget, so it errs on the safe side for English text.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.state import AgentStateStore


logger = logging.getLogger(__name__)

MAX_TURNS = 3              # owner <-> Socio exchanges (6 messages)
MAX_HISTORY_TOKENS = 2000
HISTORY_HOURS = 24
CHARS_PER_TOKEN = 4
CLIPPED = " … (shortened)"


def estimate_tokens(message: BaseMessage) -> int:
    """A rough token count of one message: its text plus any tool-call arguments."""

    text = message.content if isinstance(message.content, str) else json.dumps(message.content, default=str)
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        text += json.dumps(calls, default=str)
    return len(text) // CHARS_PER_TOKEN + 4  # + per-message overhead


def group_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """Split messages into turns: each HumanMessage starts one; everything after it belongs to it."""

    turns: list[list[BaseMessage]] = []
    for message in messages:
        if isinstance(message, HumanMessage) or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    # A leading turn without its owner message (history that began mid-turn) is dropped.
    return [turn for turn in turns if isinstance(turn[0], HumanMessage)]


def turn_tokens(turn: list[BaseMessage]) -> int:
    return sum(estimate_tokens(message) for message in turn)


def _clip_turn(turn: list[BaseMessage], budget: int) -> list[BaseMessage]:
    """The newest turn alone is over budget: shorten its last plain reply (the pairing stays intact)."""

    last = turn[-1]
    if not isinstance(last, AIMessage) or last.tool_calls or not isinstance(last.content, str):
        return turn
    excess_chars = (turn_tokens(turn) - budget) * CHARS_PER_TOKEN + len(CLIPPED)
    keep = max(0, len(last.content) - excess_chars)
    return turn[:-1] + [AIMessage(content=last.content[:keep] + CLIPPED)]


def trim_history(messages: list[BaseMessage], max_turns: int = MAX_TURNS, max_tokens: int = MAX_HISTORY_TOKENS) -> list[BaseMessage]:
    """
    The last max_turns whole turns, dropping the oldest whole turn while the total is
    over max_tokens. If only the newest turn is left and it is still too long, its
    final reply text is shortened (never its tool calls or tool results).
    """

    turns = group_turns(list(messages))[-max_turns:] if max_turns > 0 else []
    while len(turns) > 1 and sum(turn_tokens(turn) for turn in turns) > max_tokens:
        turns.pop(0)
    if turns and turn_tokens(turns[0]) > max_tokens:
        turns[0] = _clip_turn(turns[0], max_tokens)
        if turn_tokens(turns[0]) > max_tokens:
            turns = []
    return [message for turn in turns for message in turn]


def owner_turn(text: str) -> HumanMessage:
    return HumanMessage(content=f"Instruction from the owner:\n{text}")


def load_history(chat_id: int | None, store: AgentStateStore | None = None) -> list[BaseMessage]:
    """The chat's recent turns as messages, trimmed to the budget. [] on any problem (never raises)."""

    if chat_id is None:
        return []
    try:
        rows = (store or AgentStateStore()).recent_conversation_turns(int(chat_id), MAX_TURNS, HISTORY_HOURS)
    except Exception as exc:
        logger.warning("Could not load the conversation history: %s", exc)
        return []
    messages: list[BaseMessage] = []
    for row in rows:
        messages += [owner_turn(row["owner_text"]), AIMessage(content=row["socio_text"] or "Done.")]
    return trim_history(messages)


def save_turn(chat_id: int | None, owner_text: str, socio_text: str, store: AgentStateStore | None = None) -> None:
    """Store one finished turn (plain text only). Never raises."""

    if chat_id is None:
        return
    try:
        (store or AgentStateStore()).add_conversation_turn(int(chat_id), owner_text, socio_text.strip() or "Done.")
    except Exception as exc:
        logger.warning("Could not save the conversation turn: %s", exc)


def note_on_last_turn(chat_id: int | None, note: str, store: AgentStateStore | None = None) -> None:
    """Add text (e.g. the owner's card decisions) to the newest stored reply. Never raises."""

    if chat_id is None or not note:
        return
    try:
        (store or AgentStateStore()).append_to_last_conversation_turn(int(chat_id), note)
    except Exception as exc:
        logger.warning("Could not update the conversation history: %s", exc)


def archive_history(chat_id: int | None, store: AgentStateStore | None = None) -> None:
    if chat_id is not None:
        (store or AgentStateStore()).archive_conversation_turns(int(chat_id))
