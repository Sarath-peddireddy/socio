"""
Filling a reply's [OWNER: ...] gaps from the owner's answer in Telegram.

When Socio can't finish a reply to an incoming email (something only the owner knows),
the card shows the draft with its gaps and the question. The owner answers in chat
(replying to the card, or as their next message); one model call puts the answer into the
draft, keeping everything else word for word. It also says whether the message was an
answer at all, so an unrelated instruction sent right after a card still runs normally.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.approvals import GAP_MARKER
from app.agent.llm import get_llm, invoke_llm


logger = logging.getLogger(__name__)


class FilledReply(BaseModel):
    is_answer: bool = Field(description="True if the owner's message answers (part of) the question for this email.")
    body: str = Field(default="", description="The whole reply with the answered [OWNER: ...] gaps filled in; empty if not an answer.")
    still_missing: str = Field(default="", description="One short question to the owner if a gap is still unanswered, else empty.")


FILL_PROMPT = """You complete an email reply that the owner will approve before it is sent.
The reply has gaps marked [OWNER: ...]: things only the owner knows. The owner has just sent a
message in their chat with you.

Decide whether that message answers the question for this reply. It is NOT an answer if it is
about something else (a new instruction, a question to you, small talk).

If it is an answer: rewrite the reply with each gap the answer covers filled in, in natural
words that fit the sentence. Keep every other word, line break, greeting, and signature
exactly as written. Do not add facts the owner did not give. A gap the answer does not cover
stays exactly as it is, and still_missing is then one short question about it, spoken to the
owner as "you". Never put the [OWNER: ...] wording itself into the email."""


def fill_gaps(draft: str, question: str, owner_text: str) -> FilledReply:
    """One model call (the supervisor chain). Raises like any model call on an outage."""

    model = get_llm().with_structured_output(FilledReply)
    request = (
        f"Question shown to the owner: {question or '(see the gaps)'}\n\n"
        f"Reply with gaps:\n<<<\n{draft}\n>>>\n\n"
        f"Owner's message:\n<<<\n{owner_text}\n>>>"
    )
    result = invoke_llm(model, [SystemMessage(content=FILL_PROMPT), HumanMessage(content=request)])
    if isinstance(result, dict):
        result = FilledReply(**result)
    if not result.is_answer:
        return FilledReply(is_answer=False)

    body = (result.body or "").strip()
    if not body:  # said "answer" but returned nothing usable: treat as not an answer
        logger.warning("Gap fill returned an empty reply; treating the message as an instruction.")
        return FilledReply(is_answer=False)
    if GAP_MARKER.search(body) and not result.still_missing:
        result.still_missing = "Something is still missing in this reply. What should it say?"
    return FilledReply(is_answer=True, body=body, still_missing=result.still_missing.strip())
