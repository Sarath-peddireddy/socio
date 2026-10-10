from pydantic import BaseModel, Field

from app.agent.llm import get_llm


class ReplyDraft(BaseModel):
    subject: str = Field(
        description="The subject line for the reply."
    )

    body: str = Field(
        description="The complete email reply body."
    )


def draft_reply(thread: list[dict]) -> ReplyDraft:
    """
    Generate a reply draft based on the full Gmail conversation.

    This function only creates a draft.
    It does not send any email.
    """

    conversation = []

    for index, email in enumerate(thread, start=1):
        conversation.append(
            f"""
Message {index}
From: {email.get("from", "")}
To: {email.get("to", "")}
Subject: {email.get("subject", "")}
Date: {email.get("date", "")}

Body:
{email.get("body", "")}
"""
        )

    conversation_text = "\n".join(conversation)

    llm = get_llm()

    structured_llm = llm.with_structured_output(
        ReplyDraft
    )

    prompt = f"""
You are an email drafting assistant.

Your job is to draft a natural reply to the following
Gmail conversation.

Rules:

1. Read the entire conversation before drafting.
2. Reply specifically to what the sender is asking or discussing.
3. Do not invent facts, availability, commitments, or information
   that the user has not provided.
4. Keep the reply concise and natural.
5. Do not sound like a generic AI-generated email.
6. Do not use unnecessary corporate language.
7. Do not add information that is unrelated to the conversation.
8. Preserve the appropriate tone of the conversation.
9. Do not send the email. Only produce a draft.
10. If the sender asks for information that the user has not provided,
    clearly indicate that the draft needs user input rather than
    inventing an answer.

Return only the structured draft.

Gmail conversation:

{conversation_text}
"""

    response = structured_llm.invoke(prompt)

    if not isinstance(response, ReplyDraft):
        raise TypeError(
            "Structured LLM returned an unexpected response type."
        )

    return response