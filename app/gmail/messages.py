import base64
import mimetypes
from email.message import EmailMessage
from pathlib import Path

from app.gmail.client import execute_gmail_request, get_gmail_service


def _attach_file(email: EmailMessage, file_path: Path) -> None:
    """Attach a local file to an outgoing email."""

    mime_type, _ = mimetypes.guess_type(file_path.name)
    main_type, sub_type = (mime_type or "application/octet-stream").split("/", 1)

    email.add_attachment(
        file_path.read_bytes(),
        maintype=main_type,
        subtype=sub_type,
        filename=file_path.name,
    )


def list_recent_messages(max_results: int = 10):
    """
    Return recent Gmail message IDs.

    This function only retrieves message IDs and thread IDs.
    It does not retrieve the full email contents.
    """

    service = get_gmail_service()

    response = execute_gmail_request(
        service.users()
        .messages()
        .list(
            userId="me",
            maxResults=max_results,
        ),
        operation_name="list_recent_messages",
    )

    return response.get("messages", [])


def get_message(message_id: str):
    """
    Retrieve the full Gmail message for a given message ID.
    """

    service = get_gmail_service()

    message = execute_gmail_request(
        service.users()
        .messages()
        .get(
            userId="me",
            id=message_id,
            format="full",
        ),
        operation_name="get_message",
    )

    return message


def get_thread(thread_id: str):
    """
    Retrieve the complete Gmail thread for a given thread ID.

    Returns the raw Gmail thread containing all messages
    belonging to that conversation.
    """

    service = get_gmail_service()

    thread = execute_gmail_request(
        service.users()
        .threads()
        .get(
            userId="me",
            id=thread_id,
            format="full",
        ),
        operation_name="get_thread",
    )

    return thread


def search_messages(query: str, max_results: int = 10):
    """
    Search Gmail using Gmail's native search syntax.

    Examples:
        from:someone@example.com
        subject:project
        has:attachment
        is:unread
        after:2026/10/01
    """

    service = get_gmail_service()

    response = execute_gmail_request(
        service.users()
        .messages()
        .list(
            userId="me",
            q=query,
            maxResults=max_results,
        ),
        operation_name="search_messages",
    )

    return response.get("messages", [])


def download_attachment(
    message_id: str,
    attachment_id: str,
):
    """
    Download an attachment from a Gmail message.

    Returns the decoded attachment bytes.
    """

    service = get_gmail_service()

    attachment = execute_gmail_request(
        service.users()
        .messages()
        .attachments()
        .get(
            userId="me",
            messageId=message_id,
            id=attachment_id,
        ),
        operation_name="download_attachment",
    )

    data = attachment.get("data", "")

    if not data:
        return b""

    return base64.urlsafe_b64decode(
        data + "=" * (-len(data) % 4)
    )


def send_reply(
    thread_id: str,
    message_id: str,
    to: str,
    subject: str,
    body: str,
    attachment_paths: list[str] | None = None,
):
    """
    Send a reply to an existing Gmail message.

    The reply is sent inside the existing Gmail thread
    using the original message's Message-ID.

    attachment_paths must already be validated by the caller
    (see app.files.local.resolve_shareable_file).
    """

    service = get_gmail_service()

    original_message = execute_gmail_request(
        service.users()
        .messages()
        .get(
            userId="me",
            id=message_id,
            format="metadata",
            metadataHeaders=[
                "Message-ID",
                "References",
                "Subject",
            ],
        ),
        operation_name="send_reply_metadata",
    )

    headers = {
        header["name"].lower(): header["value"]
        for header in original_message
        .get("payload", {})
        .get("headers", [])
    }

    original_message_id = headers.get("message-id")

    if not original_message_id:
        raise ValueError(
            "Cannot send reply because the original "
            "email does not contain a Message-ID header."
        )

    references = headers.get("references", "")

    if references:
        references = f"{references} {original_message_id}"
    else:
        references = original_message_id

    # Gmail only keeps a reply in the thread when the subject matches,
    # so derive it from the original rather than trusting the caller.
    reply_subject = headers.get("subject", "").strip() or subject.strip()

    if not reply_subject.lower().startswith("re:"):
        reply_subject = f"Re: {reply_subject}"

    email = EmailMessage()

    email["To"] = to
    email["Subject"] = reply_subject
    email["In-Reply-To"] = original_message_id
    email["References"] = references

    email.set_content(body)

    for attachment_path in attachment_paths or []:
        _attach_file(email, Path(attachment_path))

    encoded_message = base64.urlsafe_b64encode(
        email.as_bytes()
    ).decode()

    message = {
        "raw": encoded_message,
        "threadId": thread_id,
    }

    sent_message = execute_gmail_request(
        service.users()
        .messages()
        .send(
            userId="me",
            body=message,
        ),
        operation_name="send_reply",
        retry=False,
    )

    return {
        "message_id": sent_message.get("id"),
        "thread_id": sent_message.get("threadId"),
    }

def send_new_email(
    to: str,
    subject: str,
    body: str,
    attachment_paths: list[str] | None = None,
):
    """
    Send a new email (not a reply).

    attachment_paths must already be validated by the caller.
    """

    service = get_gmail_service()

    email = EmailMessage()
    email["To"] = to
    email["Subject"] = subject
    email.set_content(body)

    for attachment_path in attachment_paths or []:
        _attach_file(email, Path(attachment_path))

    encoded_message = base64.urlsafe_b64encode(email.as_bytes()).decode()

    # Not retried: a timed-out send may still have been delivered.
    sent_message = execute_gmail_request(
        service.users().messages().send(userId="me", body={"raw": encoded_message}),
        operation_name="send_new_email",
        retry=False,
    )

    return {
        "message_id": sent_message.get("id"),
        "thread_id": sent_message.get("threadId"),
    }


def trash_message(message_id: str, whole_thread: bool = False) -> dict:
    """
    Move a message (or its whole thread) to Gmail Trash.

    This is never a permanent delete: Gmail keeps Trash for 30 days, and the
    gmail.modify scope used by this project cannot delete permanently.
    """

    service = get_gmail_service()

    if whole_thread:
        message = execute_gmail_request(
            service.users().messages().get(userId="me", id=message_id, format="minimal"),
            operation_name="trash_lookup_thread",
        )
        result = execute_gmail_request(
            service.users().threads().trash(userId="me", id=message["threadId"]),
            operation_name="trash_thread",
        )
        return {"thread_id": result.get("id"), "trashed": "thread"}

    result = execute_gmail_request(
        service.users().messages().trash(userId="me", id=message_id),
        operation_name="trash_message",
    )

    return {
        "message_id": result.get("id"),
        "thread_id": result.get("threadId"),
        "in_trash": "TRASH" in result.get("labelIds", []),
    }
