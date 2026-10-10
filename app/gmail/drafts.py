import base64
import mimetypes
from email.message import EmailMessage
from pathlib import Path

from app.gmail.client import execute_gmail_request, get_gmail_service


def create_draft(
    to: str,
    subject: str,
    body: str,
    attachment_path: str | None = None,
) -> dict:
    """
    Create a Gmail draft.

    If attachment_path is provided, the local file is attached.

    This function does NOT send the email.
    """

    service = get_gmail_service()

    message = EmailMessage()

    message["To"] = to
    message["Subject"] = subject

    message.set_content(body)

    if attachment_path:
        file_path = Path(attachment_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Attachment not found: {file_path}"
            )

        mime_type, _ = mimetypes.guess_type(file_path.name)

        if mime_type:
            main_type, sub_type = mime_type.split("/", 1)
        else:
            main_type = "application"
            sub_type = "octet-stream"

        file_data = file_path.read_bytes()

        message.add_attachment(
            file_data,
            maintype=main_type,
            subtype=sub_type,
            filename=file_path.name,
        )

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    draft_body = {
        "message": {
            "raw": encoded_message
        }
    }

    # Not retried: a timed-out create may still have created the draft.
    draft = execute_gmail_request(
        service.users()
        .drafts()
        .create(
            userId="me",
            body=draft_body,
        ),
        operation_name="create_draft",
        retry=False,
    )

    return draft

def create_reply_draft(
    message_id: str,
    body: str,
    attachment_path: str | None = None,
) -> dict:
    """
    Create a draft reply to an existing Gmail message.

    The reply is associated with the original Gmail thread.
    This function does NOT send the email.
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
                "From",
                "To",
                "Cc",
                "Subject",
                "Message-ID",
                "References",
            ],
        ),
        operation_name="reply_draft_metadata",
    )

    headers = original_message.get("payload", {}).get("headers", [])

    header_map = {
        header["name"].lower(): header["value"]
        for header in headers
    }

    original_sender = header_map.get("from", "")
    original_subject = header_map.get("subject", "")
    original_message_id = header_map.get("message-id", "")
    original_references = header_map.get("references", "")

    if original_subject.lower().startswith("re:"):
        reply_subject = original_subject
    else:
        reply_subject = f"Re: {original_subject}"

    message = EmailMessage()

    message["To"] = original_sender
    message["Subject"] = reply_subject

    if original_message_id:
        message["In-Reply-To"] = original_message_id

        if original_references:
            message["References"] = (
                f"{original_references} {original_message_id}"
            )
        else:
            message["References"] = original_message_id

    message.set_content(body)

    if attachment_path:
        file_path = Path(attachment_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Attachment not found: {file_path}"
            )

        mime_type, _ = mimetypes.guess_type(file_path.name)

        if mime_type:
            main_type, sub_type = mime_type.split("/", 1)
        else:
            main_type = "application"
            sub_type = "octet-stream"

        file_data = file_path.read_bytes()

        message.add_attachment(
            file_data,
            maintype=main_type,
            subtype=sub_type,
            filename=file_path.name,
        )

    encoded_message = base64.urlsafe_b64encode(
        message.as_bytes()
    ).decode()

    draft_body = {
        "message": {
            "threadId": original_message["threadId"],
            "raw": encoded_message,
        }
    }

    # Not retried: a timed-out create may still have created the draft.
    draft = execute_gmail_request(
        service.users()
        .drafts()
        .create(
            userId="me",
            body=draft_body,
        ),
        operation_name="create_draft",
        retry=False,
    )

    return draft