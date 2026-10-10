import base64


def decode_body(data: str) -> str:
    """
    Decode Gmail's URL-safe base64 encoded message body.
    """

    if not data:
        return ""

    decoded_bytes = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))

    return decoded_bytes.decode("utf-8", errors="replace")


def extract_headers(message: dict) -> dict:
    """
    Extract commonly used email headers from a Gmail API message.
    """

    headers = message.get("payload", {}).get("headers", [])

    result = {}

    for header in headers:
        name = header.get("name", "").lower()
        value = header.get("value", "")

        if name in {
    "from",
    "reply-to",
    "to",
    "cc",
    "bcc",
    "subject",
    "date",
    "message-id",
    "references",
    "in-reply-to",
}:
            result[name] = value

    return result


# Headers that mark bulk, mailing-list, or machine-generated email.
BULK_HEADER_NAMES = {
    "list-unsubscribe",
    "list-id",
    "precedence",
    "auto-submitted",
    "x-autoreply",
    "x-autorespond",
}


def extract_bulk_headers(message: dict) -> dict:
    """
    Extract headers that identify newsletters, mailing lists, and automated mail.
    """

    headers = message.get("payload", {}).get("headers", [])

    return {
        header.get("name", "").lower(): header.get("value", "")
        for header in headers
        if header.get("name", "").lower() in BULK_HEADER_NAMES
    }


def extract_body(payload: dict) -> str:
    """
    Extract the best available text body from a Gmail message.

    Handles both simple messages and multipart messages.
    """

    mime_type = payload.get("mimeType", "")

    body_data = payload.get("body", {}).get("data")

    if body_data and mime_type == "text/plain":
        return decode_body(body_data)

    for part in payload.get("parts", []):
        part_body = extract_body(part)

        if part_body:
            return part_body

    if body_data:
        return decode_body(body_data)

    return ""


def parse_message(message: dict) -> dict:
    """
    Convert a raw Gmail API message into a clean dictionary.
    """

    payload = message.get("payload", {})

    headers = extract_headers(message)

    return {
    "id": message.get("id"),
    "thread_id": message.get("threadId"),
    "from": headers.get("from", ""),
    "reply_to": headers.get("reply-to", ""),
    "to": headers.get("to", ""),
    "cc": headers.get("cc", ""),
    "bcc": headers.get("bcc", ""),
    "subject": headers.get("subject", ""),
    "date": headers.get("date", ""),
    "message_id": headers.get("message-id", ""),
    "references": headers.get("references", ""),
    "in_reply_to": headers.get("in-reply-to", ""),
    "body": extract_body(payload),
    "attachments": extract_attachments(payload),
    "size": message.get("sizeEstimate", 0),
    "label_ids": message.get("labelIds", []),
    "bulk_headers": extract_bulk_headers(message),
}

def extract_attachments(payload: dict) -> list[dict]:
    """
    Recursively find attachments in a Gmail message payload.
    """

    attachments = []

    filename = payload.get("filename")
    body = payload.get("body", {})

    if filename:
        attachments.append(
            {
                "filename": filename,
                "mime_type": payload.get("mimeType", ""),
                "attachment_id": body.get("attachmentId"),
                "size": body.get("size", 0),
            }
        )

    for part in payload.get("parts", []):
        attachments.extend(extract_attachments(part))

    return attachments