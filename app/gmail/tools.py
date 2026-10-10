from pathlib import Path, PureWindowsPath

from app.gmail.messages import (
    download_attachment,
    get_message,
    get_thread,
    list_recent_messages,
    search_messages,
)

from app.gmail.parser import parse_message

def get_recent_emails(max_results: int = 10) -> list[dict]:
    """
    Retrieve and parse recent Gmail messages.

    This function combines the lower-level Gmail API operations
    into one clean capability that can later become an agent tool.
    """

    messages = list_recent_messages(max_results=max_results)

    emails = []

    for message in messages:
        raw_message = get_message(message["id"])
        parsed_message = parse_message(raw_message)

        emails.append(parsed_message)

    return emails

def search_emails(query: str, max_results: int = 10) -> list[dict]:
    """
    Search Gmail and return parsed email objects.
    """

    messages = search_messages(
        query=query,
        max_results=max_results,
    )

    emails = []

    for message in messages:
        raw_message = get_message(message["id"])
        parsed_message = parse_message(raw_message)

        emails.append(parsed_message)

    return emails

def get_thread_details(thread_id: str, include_drafts: bool = False) -> list[dict]:
    """
    Retrieve and parse all messages in a Gmail thread.

    Returns a clean list of parsed email objects in the conversation.
    Unsent drafts (Gmail label DRAFT) are excluded unless include_drafts
    is True; otherwise a draft would look like the latest message.
    """

    thread = get_thread(thread_id)

    messages = thread.get("messages", [])

    emails = []

    for message in messages:
        if not include_drafts and "DRAFT" in message.get("labelIds", []):
            continue

        parsed_message = parse_message(message)
        emails.append(parsed_message)

    return emails


def _safe_attachment_filename(filename: str) -> str:
    """
    Reduce an untrusted attachment filename to a bare file name.

    Strips any directory components (both / and \\ separators, drive
    letters) so a crafted name cannot write outside the output folder.
    """

    name = PureWindowsPath(filename or "").name.strip()

    if name in {"", ".", ".."}:
        raise ValueError(f"Unsafe or empty attachment filename: {filename!r}")

    return name


def save_gmail_attachment(
    message_id: str,
    attachment: dict,
    output_directory: str = "downloads",
) -> str:
    """
    Download a Gmail attachment and save it locally.

    Returns the path of the saved file.
    """

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = _safe_attachment_filename(attachment["filename"])

    file_path = output_path / filename

    if file_path.resolve().parent != output_path.resolve():
        raise ValueError(f"Attachment path escapes output directory: {file_path}")

    data = download_attachment(
        message_id=message_id,
        attachment_id=attachment["attachment_id"],
    )

    file_path.write_bytes(data)

    return str(file_path)