"""
Sharing files Socio created, the compound "create, share, and email the link" action,
and live links to Google files in emails.

Only files in Socio's own output folder can be shared: the Workspace token has
drive.file (files Socio created), not full Drive write access. A Google Doc / Sheet /
Slides file in an email always goes as its live link, never as an exported copy; a
file Socio created is shared with the recipients (editor unless the owner said
otherwise) so the link opens. Everything here runs only from app.agent.approvals,
after the owner approves.
"""

from __future__ import annotations

import logging
import re
from email.utils import getaddresses
from typing import Any

from googleapiclient.errors import HttpError

from app.drive.client import get_drive_service
from app.files.drive import GOOGLE_TYPES, _file_id_from_reference, output_folder_id
from app.google_api import completed_write, execute_with_retry, outcome_unknown


logger = logging.getLogger(__name__)

# What the owner says -> Drive permission role.
ROLES = {"editor": "writer", "commenter": "commenter", "viewer": "reader"}
ROLE_WORDS = {
    "editor": "editor", "edit": "editor", "writer": "editor", "write": "editor",
    "commenter": "commenter", "comment": "commenter",
    "viewer": "viewer", "view": "viewer", "reader": "viewer", "read": "viewer", "read-only": "viewer",
}
# What the recipient can do, for the owner's messages ("sam@x.com can now edit it").
ROLE_VERBS = {"editor": "edit", "commenter": "comment on", "viewer": "view"}
FILE_KINDS = {"spreadsheet", "document"}
# A Google Docs / Sheets / Slides link written in an email body.
GOOGLE_FILE_URL = re.compile(r"https://docs\.google\.com/(?:spreadsheets|document|presentation)/d/[A-Za-z0-9_-]+[^\s<>()\[\]\"']*")

# Where the model marked the link in the email body: [link], [LINK], [Link to the sheet],
# {link}, {{url}}, <link>, [Google Sheet URL] ... (short, one line; not a Markdown "[text](url)").
_WORD = r"\b(?:link|url|hyperlink)\b"
# "[OWNER: ...]" is a gap only the owner fills, never a link spot, even if it says "link".
LINK_PLACEHOLDER = re.compile(
    rf"\[(?!OWNER:)[^\[\]\n]{{0,40}}?{_WORD}[^\[\]\n]{{0,40}}?\](?!\()"
    rf"|\{{\{{?[^{{}}\n]{{0,40}}?{_WORD}[^{{}}\n]{{0,40}}?\}}\}}?"
    rf"|<[^<>\n]{{0,40}}?{_WORD}[^<>\n]{{0,40}}?>",
    re.IGNORECASE,
)


def linked_file(reference: str) -> dict[str, Any] | None:
    """
    A Google Doc / Sheet / Slides file as a link for an email: id, name, url, type, and
    whether Socio may change its access (only files in the Socio folder). None for any
    other kind of file. Raises ValueError if the file can't be found.
    """

    file_id = _file_id_from_reference(reference or "")
    try:
        file = execute_with_retry(
            get_drive_service().files().get(fileId=file_id, fields="id, name, mimeType, parents, webViewLink, trashed"),
            operation_name="drive_get_link_target",
        )
    except HttpError as exc:
        raise ValueError("I couldn't find that file in your Google Drive.") from exc
    google_type = GOOGLE_TYPES.get(file.get("mimeType", ""))
    if not google_type:
        return None
    if file.get("trashed"):
        raise ValueError(f"'{file.get('name', 'That file')}' is in your Google Drive trash.")
    return {
        "file_id": file["id"],
        "name": file.get("name", ""),
        "url": file.get("webViewLink") or reference,
        "type": google_type,
        "can_share": output_folder_id() in (file.get("parents") or []),
    }


def links_in_text(text: str) -> list[str]:
    """Google Docs / Sheets / Slides URLs written in an email body, each once."""

    return list(dict.fromkeys(match.rstrip(".,;:!?") for match in GOOGLE_FILE_URL.findall(text or "")))


def with_links(body: str, links: list[dict[str, Any]]) -> str:
    """The email body with each file's link: in place of the next [link] placeholder, else below the text."""

    body = (body or "").rstrip()
    below = []
    for link in links:
        url = link["url"]
        if url in body:
            continue
        body, count = LINK_PLACEHOLDER.subn(lambda _: url, body, count=1)
        if not count:
            below.append(f"{link['name']}: {url}" if len(links) > 1 else url)
    return body + "\n\n" + "\n".join(below) if below else body


def share_links(links: list[dict[str, Any]], recipients: list[str], role: str) -> list[str]:
    """
    Give every recipient `role` access to each file Socio may share (before the email goes).
    Google sends no email of its own; ours carries the link. Returns "name -> address" for
    each grant; raises on the first failure (the grants made so far stay).
    """

    granted = []
    for link in links:
        if not link.get("can_share"):
            continue
        shareable_file(link["file_id"])  # still in the Socio folder and not trashed
        for address in recipients:
            share_file(link["file_id"], address, role, notify=False)
            granted.append(f"{link['name']} -> {address}")
            completed_write(f"gave {address} {role} access to '{link['name']}'")
    return granted


def with_link(body: str, link: str) -> str:
    """The email body with the link where the model marked it, or below the text if it didn't."""

    body = (body or "").rstrip()
    filled, count = LINK_PLACEHOLDER.subn(lambda _: link, body)
    if count or not link or link in body:
        return filled
    return f"{body}\n\n{link}"


def validate_role(role: str) -> str:
    role = (role or "editor").strip().lower()
    if role not in ROLES:
        raise ValueError(f"Access must be one of: {', '.join(ROLES)}.")
    return role


def normalize_role(role: str | None) -> str:
    """The access the owner asked for, or "" when they didn't say (the card then asks)."""

    role = (role or "").strip().lower()
    if not role:
        return ""
    if role not in ROLE_WORDS:
        raise ValueError(f"Access must be one of: {', '.join(ROLES)}.")
    return ROLE_WORDS[role]


def validate_recipient(email: str) -> str:
    """Exactly one email address."""

    addresses = [address for _, address in getaddresses([email or ""]) if address]
    if len(addresses) != 1 or "@" not in addresses[0].strip("@"):
        raise ValueError("Give exactly one email address to share with.")
    return addresses[0].lower()


def shareable_file(reference: str) -> dict[str, Any]:
    """A file in Socio's folder (id, name, url). Raises ValueError for anything else."""

    file_id = _file_id_from_reference(reference or "")
    try:
        file = execute_with_retry(
            get_drive_service().files().get(fileId=file_id, fields="id, name, parents, webViewLink, trashed"),
            operation_name="drive_get_share_target",
        )
    except HttpError as exc:
        raise ValueError("I couldn't find that file in your Google Drive.") from exc
    if file.get("trashed") or output_folder_id() not in (file.get("parents") or []):
        raise ValueError(
            f"I can only share files I created (in your Socio folder), and '{file.get('name', 'that file')}' isn't one. "
            "You can share it from Google Drive."
        )
    return {"id": file["id"], "name": file.get("name", ""), "url": file.get("webViewLink", "")}


# Drive permission role -> the owner's word, and how much each allows (to compare).
ROLE_NAMES = {"owner": "owner", "writer": "editor", "fileOrganizer": "editor", "organizer": "editor",
              "commenter": "commenter", "reader": "viewer"}
ROLE_RANK = {"viewer": 1, "commenter": 2, "editor": 3, "owner": 4}


def current_access(file_id: str, email: str) -> str | None:
    """
    The access one person already has to a file Socio created ("editor", "commenter",
    "viewer", "owner"), or None. "Anyone with the link" counts for everyone. Read-only.
    """

    address = validate_recipient(email)
    permissions = execute_with_retry(
        get_drive_service().permissions().list(fileId=file_id, fields="permissions(type, role, emailAddress)"),
        operation_name="drive_list_permissions",
    ).get("permissions", [])
    best = None
    for permission in permissions:
        applies = permission.get("type") == "anyone" or (
            permission.get("type") == "user" and (permission.get("emailAddress") or "").lower() == address
        )
        role = ROLE_NAMES.get(permission.get("role", ""))
        if applies and role and (best is None or ROLE_RANK[role] > ROLE_RANK[best]):
            best = role
    return best


def has_access(file_id: str, email: str, role: str) -> str | None:
    """The access the person has if it is at least `role`, else None."""

    have = current_access(file_id, email)
    return have if have and ROLE_RANK[have] >= ROLE_RANK[validate_role(role)] else None


def share_file(file_id: str, email: str, role: str = "editor", notify: bool = True) -> dict[str, Any]:
    """
    Give one person access to a file. With notify=False Google sends no email of its
    own, unless the address is not a Google account: Google then requires its
    notification, so the call is repeated with it (the first call changed nothing).
    """

    service = get_drive_service()
    body = {"type": "user", "role": ROLES[validate_role(role)], "emailAddress": validate_recipient(email)}

    def create(send_notification: bool) -> dict:
        return execute_with_retry(
            service.permissions().create(
                fileId=file_id, body=body, sendNotificationEmail=send_notification, fields="id, role, emailAddress",
            ),
            operation_name="drive_share",
            retry=False,
        )

    try:
        permission = create(notify)
    except HttpError as exc:
        if notify or "invalidSharingRequest" not in str(exc):
            raise
        logger.info("Recipient is not a Google account; sharing with Google's notification email.")
        permission = create(True)
    return {"permission_id": permission.get("id"), "role": role, "email": body["emailAddress"], "verified": bool(permission.get("id"))}


def create_and_share(payload: dict[str, Any]) -> dict[str, Any]:
    """
    The approved "Create & Share" pipeline: create the file, give the recipient access,
    then email them the live link. Each step runs only if the one before succeeded.

    A failed creation raises. A failure after that is returned in the result
    (share_error / email_error) so the owner is told exactly what happened.
    The email is never retried: a timed-out send may still have been delivered
    (email_unknown: the owner is told to check Sent instead of "it didn't go through").
    """

    from app.errors import report_failure, safe_error_text
    from app.gmail.messages import send_new_email, send_reply
    from app.workspace import docs, sheets

    if payload["kind"] == "spreadsheet":
        created = sheets.create_spreadsheet(payload["title"], payload.get("headers") or None)
        file_id = created["spreadsheet_id"]
    else:
        created = docs.create_document(payload["title"], payload.get("content", ""))
        file_id = created["doc_id"]
    result: dict[str, Any] = {"file": created}

    try:
        result["share"] = share_file(file_id, payload["recipient"], payload.get("role", "editor"), notify=False)
    except Exception as exc:
        report_failure(exc, "sharing.share", file=file_id)
        result["share_error"] = safe_error_text(exc)
        return result

    body = with_link(payload["body"], created["url"])
    thread = payload.get("in_reply_to")  # answering an email: reply in that conversation
    try:
        if thread:
            result["email"] = send_reply(
                thread_id=thread["thread_id"], message_id=thread["message_id"],
                to=payload["recipient"], subject="", body=body,
            )
        else:
            result["email"] = send_new_email(to=payload["recipient"], subject=payload.get("subject", ""), body=body)
    except Exception as exc:
        report_failure(exc, "sharing.email", file=file_id)
        result["email_error"] = safe_error_text(exc)
        result["email_unknown"] = outcome_unknown(exc)
    return result
