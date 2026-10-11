"""
Sharing files Socio created, and the compound "create, share, and email the link" action.

Only files in Socio's own output folder can be shared: the Workspace token has
drive.file (files Socio created), not full Drive write access. Both actions run
only from app.agent.approvals, after the owner approves.
"""

from __future__ import annotations

import logging
import re
from email.utils import getaddresses
from typing import Any

from googleapiclient.errors import HttpError

from app.drive.client import get_drive_service
from app.files.drive import _file_id_from_reference, output_folder_id
from app.google_api import execute_with_retry


logger = logging.getLogger(__name__)

# What the owner says -> Drive permission role.
ROLES = {"editor": "writer", "commenter": "commenter", "viewer": "reader"}
FILE_KINDS = {"spreadsheet", "document"}

# Where the model marked the link in the email body: [link], [LINK], [Link to the sheet],
# {link}, {{url}}, <link>, [Google Sheet URL] ... (short, one line; not a Markdown "[text](url)").
_WORD = r"\b(?:link|url|hyperlink)\b"
LINK_PLACEHOLDER = re.compile(
    rf"\[[^\[\]\n]{{0,40}}?{_WORD}[^\[\]\n]{{0,40}}?\](?!\()"
    rf"|\{{\{{?[^{{}}\n]{{0,40}}?{_WORD}[^{{}}\n]{{0,40}}?\}}\}}?"
    rf"|<[^<>\n]{{0,40}}?{_WORD}[^<>\n]{{0,40}}?>",
    re.IGNORECASE,
)


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

    A failed creation raises (nothing changed). A failure after that is returned in
    the result (share_error / email_error) so the owner is told exactly what happened.
    The email is never retried: a timed-out send may still have been delivered.
    """

    from app.errors import report_failure, safe_error_text
    from app.gmail.messages import send_new_email
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
    try:
        result["email"] = send_new_email(to=payload["recipient"], subject=payload.get("subject", ""), body=body)
    except Exception as exc:
        report_failure(exc, "sharing.email", file=file_id)
        result["email_error"] = safe_error_text(exc)
    return result
