"""
Google Drive as a document source.

By default Socio searches the owner's whole Drive (read-only). Setting
AGENT_DRIVE_FOLDER (name) or AGENT_DRIVE_FOLDER_ID optionally limits it to one
folder and its subfolders.

Files Socio creates (Docs, Sheets) go into a "Socio" folder that Socio creates
and tags itself, so it never writes into one of the owner's own folders.

Reading is unrestricted for the owner; *sending* a Drive file to someone who
emailed asking for it needs the owner's approval (see app.agent.approvals).
"""

import os
import re
from pathlib import Path, PureWindowsPath

from app.drive.client import get_drive_service
from app.google_api import execute_with_retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "downloads" / "drive_cache"
DRIVE_PREFIX = "drive:"

FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
# Google-native files are exported to a normal file type when attached.
EXPORTS = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}
# Google-native files are sent as their live link, never as an exported copy
# (developer decision 2026-10-11); EXPORTS is only used for older approvals.
GOOGLE_TYPES = {
    "application/vnd.google-apps.spreadsheet": "Google Sheet",
    "application/vnd.google-apps.document": "Google Doc",
    "application/vnd.google-apps.presentation": "Google Slides",
}
FILE_FIELDS = "id, name, mimeType, size, modifiedTime, parents, trashed"
MAX_FILES = 300

OUTPUT_FOLDER_NAME = "Socio"
OUTPUT_FOLDER_TAG = {"socio": "output"}


# ----- configuration -----------------------------------------------------------


def drive_available() -> bool:
    """True when the owner has signed in to Google Workspace (or set a folder)."""

    from app.workspace.client import TOKEN_PATH

    return TOKEN_PATH.exists() or folder_limited()


def folder_limited() -> bool:
    return bool(os.getenv("AGENT_DRIVE_FOLDER_ID", "").strip() or os.getenv("AGENT_DRIVE_FOLDER", "").strip())


# Backwards-compatible name used elsewhere in the codebase.
drive_configured = drive_available


def _limit_folder_id(service) -> str:
    folder_id = os.getenv("AGENT_DRIVE_FOLDER_ID", "").strip()
    if folder_id:
        return folder_id

    name = os.getenv("AGENT_DRIVE_FOLDER", "").strip().replace("'", "\\'")
    folders = execute_with_retry(
        service.files().list(
            q=f"name = '{name}' and mimeType = '{FOLDER_MIME}' and trashed = false",
            fields="files(id, name)",
            pageSize=5,
        ),
        operation_name="drive_find_folder",
    ).get("files", [])

    if not folders:
        raise RuntimeError(f"Google Drive folder '{name}' was not found.")
    if len(folders) > 1:
        raise RuntimeError(f"Several Drive folders are named '{name}'; set AGENT_DRIVE_FOLDER_ID instead.")

    return folders[0]["id"]


# ----- helpers -------------------------------------------------------------------


def _display_name(file: dict) -> str:
    name = file["name"]
    export = EXPORTS.get(file["mimeType"])
    if export and not name.lower().endswith(export[1]):
        name += export[1]
    return name


def _is_document(file: dict, allowed_extensions: set[str]) -> bool:
    if file["mimeType"] in (FOLDER_MIME, SHORTCUT_MIME) or file.get("trashed"):
        return False
    return file["mimeType"] in EXPORTS or Path(_display_name(file)).suffix.lower() in allowed_extensions


def _as_document(file: dict, folder: str = "") -> dict:
    google_type = GOOGLE_TYPES.get(file["mimeType"])
    return {
        "path": f"{DRIVE_PREFIX}{file['id']}",
        # A Google Sheet is listed as "Budget" (type Google Sheet), not "Budget.xlsx":
        # it is shared as a link, so an export extension would mislead the agent.
        "name": file["name"] if google_type else _display_name(file),
        **({"type": google_type} if google_type else {}),
        "folder": folder,
        "size_bytes": int(file.get("size", 0) or 0),
        "modified": (file.get("modifiedTime") or "")[:10],
        "mime_type": file["mimeType"],
        "source": "google_drive",
    }


def _file_id_from_reference(reference: str) -> str:
    """Accept "drive:<id>", a Google Docs/Sheets/Drive URL, or a bare file id."""

    reference = reference.strip()
    if reference.startswith(DRIVE_PREFIX):
        return reference[len(DRIVE_PREFIX):]

    match = re.search(r"/d/([A-Za-z0-9_-]+)", reference) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", reference)
    return match.group(1) if match else reference


# ----- listing and search -----------------------------------------------------------


def list_drive_documents(allowed_extensions: set[str]) -> list[dict]:
    """All documents in the limiting folder and its subfolders (folder-limited mode only)."""

    service = get_drive_service()
    queue = [(_limit_folder_id(service), "")]
    documents = []

    while queue and len(documents) < MAX_FILES:
        folder_id, folder_path = queue.pop(0)
        page_token = None

        while True:
            response = execute_with_retry(
                service.files().list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields=f"nextPageToken, files({FILE_FIELDS})",
                    pageSize=100,
                    pageToken=page_token,
                ),
                operation_name="drive_list_folder",
            )

            for file in response.get("files", []):
                if file["mimeType"] == FOLDER_MIME:
                    queue.append((file["id"], f"{folder_path}{file['name']}/"))
                elif _is_document(file, allowed_extensions):
                    documents.append(_as_document(file, folder_path.rstrip("/")))

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return documents


def search_drive(words: set[str], allowed_extensions: set[str], max_results: int = 25) -> list[dict]:
    """
    Search documents by name and content.

    Whole Drive by default (Drive's own search index), or the limiting folder
    when AGENT_DRIVE_FOLDER is set.
    """

    if folder_limited():
        return list_drive_documents(allowed_extensions)

    words = [word for word in words if word]
    if not words:
        return []

    clauses = " or ".join(
        f"name contains '{w}' or fullText contains '{w}'" for w in (word.replace("'", "\\'") for word in words)
    )
    # Only files the owner owns: "Shared with me" contains other people's
    # documents (e.g. their resumes), which must never be offered as "mine".
    response = execute_with_retry(
        get_drive_service().files().list(
            q=f"({clauses}) and 'me' in owners and trashed = false and mimeType != '{FOLDER_MIME}'",
            fields=f"files({FILE_FIELDS})",
            pageSize=max(1, min(max_results, 100)),
            spaces="drive",
        ),
        operation_name="drive_search",
    )

    return [_as_document(file) for file in response.get("files", []) if _is_document(file, allowed_extensions)]


def get_drive_document(reference: str, allowed_extensions: set[str]) -> dict:
    """Metadata for one document by reference, enforcing the folder limit if one is set."""

    file_id = _file_id_from_reference(reference)

    if folder_limited():
        match = next((d for d in list_drive_documents(allowed_extensions) if d["path"] == f"{DRIVE_PREFIX}{file_id}"), None)
        if match is None:
            raise ValueError("That file is outside the Drive folder Socio is limited to (AGENT_DRIVE_FOLDER).")
        return match

    try:
        file = execute_with_retry(
            get_drive_service().files().get(fileId=file_id, fields=FILE_FIELDS),
            operation_name="drive_get_file",
        )
    except Exception as exc:
        raise ValueError(f"Drive file not found or not accessible: {reference}") from exc

    if not _is_document(file, allowed_extensions):
        raise ValueError(f"'{file.get('name')}' is not a supported document type.")

    return _as_document(file)


# ----- download (for attachments) ------------------------------------------------------


def download_drive_document(path: str, allowed_extensions: set[str], max_bytes: int) -> Path:
    """Download a Drive document (by "drive:<id>") into the local cache and return its path."""

    match = get_drive_document(path, allowed_extensions)
    file_id = match["path"][len(DRIVE_PREFIX):]

    if match["size_bytes"] > max_bytes:
        raise ValueError(f"Refusing to attach {match['name']}: larger than {max_bytes // (1024 * 1024)} MB.")

    service = get_drive_service()
    export = EXPORTS.get(match["mime_type"])
    request = (
        service.files().export(fileId=file_id, mimeType=export[0])
        if export
        else service.files().get_media(fileId=file_id)
    )
    data = execute_with_retry(request, operation_name="drive_download")

    if len(data) > max_bytes:
        raise ValueError(f"Refusing to attach {match['name']}: larger than {max_bytes // (1024 * 1024)} MB.")

    safe_name = PureWindowsPath(match["name"]).name or "document"
    if export and not safe_name.lower().endswith(export[1]):
        safe_name += export[1]
    target = CACHE_DIR / file_id / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    return target


# ----- Docs / Sheets support ----------------------------------------------------------------


def resolve_drive_file(reference: str, mime_type: str) -> dict:
    """Resolve a reference to a Google Doc/Sheet the owner can access (respecting any folder limit)."""

    from app.files.local import ALLOWED_EXTENSIONS

    document = get_drive_document(reference, ALLOWED_EXTENSIONS)
    if document["mime_type"] != mime_type:
        kind = "document" if "document" in mime_type else "spreadsheet"
        raise ValueError(f"'{document['name']}' is not a Google {kind}.")

    return {"id": document["path"][len(DRIVE_PREFIX):], "name": document["name"], "folder": document["folder"]}


def output_folder_id() -> str:
    """
    Folder where Socio puts files it creates.

    The limiting folder if one is set; otherwise a "Socio" folder that Socio
    created and tagged itself (created on first use).
    """

    service = get_drive_service()

    if folder_limited():
        return _limit_folder_id(service)

    key, value = next(iter(OUTPUT_FOLDER_TAG.items()))
    existing = execute_with_retry(
        service.files().list(
            q=f"appProperties has {{ key='{key}' and value='{value}' }} and mimeType = '{FOLDER_MIME}' and trashed = false",
            fields="files(id)",
            pageSize=1,
        ),
        operation_name="drive_find_output_folder",
    ).get("files", [])

    if existing:
        return existing[0]["id"]

    folder = execute_with_retry(
        service.files().create(
            body={"name": OUTPUT_FOLDER_NAME, "mimeType": FOLDER_MIME, "appProperties": OUTPUT_FOLDER_TAG},
            fields="id",
        ),
        operation_name="drive_create_output_folder",
        retry=False,
    )
    return folder["id"]


def move_into_output_folder(file_id: str) -> None:
    """Move a file Socio just created (in My Drive) into Socio's output folder."""

    service = get_drive_service()
    current = execute_with_retry(
        service.files().get(fileId=file_id, fields="parents"),
        operation_name="drive_get_parents",
    )
    execute_with_retry(
        service.files().update(
            fileId=file_id,
            addParents=output_folder_id(),
            removeParents=",".join(current.get("parents", [])),
            fields="id, parents",
        ),
        operation_name="drive_move_to_output_folder",
    )
