"""
Google Drive as a document source.

Only files inside one configured folder (AGENT_DRIVE_FOLDER name, or
AGENT_DRIVE_FOLDER_ID) are visible to the agent. Files are downloaded to a
local cache only when they are about to be attached.
"""

import os
from pathlib import Path, PureWindowsPath

from app.drive.client import get_drive_service
from app.google_api import execute_with_retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "downloads" / "drive_cache"
DRIVE_PREFIX = "drive:"

FOLDER_MIME = "application/vnd.google-apps.folder"
# Google-native files are exported to a normal file type when attached.
EXPORTS = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}
MAX_FILES = 300


def drive_configured() -> bool:
    return bool(os.getenv("AGENT_DRIVE_FOLDER_ID", "").strip() or os.getenv("AGENT_DRIVE_FOLDER", "").strip())


def _root_folder_id(service) -> str:
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


def _display_name(file: dict) -> str:
    name = file["name"]
    export = EXPORTS.get(file["mimeType"])
    if export and not name.lower().endswith(export[1]):
        name += export[1]
    return name


def list_drive_documents(allowed_extensions: set[str]) -> list[dict]:
    """All shareable files in the configured folder and its subfolders."""

    service = get_drive_service()
    queue = [(_root_folder_id(service), "")]
    documents = []

    while queue and len(documents) < MAX_FILES:
        folder_id, folder_path = queue.pop(0)
        page_token = None

        while True:
            response = execute_with_retry(
                service.files().list(
                    q=f"'{folder_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    pageSize=100,
                    pageToken=page_token,
                ),
                operation_name="drive_list_folder",
            )

            for file in response.get("files", []):
                if file["mimeType"] == FOLDER_MIME:
                    queue.append((file["id"], f"{folder_path}{file['name']}/"))
                    continue

                name = _display_name(file)
                if file["mimeType"] not in EXPORTS and Path(name).suffix.lower() not in allowed_extensions:
                    continue

                documents.append(
                    {
                        "path": f"{DRIVE_PREFIX}{file['id']}",
                        "name": name,
                        "folder": folder_path.rstrip("/"),
                        "size_bytes": int(file.get("size", 0) or 0),
                        "modified": (file.get("modifiedTime") or "")[:10],
                        "mime_type": file["mimeType"],
                        "source": "google_drive",
                    }
                )

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return documents


def download_drive_document(path: str, allowed_extensions: set[str], max_bytes: int) -> Path:
    """
    Download a Drive document (by "drive:<id>") into the local cache and return its path.

    The id must belong to the configured folder, so an email cannot make the
    agent fetch arbitrary files from the owner's Drive.
    """

    file_id = path[len(DRIVE_PREFIX):]
    match = next((d for d in list_drive_documents(allowed_extensions) if d["path"] == path), None)

    if match is None:
        raise ValueError(f"Refusing to attach {path}: it is not in the shareable Drive folder.")

    if match["size_bytes"] > max_bytes:
        raise ValueError(f"Refusing to attach {match['name']}: larger than {max_bytes // (1024 * 1024)} MB.")

    service = get_drive_service()
    export = EXPORTS.get(match["mime_type"])

    if export:
        request = service.files().export(fileId=file_id, mimeType=export[0])
    else:
        request = service.files().get_media(fileId=file_id)

    data = execute_with_retry(request, operation_name="drive_download")

    if len(data) > max_bytes:
        raise ValueError(f"Refusing to attach {match['name']}: larger than {max_bytes // (1024 * 1024)} MB.")

    safe_name = PureWindowsPath(match["name"]).name or "document"
    target = CACHE_DIR / file_id / safe_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    return target


def socio_folder_id() -> str:
    """Id of the configured Socio Drive folder."""

    if not drive_configured():
        raise RuntimeError("No Socio Drive folder configured. Set AGENT_DRIVE_FOLDER in .env.")
    return _root_folder_id(get_drive_service())


def _file_id_from_reference(reference: str) -> str:
    """Accept "drive:<id>", a Google Docs/Sheets URL, or a bare file id."""

    reference = reference.strip()
    if reference.startswith(DRIVE_PREFIX):
        return reference[len(DRIVE_PREFIX):]

    marker = "/d/"
    if marker in reference:
        return reference.split(marker, 1)[1].split("/", 1)[0].split("?", 1)[0]

    return reference


def resolve_folder_file(reference: str, mime_type: str) -> dict:
    """
    Resolve a reference to a file of mime_type inside the Socio folder.

    Socio's Docs/Sheets permissions cover the whole Drive, so this check is
    what keeps reads and edits inside the folder the owner chose.
    """

    from app.files.local import ALLOWED_EXTENSIONS

    file_id = _file_id_from_reference(reference)
    for document in list_drive_documents(ALLOWED_EXTENSIONS):
        if document["path"] == f"{DRIVE_PREFIX}{file_id}":
            if document["mime_type"] != mime_type:
                raise ValueError(f"'{document['name']}' is not a {mime_type.rsplit('.', 1)[-1]}.")
            return {"id": file_id, "name": document["name"], "folder": document["folder"]}

    raise ValueError("That file is not in the Socio Drive folder. Use find_documents to locate it first.")


def move_into_socio_folder(file_id: str) -> None:
    """Move a file Socio just created (in My Drive) into the Socio folder."""

    service = get_drive_service()
    current = execute_with_retry(
        service.files().get(fileId=file_id, fields="parents"),
        operation_name="drive_get_parents",
    )
    execute_with_retry(
        service.files().update(
            fileId=file_id,
            addParents=socio_folder_id(),
            removeParents=",".join(current.get("parents", [])),
            fields="id, parents",
        ),
        operation_name="drive_move_to_folder",
    )
