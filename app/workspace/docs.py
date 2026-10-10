"""
Google Docs: read, create, and append. Nothing is ever deleted or overwritten.

Reads and edits are limited to the Socio Drive folder; new documents are
created there.
"""

from app.files.drive import move_into_socio_folder, resolve_folder_file
from app.google_api import execute_with_retry
from app.workspace.client import get_workspace_service


DOC_MIME = "application/vnd.google-apps.document"
MAX_READ_CHARS = 20000


def _service():
    return get_workspace_service("docs", "v1")


def _doc_url(doc_id: str) -> str:
    return f"https://docs.google.com/document/d/{doc_id}/edit"


def _text_from_elements(elements: list[dict]) -> str:
    parts = []
    for element in elements:
        if "paragraph" in element:
            for run in element["paragraph"].get("elements", []):
                parts.append(run.get("textRun", {}).get("content", ""))
        elif "table" in element:
            for row in element["table"].get("tableRows", []):
                cells = [_text_from_elements(cell.get("content", [])).strip() for cell in row.get("tableCells", [])]
                parts.append(" | ".join(cells) + "\n")
    return "".join(parts)


def read_document(reference: str) -> dict:
    """Plain text of a Google Doc in the Socio folder (truncated for very long docs)."""

    file = resolve_folder_file(reference, DOC_MIME)
    document = execute_with_retry(_service().documents().get(documentId=file["id"]), operation_name="docs_get")
    text = _text_from_elements(document.get("body", {}).get("content", []))

    return {
        "doc_id": file["id"],
        "title": document.get("title", file["name"]),
        "url": _doc_url(file["id"]),
        "text": text[:MAX_READ_CHARS],
        "truncated": len(text) > MAX_READ_CHARS,
    }


def create_document(title: str, content: str) -> dict:
    """Create a Google Doc with the given text inside the Socio folder."""

    if not title.strip():
        raise ValueError("A document needs a title.")

    service = _service()

    # Not retried: a timed-out create may still have created the document.
    document = execute_with_retry(
        service.documents().create(body={"title": title.strip()}),
        operation_name="docs_create",
        retry=False,
    )
    doc_id = document["documentId"]

    if content:
        execute_with_retry(
            service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": [{"insertText": {"location": {"index": 1}, "text": content}}]},
            ),
            operation_name="docs_insert_text",
            retry=False,
        )

    move_into_socio_folder(doc_id)
    return {"doc_id": doc_id, "title": title.strip(), "url": _doc_url(doc_id), "verified": True}


def append_to_document(reference: str, text: str) -> dict:
    """Append text to the end of a Google Doc in the Socio folder (existing content is untouched)."""

    if not text:
        raise ValueError("Nothing to append.")

    file = resolve_folder_file(reference, DOC_MIME)
    service = _service()

    # endOfSegmentLocation appends at the end of the body without touching existing content.
    execute_with_retry(
        service.documents().batchUpdate(
            documentId=file["id"],
            body={"requests": [{"insertText": {"endOfSegmentLocation": {}, "text": "\n" + text}}]},
        ),
        operation_name="docs_append",
        retry=False,
    )
    return {"doc_id": file["id"], "title": file["name"], "url": _doc_url(file["id"]), "appended_chars": len(text), "verified": True}
