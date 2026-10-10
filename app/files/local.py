"""
Local file discovery for document requests.

Only files inside the allowlisted folder(s) can be found or attached, and only
common document types. Email content is untrusted, so the agent must never be
able to reach .env, OAuth tokens, or arbitrary files on disk.
"""

import os
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILES_DIR = PROJECT_ROOT / "documents"

ALLOWED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt",
    ".ppt", ".pptx", ".xls", ".xlsx", ".csv",
    ".png", ".jpg", ".jpeg",
}

# Gmail rejects messages over 25 MB; leave room for base64 overhead.
MAX_ATTACHMENT_BYTES = 18 * 1024 * 1024


def shareable_roots() -> list[Path]:
    """Folders the agent may search and attach from (AGENT_FILES_DIR, ';'-separated)."""

    configured = os.getenv("AGENT_FILES_DIR", "").strip()
    folders = [Path(p.strip()) for p in configured.split(";") if p.strip()] if configured else [DEFAULT_FILES_DIR]
    return [folder.resolve() for folder in folders if folder.is_dir()]


def resolve_shareable_file(path: str) -> Path:
    """
    Return a local path for an allowed document.

    "drive:<id>" paths are downloaded from the configured Google Drive folder;
    anything else must be an existing file inside a local shareable folder.
    """

    if path.startswith("drive:"):
        from app.files.drive import download_drive_document

        return download_drive_document(path, ALLOWED_EXTENSIONS, MAX_ATTACHMENT_BYTES)

    candidate = Path(path).resolve()
    roots = shareable_roots()

    if not any(candidate.is_relative_to(root) for root in roots):
        raise ValueError(f"Refusing to attach {path}: it is outside the shareable documents folder.")

    if not candidate.is_file():
        raise ValueError(f"Attachment not found: {path}")

    if candidate.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Refusing to attach {candidate.name}: file type {candidate.suffix} is not allowed.")

    if candidate.stat().st_size > MAX_ATTACHMENT_BYTES:
        raise ValueError(f"Refusing to attach {candidate.name}: larger than {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB.")

    return candidate


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 2}


SYNONYMS = {"resume": {"cv"}, "cv": {"resume"}}


def _query_tokens(query: str) -> set[str]:
    tokens = _tokens(query)
    for token in list(tokens):
        tokens |= SYNONYMS.get(token, set())
    return tokens


def search_documents(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the Google Drive folder (if configured) and local shareable folders.

    Results are ranked by how many query words match the file or folder name.
    If Drive is configured but unavailable, a {"note": ...} entry explains why.
    """

    matches = search_local_files(query, max_results=max_results)
    notes = []

    from app.files.drive import drive_configured, list_drive_documents

    if drive_configured():
        try:
            query_tokens = _query_tokens(query)
            for document in list_drive_documents(ALLOWED_EXTENSIONS):
                name_tokens = _tokens(f"{document['folder']} {Path(document['name']).stem}")
                score = len(query_tokens & name_tokens)
                if score == 0 and query_tokens:
                    continue
                matches.append(
                    {
                        "path": document["path"],
                        "name": document["name"],
                        "folder": document["folder"],
                        "source": "google_drive",
                        "size_kb": round(document["size_bytes"] / 1024, 1),
                        "modified": document["modified"],
                        "match_score": score,
                        "too_large_to_attach": document["size_bytes"] > MAX_ATTACHMENT_BYTES,
                    }
                )
        except Exception as exc:
            notes.append({"note": f"Google Drive could not be searched: {exc}"})

    matches.sort(key=lambda match: (-match["match_score"], match["source"] != "google_drive", match["name"].lower()))
    return matches[:max_results] + notes


def search_local_files(query: str, max_results: int = 5) -> list[dict]:
    """
    Find files in the local shareable folders whose name or folder matches the query words.

    Results are ranked by how many query words match.
    """

    query_tokens = _query_tokens(query)
    matches = []

    for root in shareable_roots():
        for file_path in root.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            relative = file_path.relative_to(root)
            name_tokens = _tokens(str(relative.with_suffix("")))
            score = len(query_tokens & name_tokens)

            if score == 0 and query_tokens:
                continue

            stat = file_path.stat()
            matches.append(
                {
                    "path": str(file_path),
                    "name": file_path.name,
                    "folder": str(relative.parent) if str(relative.parent) != "." else "",
                    "source": "local",
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
                    "match_score": score,
                    "too_large_to_attach": stat.st_size > MAX_ATTACHMENT_BYTES,
                }
            )

    matches.sort(key=lambda match: (-match["match_score"], match["name"].lower()))
    return matches[:max_results]
