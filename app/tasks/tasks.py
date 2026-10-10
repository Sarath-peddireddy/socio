"""
Google Tasks: list, create, and complete tasks in the owner's default list.

Tasks are never deleted by Socio; completing one is reversible in the app.
"""

from datetime import date, datetime, timezone

from app.google_api import execute_with_retry
from app.workspace.client import get_workspace_service


DEFAULT_LIST = "@default"


def _service():
    return get_workspace_service("tasks", "v1")


def _summarize(task: dict) -> dict:
    return {
        "task_id": task.get("id"),
        "title": task.get("title", ""),
        "notes": task.get("notes", ""),
        "due": (task.get("due") or "")[:10],
        "status": task.get("status"),
    }


def _due_rfc3339(due: str) -> str:
    """Tasks only stores the date part of 'due'; accept YYYY-MM-DD or an ISO datetime."""

    try:
        parsed = date.fromisoformat(due[:10])
    except ValueError as exc:
        raise ValueError(f"Invalid due date {due!r}. Use YYYY-MM-DD.") from exc
    return f"{parsed.isoformat()}T00:00:00.000Z"


def list_tasks(include_completed: bool = False, max_results: int = 50) -> list[dict]:
    response = execute_with_retry(
        _service().tasks().list(
            tasklist=DEFAULT_LIST,
            showCompleted=include_completed,
            showHidden=include_completed,
            maxResults=max(1, min(int(max_results), 100)),
        ),
        operation_name="tasks_list",
    )
    return [_summarize(task) for task in response.get("items", [])]


def create_task(title: str, notes: str = "", due: str = "") -> dict:
    if not title.strip():
        raise ValueError("A task needs a title.")

    body = {"title": title.strip()[:1024], "notes": notes[:8000]}
    if due:
        body["due"] = _due_rfc3339(due)

    # Not retried: a timed-out insert may still have created the task.
    task = execute_with_retry(
        _service().tasks().insert(tasklist=DEFAULT_LIST, body=body),
        operation_name="tasks_create",
        retry=False,
    )
    return {**_summarize(task), "verified": bool(task.get("id"))}


def complete_task(task_id: str) -> dict:
    task = execute_with_retry(
        _service().tasks().patch(
            tasklist=DEFAULT_LIST,
            task=task_id,
            body={"status": "completed", "completed": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")},
        ),
        operation_name="tasks_complete",
    )
    return {**_summarize(task), "verified": task.get("status") == "completed"}
