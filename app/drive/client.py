from app.workspace.client import get_workspace_service


def get_drive_service(allow_browser_login: bool = False):
    """
    Return an authenticated Google Drive API service.

    Uses the shared Workspace token (Drive, Docs, Sheets, Tasks). The first
    sign-in runs only when allow_browser_login is True
    (python -m app.main --google-login).
    """

    return get_workspace_service("drive", "v3", allow_browser_login=allow_browser_login)
