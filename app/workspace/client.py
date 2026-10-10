"""
One authenticated client factory for Drive, Docs, Sheets, and Tasks.
"""

from googleapiclient.discovery import build

from app.google_api import build_authorized_http
from app.google_auth import PROJECT_ROOT, load_credentials


WORKSPACE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",  # find and read files in the Socio folder
    "https://www.googleapis.com/auth/drive.file",      # move files Socio creates into the Socio folder
    "https://www.googleapis.com/auth/documents",       # read / append to Google Docs
    "https://www.googleapis.com/auth/spreadsheets",    # read / append rows to Google Sheets
    "https://www.googleapis.com/auth/tasks",           # Google Tasks
]

TOKEN_PATH = PROJECT_ROOT / "workspace_token.json"
LOGIN_COMMAND = "python -m app.main --google-login"


def get_workspace_service(api: str, version: str, allow_browser_login: bool = False):
    """Build a Google API client (drive v3, docs v1, sheets v4, tasks v1) with the shared token."""

    credentials = load_credentials(
        TOKEN_PATH,
        WORKSPACE_SCOPES,
        allow_browser_login=allow_browser_login,
        login_command=LOGIN_COMMAND,
    )

    return build(api, version, http=build_authorized_http(credentials), cache_discovery=False)
