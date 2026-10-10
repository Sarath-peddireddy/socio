from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.google_api import build_authorized_http, refresh_credentials


# Read-only: the agent can find and download files, never change or delete them.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOKEN_PATH = PROJECT_ROOT / "drive_token.json"
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"


def get_drive_service(allow_browser_login: bool = False):
    """
    Return an authenticated Google Drive API service.

    The first login needs a browser; it only runs when allow_browser_login
    is True (python -m app.main --drive-login), so a background run never
    hangs waiting for a login page.
    """

    credentials = None

    if TOKEN_PATH.exists():
        credentials = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            refresh_credentials(credentials)
        elif allow_browser_login:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            credentials = flow.run_local_server(port=0)
        else:
            raise RuntimeError(
                "Google Drive is not connected yet. Run: python -m app.main --drive-login"
            )

        TOKEN_PATH.write_text(credentials.to_json(), encoding="utf-8")

    return build(
        "drive",
        "v3",
        http=build_authorized_http(credentials),
        cache_discovery=False,
    )
