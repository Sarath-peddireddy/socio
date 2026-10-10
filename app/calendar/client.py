from pathlib import Path
import os

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from app.google_api import build_authorized_http, refresh_credentials


SCOPES = [
    "https://www.googleapis.com/auth/calendar"
]

# Resolve from the project root (like app/gmail/auth.py) so the token is
# found regardless of the directory the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKEN_PATH = PROJECT_ROOT / "calendar_token.json"
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"


def get_calendar_service():
    """
    Authenticate with Google Calendar and return the API service.
    """

    credentials = None

    if TOKEN_PATH.exists():
        credentials = Credentials.from_authorized_user_file(
            TOKEN_PATH,
            SCOPES,
        )

    if not credentials or not credentials.valid:

        if credentials and credentials.expired and credentials.refresh_token:
            refresh_credentials(credentials)

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH,
                SCOPES,
            )

            credentials = flow.run_local_server(
                port=0
            )

        TOKEN_PATH.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

    return build(
        "calendar",
        "v3",
        http=build_authorized_http(credentials),
        cache_discovery=False,
    )