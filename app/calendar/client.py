from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.google_api import LOGIN_COMMAND, browser_login_allowed, build_authorized_http
from app.google_auth import load_credentials


SCOPES = [
    "https://www.googleapis.com/auth/calendar"
]

# Resolve from the project root (like app/gmail/auth.py) so the token is
# found regardless of the directory the script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOKEN_PATH = PROJECT_ROOT / "calendar_token.json"
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"


def get_calendar_credentials(allow_browser_login: bool | None = None) -> Credentials:
    """
    Calendar credentials from calendar_token.json (refreshed when expired). A browser
    sign-in only opens from an interactive command, never in the Telegram bot: there a
    missing or refused sign-in raises GoogleSignInRequired (audit H4).
    """

    allowed = browser_login_allowed() if allow_browser_login is None else allow_browser_login
    return load_credentials(TOKEN_PATH, SCOPES, allow_browser_login=allowed, login_command=LOGIN_COMMAND)


def get_calendar_service():
    """
    Authenticate with Google Calendar and return the API service.
    """

    return build(
        "calendar",
        "v3",
        http=build_authorized_http(get_calendar_credentials()),
        cache_discovery=False,
    )
