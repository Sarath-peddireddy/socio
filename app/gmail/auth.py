from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from app.google_api import credentials_locked, refresh_credentials


# Gmail permissions required by this project.
# We are using gmail.modify because the final agent will need to
# read, compose, modify, and send emails.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# Project root (two levels above this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"


@credentials_locked
def get_gmail_credentials() -> Credentials:
    """
    Authenticate the user with Google and return Gmail API credentials.

    The first run opens a browser for Google OAuth.
    After successful authorization, token.json is saved locally
    so future runs don't require authentication every time.
    """

    credentials = None

    # If we already authenticated previously, load the saved token.
    if TOKEN_FILE.exists():
        credentials = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES,
        )

    # If credentials don't exist or are no longer valid, authenticate.
    if not credentials or not credentials.valid:

        # Refresh an expired access token if possible.
        if credentials and credentials.expired and credentials.refresh_token:
            refresh_credentials(credentials)

        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Google OAuth credentials not found at: "
                    f"{CREDENTIALS_FILE}"
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES,
            )

            credentials = flow.run_local_server(port=0)

        # Save credentials for future runs.
        TOKEN_FILE.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

    return credentials