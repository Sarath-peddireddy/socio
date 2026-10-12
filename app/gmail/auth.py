from pathlib import Path

from google.oauth2.credentials import Credentials

from app.google_api import LOGIN_COMMAND, browser_login_allowed
from app.google_auth import load_credentials


# Gmail permissions required by this project.
# We are using gmail.modify because the final agent will need to
# read, compose, modify, and send emails.
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


# Project root (two levels above this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"


def get_gmail_credentials(allow_browser_login: bool | None = None) -> Credentials:
    """
    Authenticate the user with Google and return Gmail API credentials.

    token.json is reused and refreshed. A browser sign-in only opens from an interactive
    command (CLI, `--google-login`), never in the Telegram bot: there a missing or refused
    sign-in raises GoogleSignInRequired instead of waiting for a browser forever (audit H4).
    """

    allowed = browser_login_allowed() if allow_browser_login is None else allow_browser_login
    return load_credentials(TOKEN_FILE, SCOPES, allow_browser_login=allowed, login_command=LOGIN_COMMAND)
