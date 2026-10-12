"""
Shared OAuth credential loading for Google services.

Gmail and Calendar keep their own token files; Drive, Docs, Sheets, and Tasks
share one Workspace token so a single sign-in covers all of them.
"""

import json
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from app.google_api import GoogleSignInRequired, credentials_locked, refresh_credentials


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"


@credentials_locked
def load_credentials(token_path: Path, scopes: list[str], *, allow_browser_login: bool, login_command: str) -> Credentials:
    """
    Load, refresh, or (if allowed) create OAuth credentials for the given scopes.

    The browser sign-in only runs when allow_browser_login is True, so a
    background process never hangs waiting for a login page.
    """

    credentials = None

    if token_path.exists():
        # from_authorized_user_file reports the scopes we pass in, not the ones the
        # token was granted, so check the granted scopes stored in the file. A token
        # created for fewer scopes cannot be refreshed into more.
        granted = set(json.loads(token_path.read_text(encoding="utf-8")).get("scopes") or [])
        if set(scopes) <= granted:
            credentials = Credentials.from_authorized_user_file(token_path, scopes)

    if credentials and credentials.valid:
        return credentials

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            refresh_credentials(credentials)
        except GoogleSignInRequired:
            # The saved sign-in was revoked or expired (invalid_grant): only a new sign-in helps.
            if not allow_browser_login:
                raise
            credentials = None
    if credentials and credentials.valid:
        pass
    elif allow_browser_login:
        if not CREDENTIALS_PATH.exists():
            raise FileNotFoundError(f"Google OAuth client file not found: {CREDENTIALS_PATH}")
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, scopes)
        credentials = flow.run_local_server(port=0)
    else:
        raise GoogleSignInRequired(f"Google sign-in needed. Run: {login_command}")

    token_path.write_text(credentials.to_json(), encoding="utf-8")
    return credentials
