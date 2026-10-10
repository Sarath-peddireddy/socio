from googleapiclient.discovery import build

from app.gmail.auth import get_gmail_credentials
from app.google_api import build_authorized_http, execute_with_retry


def execute_gmail_request(request, *, operation_name: str, retry: bool = True):
    """
    Execute a Gmail API request with bounded retries for transient failures.

    Pass retry=False for non-idempotent requests such as sending an email.
    See app.google_api.execute_with_retry.
    """

    return execute_with_retry(request, operation_name=operation_name, retry=retry)


def get_gmail_service():
    """
    Create and return an authenticated Gmail API service.

    The Gmail API can intermittently time out on managed networks, so every
    request uses a bounded socket timeout and call sites add explicit retries.
    """

    credentials = get_gmail_credentials()

    service = build(
        "gmail",
        "v1",
        http=build_authorized_http(credentials),
        cache_discovery=False,
    )

    return service
