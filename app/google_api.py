"""
Shared timeout, retry, and error handling for Google API calls
(Gmail, Calendar, and OAuth token refresh).
"""

import functools
import logging
import socket
import threading
import time

import httplib2
from google.auth.exceptions import TransportError
from google.auth.transport.requests import Request
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.errors import HttpError
from requests.exceptions import RequestException


logger = logging.getLogger(__name__)

GOOGLE_API_TIMEOUT_SECONDS = 30
MAX_GOOGLE_API_RETRIES = 3


# Token files are read, refreshed, and rewritten by whichever thread needs a
# client; supervisor workers run in parallel, so serialize that per process.
CREDENTIALS_LOCK = threading.RLock()


def credentials_locked(func):
    """Run a credential load/refresh/save function under CREDENTIALS_LOCK."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with CREDENTIALS_LOCK:
            return func(*args, **kwargs)

    return wrapper


class GoogleApiUnavailableError(RuntimeError):
    """Raised when a Google API cannot be reached after all retries."""


def is_retryable_error(exc: Exception) -> bool:
    """Return True when the error suggests a transient network or server failure."""

    if isinstance(exc, HttpError):
        # httplib2 stores resp["status"] as a string; resp.status is the int.
        status_code = getattr(getattr(exc, "resp", None), "status", None)
        return status_code in {429, 500, 502, 503, 504}

    retryable_types = (
        TimeoutError,  # includes WinError 10060
        socket.timeout,
        RequestException,
        TransportError,  # OAuth token refresh could not reach Google
        httplib2.ServerNotFoundError,  # DNS lookup failed
        OSError,
    )

    return isinstance(exc, retryable_types)


def _unavailable(operation_name: str, exc: Exception) -> GoogleApiUnavailableError:
    return GoogleApiUnavailableError(
        f"Could not reach Google API during '{operation_name}' after "
        f"{MAX_GOOGLE_API_RETRIES} attempts ({type(exc).__name__}: {exc}). "
        "Check the internet connection, VPN, proxy, or firewall."
    )


def execute_with_retry(request, *, operation_name: str, retry: bool = True):
    """
    Execute a googleapiclient request with bounded retries for transient failures.

    Pass retry=False for non-idempotent requests such as sending an email or
    creating an event: a timeout there does not prove the request failed, and
    retrying could create a duplicate.
    """

    if not retry:
        return request.execute()

    for attempt in range(1, MAX_GOOGLE_API_RETRIES + 1):
        try:
            return request.execute()
        except Exception as exc:
            if not is_retryable_error(exc):
                raise

            if attempt == MAX_GOOGLE_API_RETRIES:
                if isinstance(exc, HttpError):
                    raise
                raise _unavailable(operation_name, exc) from exc

            logger.warning(
                "%s failed (attempt %d/%d): %s: %s. Retrying.",
                operation_name, attempt, MAX_GOOGLE_API_RETRIES, type(exc).__name__, exc,
            )
            time.sleep(2 ** (attempt - 1))


def refresh_credentials(credentials) -> None:
    """Refresh an expired OAuth access token, retrying transient network failures."""

    for attempt in range(1, MAX_GOOGLE_API_RETRIES + 1):
        try:
            credentials.refresh(Request())
            return
        except Exception as exc:
            if not is_retryable_error(exc):
                raise

            if attempt == MAX_GOOGLE_API_RETRIES:
                raise _unavailable("oauth_token_refresh", exc) from exc

            logger.warning("OAuth token refresh failed (attempt %d): %s. Retrying.", attempt, exc)
            time.sleep(2 ** (attempt - 1))


def build_authorized_http(credentials) -> AuthorizedHttp:
    """Return an HTTP transport that applies a bounded socket timeout to every request."""

    return AuthorizedHttp(
        credentials,
        http=httplib2.Http(timeout=GOOGLE_API_TIMEOUT_SECONDS),
    )
