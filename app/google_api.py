"""
Shared timeout, retry, and error handling for Google API calls
(Gmail, Calendar, and OAuth token refresh).
"""

import functools
import logging
import socket
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

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


def outcome_unknown(exc: Exception) -> bool:
    """
    True when a failed write may still have taken effect: the connection dropped or
    timed out, or Google answered 5xx, after the request may have reached it. Such a
    write is never retried (execute_with_retry(retry=False)), so nobody knows.

    False when the failure proves nothing happened: 429 (refused before acting), 4xx,
    our own validation errors, a refused sign-in, and GoogleApiUnavailableError (only
    raised by retried calls, which are reads or idempotent).
    """

    if isinstance(exc, GoogleApiUnavailableError):
        return False
    if isinstance(exc, HttpError):
        return getattr(getattr(exc, "resp", None), "status", None) in {500, 502, 503, 504}
    return is_retryable_error(exc)


# Writes that took effect during one approved action ("created the Google Doc 'X'"), so a
# failure in a later step can say exactly what is already done (app.agent.approvals.approve).
_completed_writes: ContextVar[list[str] | None] = ContextVar("socio_completed_writes", default=None)


@contextmanager
def recording_writes() -> Iterator[list[str]]:
    """Collect completed_write() steps made inside the block (in this thread)."""

    token = _completed_writes.set([])
    try:
        yield _completed_writes.get()
    finally:
        _completed_writes.reset(token)


def completed_write(step: str) -> None:
    """Note a write that succeeded (no-op outside recording_writes())."""

    steps = _completed_writes.get()
    if steps is not None:
        steps.append(step)


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


LOGIN_COMMAND = "python -m app.main --google-login"


class GoogleSignInRequired(RuntimeError):
    """
    The saved Google sign-in is gone or refused (expired or revoked refresh token:
    'invalid_grant'). Only the owner can fix it, by signing in again (LOGIN_COMMAND).
    """


# A background process (the Telegram bot) must never open a browser sign-in: it would wait
# forever while holding CREDENTIALS_LOCK, and every Google call would wait behind it (audit H4).
_browser_login = {"allowed": True}


def disable_browser_login() -> None:
    _browser_login["allowed"] = False


def browser_login_allowed() -> bool:
    return _browser_login["allowed"]


def refresh_credentials(credentials) -> None:
    """Refresh an expired OAuth access token, retrying transient network failures."""

    from google.auth.exceptions import RefreshError

    for attempt in range(1, MAX_GOOGLE_API_RETRIES + 1):
        try:
            credentials.refresh(Request())
            return
        except RefreshError as exc:
            # invalid_grant & co.: retrying cannot help; the owner has to sign in again.
            raise GoogleSignInRequired(f"Google refused the saved sign-in ({exc}). Run: {LOGIN_COMMAND}") from exc
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
