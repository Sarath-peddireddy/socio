"""
Error gateway: full details go to logs/agent.log, the owner gets one short fixed message.

    try:
        ...
    except Exception as exc:
        bot.send(report_failure(exc, "telegram.callback", approval_id=7))

- report_failure() writes the stack trace, where it happened, the given context
  (ids and command names only; never email bodies or tool argument values), and
  a timestamp to logs/agent.log (AGENT_LOG_DIR to move it), then returns
  user_message(exc).
- user_message() maps an exception to one of a few predefined strings. They never
  contain tracebacks, class names, or file paths, and they only promise what
  actually happens (nothing is queued or re-run behind the owner's back).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

GOOGLE_BUSY = "⚠️ Google services are busy or unreachable right now. Please try again in a few minutes."
SYSTEM_BUSY = "⚠️ System is handling another task. Please try again in a moment."
AI_SLOW = "⚠️ The AI model took too long to respond. Please tap /run or retry."
AI_QUOTA = "⚠️ The AI model's daily limit is used up. Please try again later."
GENERIC = "⚠️ Unable to complete that task right now. I've logged the issue for review."

# Google HTTP statuses that mean "busy / try later" (not "you asked for something wrong").
GOOGLE_BUSY_STATUSES = {429, 500, 502, 503, 504}

_handler: logging.Handler | None = None
_handler_lock = threading.Lock()
_gateway_logger = logging.getLogger("socio.errors")


def log_path() -> Path:
    return Path(os.getenv("AGENT_LOG_DIR") or PROJECT_ROOT / "logs") / "agent.log"


def _file_handler() -> logging.Handler:
    """One handler per process for logs/agent.log, shared by the root logger and the gateway."""

    global _handler
    with _handler_lock:
        if _handler is None:
            path = log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            _handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8", delay=True)
            _handler.setFormatter(logging.Formatter(LOG_FORMAT))
            _handler.setLevel(logging.INFO)
        return _handler


def _ensure_gateway_logger() -> logging.Logger:
    handler = _file_handler()
    if handler not in _gateway_logger.handlers:
        _gateway_logger.addHandler(handler)
        _gateway_logger.setLevel(logging.INFO)
        _gateway_logger.propagate = False  # the root logger may share the same handler
    return _gateway_logger


def setup_logging(console_level: int = logging.WARNING) -> None:
    """Console output for the developer plus warnings and errors from every module in logs/agent.log."""

    logging.basicConfig(level=console_level, format=LOG_FORMAT)
    root = logging.getLogger()
    handler = _file_handler()
    if handler not in root.handlers:
        root.addHandler(handler)
    _ensure_gateway_logger()


# ----- classification -----------------------------------------------------------------


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _http_status(exc: BaseException) -> int | None:
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def classify(exc: BaseException) -> str:
    """'google_busy', 'system_busy', 'ai_timeout', 'ai_quota', or 'generic'."""

    from app.agent.llm import LLMUnavailableError
    from app.google_api import GoogleApiUnavailableError

    try:
        import httpx
        ai_timeouts: tuple[type, ...] = (httpx.TimeoutException,)
    except ImportError:  # pragma: no cover
        ai_timeouts = ()

    try:
        from googleapiclient.errors import HttpError
    except ImportError:  # pragma: no cover
        HttpError = ()  # type: ignore[assignment]

    try:
        import httplib2
        from google.auth.exceptions import TransportError
        google_network: tuple[type, ...] = (TransportError, httplib2.ServerNotFoundError)
    except ImportError:  # pragma: no cover
        google_network = ()

    for error in _chain(exc):
        if isinstance(error, LLMUnavailableError):
            return "ai_quota" if "quota" in str(error).lower() else "ai_timeout"
        if ai_timeouts and isinstance(error, ai_timeouts):
            return "ai_timeout"
        if isinstance(error, GoogleApiUnavailableError):
            return "google_busy"
        if HttpError and isinstance(error, HttpError):
            return "google_busy" if _http_status(error) in GOOGLE_BUSY_STATUSES else "generic"
        if google_network and isinstance(error, google_network):
            return "google_busy"
        if isinstance(error, sqlite3.OperationalError) and re.search(r"locked|busy", str(error), re.IGNORECASE):
            return "system_busy"
    return "generic"


MESSAGES = {
    "google_busy": GOOGLE_BUSY,
    "system_busy": SYSTEM_BUSY,
    "ai_timeout": AI_SLOW,
    "ai_quota": AI_QUOTA,
    "generic": GENERIC,
}


def user_message(exc: BaseException) -> str:
    """The predefined owner-facing text for an exception (with the ⚠️ prefix)."""

    return MESSAGES[classify(exc)]


def plain_message(exc: BaseException) -> str:
    """user_message() without the ⚠️ prefix, for text that already has its own marker."""

    return user_message(exc).removeprefix("⚠️").strip()


# ----- reporting ------------------------------------------------------------------------


def _context_text(context: dict[str, Any]) -> str:
    parts = []
    for key, value in context.items():
        text = " ".join(str(value).split())
        parts.append(f"{key}={text[:200]}")
    return " ".join(parts)


def report_failure(exc: BaseException, where: str, **context: Any) -> str:
    """Log the full stack trace with context to logs/agent.log; return the owner-facing text."""

    try:
        logger = _ensure_gateway_logger()
        logger.error(
            "%s failed (%s)%s", where, classify(exc), f" | {_context_text(context)}" if context else "",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    except Exception:  # logging must never turn one failure into two
        logging.getLogger(__name__).exception("Could not write to %s", log_path())
    return user_message(exc)


def is_validation_error(exc: BaseException) -> bool:
    """
    Our own refusals: ValueError and subclasses (SkillError, PlanError) raised with a
    message written for people ("that slot overlaps 'Dentist'"). Python's decoding
    errors are ValueErrors too, but their text is not meant for the owner.
    """

    return (
        isinstance(exc, ValueError)
        and not isinstance(exc, (UnicodeError, json.JSONDecodeError))
        and classify(exc) == "generic"
    )


def safe_error_text(exc: BaseException) -> str:
    """
    Why something failed, safe to show the owner (or to the agent model, whose reply
    the owner reads).

    Validation errors keep their message, which is needed to act on them; anything
    else (Google, network, SQLite, bugs) becomes the predefined text without the ⚠️.
    The full error is in the log either way.
    """

    if is_validation_error(exc):
        return " ".join(str(exc).split())
    return plain_message(exc)


# ----- last line of defence for outgoing chat text ---------------------------------------

_TRACEBACK_LINE = re.compile(r'^\s*(Traceback \(most recent call last\):|File "[^"]+", line \d+)')


def scrub(text: str) -> str:
    """Drop traceback lines from text bound for the owner's chat (never expected; defensive)."""

    lines = [line for line in (text or "").splitlines() if not _TRACEBACK_LINE.match(line)]
    return "\n".join(lines)
