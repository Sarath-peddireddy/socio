"""
Langfuse tracing (optional): one trace per owner message / email / routine.

    telegram_interaction (session = chat id)
      supervisor_plan                 the planner's model call, output = the task list
      worker_<domain>_<task_id>       one per worker thread (type agent)
        worker_llm                    a model call (generation: model, tokens, messages)
        worker_llm                    ...the next provider after a failover (ERROR before it)
        find_calendar_availability    a tool call (type tool), proposals and refusals too

Model calls are recorded by Langfuse's LangChain callback, attached per provider
attempt in llm_router.route as a "worker_llm" / "supervisor_llm" generation (a failed
attempt is ERROR, followed by a "model_failover" event and the next provider's
generation, whose metadata says failover_from). Everything else is
a span opened here. Nothing else creates generations, so no model call is counted twice.

Off unless asked for: tracing runs only with LANGFUSE_TRACING_ENABLED=true AND
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY set (keys alone do nothing). Off, or when the
SDK itself fails, every function here is a no-op and Socio runs exactly as before. Spans are exported in a background thread, so
an unreachable Langfuse never delays a reply; the bot does not flush per message.

Worker threads keep their parent: supervisor.run_plan starts each worker in
contextvars.copy_context(), which carries OpenTelemetry's current span.

What is recorded: prompts and tool results as the models saw them (email text
included, already wrapped as untrusted). Base64 blobs and data: URIs are replaced and
long strings clipped before anything leaves the machine (scrub()).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

# Longest string kept in a trace; a whole thread transcript fits, a raw attachment does not.
MAX_TEXT_CHARS = 20_000
_DATA_URI = re.compile(r"data:[\w/+.\-]+;base64,[A-Za-z0-9+/=\s]+")
# MIME / attachment payloads: long runs of base64 (URL-safe too) with no spaces.
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/_\-]{400,}={0,2}")

_client: Any = None
_state = "unset"            # "unset" -> "on" | "off"
_init_lock = threading.Lock()
_warned: set[str] = set()

# The trace's root observation, so deep code can add to it (intent, reply texts).
_root: ContextVar["Observation | None"] = ContextVar("socio_trace_root", default=None)


def _warn_once(key: str, message: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


# ----- privacy / size ------------------------------------------------------------------


def _scrub_text(text: str) -> str:
    text = _DATA_URI.sub(lambda m: f"[base64 data removed: {len(m.group(0))} chars]", text)
    text = _BASE64_RUN.sub(lambda m: f"[base64 removed: {len(m.group(0))} chars]", text)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + f"... [clipped: {len(text) - MAX_TEXT_CHARS} more chars]"
    return text


def scrub(data: Any, _depth: int = 0) -> Any:
    """Data safe to send: no base64 payloads or raw bytes, no huge strings."""

    if _depth > 20:
        return "[nested too deep]"
    if isinstance(data, str):
        return _scrub_text(data)
    if isinstance(data, (bytes, bytearray)):
        return f"[{len(data)} bytes removed]"
    if isinstance(data, dict):
        return {key: scrub(value, _depth + 1) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [scrub(value, _depth + 1) for value in data]
    return data


def _mask(*, data: Any, **_kwargs) -> Any:
    """Langfuse's mask hook: also covers what the LangChain callback records."""

    try:
        return scrub(data)
    except Exception:
        return "[not recorded]"


# ----- client ----------------------------------------------------------------------------


def enabled() -> bool:
    """True when traces are being sent (keys present, not switched off, SDK started)."""

    return _get_client() is not None


def _get_client():
    global _client, _state
    if _state != "unset":
        return _client
    with _init_lock:
        if _state != "unset":
            return _client
        _state = "off"
        # Opt-in (developer decision 2026-10-12: no personal data leaves the machine by
        # accident): keys alone are not enough, LANGFUSE_TRACING_ENABLED=true must be set too.
        if os.getenv("LANGFUSE_TRACING_ENABLED", "").strip().lower() != "true":
            return None
        if not (os.getenv("LANGFUSE_PUBLIC_KEY", "").strip() and os.getenv("LANGFUSE_SECRET_KEY", "").strip()):
            _warn_once("keys", "[Observability] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set; tracing is off.")
            return None
        # Socio's .env names the server LANGFUSE_BASE_URL; older setups use LANGFUSE_HOST.
        base_url = (os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "").strip() or None
        # Test runs set LANGFUSE_TRACING_ENVIRONMENT=development so they stay out of production views.
        environment = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "").strip() or "production"
        try:
            from langfuse import Langfuse

            _client = Langfuse(base_url=base_url, mask=_mask, environment=environment)
            _state = "on"
        except Exception as exc:
            _warn_once("init", "[Observability] Langfuse could not start (%s); tracing is off.", type(exc).__name__)
            _client = None
    return _client


def disable(reason: str) -> None:
    """Stop tracing for the rest of the process (e.g. the keys were rejected)."""

    global _state, _client
    with _init_lock:
        _state, _client = "off", None
    _warn_once("disabled", "[Observability] %s Tracing is off until restart.", reason)


def check_connection() -> bool | None:
    """
    One authenticated request to Langfuse. True = keys accepted, False = keys rejected
    (tracing is turned off), None = unreachable or not configured (traces are retried
    by the exporter; nothing else changes). Blocking: call it off the main path.
    """

    client = _get_client()
    if client is None:
        return None
    try:
        return bool(client.auth_check())
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status in {401, 403}:
            disable("Langfuse rejected the keys.")
            return False
        _warn_once("unreachable", "[Observability] Langfuse is not reachable right now (%s); Socio carries on.", type(exc).__name__)
        return None


def check_connection_in_background() -> None:
    threading.Thread(target=check_connection, name="langfuse-auth-check", daemon=True).start()


def flush() -> None:
    """Send what is buffered (blocks up to the exporter's timeout). For CLI runs and scripts."""

    client = _client if _state == "on" else None
    if client is not None:
        try:
            client.flush()
        except Exception as exc:
            _warn_once("flush", "[Observability] Could not send traces (%s).", type(exc).__name__)


def shutdown() -> None:
    client = _client if _state == "on" else None
    if client is not None:
        try:
            client.shutdown()
        except Exception as exc:
            _warn_once("shutdown", "[Observability] Could not send the last traces (%s).", type(exc).__name__)


def trace_url(trace_id: str | None = None) -> str | None:
    client = _get_client()
    if client is None:
        return None
    try:
        return client.get_trace_url(trace_id=trace_id)
    except Exception:
        return None


# ----- observations ----------------------------------------------------------------------


class Observation:
    """A Langfuse observation that never raises; the no-op one when tracing is off."""

    def __init__(self, span: Any = None):
        self._span = span
        self.replies: list[str] = []
        self.output_set = False

    @property
    def trace_id(self) -> str | None:
        return getattr(self._span, "trace_id", None)

    def update(self, **fields) -> None:
        if self._span is None:
            return
        if "output" in fields:
            self.output_set = True
        try:
            self._span.update(**_clean(fields))
        except Exception as exc:
            _warn_once("update", "[Observability] Could not update a trace (%s).", type(exc).__name__)

    def event(self, name: str, **fields) -> None:
        if self._span is None:
            return
        try:
            self._span.create_event(name=name, **_clean(fields))
        except Exception as exc:
            _warn_once("event", "[Observability] Could not record an event (%s).", type(exc).__name__)


NOOP = Observation()


def _clean(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop empty values; scrub what goes out."""

    cleaned = {}
    for key, value in fields.items():
        if value is None:
            continue
        cleaned[key] = scrub(value) if key in {"input", "output", "metadata"} else value
    return cleaned


def error_text(exc: BaseException) -> str:
    """Status message for an ERROR observation (developer-facing, kept short)."""

    detail = " ".join(str(exc).split())[:300]
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


@contextmanager
def observation(name: str, *, as_type: str = "span", **fields) -> Iterator[Observation]:
    """
    A child of the current observation (or a new trace), current for everything inside.
    An exception from the body marks it ERROR and propagates unchanged.
    """

    client = _get_client()
    if client is None:
        yield NOOP
        return
    try:
        manager = client.start_as_current_observation(name=name, as_type=as_type, **_clean(fields))
        span = manager.__enter__()
    except Exception as exc:
        _warn_once("start", "[Observability] Could not start a trace (%s).", type(exc).__name__)
        yield NOOP
        return

    current = Observation(span)
    try:
        yield current
    except BaseException as exc:
        current.update(level="ERROR", status_message=error_text(exc))
        raise
    finally:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:
            _warn_once("end", "[Observability] Could not close a trace (%s).", type(exc).__name__)


@contextmanager
def interaction(
    name: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    nest: bool = True,
    **fields,
) -> Iterator[Observation]:
    """
    The root of a trace: one owner message, one email, one routine. Inside another
    interaction it is a child span (an email handled during /run), or nothing at all
    with nest=False (the CLI's owner_command when the bot's trace is already open).
    Texts passed to record_reply() become its output unless one was set.
    """

    if _root.get() is not None:
        if not nest:
            yield NOOP
            return
        with observation(name, **fields) as child:
            yield child
        return

    with observation(name, **fields) as root:
        token = _root.set(root)
        try:
            with _propagated(trace_name=name, session_id=session_id, user_id=user_id, tags=tags):
                yield root
        finally:
            _root.reset(token)
            if root.replies and not root.output_set:
                root.update(output={"replies": root.replies})


@contextmanager
def _propagated(**attributes) -> Iterator[None]:
    """Trace name / session / user on every observation below (no-op if that fails)."""

    attributes = {key: value for key, value in attributes.items() if value}
    manager = None
    if _get_client() is not None and attributes:
        try:
            from langfuse import propagate_attributes

            manager = propagate_attributes(**attributes)
            manager.__enter__()
        except Exception as exc:
            _warn_once("propagate", "[Observability] Could not set the trace's session (%s).", type(exc).__name__)
            manager = None
    try:
        yield
    finally:
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                pass


def annotate_root(**metadata) -> None:
    """Add metadata to the trace's root (e.g. which route the request took)."""

    root = _root.get()
    if root is not None:
        root.update(metadata=metadata)


def current_trace_id() -> str | None:
    """The id of the trace being recorded now (None when tracing is off or outside one)."""

    root = _root.get()
    return root.trace_id if root is not None else None


def score_trace(trace_id: str | None, name: str, value: float, *, comment: str | None = None, metadata: dict | None = None) -> None:
    """
    A score on a trace, e.g. negative owner feedback found later (a Reject tap is its own trace,
    so the trace that proposed the card gets the score). No-op without a trace id or client.
    """

    client = _get_client()
    if client is None or not trace_id:
        return
    try:
        client.create_score(
            trace_id=trace_id, name=name, value=value, data_type="NUMERIC",
            comment=_scrub_text(comment)[:500] if comment else None, metadata=scrub(metadata) if metadata else None,
        )
    except Exception as exc:
        _warn_once("score", "[Observability] Could not record a score (%s).", type(exc).__name__)


def record_reply(text: str) -> None:
    """A message delivered to the owner: becomes the root's output."""

    root = _root.get()
    if root is not None and root is not NOOP:
        root.replies.append(text)


def current_event(name: str, **fields) -> None:
    """A point-in-time event under the current observation (e.g. a model failover)."""

    client = _get_client()
    if client is None:
        return
    try:
        client.create_event(name=name, **_clean(fields))
    except Exception as exc:
        _warn_once("event", "[Observability] Could not record an event (%s).", type(exc).__name__)


# ----- LangChain -------------------------------------------------------------------------


def llm_config(config: dict | None, *, run_name: str, metadata: dict[str, Any]) -> dict | None:
    """
    The config for one model attempt with Langfuse's callback added, so the call is a
    generation (model, tokens, messages, ERROR on failure) under the current observation.
    Unchanged when tracing is off.
    """

    if _get_client() is None:
        return config
    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()   # one per attempt: it keeps per-run state
    except Exception as exc:
        _warn_once("handler", "[Observability] Could not trace a model call (%s).", type(exc).__name__)
        return config
    merged = dict(config or {})
    callbacks = merged.get("callbacks")
    if callbacks is None:
        merged["callbacks"] = [handler]
    elif isinstance(callbacks, list):
        merged["callbacks"] = [*callbacks, handler]
    else:  # a CallbackManager: leave it alone rather than replace it
        return config
    merged["run_name"] = run_name    # stable name per role; the provider is metadata, the model an attribute
    merged["metadata"] = {**(merged.get("metadata") or {}), **metadata}
    return merged
