"""
Which model answers a call, and where the call goes when that model cannot.

Two chains:
- supervisor (email handling, the owner agent, the planner, triage, rules,
  skills): Gemini primary key -> Gemini fallback key -> Mistral.
- worker (supervisor subtasks): Mistral -> Groq -> Gemini fallback key ->
  Gemini primary key, so parallel subtasks do not spend the supervisor's
  Gemini quota (the primary key is their last resort).

When a provider answers 429 (rate limit / quota) or 503, or cannot be reached,
the same request goes to the next provider at once and the provider is skipped
for a cool-down. A model call has no side effects, so repeating it elsewhere is
safe. Providers without a key in .env are left out of the chain.
"""

import asyncio
import logging
import os
import re
import socket
import threading
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import httpx
from dotenv import load_dotenv
from langchain_core.rate_limiters import BaseRateLimiter

from app import observability


load_dotenv()

logger = logging.getLogger(__name__)

SUPERVISOR = "supervisor"
WORKER = "worker"

GEMINI_MODEL = os.getenv("AGENT_GEMINI_MODEL", "gemini-3.5-flash-lite")
# The third key may use another Gemini model: a 503 "high demand" is per model, and on 2026-10-16
# both keys on gemini-3.5-flash-lite failed together while other Gemini models answered.
GEMINI_FALLBACK_2_MODEL = os.getenv("AGENT_GEMINI_FALLBACK_2_MODEL", GEMINI_MODEL)
GROQ_MODEL = os.getenv("AGENT_GROQ_MODEL", "openai/gpt-oss-120b")
# mistral-small/medium get 0 requests/minute on the free tier; ministral-14b gets 30.
MISTRAL_MODEL = os.getenv("AGENT_MISTRAL_MODEL", "ministral-14b-latest")

LLM_TIMEOUT_SECONDS = 60

# When every provider is cooling down, wait for the first one to come back
# (bounded), at most this many times per call.
RATE_LIMIT_MAX_WAITS = 3
RATE_LIMIT_MAX_WAIT_SECONDS = 70

# How long a provider is skipped after each kind of failure. Request-specific
# failures (too large for Groq's per-minute token budget, a malformed tool call)
# only move this one request on.
COOLDOWN_SECONDS = {
    "daily": 3600,       # retried hourly; a 429 costs no quota
    "rejected": 3600,    # key or model refused (401/403/404)
    "unavailable": 10,
    "network": 10,
    "too_large": 0,
    "bad_tool_call": 0,
    "bad_output": 0,
}

FAILOVER_REASONS = {
    "daily": "Daily quota exhausted",
    "rate": "Rate limit hit",
    "unavailable": "Service unavailable",
    "network": "Network error",
    "rejected": "Key or model rejected",
    "too_large": "Request too large",
    "bad_tool_call": "Malformed tool call",
    "bad_output": "Unreadable answer",
}


class LLMUnavailableError(RuntimeError):
    """Raised when no model can be used right now (quota exhausted or unreachable)."""


class UnreadableAnswer(ValueError):
    """A provider answered, but not in the structured form asked for (None from the parser)."""


class DeadlineExceeded(LLMUnavailableError):
    """The caller's time budget (deadline()) ran out before a model could answer."""


# The caller's wall-clock budget as a time.monotonic() value (a supervisor worker's 180 s).
# route() starts no attempt and no wait past it, so a network outage cannot hold a worker
# for many minutes of failovers and cool-down sleeps (audit H3). A call already sent keeps
# its own 60 s timeout.
_DEADLINE: ContextVar[float | None] = ContextVar("socio_llm_deadline", default=None)


@contextmanager
def deadline(at: float | None) -> Iterator[None]:
    """Model calls inside the block give up at `at` (time.monotonic()); None = no budget."""

    token = _DEADLINE.set(at)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def _time_left() -> float | None:
    at = _DEADLINE.get()
    return None if at is None else at - time.monotonic()


class SlidingWindowRateLimiter(BaseRateLimiter):
    """
    At most `per_minute` requests in any 60 seconds, shared by every thread.

    Supervisor workers call the model in parallel; without this they would burst
    past the free-tier limit and spend their time on 429s instead.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._sent: deque[float] = deque()
        self._lock = threading.Lock()

    def _try(self) -> float:
        """Take a slot and return 0, or return how long to wait for the next one."""

        with self._lock:
            now = time.monotonic()
            while self._sent and now - self._sent[0] >= 60:
                self._sent.popleft()
            if len(self._sent) < self.per_minute:
                self._sent.append(now)
                return 0.0
            return 60 - (now - self._sent[0]) + 0.05

    def acquire(self, *, blocking: bool = True) -> bool:
        while (wait := self._try()) > 0:
            if not blocking:
                return False
            time.sleep(wait)
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        while (wait := self._try()) > 0:
            if not blocking:
                return False
            await asyncio.sleep(wait)
        return True


@dataclass(frozen=True)
class Provider:
    name: str
    kind: str          # "gemini", "groq" or "mistral"
    key_env: str
    model: str
    rpm_env: str
    default_rpm: int
    optional: bool = False  # no "is not set" warning when its key is missing

    def api_key(self) -> str:
        return os.getenv(self.key_env, "").strip()


# Free tiers: Gemini 15 requests/minute per project, Groq 30, ministral-14b 30.
# Keep some in reserve; 0 turns a limiter off. Each key has its own limiter.
GEMINI_PRIMARY = Provider("Gemini (primary key)", "gemini", "GEMINI_API_KEY", GEMINI_MODEL, "AGENT_LLM_RPM", 14)
GEMINI_FALLBACK = Provider("Gemini (fallback key)", "gemini", "GEMINI_FALLBACK_API_KEY", GEMINI_MODEL, "AGENT_LLM_RPM", 14)
GEMINI_FALLBACK_2 = Provider(
    "Gemini (fallback key 2)", "gemini", "GEMINI_FALLBACK_2_API_KEY", GEMINI_FALLBACK_2_MODEL, "AGENT_LLM_RPM", 14,
    optional=True,
)
GROQ = Provider("Groq", "groq", "GROQ_API_KEY", GROQ_MODEL, "AGENT_GROQ_RPM", 28)
MISTRAL = Provider("Mistral", "mistral", "MISTRAL_API_KEY", MISTRAL_MODEL, "AGENT_MISTRAL_RPM", 28)

# A provider without a key is left out (configured_chain), so the third key is optional.
CHAINS: dict[str, tuple[Provider, ...]] = {
    SUPERVISOR: (GEMINI_PRIMARY, GEMINI_FALLBACK, GEMINI_FALLBACK_2, MISTRAL),
    WORKER: (MISTRAL, GROQ, GEMINI_FALLBACK_2, GEMINI_FALLBACK, GEMINI_PRIMARY),
}


# ----- model construction --------------------------------------------------------------

_LIMITERS: dict[str, SlidingWindowRateLimiter | None] = {}
_MODELS: dict[tuple[str, str, str], Any] = {}
_BUILD_LOCK = threading.Lock()
_warned_missing: set[str] = set()


def _limiter(provider: Provider) -> SlidingWindowRateLimiter | None:
    if provider.name not in _LIMITERS:
        rpm = int(os.getenv(provider.rpm_env, str(provider.default_rpm)))
        _LIMITERS[provider.name] = SlidingWindowRateLimiter(rpm) if rpm > 0 else None
    return _LIMITERS[provider.name]


def build_chat_model(provider: Provider, api_key: str, *, rate_limited: bool = True, **overrides):
    """A chat model for one provider that makes exactly one attempt per call (failover is ours)."""

    limiter = _limiter(provider) if rate_limited else None
    if provider.kind == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=provider.model,
            temperature=0,
            google_api_key=api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            # Becomes the SDK's attempts=1: one try, no internal 429 back-off.
            max_retries=1,
            rate_limiter=limiter,
            **overrides,
        )
    if provider.kind == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=provider.model,
            temperature=0,
            api_key=api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,
            rate_limiter=limiter,
            **overrides,
        )
    if provider.kind == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=provider.model,
            temperature=0,
            api_key=api_key,
            timeout=LLM_TIMEOUT_SECONDS,
            # tenacity stop_after_attempt(1): one try (it only retries network errors anyway).
            max_retries=1,
            rate_limiter=limiter,
            **overrides,
        )
    raise ValueError(f"Unknown model provider kind: {provider.kind}")


def _base_model(provider: Provider):
    """One shared model per provider and key (thread-safe; LangChain models are)."""

    key = provider.api_key()
    cache_key = (provider.name, provider.model, key)
    with _BUILD_LOCK:
        if cache_key not in _MODELS:
            _MODELS[cache_key] = build_chat_model(provider, key)
        return _MODELS[cache_key]


def configured_chain(role: str) -> list[Provider]:
    """The role's providers that have a key, in priority order."""

    if role not in CHAINS:
        raise ValueError(f"Unknown model role: {role}")
    chain = []
    for provider in CHAINS[role]:
        if provider.api_key():
            chain.append(provider)
        elif not provider.optional and provider.key_env not in _warned_missing:
            _warned_missing.add(provider.key_env)
            logger.warning("[ModelRouter] %s is not set; %s is left out of the fallback chain.", provider.key_env, provider.name)
    if not chain:
        raise ValueError("No model API key is configured: set GEMINI_API_KEY (and optionally GEMINI_FALLBACK_API_KEY, GEMINI_FALLBACK_2_API_KEY, MISTRAL_API_KEY, GROQ_API_KEY) in the .env file.")
    return chain


# ----- failure classification ----------------------------------------------------------


def _error_chain(exc: BaseException | None):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _status(error: BaseException) -> int | None:
    # Mistral raises httpx.HTTPStatusError, whose status lives on .response.
    for value in (getattr(error, "status_code", None), getattr(error, "code", None),
                  getattr(getattr(error, "response", None), "status_code", None)):
        if isinstance(value, int) and 100 <= value < 600:
            return value
    return None


_DAILY = re.compile(r"PerDay|per day|\(RPD\)|\(TPD\)", re.IGNORECASE)


def failure_kind(exc: BaseException) -> str | None:
    """Why a provider failed in a way another provider can fix, or None (a real error)."""

    from google.api_core import exceptions as api_core
    from langchain_core.exceptions import ModelConnectionError, ModelRateLimitError, ModelTimeoutError, OutputParserException

    try:
        import groq
        groq_rate: tuple[type, ...] = (groq.RateLimitError,)
        groq_network: tuple[type, ...] = (groq.APIConnectionError,)
    except ImportError:  # pragma: no cover
        groq_rate = groq_network = ()

    text = " ".join(str(error) for error in _error_chain(exc))
    statuses = {status for error in _error_chain(exc) if (status := _status(error)) is not None}
    errors = list(_error_chain(exc))

    rate_limited = (
        429 in statuses
        or any(isinstance(error, (api_core.ResourceExhausted, ModelRateLimitError, *groq_rate)) for error in errors)
        or "RESOURCE_EXHAUSTED" in text
    )
    if rate_limited:
        return "daily" if _DAILY.search(text) else "rate"
    if 413 in statuses:
        return "too_large"
    if statuses & {500, 502, 503, 504} or any(isinstance(error, (api_core.ServiceUnavailable, api_core.InternalServerError)) for error in errors):
        return "unavailable"
    if statuses & {401, 403, 404}:
        return "rejected"
    if 400 in statuses and "tool_use_failed" in text:
        return "bad_tool_call"
    # An answer the structured parser could not read: another provider may well answer it
    # (audit M7). A 422 / 400 is not: it means our request is wrong and must surface (test A8).
    if any(isinstance(error, (OutputParserException, UnreadableAnswer)) for error in errors):
        return "bad_output"
    if any(isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError, socket.gaierror, ModelConnectionError, ModelTimeoutError, *groq_network)) for error in errors):
        return "network"
    return None


def _suggested_delay(exc: BaseException) -> float:
    """Seconds the provider asked us to wait (Groq: "try again in 1m2.5s", Gemini: "retry in 41s")."""

    text = " ".join(str(error) for error in _error_chain(exc))
    if match := re.search(r"try again in (?:(\d+)m)?([\d.]+)s", text):
        delay = float(match.group(1) or 0) * 60 + float(match.group(2))
    elif match := (re.search(r"retry in ([\d.]+)s", text) or re.search(r"retryDelay'?\"?:\s*'?\"?([\d.]+)s", text)):
        delay = float(match.group(1))
    else:
        delay = 29.0
    return min(delay + 1, RATE_LIMIT_MAX_WAIT_SECONDS)


# ----- cool-downs ------------------------------------------------------------------------

_cooldowns: dict[str, tuple[float, str]] = {}
_COOLDOWN_LOCK = threading.Lock()


def _cooling(provider: Provider, now: float) -> tuple[float, str] | None:
    with _COOLDOWN_LOCK:
        entry = _cooldowns.get(provider.name)
        if entry and entry[0] > now:
            return entry
        _cooldowns.pop(provider.name, None)
        return None


def _cool_down(provider: Provider, kind: str, exc: BaseException) -> None:
    seconds = _suggested_delay(exc) if kind == "rate" else COOLDOWN_SECONDS[kind]
    if seconds > 0:
        with _COOLDOWN_LOCK:
            _cooldowns[provider.name] = (time.monotonic() + seconds, kind)


def reset_cooldowns() -> None:
    with _COOLDOWN_LOCK:
        _cooldowns.clear()


# ----- the routed model ------------------------------------------------------------------


@dataclass
class RoutedModel:
    """
    A chat model for one role that tries each configured provider in order.

    Supports what Socio uses: `bind_tools`, `with_structured_output`, `invoke`.
    Each provider gets the same transforms, so a failover sees the same tools
    and output schema.
    """

    role: str
    chain: list[Provider]
    transforms: list[Callable[[Any], Any]] = field(default_factory=list)
    # with_structured_output: an empty (None) answer is unreadable, so the next provider is tried.
    structured: bool = False
    _runnables: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Built up front so a tool schema one provider cannot take fails here, not mid-failover.
        for provider in self.chain:
            runnable = _base_model(provider)
            for transform in self.transforms:
                runnable = transform(runnable)
            self._runnables[provider.name] = runnable

    def _with(self, transform: Callable[[Any], Any]) -> "RoutedModel":
        return RoutedModel(self.role, self.chain, [*self.transforms, transform], self.structured)

    def bind_tools(self, tools, **kwargs) -> "RoutedModel":
        return self._with(lambda model: model.bind_tools(tools, **kwargs))

    def with_structured_output(self, schema, **kwargs) -> "RoutedModel":
        routed = self._with(lambda model: model.with_structured_output(schema, **kwargs))
        return RoutedModel(routed.role, routed.chain, routed.transforms, True)

    def runnable_for(self, provider: Provider):
        return self._runnables[provider.name]

    def invoke(self, model_input, config=None, **kwargs):
        return route(self.chain, self.runnable_for, model_input, role=self.role, config=config,
                     reject_none=self.structured, **kwargs)


def _next_available(chain: list[Provider], after: int, skipped: set[str], now: float) -> Provider | None:
    for provider in chain[after + 1:]:
        if provider.name not in skipped and _cooling(provider, now) is None:
            return provider
    return None


def _label(role: str, provider: Provider, *, lead: bool) -> str:
    """'Worker priority 1 (Mistral)' / 'priority 2 (Groq)'; just the name outside a role chain."""

    if role not in CHAINS or provider not in CHAINS[role]:
        return f"provider {provider.name}" if lead else provider.name
    priority = f"priority {_priority(role, provider)} ({provider.name})"
    return f"{role.capitalize()} {priority}" if lead else priority


def _priority(role: str, provider: Provider) -> int:
    """Position among the providers that have a key (an unset optional key takes no number)."""

    return [p for p in CHAINS[role] if p is provider or p.api_key()].index(provider) + 1


def route(chain: list[Provider], runnable_for: Callable[[Provider], Any], model_input, *, role: str = "",
          reject_none: bool = False, **kwargs):
    """
    Invoke the first provider that answers.

    Raises LLMUnavailableError when every provider is out of quota or unreachable
    (after waiting for a short cool-down, bounded), so callers can stop a batch
    instead of failing every remaining email. Other errors propagate unchanged.
    """

    skipped: set[str] = set()     # failed for this request only
    last_kind = ""
    # Each attempt is its own Langfuse generation (a failed one is ERROR); the metadata
    # links a failover attempt to the provider it replaced.
    traced = role in CHAINS
    config = kwargs.pop("config", None)
    attempt = 0
    failed_over_from: tuple[str, str] | None = None

    for _round in range(RATE_LIMIT_MAX_WAITS + 1):
        for index, provider in enumerate(chain):
            if provider.name in skipped or _cooling(provider, time.monotonic()):
                continue
            left = _time_left()
            if left is not None and left <= 0:
                raise DeadlineExceeded("The time budget ran out before an AI model answered.")
            attempt += 1
            call_kwargs = dict(kwargs)
            if traced:
                call_kwargs["config"] = observability.llm_config(config, run_name=f"{role}_llm", metadata={
                    "router_role": role,
                    "provider": provider.name,
                    "priority": _priority(role, provider) if provider in CHAINS[role] else None,
                    "attempt": attempt,
                    **({"failover_from": failed_over_from[0], "failover_reason": failed_over_from[1]} if failed_over_from else {}),
                })
            elif config is not None:
                call_kwargs["config"] = config
            try:
                answer = runnable_for(provider).invoke(model_input, **call_kwargs)
                if reject_none and answer is None:
                    raise UnreadableAnswer(f"{provider.name} returned no structured answer.")
                return answer
            except Exception as exc:
                kind = failure_kind(exc)
                if kind is None:
                    raise
                last_kind = kind
                failed_over_from = (provider.name, FAILOVER_REASONS[kind])
                _cool_down(provider, kind, exc)
                if COOLDOWN_SECONDS.get(kind) == 0:
                    skipped.add(provider.name)
                successor = _next_available(chain, index, skipped, time.monotonic())
                if traced:
                    observability.current_event(
                        "model_failover", level="WARNING",
                        status_message=f"{FAILOVER_REASONS[kind]} on {provider.name}",
                        metadata={"router_role": role, "from": provider.name, "to": successor.name if successor else None,
                                  "reason": kind, "status": observability.error_text(exc)},
                    )
                # Message only: no stack trace, no provider error text (agent.log stays readable).
                if successor is not None:
                    logger.warning("⚠️ [ModelRouter] %s on %s. Seamlessly failing over to %s.",
                                   FAILOVER_REASONS[kind], _label(role, provider, lead=True), _label(role, successor, lead=False))
                else:
                    logger.warning("⚠️ [ModelRouter] %s on %s. No other provider is available right now.",
                                   FAILOVER_REASONS[kind], _label(role, provider, lead=True))

        # Every provider failed or is cooling down: wait for the first to come back, if soon.
        now = time.monotonic()
        waits = []
        kinds = []
        for provider in chain:
            if provider.name in skipped:
                continue
            entry = _cooling(provider, now)
            if entry is None:
                waits.append(0.0)
            else:
                waits.append(entry[0] - now)
                kinds.append(entry[1])
        if not waits:
            break
        wait = min(waits)
        if wait > RATE_LIMIT_MAX_WAIT_SECONDS:
            # Quota gone (a rejected key cannot help either): the owner sees the daily-limit card.
            if "daily" in kinds and set(kinds) <= {"daily", "rejected"}:
                raise LLMUnavailableError("AI daily quota exhausted on every configured provider.")
            break
        left = _time_left()
        if _round < RATE_LIMIT_MAX_WAITS and wait > 0 and left is not None and wait >= left:
            raise DeadlineExceeded("The time budget ran out while every AI provider was busy.")
        if _round < RATE_LIMIT_MAX_WAITS and wait > 0:
            logger.warning("[ModelRouter] Every provider is busy; waiting %.0fs.", wait)
            time.sleep(wait)

    if traced:
        observability.current_event("model_router_exhausted", level="ERROR",
                                    status_message=f"No provider could answer (last: {last_kind or 'all cooling down'})",
                                    metadata={"router_role": role, "attempts": attempt})
    if reject_none and last_kind == "bad_output":
        return None  # nobody gave a readable answer: callers handle None as they always did
    if last_kind == "network":
        raise LLMUnavailableError("Could not reach any AI provider.")
    raise LLMUnavailableError("No AI provider could answer (rate limits or outages persisted).")
