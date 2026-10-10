import asyncio
import logging
import os
import re
import threading
import time
from collections import deque

import httpx
from dotenv import load_dotenv
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 3

# Free-tier Gemini allows a small number of requests per minute. When the
# per-minute quota is hit, wait for the delay Google suggests (bounded).
RATE_LIMIT_MAX_WAITS = 3
RATE_LIMIT_MAX_WAIT_SECONDS = 70
NETWORK_RETRY_WAIT_SECONDS = 5


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be used right now (quota exhausted or unreachable)."""


class SlidingWindowRateLimiter(BaseRateLimiter):
    """
    At most `per_minute` Gemini requests in any 60 seconds, shared by every thread.

    Supervisor workers call Gemini in parallel; without this they would burst past
    the free-tier limit and spend their time waiting out 429s instead.
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


# Free tier: 15 requests/minute; keep one in reserve. AGENT_LLM_RPM=0 turns it off.
LLM_REQUESTS_PER_MINUTE = int(os.getenv("AGENT_LLM_RPM", "14"))
RATE_LIMITER = SlidingWindowRateLimiter(LLM_REQUESTS_PER_MINUTE) if LLM_REQUESTS_PER_MINUTE > 0 else None


def get_llm():
    """
    Create and return the Gemini LLM used by the agent.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured in the .env file."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=api_key,
        # Bound each call so a network stall cannot hang the agent loop.
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
        # One limiter per process, shared by every model instance and thread.
        rate_limiter=RATE_LIMITER,
    )


def _error_chain(exc: BaseException):
    while exc is not None:
        yield exc
        exc = exc.__cause__ or exc.__context__


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or " 429 " in f" {text} "


def _is_daily_quota(exc: Exception) -> bool:
    return re.search(r"PerDay", str(exc)) is not None


def _is_network_error(exc: Exception) -> bool:
    return any(isinstance(error, (httpx.TransportError, TimeoutError, OSError)) for error in _error_chain(exc))


def _suggested_delay(exc: Exception) -> float:
    match = re.search(r"retry in ([\d.]+)s", str(exc)) or re.search(r"retryDelay'?:\s*'?(\d+)s", str(exc))
    delay = float(match.group(1)) + 1 if match else 30.0
    return min(delay, RATE_LIMIT_MAX_WAIT_SECONDS)


def invoke_llm(runnable, model_input):
    """
    Invoke an LLM runnable, waiting out per-minute rate limits.

    Raises LLMUnavailableError when the daily quota is exhausted, the
    rate limit persists, or Gemini cannot be reached, so callers can stop
    a batch instead of failing every remaining email.
    """

    network_retried = False

    for attempt in range(RATE_LIMIT_MAX_WAITS + 1):
        try:
            return runnable.invoke(model_input)
        except Exception as exc:
            if _is_rate_limit(exc):
                if _is_daily_quota(exc):
                    raise LLMUnavailableError(f"Gemini daily quota exhausted: {exc}") from exc
                if attempt == RATE_LIMIT_MAX_WAITS:
                    raise LLMUnavailableError(f"Gemini rate limit persisted after {RATE_LIMIT_MAX_WAITS} waits.") from exc

                delay = _suggested_delay(exc)
                logger.warning("Gemini rate limit hit; waiting %.0fs (wait %d/%d).", delay, attempt + 1, RATE_LIMIT_MAX_WAITS)
                time.sleep(delay)
                continue

            if _is_network_error(exc):
                if network_retried:
                    raise LLMUnavailableError(f"Could not reach Gemini: {type(exc).__name__}: {exc}") from exc
                network_retried = True
                logger.warning("Gemini network error (%s); retrying once.", type(exc).__name__)
                time.sleep(NETWORK_RETRY_WAIT_SECONDS)
                continue

            raise

    raise LLMUnavailableError("Gemini call failed after retries.")
