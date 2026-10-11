"""
The models Socio talks to. Routing and failover live in `llm_router.py`.

`get_llm(role)` returns a model for the supervisor chain (Gemini first) or the
worker chain (Mistral first); `invoke_llm` calls it, failing over between
providers and raising LLMUnavailableError when none can answer.
"""

from app.agent.llm_router import (  # noqa: F401  (re-exported for callers and tests)
    LLM_TIMEOUT_SECONDS,
    RATE_LIMIT_MAX_WAIT_SECONDS,
    RATE_LIMIT_MAX_WAITS,
    SUPERVISOR,
    WORKER,
    LLMUnavailableError,
    Provider,
    RoutedModel,
    SlidingWindowRateLimiter,
    configured_chain,
    failure_kind,
    route,
)


def get_llm(role: str = SUPERVISOR) -> RoutedModel:
    """
    The chat model for a role: SUPERVISOR (Gemini primary -> Gemini fallback ->
    Mistral) or WORKER (Mistral -> Groq -> Gemini fallback -> Gemini primary).
    """

    return RoutedModel(role, configured_chain(role))


_SINGLE = Provider("model", "", "", "", "", 0)


def invoke_llm(runnable, model_input):
    """
    Invoke an LLM runnable with failover.

    A routed model (from get_llm) tries its providers in order. Any other
    runnable (tests use fakes) gets the same handling as a one-provider chain.
    Raises LLMUnavailableError when quota is exhausted or the model cannot be
    reached, so callers can stop a batch instead of failing every remaining email.
    """

    if isinstance(runnable, RoutedModel):
        return runnable.invoke(model_input)
    return route([_SINGLE], lambda _provider: runnable, model_input)
