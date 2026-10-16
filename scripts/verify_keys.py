"""
Check each model API key with one tiny request ("1+1=", a few output tokens).

    .venv\\Scripts\\python scripts\\verify_keys.py

Uses the same models and settings as Socio (llm_router), so a wrong model name
shows up here too. Exit code 1 if any key is missing, rejected, or unreachable.
A key that is only rate-limited right now is valid (Google, Mistral and Groq
check the key before the quota) and does not fail the check.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import llm_router as lr  # noqa: E402

PROMPT = [{"role": "user", "content": "1+1="}]


def probe(provider: lr.Provider) -> bool:
    key = provider.api_key()
    if not key:
        print(f"[FAIL] {provider.name} - {provider.key_env} is not set in .env")
        return False

    # Smallest useful request, no retries. gpt-oss reasons before answering, so it
    # gets low effort and a little more room (Gemini 3 rejects thinking_budget=0).
    limits = {
        "gemini": {"max_output_tokens": 5},
        "mistral": {"max_tokens": 5},
        "groq": {"max_tokens": 32, "reasoning_effort": "low"},
    }[provider.kind]
    model = lr.build_chat_model(provider, key, rate_limited=False, **limits)
    try:
        reply = model.invoke(PROMPT)
    except Exception as exc:
        kind = lr.failure_kind(exc)
        if kind in {"rate", "daily"}:
            print(f"[OK] {provider.name} - Auth Verified (rate-limited right now: {'daily' if kind == 'daily' else 'per-minute'} quota; no tokens consumed)")
            return True
        reason = {
            "rejected": "key or model rejected (check the key, and that the model exists)",
            "network": "could not reach the provider",
            "unavailable": "provider temporarily unavailable",
        }.get(kind, "request failed")
        print(f"[FAIL] {provider.name} - {reason}: {type(exc).__name__}: {str(exc)[:200]}")
        return False

    usage = getattr(reply, "usage_metadata", None) or {}
    tokens = usage.get("total_tokens")
    consumed = f"consumed {tokens} tokens" if tokens is not None else "token count not reported"
    print(f"[OK] {provider.name} - Auth Verified ({consumed}; model {provider.model}; answer {reply.text.strip()!r})")
    return True


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results = [probe(provider) for provider in (lr.GEMINI_PRIMARY, lr.GEMINI_FALLBACK, lr.MISTRAL, lr.GROQ)]
    # Optional third Gemini key: checked only when it is set.
    if lr.GEMINI_FALLBACK_2.api_key():
        results.append(probe(lr.GEMINI_FALLBACK_2))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
