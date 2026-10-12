"""
Check the Langfuse setup end to end with one small test trace (no model call, no quota).

    .venv\\Scripts\\python scripts\\verify_langfuse.py

1. Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL from .env and
   checks the keys (an authenticated API request must answer HTTP 200).
2. Sends a trace through Socio's own tracing module (app.observability): a parent
   span, a child generation, and a tool span whose output carries a fake base64
   attachment (it must arrive stripped).
3. Flushes, then reads the trace back from the Langfuse API (ingestion takes a few
   seconds) and checks its observations. Prints the trace URL.

Exit code 1 if the keys are missing or rejected, Langfuse is unreachable, or the
trace did not arrive as sent.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "development")   # keep checks out of production views

from app import observability  # noqa: E402  (loads .env)

FAKE_ATTACHMENT = "QUJD" * 300   # 1,200 characters of base64
WAIT_SECONDS = 90


def main() -> int:
    client = observability._get_client()
    if client is None:
        print("[FAIL] Tracing is off: it needs LANGFUSE_TRACING_ENABLED=true plus LANGFUSE_PUBLIC_KEY "
              "and LANGFUSE_SECRET_KEY in .env (off by default: traces carry prompts and email text).")
        return 1

    try:
        client.auth_check()
        print(f"[OK]   Keys accepted by {client._base_url} (HTTP 200)")
    except Exception as exc:
        print(f"[FAIL] Langfuse refused or could not be reached: {observability.error_text(exc)}")
        return 1

    with observability.interaction(
        "verify_langfuse", session_id="verify-langfuse", user_id="developer",
        input={"check": "parent span, child generation, tool span"},
    ) as root:
        with observability.observation(
            "verify_generation", as_type="generation", model="socio-verify",
            input=[{"role": "user", "content": "1+1="}],
        ) as generation:
            generation.update(output="2", usage_details={"input": 4, "output": 1})
        with observability.observation("verify_tool", as_type="tool", input={"query": "test"}) as tool:
            tool.update(output={"attachment": FAKE_ATTACHMENT, "note": "the attachment above must be stripped"})
        root.update(output={"result": "sent"})
        trace_id = root.trace_id

    observability.flush()
    print(f"[OK]   Trace {trace_id} sent (flushed)")

    # Read it back: proves the trace was stored, not just sent. Organizations created
    # since 16 Sep 2026 only have the v2 observations API (GET /api/public/traces answers 410).
    expected = {"verify_langfuse", "verify_generation", "verify_tool"}
    deadline = time.monotonic() + WAIT_SECONDS
    last_error = ""
    observations = []
    while time.monotonic() < deadline:
        try:
            observations = client.api.observations.get_many(trace_id=trace_id, fields="core,basic,io", limit=50).data
            # Ingestion is eventually consistent: an observation can appear before its output.
            if expected <= {o.name for o in observations} and all(o.output for o in observations if o.name in expected):
                break
        except Exception as exc:
            last_error = observability.error_text(exc)
        time.sleep(3)   # ingestion is asynchronous: a few seconds
    else:
        print(f"[FAIL] The trace did not show up in Langfuse within {WAIT_SECONDS} s. {last_error}")
        return 1

    by_name = {o.name: o for o in observations}
    types = {name: str(by_name[name].type) for name in sorted(expected)}
    root_id = by_name["verify_langfuse"].id
    nested = all(by_name[name].parent_observation_id == root_id for name in ("verify_generation", "verify_tool"))
    tool_output = str(by_name["verify_tool"].output)
    stripped = FAKE_ATTACHMENT not in tool_output and "base64 removed" in tool_output
    root = by_name["verify_langfuse"]

    print(f"[{'OK' if nested else 'FAIL'}]   Read back (HTTP 200): {len(observations)} observations, "
          f"types {types}, children nested under the parent: {nested}")
    print(f"[{'OK' if stripped else 'FAIL'}]   Base64 attachment stripped before upload: {stripped}")
    print(f"       Session: {root.session_id}  User: {root.user_id}")
    print(f"       {observability.trace_url(trace_id) or '(trace URL unavailable)'}")
    return 0 if nested and stripped else 1


if __name__ == "__main__":
    sys.exit(main())
