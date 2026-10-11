"""
Runs due routines (called by the Telegram bot's loop once a minute).

For each due job:
1. Claim it: move next_run_at to the following slot (or deactivate a one-off) BEFORE
   running, with a conditional UPDATE. A crash mid-run or a second process can never
   run the same slot twice; the price is that a failed run is not retried (the owner
   is told it failed, and the next slot runs as usual).
2. Run prompt_query like an owner instruction (callback `run`), with no chat history:
   a routine stands on its own. Anything consequential it proposes still becomes an
   approval card.
3. Hand the result to `deliver` (the bot formats and sends it).

Missed runs (the PC was off at 9:00): a job that is late by at most GRACE runs once
when the bot is back; later than that it is skipped to its next slot and the owner is
told it was missed. Several missed days never pile up into several runs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from app.agent.state import AgentStateStore
from app.errors import report_failure
from app.scheduler.jobs import _ROUTINE_RUNNING, following_run, now_utc, parse_utc, utc_text


logger = logging.getLogger(__name__)

TICK_SECONDS = 60


def grace() -> timedelta:
    try:
        hours = float(os.getenv("AGENT_SCHEDULE_GRACE_HOURS", "6"))
    except ValueError:
        hours = 6.0
    return timedelta(hours=max(0.0, hours))


@dataclass
class RoutineRun:
    job: dict[str, Any]
    status: str                 # "ok", "failed", or "missed"
    result: Any = None          # what `run` returned (status "ok")
    error_html: str = ""        # the gateway's status card (status "failed"); never a trace


def run_due_jobs(
    run: Callable[[dict[str, Any]], Any],
    deliver: Callable[[RoutineRun], None],
    *,
    store: AgentStateStore | None = None,
    now: datetime | None = None,
) -> list[RoutineRun]:
    """Run every due job once. `run(job)` does its prompt_query; `deliver(outcome)` tells the owner."""

    store = store or AgentStateStore()
    now = now or now_utc()
    outcomes = []

    for job in store.due_scheduled_jobs(utc_text(now)):
        late = now - parse_utc(job["next_run_at"])
        next_run = following_run(job, now)
        missed = late > grace()
        status = "missed" if missed else "running"
        if not store.claim_scheduled_job(job["id"], job["next_run_at"], utc_text(now), next_run, status):
            continue  # another process took this slot, or it was cancelled meanwhile
        job = {**job, "next_run_at": next_run or job["next_run_at"], "is_active": int(next_run is not None)}

        if missed:
            logger.warning("Routine %s skipped: its slot was %s ago (the bot was not running).", job["id"], late)
            outcome = RoutineRun(job, "missed")
        else:
            _ROUTINE_RUNNING.set()
            try:
                result = run(job)
            except Exception as exc:
                card = report_failure(exc, "scheduler.run_job", html=True, job_id=job["id"])
                store.set_scheduled_job_status(job["id"], "failed")
                outcome = RoutineRun(job, "failed", error_html=card)
            else:
                store.set_scheduled_job_status(job["id"], "ok")
                outcome = RoutineRun(job, "ok", result=result)
            finally:
                _ROUTINE_RUNNING.clear()

        outcomes.append(outcome)
        try:
            deliver(outcome)
        except Exception as exc:
            report_failure(exc, "scheduler.deliver", job_id=job["id"])

    return outcomes
